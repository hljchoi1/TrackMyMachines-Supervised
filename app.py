import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import joblib
import os
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CNC Maintenance Audit", layout="wide", page_icon="⚙️")

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    """Loads the trained Random Forest model."""
    model_path = 'models/rf_tool_wear_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

def parse_range(r):
    """Extracts start and end times from the timeline string."""
    matches = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', str(r))
    if len(matches) >= 2:
        return pd.to_datetime(matches[0]), pd.to_datetime(matches[1])
    return pd.NaT, pd.NaT

@st.cache_data
def process_data(df_current, df_timeline):
    """Engineers features to match the Random Forest training pipeline."""
    # Process Timelines
    df_timeline[['start_time', 'end_time']] = df_timeline['range'].apply(lambda x: pd.Series(parse_range(x)))
    df_timeline['is_loading'] = (df_timeline['activity_reason_id'] == 189.0).astype(int)
    df_timeline = df_timeline.dropna(subset=['start_time', 'end_time'])
    
    # Process Current
    df_current['created_at'] = pd.to_datetime(df_current['created_at'])
    df_current = df_current.rename(columns={'current1': 'avg_current'})
    df_current = df_current.sort_values('created_at')
    
    # Feature Engineering (Matching train_pipeline.py)
    df_current['rolling_baseline'] = df_current['avg_current'].rolling(window=150, min_periods=1).mean().shift(1).fillna(method='bfill')
    df_current['current_delta_pct'] = (df_current['avg_current'] - df_current['rolling_baseline']) / (df_current['rolling_baseline'] + 1e-9)
    df_current['current_lag_2'] = df_current['avg_current'].shift(2).fillna(method='bfill')
    df_current['current_lead_2'] = df_current['avg_current'].shift(-2).fillna(method='ffill')
    df_current['rolling_variance'] = df_current['avg_current'].rolling(window=10, min_periods=1).var().fillna(0)
    
    # Map ground truth manual logs to the high-frequency current data
    df_current['manual_log_event'] = 0
    for _, row in df_timeline[df_timeline['is_loading'] == 1].iterrows():
        mask = (df_current['created_at'] >= row['start_time']) & (df_current['created_at'] <= row['end_time'])
        df_current.loc[mask, 'manual_log_event'] = 1
        
    # Drop NAs to ensure model predict doesn't fail
    features = ['avg_current', 'rolling_baseline', 'current_delta_pct', 'current_lag_2', 'current_lead_2', 'rolling_variance']
    df_current = df_current.dropna(subset=features)
    
    return df_current, features

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ System Configuration")

st.sidebar.markdown("### 1. Load Data")
data_source = st.sidebar.radio("Select Data Source", ["Use Synthetic Data (GitHub Demo)", "Upload Original Data"])

df_current_raw, df_timeline_raw = None, None

if data_source == "Upload Original Data":
    st.sidebar.info("Upload your private Machine 290 logs.")
    file_curr = st.sidebar.file_uploader("Upload current_data.csv", type=['csv'])
    file_time = st.sidebar.file_uploader("Upload machine_timelines.csv", type=['csv'])
    if file_curr and file_time:
        df_current_raw = pd.read_csv(file_curr)
        df_timeline_raw = pd.read_csv(file_time)
elif data_source == "Use Synthetic Data (GitHub Demo)":
    try:
        df_current_raw = pd.read_csv('data/current_data.csv')
        df_timeline_raw = pd.read_csv('data/machine_timelines_v2.csv')
    except FileNotFoundError:
        st.sidebar.error("Synthetic data not found. Please run `data/generate_synthetic_data.py` first.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Parameters")

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.50, max_value=0.99, value=0.86, step=0.01,
    help="Default is 0.50. Move to 0.86 to filter out aggressive SMOTE bias."
)

# --- MAIN DASHBOARD INTERFACE ---
st.title("Autonomous Maintenance Verification Dashboard")
st.markdown("Bridging the gap in industrial logging through intelligent spindle current analysis.")

model = load_model()

if not model:
    st.error("⚠️ Trained model not found. Please ensure `models/rf_tool_wear_model.pkl` exists or run `train_pipeline.py` first.")
elif df_current_raw is not None and df_timeline_raw is not None:
    
    with st.spinner("Engineering features and running model inference..."):
        df, feature_cols = process_data(df_current_raw, df_timeline_raw)
        
        # Run Model Prediction
        X = df[feature_cols]
        # Get probability of class 1 (Tool Loading)
        df['ai_probability'] = model.predict_proba(X)[:, 1] 
        
        # Apply user threshold
        df['ai_detected_event'] = (df['ai_probability'] >= threshold).astype(int)
        df['missing_from_log'] = ((df['ai_detected_event'] == 1) & (df['manual_log_event'] == 0)).astype(int)

    # --- KPIs ---
    st.subheader("Audit Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_logged = df[df['manual_log_event'] == 1]['created_at'].dt.floor('Min').nunique()
    total_detected = df[df['ai_detected_event'] == 1]['created_at'].dt.floor('Min').nunique()
    unrecorded_flags = df['missing_from_log'].sum()

    col1.metric("Manually Logged Changes", total_logged)
    col2.metric("AI Detected Changes", total_detected)
    col3.metric("Unrecorded High-Risk Events", unrecorded_flags, delta=int(unrecorded_flags), delta_color="inverse")
    col4.metric("Active Threshold", f"{threshold:.2f}")

    st.markdown("---")

    # --- VISUALIZATION ---
    st.subheader("Spindle Current Analysis & Event Verification")

    fig = go.Figure()

    # Raw Current
    fig.add_trace(go.Scatter(
        x=df['created_at'], y=df['avg_current'], 
        mode='lines', name='Motor Current (Amps)', 
        line=dict(color='lightgrey', width=1)
    ))

    # Manual Logs
    manual_events = df[df['manual_log_event'] == 1]
    fig.add_trace(go.Scatter(
        x=manual_events['created_at'], y=manual_events['avg_current'],
        mode='markers', name='Logged by Operator',
        marker=dict(color='blue', size=8, symbol='circle')
    ))

    # AI Detected Unrecorded Events
    missing_events = df[df['missing_from_log'] == 1]
    fig.add_trace(go.Scatter(
        x=missing_events['created_at'], y=missing_events['avg_current'],
        mode='markers', name='Unrecorded Event (AI Flag)',
        marker=dict(color='red', size=10, symbol='x')
    ))

    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # --- DIAGNOSTICS TABLE ---
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Actionable Maintenance Gap Log")
        st.markdown("Timestamps where the AI detected a tool change with high probability, but no manual log exists.")
        
        audit_df = df[df['missing_from_log'] == 1].copy()
        if not audit_df.empty:
            audit_df = audit_df[['created_at', 'avg_current', 'ai_probability']]
            audit_df['ai_probability'] = audit_df['ai_probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(audit_df.sort_values(by='created_at', ascending=False), use_container_width=True)
        else:
            st.success("No unrecorded maintenance events found at this threshold!")

    with colB:
        st.subheader("Model Diagnostics")
        st.markdown("Dynamic Confusion Matrix (Timestamp Level)")
        
        cm = confusion_matrix(df['manual_log_event'], df['ai_detected_event'])
        if cm.shape == (2,2):
            st.table(pd.DataFrame(
                cm,
                columns=["Predicted Normal", "Predicted Event"],
                index=["Actual Normal", "Actual Event"]
            ))
        else:
            st.info("Adjust threshold to populate matrix.")

else:
    st.info("👈 Please select a data source from the sidebar to begin.")
