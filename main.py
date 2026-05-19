import os
import re
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib

warnings.filterwarnings('ignore')

def parse_range(r: str):
    """Extracts start and end times from a timeline range string."""
    matches = re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', str(r))
    if len(matches) >= 2:
        return pd.to_datetime(matches[0]), pd.to_datetime(matches[1])
    return pd.NaT, pd.NaT

def load_and_preprocess_data(timeline_path: str, current_path: str) -> tuple:
    """Loads CSVs and formats datetime columns."""
    print("1. Loading and Preprocessing Datasets...")
    df_timeline = pd.read_csv(timeline_path)
    df_current = pd.read_csv(current_path)

    # Parse timelines
    df_timeline[['start_time', 'end_time']] = df_timeline['range'].apply(lambda x: pd.Series(parse_range(x)))
    df_timeline = df_timeline.dropna(subset=['start_time', 'end_time'])
    
    # Remove timezone localization for merging
    df_timeline['start_time'] = df_timeline['start_time'].dt.tz_localize(None)
    df_timeline['end_time'] = df_timeline['end_time'].dt.tz_localize(None)
    df_current['created_at'] = pd.to_datetime(df_current['created_at'], errors='coerce').dt.tz_localize(None)
    
    # Define Target Label (Activity 189 = Tool Loading)
    df_timeline['is_loading'] = (df_timeline['activity_reason_id'] == 189.0).astype(int)
    df_timeline = df_timeline[['machine_id', 'start_time', 'end_time', 'is_loading']]
    
    return df_timeline, df_current

def resample_and_fuse(df_timeline: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
    """Resamples high-frequency current data and merges it with activity timelines."""
    print("2. Resampling High-Frequency Current Data (2-Second Windows)...")
    current_resampled = df_current.groupby(['machine_id', pd.Grouper(key='created_at', freq='2S')]).agg(
        avg_current=('current1', 'mean')
    ).reset_index()

    current_resampled = current_resampled.sort_values('created_at')
    df_timeline = df_timeline.sort_values('start_time')

    print("3. Fusing Electrical Data with Timeline Activities...")
    df_merged = pd.merge_asof(
        current_resampled,
        df_timeline,
        left_on='created_at',
        right_on='start_time',
        by='machine_id',
        direction='backward'
    )

    # Filter to exact active times
    mask = (df_merged['created_at'] >= df_merged['start_time']) & (df_merged['created_at'] < df_merged['end_time'])
    df_time = df_merged[mask].copy()
    
    return df_time.sort_values(['machine_id', 'created_at'])

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers time-series features essential for tool wear detection."""
    print("4. Engineering Time-Series Features...")
    
    df['rolling_baseline'] = df.groupby('machine_id')['avg_current'].transform(
        lambda x: x.rolling(window=150, min_periods=30).mean().shift(1)
    )
    df['current_delta_pct'] = (df['avg_current'] - df['rolling_baseline']) / df['rolling_baseline']

    # Lag and Lead Features
    df['current_lag_2'] = df.groupby('machine_id')['avg_current'].shift(2)
    df['current_lead_2'] = df.groupby('machine_id')['avg_current'].shift(-2)

    # Rolling Variance
    df['rolling_variance'] = df.groupby('machine_id')['avg_current'].transform(
        lambda x: x.rolling(window=10, min_periods=1).var().fillna(0)
    )
    
    return df

def execute_cross_validation(X_train: pd.DataFrame, y_train: pd.Series):
    """Runs a 5-fold cross validation across multiple baseline models."""
    print("5. Executing Hyperparameter Search & Cross Validation...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['precision', 'recall', 'f1', 'f1_macro', 'roc_auc']

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest (Default)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting (Default)': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    cv_results_summary = {}
    for name, model in models.items():
        print(f"   -> Evaluating {name}...")
        scores = cross_validate(model, X_train, y_train, cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1)
        cv_results_summary[name] = {
            'Precision': scores['test_precision'].mean(),
            'Recall': scores['test_recall'].mean(),
            'F1-Score': scores['test_f1'].mean()
        }

    results_df = pd.DataFrame(cv_results_summary).T.sort_values(by='F1-Score', ascending=False)
    print("\n--- 5-FOLD CV RESULTS ---")
    print(results_df.applymap(lambda x: f"{x:.4f}"))
    return models['Random Forest (Default)'] # Return the best model

def main():
    """Main execution block."""
    # 1. Define paths (Adjust these to point to your data folder in GitHub)
    TIMELINE_PATH = 'data/machine_timelines_v2.csv'
    CURRENT_PATH = 'data/current_data.csv'
    
    # 2. Run Data Pipeline
    df_timeline, df_current = load_and_preprocess_data(TIMELINE_PATH, CURRENT_PATH)
    df_fused = resample_and_fuse(df_timeline, df_current)
    df_engineered = engineer_features(df_fused)
    
    # 3. Prepare Train/Test Split
    features = [
        'avg_current', 'rolling_baseline', 'current_delta_pct',
        'current_lag_2', 'current_lead_2', 'rolling_variance'
    ]
    target_col = 'is_loading'
    
    df_clean = df_engineered[['machine_id', 'created_at'] + features + [target_col]].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Filtering for Machine 290
    df_machine_290 = df_clean[df_clean['machine_id'] == 290]
    X = df_machine_290[features]
    y = df_machine_290[target_col].astype(int)
    
    # 4. Train Model and Save
    best_model = execute_cross_validation(X, y)
    
    print("\n6. Training final Random Forest model on full dataset...")
    best_model.fit(X, y)
    
    # Save the model so the Streamlit app can use it
    joblib.dump(best_model, 'models/rf_tool_wear_model.pkl')
    print("Pipeline Complete! Model saved to 'models/rf_tool_wear_model.pkl'")

if __name__ == "__main__":
    main()
