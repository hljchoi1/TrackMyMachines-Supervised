import os
import pandas as pd
import numpy as np
import datetime

def generate_synthetic_factory_data(output_dir="data"):
    """
    Generates synthetic industrial datasets mimicking the exact schemas,
    high-frequency sampling noise, and severe class imbalance of the 
    original proprietary CNC machine logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic datasets in './{output_dir}'...")

    np.random.seed(42)
    machine_id = 290
    base_time = datetime.datetime(2026, 5, 1)
    
    # -------------------------------------------------------------
    # 1. GENERATE SYNTHETIC MACHINE TIMELINES (Activity Blocks)
    # -------------------------------------------------------------
    print(" -> Creating synthetic timeline logs...")
    timeline_records = []
    current_time = base_time
    
    # Generate 50 consecutive activity blocks over a few days
    for i in range(50):
        duration_minutes = np.random.randint(15, 120)
        end_time = current_time + datetime.timedelta(minutes=duration_minutes)
        
        # Format matching the regex: 'YYYY-MM-DD HH:MM:SS'
        range_str = f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}]"
        
        # Inject extreme class imbalance: 
        # Activity 189 (Tool Loading) happens very rarely (approx ~6% of blocks)
        if i in [12, 28, 41]:
            activity_reason = 189.0  # The target event
        else:
            activity_reason = float(np.random.choice([101, 102, 105, 201])) # Normal operations
            
        timeline_records.append({
            'machine_id': machine_id,
            'range': range_str,
            'activity_reason_id': activity_reason
        })
        current_time = end_time + datetime.timedelta(seconds=np.random.randint(5, 30)) # Brief idle gaps
        
    df_timeline = pd.DataFrame(timeline_records)
    timeline_file = os.path.join(output_dir, "machine_timelines_v2.csv")
    df_timeline.to_csv(timeline_file, index=False)
    print(f"    [SUCCESS] Saved {len(df_timeline)} timeline entries to {timeline_file}")

    # -------------------------------------------------------------
    # 2. GENERATE HIGH-FREQUENCY MOTOR CURRENT DATA
    # -------------------------------------------------------------
    print(" -> Creating high-frequency spindle current signals...")
    current_records = []
    
    # Generate sub-second current readings spanned across the timeline boundaries
    for _, row in df_timeline.iterrows():
        # Re-extract actual timestamps to populate the current sensor readings
        start_ts = pd.to_datetime(row['range'].split(' to ')[0].replace('[',''))
        end_ts = pd.to_datetime(row['range'].split(' to ')[1].replace(']',''))
        
        # Generate readings at 1-second raw intervals (which later group into 2-second windows)
        duration_seconds = int((end_ts - start_ts).total_seconds())
        timestamps = [start_ts + datetime.timedelta(seconds=s) for s in range(duration_seconds)]
        
        if row['activity_reason_id'] == 189.0:
            # Tool replacement event profile: massive upward current spike & intense variance
            raw_current = np.random.normal(loc=38.0, scale=6.5, size=len(timestamps))
        else:
            # Normal cutting profile: lower steady state baseline noise
            raw_current = np.random.normal(loc=14.5, scale=1.8, size=len(timestamps))
            
        for ts, val in zip(timestamps, raw_current):
            current_records.append({
                'machine_id': machine_id,
                'created_at': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'current1': max(0.0, val) # Sensor current cannot be negative
            })
            
    df_current = pd.DataFrame(current_records)
    current_file = os.path.join(output_dir, "current_data.csv")
    df_current.to_csv(current_file, index=False)
    print(f"    [SUCCESS] Saved {len(df_current)} raw signal samples to {current_file}")
    print("Synthetic data pipeline ready.")

if __name__ == "__main__":
    generate_synthetic_factory_data()
