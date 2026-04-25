import pandas as pd
import numpy as np

def load_and_engineer_features(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    """
    Loads raw data and generates purely local, physics-based interaction features.
    NO API CALLS.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Target columns
    target_cols = ['event', 'time_to_hit_hours']
    y_train = train[target_cols].copy()
    y_train['event'] = y_train['event'].astype(bool)
    
    # Convert to structured array for sksurv
    y_train_struct = np.array([(row['event'], row['time_to_hit_hours']) for _, row in y_train.iterrows()],
                              dtype=[('event', '?'), ('time_to_hit_hours', '<f8')])
                              
    # Drop identifiers and targets from train
    drop_cols = ['event_id', 'event', 'time_to_hit_hours']
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns]).copy()
    
    # Store test ids but drop from features
    test_ids = test['event_id'].copy() if 'event_id' in test.columns else None
    X_test = test.drop(columns=['event_id']).copy() if 'event_id' in test.columns else test.copy()
    
    # Fill NAs with median (since it's a small dataset, keep it robust)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median()) # Avoid leakage
    
    # Feature Engineering: Physics Interactions
    for df in [X_train, X_test]:
        # Danger Index: How fast it's closing * how aligned it is
        df['danger_index'] = df['closing_speed_m_per_h'] * df['alignment_cos']
        
        # Time to impact estimate (distance / closing speed). Clip speed to avoid div by zero
        safe_close_speed = df['closing_speed_m_per_h'].clip(lower=0.1)
        df['time_to_impact_est'] = df['dist_min_ci_0_5h'] / safe_close_speed
        
        # Growth impact metrics
        df['growth_impact'] = df['area_growth_rate_ha_per_h'] * df['alignment_cos']
        df['dist_accel_impact'] = df['dist_accel_m_per_h2'] / safe_close_speed
        
    return X_train, y_train_struct, X_test, test_ids, y_train

