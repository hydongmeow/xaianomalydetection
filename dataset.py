
import pandas as pd
import numpy as np
import re

def complete_swat_preprocessing(df):
    print(f"Initial shape: {df.shape}")

    df_proc = df
    df_proc.columns = df_proc.columns.str.strip()

    initial_len = len(df_proc)
    df_proc = df_proc.drop_duplicates()
    print(f"Removed {initial_len - len(df_proc)} duplicate rows")

    target   = df_proc['Normal/Attack']
    features = df_proc.drop(columns=['Normal/Attack', 'Timestamp'])
    features.columns = features.columns.str.strip()

    # --- Binary actuators ---
    binary_actuators = ['MV101', 'MV201', 'P201', 'P202', 'P204', 'MV303']
    for col in binary_actuators:
        matching = [c for c in features.columns if c.strip() == col]
        if matching:
            features[matching[0]] = features[matching[0]].fillna(0).astype(int)

    # --- Critical sensors ---
    critical_sensors = ['LIT101', 'AIT201', 'AIT202', 'FIT401', 'PIT501']
    for col in critical_sensors:
        matching = [c for c in features.columns if c.strip() == col]
        if matching:
            features[matching[0]] = features[matching[0]].interpolate(
                method='linear', limit_direction='both'
            )

    numeric_cols    = features.select_dtypes(include=[np.number]).columns.tolist()
    other_sensors   = [c for c in numeric_cols
                       if c not in binary_actuators + critical_sensors]
    if other_sensors:
        medians = features[other_sensors].median()
        features[other_sensors] = features[other_sensors].ffill().fillna(medians)

    # --- Encode target ---
    if not pd.api.types.is_numeric_dtype(target):
        target_encoded = (target.str.strip() == 'Attack').astype(int)
    else:
        target_encoded = target.astype(int)

    attack_count = int(target_encoded.sum())
    print(f"\nFinal shape:    {features.shape}")
    print(f"Missing values: {features.isnull().sum().sum()}")  # safe now
    print(f"Attack samples: {attack_count}")
    print(f"Normal samples: {len(target_encoded) - attack_count}")

    return features, target_encoded

# ===================== for WADi dataset ========================

def fix_df(test_df: pd.DataFrame, train_normal: pd.DataFrame) -> pd.DataFrame:
    """
    1. Strip surrounding quotes from string values  (e.g. "'1.23'" → "1.23")
    2. Cast each column to the same dtype as train_normal
    """
    df = test_df.copy()
    df['Attack LABLE (1:No Attack, -1:Attack)'] = pd.to_numeric(
    test_df['Attack LABLE (1:No Attack, -1:Attack)'], errors='coerce'
).astype(int).map({1:0, -1:1}).values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.strip("'\"")
                .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            )

        if col not in train_normal.columns:
            continue

        target = train_normal[col].dtype

        try:
            if target == np.float64:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)

            elif target == np.int64:
                df[col] = (
                    pd.to_numeric(df[col], errors='coerce')
                      .fillna(0)
                      .astype(np.int64)
                )

            else:
                df[col] = df[col].astype(target)

        except Exception as e:
            print(f"  ❌ Could not convert '{col}': {e}")

    return df

def get_columns(df: pd.DataFrame, discrete_threshold: int = 4):
    print(f"Shape: {df.shape}\n")

    # ── Identify empty columns ──────────────────────────────────────────────
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        print(f"⚠️  Empty columns ({len(empty_cols)}): {empty_cols}\n")
    else:
        print("✅ No empty columns\n")

    non_numeric_cols = []
    discrete_cols    = []
    continuous_cols  = []

    for col in df.columns:
        if col in empty_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
        elif df[col].nunique(dropna=False) <= discrete_threshold:
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

    return discrete_cols, continuous_cols


def complete_wadi_preprocessing(df, discrete_cols, continuous_cols):
    print(f"Initial shape: {df.shape}")
    features = df.copy()

    # Step 5: Handle missing values based on SWaT domain knowledge
    # Binary actuators (fill with 0 for OFF state)
    binary_actuators = discrete_cols
    for col in binary_actuators:
        matching_cols = [c for c in features.columns if c.strip() == col]
        if matching_cols:
            actual_col = matching_cols[0]
            features[actual_col] = features[actual_col].fillna(0)
            features[actual_col] = features[actual_col].astype(int)
            if actual_col != col:
                features = features.rename(columns={actual_col: col})

    CRITICAL_PATTERNS = [
        r'_PV$',  # Process Variable  (sensor readings)
        r'_FQ_',  # Flow Quantity
        r'_SPEED$',  # Pump speed
        r'^LEAK_',  # Leak differential pressure
        r'^TOTAL_',  # Total consumption
    ]

    def is_critical(col: str) -> bool:
        return any(re.search(pat, col) for pat in CRITICAL_PATTERNS)

    critical_cols = [c for c in continuous_cols if is_critical(c)]
    other_cols = [c for c in continuous_cols if not is_critical(c)]

    print(f"Critical (interpolate) : {len(critical_cols)} cols")
    print(f"Other    (median fill) : {len(other_cols)} cols")

    for col in critical_cols:
        matching_cols = [c for c in features.columns if c.strip() == col]
        if matching_cols:
            actual_col = matching_cols[0]
            features[actual_col] = features[actual_col].interpolate(method='linear', limit_direction='both')
            # Linear interpolation fills missing values based on surrounding values.
            if actual_col != col:
                features = features.rename(columns={actual_col: col})
    for col in other_cols:
        features[col] = features[col].ffill().fillna(features[col].median())

    print(f"\nFinal shape: {features.shape}")
    print(f"Missing values: {features.isnull().sum().sum()}")

    return features

