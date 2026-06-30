import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join("data", "raw")
CSV_FILENAME = "smoke_detection_iot.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

def download_data():
    if os.path.exists(CSV_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Attempting to download dataset using kagglehub...")
    try:
        path = kagglehub.dataset_download("deepcontractor/smoke-detection-dataset")
        import shutil
        downloaded_csv = os.path.join(path, CSV_FILENAME)
        if os.path.exists(downloaded_csv):
            shutil.copy(downloaded_csv, CSV_PATH)
        else:
            raise FileNotFoundError(f"Could not find {CSV_FILENAME}")
    except Exception as e:
        raise FileNotFoundError(f"Missing dataset due to download failure: {e}") from e

def load_raw_data(filepath=CSV_PATH):
    if not os.path.exists(filepath):
        download_data()
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    unnamed_cols = [col for col in df.columns if "Unnamed" in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df.dropna()

def load_and_segment(filepath=CSV_PATH):
    df = load_raw_data(filepath)
    cnt_diff = df['CNT'].diff().fillna(1)
    seq_starts = df.index[ (cnt_diff <= 0) | (cnt_diff > 1) ].tolist()
    seq_starts = [0] + seq_starts + [len(df)]
    seq_starts = sorted(list(set(seq_starts)))
    
    segments = []
    for i in range(len(seq_starts) - 1):
        seg = df.iloc[seq_starts[i]:seq_starts[i+1]].copy()
        segments.append(seg)
    return segments

def apply_labels_per_segment(segments):
    for i, df in enumerate(segments):
        neg_mask = df["Fire Alarm"] == 0
        if neg_mask.sum() == 0:
            tvoc_thresh = float('inf')
            eco2_thresh = float('inf')
            pm25_thresh = float('inf')
        else:
            tvoc_thresh = df.loc[neg_mask, "TVOC[ppb]"].quantile(0.90)
            eco2_thresh = df.loc[neg_mask, "eCO2[ppm]"].quantile(0.90)
            pm25_thresh = df.loc[neg_mask, "PM2.5"].quantile(0.90)
            
        nuisance_mask = neg_mask & (
            (df["TVOC[ppb]"] > tvoc_thresh) |
            (df["eCO2[ppm]"] > eco2_thresh) |
            (df["PM2.5"] > pm25_thresh)
        )
        df["Class"] = 0
        df.loc[nuisance_mask, "Class"] = 1
        df.loc[df["Fire Alarm"] == 1, "Class"] = 2
        df.drop(columns=["Fire Alarm"], inplace=True)
    return segments

def engineer_features_and_normalize(segments):
    for df in segments:
        num_cols = [c for c in df.columns if c not in ["UTC", "CNT", "Class"]]
        
        # 1. Compute Derivatives (Rate of Change)
        for col in num_cols:
            df[f"{col}_delta"] = df[col].diff().fillna(0)
            
        # 2. Per-segment Normalization
        all_features = [c for c in df.columns if c not in ["UTC", "CNT", "Class"]]
        scaler = StandardScaler()
        df[all_features] = scaler.fit_transform(df[all_features])
        
    return segments

def split_segments(segments):
    train_dfs = [segments[0], segments[2], segments[3]]
    seg1 = segments[1]
    seg1_mid = len(seg1) // 2
    seg4 = segments[4]
    seg4_mid = len(seg4) // 2
    
    val_dfs = [seg1.iloc[:seg1_mid], seg4.iloc[:seg4_mid]]
    test_dfs = [seg1.iloc[seg1_mid:], seg4.iloc[seg4_mid:]]
    
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    print(f"Train Class distribution:\n{train_df['Class'].value_counts().sort_index()}")
    print(f"Val Class distribution:\n{val_df['Class'].value_counts().sort_index()}")
    print(f"Test Class distribution:\n{test_df['Class'].value_counts().sort_index()}")
    
    return train_df, val_df, test_df

def prepare_pipeline(filepath=CSV_PATH):
    segments = load_and_segment(filepath)
    segments = apply_labels_per_segment(segments)
    segments = engineer_features_and_normalize(segments)
    train_df, val_df, test_df = split_segments(segments)
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_pipeline()
