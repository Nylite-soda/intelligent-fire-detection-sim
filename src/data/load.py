import os
import pandas as pd
import kagglehub
import numpy as np

DATA_DIR = os.path.join("data", "raw")
CSV_FILENAME = "smoke_detection_iot.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

def download_data():
    if os.path.exists(CSV_PATH):
        print(f"Data already exists at {CSV_PATH}")
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Attempting to download dataset using kagglehub...")
    try:
        path = kagglehub.dataset_download("deepcontractor/smoke-detection-dataset")
        import shutil
        downloaded_csv = os.path.join(path, CSV_FILENAME)
        if os.path.exists(downloaded_csv):
            shutil.copy(downloaded_csv, CSV_PATH)
            print(f"Successfully downloaded and moved dataset to {CSV_PATH}")
        else:
            raise FileNotFoundError(f"Could not find {CSV_FILENAME} in {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Kaggle auth might not be available or network error.")
        print("Please manually download the dataset from:")
        print("https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset")
        print(f"And place the '{CSV_FILENAME}' file in the '{DATA_DIR}' directory.")
        raise FileNotFoundError(f"Missing dataset due to download failure: {e}") from e

def load_raw_data(filepath=CSV_PATH):
    if not os.path.exists(filepath):
        download_data()
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    unnamed_cols = [col for col in df.columns if "Unnamed" in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    df = df.dropna()
    
    # Keep original chronological/session order
    return df

def split_contiguous(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    return train_df, val_df, test_df

def apply_labels(train_df, val_df, test_df):
    neg_mask_train = train_df["Fire Alarm"] == 0
    
    tvoc_thresh = train_df.loc[neg_mask_train, "TVOC[ppb]"].quantile(0.90)
    eco2_thresh = train_df.loc[neg_mask_train, "eCO2[ppm]"].quantile(0.90)
    pm25_thresh = train_df.loc[neg_mask_train, "PM2.5"].quantile(0.90)
    
    print(f"Nuisance Thresholds (Computed on TRAIN ONLY):")
    print(f"  TVOC: {tvoc_thresh:.2f}")
    print(f"  eCO2: {eco2_thresh:.2f}")
    print(f"  PM2.5: {pm25_thresh:.2f}")
    
    for split_name, df in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        neg_mask = df["Fire Alarm"] == 0
        nuisance_mask = neg_mask & (
            (df["TVOC[ppb]"] > tvoc_thresh) |
            (df["eCO2[ppm]"] > eco2_thresh) |
            (df["PM2.5"] > pm25_thresh)
        )
        
        df["Class"] = 0
        df.loc[nuisance_mask, "Class"] = 1
        df.loc[df["Fire Alarm"] == 1, "Class"] = 2
        df.drop(columns=["Fire Alarm"], inplace=True)
        
        print(f"{split_name} Class distribution:\n{df['Class'].value_counts().sort_index()}")
        
    return train_df, val_df, test_df

def prepare_pipeline(filepath=CSV_PATH):
    df = load_raw_data(filepath)
    train_df, val_df, test_df = split_contiguous(df)
    train_df, val_df, test_df = apply_labels(train_df, val_df, test_df)
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_pipeline()
