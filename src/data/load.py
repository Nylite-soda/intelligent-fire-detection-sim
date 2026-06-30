import os
import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("data", "raw")
CSV_FILENAME = "smoke_detection_iot.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)

def download_data():
    """
    Downloads the dataset from Kaggle if not already present.
    """
    if os.path.exists(CSV_PATH):
        print(f"Data already exists at {CSV_PATH}")
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Attempting to download dataset using kagglehub...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("deepcontractor/smoke-detection-dataset")
        
        # The downloaded path usually contains the CSV. We need to copy/move it to our DATA_DIR.
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

def load_and_clean_data(filepath=CSV_PATH):
    """
    Loads the CSV, drops unnamed columns, handles nulls, and maps labels to 3 classes.
    """
    if not os.path.exists(filepath):
        download_data()
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop Unnamed columns (usually index)
    unnamed_cols = [col for col in df.columns if "Unnamed" in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    # Handle nulls
    df = df.dropna()
    
    # Map binary Fire Alarm to THREE classes
    # 0: Normal
    # 1: Nuisance (Borderline readings but no actual fire)
    # 2: Active Fire
    
    # Negative class mask
    neg_mask = df["Fire Alarm"] == 0
    
    # Nuisance class definition:
    # Elevated TVOC, eCO2, PM2.5 or Humidity within the negative class.
    # We will use 90th percentile of these features within the negative class as a threshold.
    # If any of these are above the threshold, we consider it a nuisance.
    tvoc_thresh = df.loc[neg_mask, "TVOC[ppb]"].quantile(0.90)
    eco2_thresh = df.loc[neg_mask, "eCO2[ppm]"].quantile(0.90)
    pm25_thresh = df.loc[neg_mask, "PM2.5"].quantile(0.90)
    
    print(f"Nuisance Thresholds (90th percentile of neg class):")
    print(f"  TVOC: {tvoc_thresh:.2f}")
    print(f"  eCO2: {eco2_thresh:.2f}")
    print(f"  PM2.5: {pm25_thresh:.2f}")
    
    nuisance_mask = neg_mask & (
        (df["TVOC[ppb]"] > tvoc_thresh) |
        (df["eCO2[ppm]"] > eco2_thresh) |
        (df["PM2.5"] > pm25_thresh)
    )
    
    # Initialize all as Normal (0)
    df["Class"] = 0 
    
    # Set Nuisance (1)
    df.loc[nuisance_mask, "Class"] = 1
    
    # Set Active Fire (2)
    df.loc[df["Fire Alarm"] == 1, "Class"] = 2
    
    # Drop original label
    df = df.drop(columns=["Fire Alarm"])
    
    print(f"Class distribution:\n{df['Class'].value_counts().sort_index()}")
    
    return df

def get_train_val_test_splits(df):
    """
    Splits the dataframe into train/val/test (70/15/15), stratified by class.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    # Second split: split temp in half (15% val, 15% test of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = load_and_clean_data()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_splits(df)
