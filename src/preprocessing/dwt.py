import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

def compute_dwt_features(signal, wavelet='db4', level=3):
    """
    Applies DWT to a 1D signal and extracts feature statistics.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] is approximation, coeffs[1:] are details
    
    features = {}
    for i, coeff in enumerate(coeffs):
        coeff_name = f"A{level}" if i == 0 else f"D{level - i + 1}"
        features[f"{coeff_name}_mean"] = np.mean(coeff)
        features[f"{coeff_name}_std"] = np.std(coeff)
        features[f"{coeff_name}_energy"] = np.sum(np.square(coeff))
        
    return features

def process_window(window_df, feature_cols, wavelet='db4', level=3):
    """
    Processes a single window of data, computing DWT features and raw aggregations.
    """
    window_features = {}
    
    for col in feature_cols:
        signal = window_df[col].values
        
        # Raw aggregations
        window_features[f"{col}_raw_mean"] = np.mean(signal)
        window_features[f"{col}_raw_std"] = np.std(signal)
        window_features[f"{col}_raw_max"] = np.max(signal)
        window_features[f"{col}_raw_min"] = np.min(signal)
        
        # DWT features
        try:
            dwt_feats = compute_dwt_features(signal, wavelet, level)
            for k, v in dwt_feats.items():
                window_features[f"{col}_{k}"] = v
        except Exception as e:
            raise ValueError(f"DWT feature extraction failed for column {col}: {e}")
            
    # Include target class if 'Class' is present
    if "Class" in window_df.columns:
        # Taking the max class ensures we don't miss fire if it occurs in the window
        window_features["Class"] = window_df["Class"].max()
        
    return window_features

def preprocess_dataframe_into_windows(df, window_size=32, step_size=16, wavelet='db4', level=3):
    """
    Groups dataframe by contiguous CNT sequences, then applies sliding window.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Find where sequence breaks (CNT doesn't strictly increase by 1, or is lower)
    cnt_diff = df['CNT'].diff()
    # Fill NA for the first row to be > 0 so it's not considered a break
    cnt_diff = cnt_diff.fillna(1)
    
    # We define a break if the difference is <= 0 or perhaps > 10 (large jump)
    # To be safe, let's just find drops in CNT or large jumps
    seq_starts = df.index[ (cnt_diff <= 0) | (cnt_diff > 1) ].tolist()
    
    seq_starts = [df.index[0]] + seq_starts + [df.index[-1] + 1]
    seq_starts = sorted(list(set(seq_starts)))
    
    feature_cols = [c for c in df.columns if c not in ['CNT', 'Class', 'Fire Alarm']]
    
    processed_windows = []
    
    for i in range(len(seq_starts) - 1):
        start_idx = seq_starts[i]
        end_idx = seq_starts[i+1]
        
        # seq_df has contiguous CNT
        seq_df = df.loc[start_idx:end_idx-1]
        if len(seq_df) < window_size:
            continue
            
        for w_start in range(0, len(seq_df) - window_size + 1, step_size):
            w_df = seq_df.iloc[w_start : w_start + window_size]
            win_feats = process_window(w_df, feature_cols, wavelet, level)
            processed_windows.append(win_feats)
            
    return pd.DataFrame(processed_windows)

def demo():
    np.random.seed(42)
    time = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.random.randn(100)
    
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Original Signal')
    
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    cA3, cD3, cD2, cD1 = coeffs
    
    # Reconstruct denoised signal (zero out D1 noise)
    coeffs_denoised = [cA3, cD3, cD2, np.zeros_like(cD1)]
    reconstructed = pywt.waverec(coeffs_denoised, 'db4')
    
    plt.plot(reconstructed[:len(signal)], label='Denoised Signal (D1 zeroed)')
    plt.legend()
    plt.title('DWT Denoising Demo')
    plt.tight_layout()
    plt.savefig('dwt_demo.png')
    print("Saved demo plot to dwt_demo.png")
    
    df = pd.DataFrame({'Temperature[C]': signal, 'Class': np.zeros(100)})
    feats = process_window(df, ['Temperature[C]'])
    print("Extracted Features:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
