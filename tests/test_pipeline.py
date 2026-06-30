import pytest
import numpy as np
import pandas as pd
from src.preprocessing.dwt import process_window

def test_dwt_process_window():
    # Create synthetic window
    df = pd.DataFrame({
        'Temperature[C]': np.random.randn(32),
        'Humidity[%]': np.random.randn(32)
    })
    
    feats = process_window(df, ['Temperature[C]', 'Humidity[%]'])
    
    # Should contain raw stats
    assert 'Temperature[C]_raw_mean' in feats
    assert 'Humidity[%]_raw_max' in feats
    
    # Should contain DWT stats (db4 level 3)
    assert 'Temperature[C]_A3_energy' in feats
    assert 'Humidity[%]_D1_std' in feats

def test_inference_pipeline_end_to_end():
    import os
    if not os.path.exists("models/fire_ann.pt") or not os.path.exists("models/scaler.pkl"):
        pytest.skip("Models not trained yet")
        
    from src.inference.predict import FireDetector
    detector = FireDetector("models/fire_ann.pt", "models/scaler.pkl")
    df = pd.DataFrame({
        'Temperature[C]': np.random.randn(32),
        'Humidity[%]': np.random.randn(32),
        'TVOC[ppb]': np.random.randn(32),
        'eCO2[ppm]': np.random.randn(32),
        'PM1.0': np.random.randn(32),
        'PM2.5': np.random.randn(32),
        'NC0.5': np.random.randn(32),
        'NC1.0': np.random.randn(32),
        'NC2.5': np.random.randn(32),
        'CNT': np.arange(32),
        'Class': np.zeros(32)
    })
    
    pred_class, conf = detector.predict_window(df)
    
    
    assert pred_class in ["Normal", "Nuisance", "Active Fire", "Possible Fire (Low Confidence)"]
    assert 0.0 <= conf <= 1.0

def test_dwt_shuffled_sequence():
    from src.preprocessing.dwt import preprocess_dataframe_into_windows
    
    # Create contiguous sequence
    df = pd.DataFrame({
        'UTC': np.arange(1000, 1100),
        'CNT': np.arange(100),
        'Temperature[C]': np.random.randn(100),
        'Class': np.zeros(100)
    })
    
    # Shuffle completely out of order
    df_shuffled = df.sample(frac=1.0, random_state=42)
    
    # Process (explicit sort logic in dwt.py should recover the order and yield 5 windows)
    feats = preprocess_dataframe_into_windows(df_shuffled, window_size=32, step_size=16)
    
    assert len(feats) == 5
    assert "Temperature[C]_A3_energy" in feats.columns
