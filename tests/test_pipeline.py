import pytest
import numpy as np
import pandas as pd

def test_inference_pipeline_end_to_end():
    import os
    if not os.path.exists("models/fire_cnn_production.pt"):
        pytest.skip("Models not trained yet")
        
    from src.inference.predict import FireDetector
    detector = FireDetector()
    
    # 64 rows of raw sensor data
    df = pd.DataFrame({
        'UTC': np.arange(1000, 1064),
        'CNT': np.arange(64),
        'Temperature[C]': np.random.randn(64),
        'Humidity[%]': np.random.randn(64),
        'TVOC[ppb]': np.random.randn(64),
        'eCO2[ppm]': np.random.randn(64),
        'Raw H2': np.random.randn(64),
        'Raw Ethanol': np.random.randn(64),
        'Pressure[hPa]': np.random.randn(64),
        'PM1.0': np.random.randn(64),
        'PM2.5': np.random.randn(64),
        'NC0.5': np.random.randn(64),
        'NC1.0': np.random.randn(64),
        'NC2.5': np.random.randn(64),
        'Fire Alarm': np.zeros(64)
    })
    
    pred_class, conf = detector.predict_window(df)
    
    assert pred_class in ["Normal", "Nuisance", "Active Fire", "Possible Fire (Low Confidence)"]
    assert 0.0 <= conf <= 1.0

def test_delta_features():
    from src.data.load import engineer_features_and_normalize
    
    df1 = pd.DataFrame({
        'UTC': [1, 2, 3],
        'CNT': [1, 2, 3],
        'Temperature[C]': [10.0, 12.0, 11.0],
        'Class': [0, 0, 0]
    })
    
    segments = engineer_features_and_normalize([df1])
    processed = segments[0]
    
    assert "Temperature[C]_delta" in processed.columns
