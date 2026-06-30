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
