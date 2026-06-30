import pytest
import pandas as pd
from src.data.load import load_and_clean_data

def test_label_mapping():
    # Load data (this will download if not present)
    df = load_and_clean_data()
    
    assert "Class" in df.columns
    assert "Fire Alarm" not in df.columns
    
    # Check classes
    unique_classes = df["Class"].unique()
    assert set(unique_classes).issubset({0, 1, 2})
