import os
import torch
import joblib
import pandas as pd
import numpy as np
from src.model.ann import FireANN
from src.preprocessing.dwt import process_window

class FireDetector:
    def __init__(self, model_path="models/fire_ann.pt", scaler_path="models/scaler.pkl"):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler not found. Please run training first.")
            
        self.scaler = joblib.load(scaler_path)
        
        # Determine input size from scaler
        input_size = self.scaler.n_features_in_
        self.model = FireANN(input_size=input_size, num_classes=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.classes = {0: "Normal", 1: "Nuisance", 2: "Active Fire"}
        
    def predict_window(self, window_df):
        """
        Takes a raw pandas DataFrame representing a window of sensor data.
        Returns the class name and confidence.
        """
        feature_cols = [c for c in window_df.columns if c not in ['CNT', 'Class', 'Fire Alarm']]
        
        # Extract features
        feats = process_window(window_df, feature_cols)
        
        # Drop Class if it somehow got in
        if 'Class' in feats:
            del feats['Class']
            
        feature_names = getattr(self.scaler, "feature_names_in_", list(feats.keys()))
        
        # Create array with correct order
        feat_array = np.array([[feats.get(k, 0.0) for k in feature_names]])
        
        # Scale
        scaled_feats = self.scaler.transform(feat_array)
        
        # Predict
        with torch.no_grad():
            tensor_feats = torch.tensor(scaled_feats, dtype=torch.float32)
            outputs = self.model(tensor_feats)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_class = torch.max(probs, 0)
            
        return self.classes[pred_class.item()], confidence.item()

if __name__ == "__main__":
    print("Inference module ready.")
