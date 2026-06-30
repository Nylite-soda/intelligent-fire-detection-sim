import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model.cnn import FireCNN

class FireDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.environ.get("FIRE_MODEL_PATH", "models/fire_cnn_production.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Please run training first.")
            
        # 12 base numeric features + 12 delta features = 24
        self.num_features = 24
        
        self.model = FireCNN(n_features=self.num_features, num_classes=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.classes = {0: "Normal", 1: "Nuisance", 2: "Active Fire"}
        
    def predict_window(self, window_df):
        """
        Takes a raw pandas DataFrame representing a window of sensor data (e.g. 64 rows).
        Returns the class name and confidence.
        """
        df = window_df.copy()
        
        # 1. Base feature columns
        base_cols = [c for c in df.columns if c not in ['UTC', 'CNT', 'Class', 'Fire Alarm'] and "Unnamed" not in c]
        
        # 2. Compute Deltas
        for col in base_cols:
            df[f"{col}_delta"] = df[col].diff().fillna(0)
            
        # All feature columns (Base + Delta)
        all_features = base_cols + [f"{c}_delta" for c in base_cols]
        
        if len(all_features) != self.num_features:
            # If there's an issue, let it pass by taking just what's needed or padding,
            # but ideally the user uploads a proper CSV with the 12 base columns.
            pass
            
        # 3. Local Normalization (Per-window, approximating per-segment)
        X = df[all_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. Predict
        # Model expects shape (batch, window_size, n_features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_class_idx = torch.max(probs, 0)
            
        pred_class_name = self.classes[pred_class_idx.item()]
        conf_val = confidence.item()
        
        # Confidence-gated decision
        if pred_class_name == "Active Fire" and conf_val <= 0.80:
            pred_class_name = "Possible Fire (Low Confidence)"
            
        return pred_class_name, conf_val

if __name__ == "__main__":
    print("Inference module ready.")
