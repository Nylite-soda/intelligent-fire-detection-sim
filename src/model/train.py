import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.data.load import prepare_pipeline

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

def extract_3d_windows(df, window_size=32, step_size=16):
    if df.empty:
        return np.array([]), np.array([])
    cnt_diff = df['CNT'].diff().fillna(1)
    seq_starts = df.index[ (cnt_diff <= 0) | (cnt_diff > 1) ].tolist()
    seq_starts = [0] + seq_starts + [len(df)]
    seq_starts = sorted(list(set(seq_starts)))
    
    feature_cols = [c for c in df.columns if c not in ['CNT', 'Class', 'Fire Alarm', 'UTC']]
    
    X_list = []
    y_list = []
    
    for i in range(len(seq_starts) - 1):
        seq_df = df.iloc[seq_starts[i]:seq_starts[i+1]]
        if len(seq_df) < window_size:
            continue
        for w_start in range(0, len(seq_df) - window_size + 1, step_size):
            w_df = seq_df.iloc[w_start : w_start + window_size]
            X_list.append(w_df[feature_cols].values)
            if 'Class' in w_df.columns:
                y_list.append(w_df['Class'].max())
                
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)

class FireCNN(nn.Module):
    def __init__(self, n_features, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)

def train_model():
    torch.manual_seed(42)
    np.random.seed(42)

    logging.info("Loading data and preparing pipeline...")
    train_raw, val_raw, test_raw = prepare_pipeline()
    
    logging.info("Extracting 3D features with window_size=64...")
    X_train, y_train = extract_3d_windows(train_raw, window_size=64, step_size=16)
    X_val, y_val = extract_3d_windows(val_raw, window_size=64, step_size=16)
    X_test, y_test = extract_3d_windows(test_raw, window_size=64, step_size=16)
    
    logging.info(f"Total windows: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (3 * class_counts)
    
    # Increase penalty for misclassifying Normal and Nuisance
    class_weights[0] *= 2.0
    class_weights[1] *= 3.0
    
    logging.info(f"Computed Class Weights: {class_weights}")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    
    model = FireCNN(n_features=X_train.shape[2], num_classes=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs("models", exist_ok=True)
    best_val = float('inf')
    patience = 5
    patience_counter = 0
    num_epochs = 50
    
    logging.info("Starting full training run...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                out = model(X_b)
                val_loss += criterion(out, y_b).item()
                
        t_l = train_loss/len(train_loader)
        v_l = val_loss/len(val_loader)
        logging.info(f"Epoch {epoch+1:02d} - Train Loss: {t_l:.4f} - Val Loss: {v_l:.4f}")
        
        if v_l < best_val:
            best_val = v_l
            patience_counter = 0
            torch.save(model.state_dict(), "models/fire_cnn.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
    train_time = time.time() - start_time
    logging.info(f"Training time: {train_time:.2f} seconds")
    
    logging.info("Evaluating on test set...")
    model.load_state_dict(torch.load("models/fire_cnn.pt"))
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), 1).numpy()
        
    logging.info("\nClassification Report:")
    report = classification_report(y_test, preds, target_names=["Normal", "Nuisance", "Active Fire"], zero_division=0)
    logging.info("\n" + report)
    
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    logging.info("\nRaw Confusion Matrix ([Normal, Nuisance, Active Fire]):")
    logging.info("\n" + str(cm))
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        f.write("Full Training Run\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        
    logging.info("Training complete. Results saved to logs/training_log.txt and reports/metrics.txt.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_model()
