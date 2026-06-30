import os
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

from src.data.load import load_and_clean_data, get_train_val_test_splits
from src.preprocessing.dwt import preprocess_dataframe_into_windows
from src.model.ann import FireANN

def train_model():
    print("Loading data...")
    df = load_and_clean_data()
    
    print("Extracting DWT features. This might take a while...")
    features_df = preprocess_dataframe_into_windows(df, window_size=32, step_size=16)
    
    print(f"Total feature windows extracted: {len(features_df)}")
    
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = get_train_val_test_splits(features_df)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)
    
    # Save scaler for inference
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    y_train = y_train_s.values
    y_val = y_val_s.values
    y_test = y_test_s.values
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    
    model = FireANN(input_size=X_train.shape[1], num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/fire_ann.pt")
            
    print("Training complete. Evaluating on test set...")
    
    model.load_state_dict(torch.load("models/fire_ann.pt"))
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, preds = torch.max(test_outputs, 1)
        
    y_pred = preds.numpy()
    
    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Normal", "Nuisance", "Active Fire"])
    print(report)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save confusion matrix plot
    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal", "Nuisance", "Active Fire"],
                yticklabels=["Normal", "Nuisance", "Active Fire"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    print("Saved confusion matrix to reports/confusion_matrix.png")
    
    with open("reports/metrics.txt", "w") as f:
        f.write(report)
        f.write(f"\nTest Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    train_model()
