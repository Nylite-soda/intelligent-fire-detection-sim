# Intelligent Fire Detection System (Software Simulation)

## Project Overview
This project is a software-simulated end-to-end implementation of an Intelligent Fire Detection System, developed as a final-year CSC 476 project (Group 21/18).

The original proposal specifies an embedded hardware system incorporating multi-sensor fusion (PIR flame sensor, electrochemical smoke sensor, MLX90614 thermal sensor) and GSM/IoT alerting capabilities. Due to current hardware unavailability, this repository provides a **fully functional software simulation** of the entire pipeline. The machine learning and signal processing core functions on real datasets, and the physical sensor/alert hardware layer is cleanly mocked via interfaces for future drop-in replacement.

## Hardware Integration Status
**Current Build:** Full Software Simulation
Physical sensor procurement is pending. The system currently uses the Kaggle "Smoke Detection Dataset" to simulate live sensor data streaming. The alerting mechanism (GSM/IoT) is mocked using the `AlertDispatcher` interface, which logs events and prints console alerts. A future physical GSM/IoT module can be easily integrated by extending this interface.

## Pipeline Architecture
1. **Data Loading & Cleaning:** Fetches and processes raw sensor data, mapping binary labels into three classes (Normal, Nuisance, Active Fire).
2. **Signal Preprocessing (DWT):** Applies Discrete Wavelet Transform (DWT) to sliding windows of sensor data to denoise and extract approximation and detail coefficients.
3. **ANN Classifier:** A lightweight Multilayer Perceptron (PyTorch, CPU-only) trained on DWT features to classify the state.
4. **Inference & Alert Simulation:** Predicts on new data windows and triggers simulated alerts via the mocked hardware chain.
5. **Live Dashboard:** A Streamlit interface simulating live sensor streaming, classification, and alerting.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
The data loader uses `kagglehub` to automatically download the [Kaggle Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset). 
If this fails, manually download the dataset and place `smoke_detection_iot.csv` in `data/raw/`.

### 3. Train the Model
Run the training script to preprocess data, train the ANN, and generate evaluation metrics:
```bash
python -m src.model.train
```

### 4. Run the Dashboard
Launch the live simulation dashboard:
```bash
streamlit run app/dashboard.py
```

## Model Performance Metrics
*(Run `python -m src.model.train` to populate `reports/metrics.txt` and `reports/confusion_matrix.png` with genuine results.)*
Targeting ≥90% accuracy on Active Fire detection. See `reports/` for detailed breakdown after training.
