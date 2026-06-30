import streamlit as st
import pandas as pd
import time
import os

from src.inference.predict import FireDetector
from src.alert.simulator import MockHardwareAlert

# Configure page
st.set_page_config(page_title="Intelligent Fire Detection System", layout="wide")
st.title("🔥 Intelligent Fire Detection System Dashboard")
st.markdown("""
This dashboard simulates a live hardware sensor stream using software simulation.
It uses an Artificial Neural Network trained on Discrete Wavelet Transform (DWT) features extracted from sensor data.
""")

@st.cache_resource
def load_detector():
    try:
        return FireDetector()
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None

detector = load_detector()
alert_system = MockHardwareAlert()

# Sidebar
st.sidebar.header("Controls")
data_source = st.sidebar.selectbox("Data Source", ["Test Set Replay", "Upload CSV Window"])

df = None
if data_source == "Test Set Replay":
    if os.path.exists("data/raw/smoke_detection_iot.csv"):
        # Load a small chunk to replay
        full_df = pd.read_csv("data/raw/smoke_detection_iot.csv")
        # take a random contiguous chunk of say 500 rows
        start_idx = st.sidebar.number_input("Start Index", min_value=0, max_value=len(full_df)-500, value=1000)
        df = full_df.iloc[start_idx:start_idx+500].reset_index(drop=True)
        st.sidebar.success(f"Loaded 500 rows from test set.")
    else:
        st.sidebar.error("Data not found. Please ensure data is loaded.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'PM2.5'}
            if not required_cols.issubset(set(df.columns)):
                st.sidebar.error(f"Invalid CSV schema. Missing columns: {required_cols - set(df.columns)}")
                df = None
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")
            df = None
        
if df is not None and not df.empty:
    st.subheader("Live Sensor Stream Simulator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chart_placeholder = st.empty()
        
    with col2:
        status_placeholder = st.empty()
        log_placeholder = st.empty()
        
    # Replay controls
    if st.sidebar.button("Start Live Replay"):
        if detector is None:
            st.error("Detector not initialized.")
            st.stop()
            
        window_size = 32
        step_size = 16 # Step 7: aligned with training distribution
        
        # Display logs
        logs = []
        
        for i in range(0, len(df) - window_size, step_size):
            window_df = df.iloc[i : i + window_size]
            
            # Predict
            pred_class, conf = detector.predict_window(window_df)
            
            # Plot
            chart_placeholder.line_chart(window_df[['Temperature[C]', 'TVOC[ppb]', 'eCO2[ppm]', 'PM2.5']])
            
            # Status Box
            if pred_class == "Normal":
                color = "green"
            elif pred_class == "Nuisance":
                color = "orange"
            elif "Possible Fire" in pred_class:
                color = "darkorange"
            else:
                color = "red"
                
            status_html = f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2>{pred_class}</h2>
                <p>Confidence: {conf*100:.1f}%</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            
            # Alert
            if pred_class == "Active Fire":
                alert_system.dispatch(zone_id="Zone-Alpha", confidence=conf, alert_type="Active Fire")
                logs.insert(0, f"🔥 ALERT DISPATCHED (Conf: {conf*100:.1f}%)")
            elif "Possible Fire" in pred_class:
                logs.insert(0, f"⚠️ Possible Fire Logged (Conf: {conf*100:.1f}%)")
                
            # Keep logs short
            if len(logs) > 5:
                logs.pop()
                
            log_html = "<h4>Alert Logs</h4><ul>" + "".join([f"<li>{l}</li>" for l in logs]) + "</ul>"
            log_placeholder.markdown(log_html, unsafe_allow_html=True)
            
            # Delay to simulate live streaming
            time.sleep(0.1)
