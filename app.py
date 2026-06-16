import sys
import os
import subprocess
import pandas as pd
import requests
import plotly.express as px
import streamlit as st
from datetime import datetime

# Set up global page configuration
st.set_page_config(page_title="MLOps Control Panel", layout="wide")
st.title("Flight Delay MLOps Master Dashboard")
st.markdown("Comprehensive control panel for orchestrating the Machine Learning lifecycle from data ingestion to observability.")

# Sidebar Configurations
st.sidebar.header("System Configuration")
API_URL = st.sidebar.text_input("FastAPI Endpoint", value="http://localhost:8000")
GITHUB_TOKEN = st.sidebar.text_input("GitHub PAT", type="password", help="Required to trigger remote GitHub Actions workflows.")
GITHUB_REPO = st.sidebar.text_input("GitHub Repository", value="username/mlops-flight-delay-prediction")

# Define Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Pipeline", 
    "Model & Registry", 
    "Serving & Inference", 
    "Observability & Automation"
])

# --- TAB 1: DATA PIPELINE ---
with tab1:
    st.header("Data Ingestion and Preprocessing")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pipeline Execution")
        if st.button("Execute Data Ingestion"):
            with st.spinner("Fetching data from AviationStack..."):
                try:
                    subprocess.run([sys.executable, "src/ingest_data.py"], check=True)
                    st.success("Data successfully fetched and stored.")
                except Exception as e:
                    st.error(f"Data ingestion failed: {e}")
                    
        if st.button("Execute Data Preprocessing"):
            with st.spinner("Aggregating and processing raw data..."):
                try:
                    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
                    st.success("Data preprocessing and deduplication completed.")
                except Exception as e:
                    st.error(f"Data preprocessing failed: {e}")

    with col2:
        st.subheader("Processed Data Preview")
        try:
            target_file = 'data/processed/processed_flights.csv'
            if os.path.exists(target_file):
                df = pd.read_csv(target_file)
                st.write(f"Displaying latest merged data from: `{os.path.basename(target_file)}`")
                st.write(f"**Total Historical Records:** {len(df)}")
                st.dataframe(df.tail(5))
            else:
                st.info("Consolidated processed data file not found. Please run preprocessing.")
        except Exception as e:
            st.warning(f"Unable to load data preview: {e}")

# --- TAB 2: MODEL & REGISTRY ---
with tab2:
    st.header("Model Training and MLflow Registry")
    st.markdown("Configure hyperparameters and initialize the Random Forest model training process.")
    
    n_estimators = st.slider("Number of Estimators", 50, 300, 100)
    max_depth = st.slider("Maximum Depth", 5, 50, 15)
    
    if st.button("Initialize Model Training"):
        with st.spinner("Training model, evaluating thresholds, and registering to MLflow..."):
            try:
                subprocess.run(
                    [sys.executable, "src/train.py", "--n_estimators", str(n_estimators), "--max_depth", str(max_depth)], 
                    check=True
                )
                st.success("Training script executed. Check MLflow for registration status.")
            except Exception as e:
                st.error(f"Model training pipeline failed: {e}")
                
    st.markdown("[Access MLflow Tracking Server](http://localhost:5000)")

# --- TAB 3: SERVING & INFERENCE ---
with tab3:
    st.header("Inference API Testing")
    st.markdown("Send HTTP POST requests to the production-grade FastAPI service.")
    
    # Define valid constraints based on the API Validation Logic
    valid_airlines = ["Batik Air", "Citilink", "Garuda Indonesia", "Lion Air", "Pelita Air", "Super Air Jet"]
    
    with st.form("predict_form"):
        st.subheader("Flight Parameters")
        
        # Use columns for a cleaner UI layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            airline = st.selectbox("Airline Operator", valid_airlines, index=3) # Default to Lion Air
            departure = st.selectbox("Departure Airport (IATA)", ["SUB"]) # Locked to SUB
            arrival = st.selectbox("Arrival Airport (IATA)", ["BDJ"])     # Locked to BDJ
            
        with col_right:
            flight_date = st.date_input("Scheduled Flight Date", value=datetime.today())
            scheduled_hour = st.slider("Scheduled Departure Hour", 0, 23, 19)
        
        submit_button = st.form_submit_button(label="Execute Prediction")
        
    if submit_button:
        # Construct the payload matching the new FastAPI Pydantic schema
        payload = {
            "airline": airline,
            "departure": departure,
            "arrival": arrival,
            "flight_date": flight_date.strftime("%Y-%m-%d"), # Format date to string
            "scheduled_hour": scheduled_hour
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction retrieved successfully.")
                
                # Display the formatted outputs from the API
                col_a, col_b = st.columns(2)
                col_a.metric("Flight Info", result.get("flight_info"))
                
                # Dynamic color warning based on status
                status = result.get("prediction_status")
                if "DELAYED" in status:
                    col_b.error(f"Status: {status}")
                else:
                    col_b.success(f"Status: {status}")
                    
                st.info(f"Schedule Detail: {result.get('schedule')}")
            else:
                st.error(f"API Error ({response.status_code}): {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Connection failed: {e}. Please ensure the FastAPI Docker container is actively running.")

# --- TAB 4: OBSERVABILITY & AUTOMATION ---
with tab4:
    st.header("System Observability and CI/CD Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CI/CD Remote Trigger")
        st.markdown("Trigger automated workflows via GitHub Actions REST API.")
        
        def trigger_github_workflow(workflow_id):
            if not GITHUB_TOKEN or "username" in GITHUB_REPO:
                st.warning("Please configure your GitHub PAT and Repository in the sidebar.")
                return
            
            url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_id}/dispatches"
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {GITHUB_TOKEN}"
            }
            res = requests.post(url, headers=headers, json={"ref": "main"})
            if res.status_code == 204:
                st.success(f"Workflow '{workflow_id}' triggered successfully.")
            else:
                st.error(f"Failed to trigger workflow: {res.status_code} - {res.text}")

        if st.button("Trigger Daily Data Ingestion"):
            trigger_github_workflow("daily_ingest.yaml")
            
        if st.button("Trigger Scheduled Model Retraining"):
            trigger_github_workflow("tri_daily_train.yaml")
            
    with col2:
        st.subheader("Performance Metrics Visualization")
        st.markdown("Real-time simulation of API throughput.")
        
        if st.button("Load Live Metrics"):
            try:
                chart_data = pd.DataFrame({
                    "Timestamp": pd.date_range(start="today", periods=20, freq="1min"),
                    "Requests Per Minute": [2, 5, 3, 10, 45, 60, 55, 12, 4, 3, 2, 8, 15, 20, 18, 5, 2, 1, 0, 1]
                })
                fig = px.line(
                    chart_data, 
                    x="Timestamp", 
                    y="Requests Per Minute", 
                    title="API Throughput (Simulated)", 
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.error("Failed to render performance metrics chart.")