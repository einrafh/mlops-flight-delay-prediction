from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

# Initialize API 
app = FastAPI(
    title="Flight Delay Inference API",
    description="API for predicting flight delays with integrated Prometheus monitoring.",
    version="1.0.0"
)

# Instrument app to expose default HTTP metrics (latency, RPS) at /metrics
Instrumentator().instrument(app).expose(app)

# Custom metric to monitor data drift (prediction distribution)
prediction_gauge = Gauge(
    "flight_delay_prediction_output", 
    "Output of the flight delay prediction (0 or 1)"
)

# Configure MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
MODEL_NAME = os.environ.get("MODEL_NAME", "FlightDelayModel")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
model = None

@app.on_event("startup")
def load_model():
    global model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    print(f"[INFO] Attempting to load model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("[SUCCESS] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

class FlightData(BaseModel):
    airline_encoded: int

@app.get("/")
def health_check():
    return {
        "status": "Service is running", 
        "model_ready": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(data: FlightData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded or unavailable.")
    
    input_df = pd.DataFrame([{"airline_encoded": data.airline_encoded}])
    
    try:
        prediction = model.predict(input_df)
        pred_value = int(prediction[0])
        
        # Export prediction output to Prometheus metric
        prediction_gauge.set(pred_value)
        
        return {
            "airline_encoded": data.airline_encoded,
            "is_delayed_prediction": pred_value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")