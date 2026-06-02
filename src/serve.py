from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

# Initialize FastAPI application
app = FastAPI(
    title="Flight Delay Inference API",
    description="API for predicting flight delays using a trained Random Forest model.",
    version="1.0.0"
)

# Configure MLflow Tracking URI from environment variables
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

MODEL_NAME = os.environ.get("MODEL_NAME", "FlightDelayModel")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

# Global variable to hold the loaded model
model = None

@app.on_event("startup")
def load_model():
    """
    Load the machine learning model from MLflow Model Registry upon application startup.
    """
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"[INFO] Attempting to load model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("[SUCCESS] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

# Define the input data schema using Pydantic
class FlightData(BaseModel):
    airline_encoded: int

@app.get("/")
def health_check():
    """
    Health check endpoint to verify service availability and model status.
    """
    return {
        "status": "Service is running", 
        "model_ready": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(data: FlightData):
    """
    Inference endpoint to predict flight delays based on input features.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded or unavailable.")
    
    # Format input data into a Pandas DataFrame expected by the MLflow model
    input_df = pd.DataFrame([{"airline_encoded": data.airline_encoded}])
    
    try:
        # Perform inference
        prediction = model.predict(input_df)
        
        return {
            "airline_encoded": data.airline_encoded,
            "is_delayed_prediction": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")