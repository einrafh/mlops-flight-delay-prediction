from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

# Initialize the FastAPI application instance for production deployment
app = FastAPI(
    title="Flight Delay Inference API",
    description="Production-grade API for predicting domestic flight delay probabilities.",
    version="2.0.0"
)

# Instrument the application to expose default HTTP metrics (e.g., latency, RPS) at /metrics
Instrumentator().instrument(app).expose(app)

# Define a custom Prometheus Gauge to monitor prediction output distribution for data drift detection
prediction_gauge = Gauge(
    "flight_delay_prediction_output", 
    "Output of the flight delay prediction (0: On Time, 1: Delayed)"
)

# Configure MLflow tracking URI and model registry parameters dynamically via environment variables
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
MODEL_NAME = os.environ.get("MODEL_NAME", "FlightDelayModel")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
model = None

@app.on_event("startup")
def load_model():
    """
    Load the machine learning model from the MLflow Model Registry during application startup.
    """
    global model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    print(f"[INFO] Attempting to load model from registry URI: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("[SUCCESS] Model artifact loaded successfully into memory.")
    except Exception as e:
        print(f"[ERROR] Failed to load the model artifact: {e}")

# Define the data validation schema for incoming inference requests
class FlightData(BaseModel):
    airline: str = Field(..., example="Lion Air", description="Name of the operating airline")
    departure: str = Field(..., example="SUB", description="Departure airport IATA code")
    arrival: str = Field(..., example="BDJ", description="Arrival airport IATA code")
    flight_date: str = Field(..., example="2026-06-12", description="Scheduled date of the flight (YYYY-MM-DD)")
    scheduled_hour: int = Field(..., example=19, description="Scheduled hour of departure (0-23)")

# Define a static mapping dictionary for categorical label encoding.
# Note: In a fully automated pipeline, this should ideally be loaded from a serialized artifact (.pkl) generated during the training phase.
AIRLINE_MAPPING = {
    "Batik Air": 0,
    "Citilink": 1,
    "Garuda Indonesia": 2,
    "Lion Air": 3,
    "Pelita Air": 4,
    "Super Air Jet": 5
}

@app.get("/")
def health_check():
    """
    Provide a health check endpoint to verify service operational status and model availability.
    """
    return {
        "status": "Service is operational", 
        "model_ready": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }

@app.post("/predict")
def predict(data: FlightData):
    """
    Execute the inference pipeline: validate inputs, preprocess features, generate predictions, and export metrics.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="The inference model is currently not loaded or unavailable.")
    
    # Defensive Programming: Restrict inference to the supported origin-destination route
    if data.departure.upper() != "SUB" or data.arrival.upper() != "BDJ":
        raise HTTPException(
            status_code=400, 
            detail="Validation Error: The model currently only supports flights operating on the SUB to BDJ route."
        )

    # Defensive Programming: Validate the requested airline against the supported mapping
    if data.airline not in AIRLINE_MAPPING:
        valid_airlines = [k for k in AIRLINE_MAPPING.keys() if k != ""]
        raise HTTPException(
            status_code=400, 
            detail=f"Validation Error: Unrecognized airline '{data.airline}'. Supported airlines are: {valid_airlines}"
        )
    
    # Feature Preprocessing: Apply label encoding to the categorical airline feature
    encoded_airline = AIRLINE_MAPPING[data.airline]
    
    # Construct the input DataFrame to match the schema expected by the trained Random Forest model
    input_df = pd.DataFrame([{"airline_encoded": encoded_airline}])
    
    try:
        # Execute model inference
        prediction = model.predict(input_df)
        pred_value = int(prediction[0])
        
        # Export the prediction result to the Prometheus gauge for monitoring
        prediction_gauge.set(pred_value)
        
        # Construct a human-readable JSON response
        status_text = "DELAYED (High Risk)" if pred_value == 1 else "ON TIME (Low Risk)"
        
        return {
            "flight_info": f"{data.airline} | {data.departure} -> {data.arrival}",
            "schedule": f"{data.flight_date} at {data.scheduled_hour}:00",
            "prediction_status": status_text,
            "prediction_code": pred_value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failed: {e}")