import mlflow.pyfunc
import pandas as pd

# Connect to local MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "FlightDelayModel"
stage = "Production"
model_uri = f"models:/{model_name}/{stage}"

print(f"Loading model from registry: {model_uri}...")

try:
    # Load the model from MLflow Model Registry
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Create dummy inference data (Encoded airline IDs)
    dummy_data = pd.DataFrame({"airline_encoded": [0, 1, 2, 3]})
    
    # Run prediction
    predictions = model.predict(dummy_data)
    
    print("\n[SUCCESS] Model loaded successfully.")
    print(f"Inference Results: {predictions}")

except Exception as e:
    print(f"[ERROR] Failed to load or run model: {e}")