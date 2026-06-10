import os
import pandas as pd
import mlflow.pyfunc
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

def run_batch_inference():
    """
    Executes batch inference using the registered production model from MLflow.
    Dynamically connects to the tracking server based on the environment configuration.
    """
    # Retrieve MLflow tracking URI from environment or fallback to local SQLite
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    model_name = config["mlflow"]["model_name"]
    stage = "Production"

    model_uri = f"models:/{model_name}@{stage}"

    print(f"Connecting to MLflow Tracking Server at: {tracking_uri}")
    print(f"Loading model from registry: {model_uri}...")

    try:
        # Load the model artifact from the MLflow Model Registry
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Generate synthetic inference data (Encoded airline IDs)
        dummy_data = pd.DataFrame({"airline_encoded": [0, 1, 2, 3]})
        
        # Execute model prediction
        predictions = model.predict(dummy_data)
        
        print("\n[SUCCESS] Model loaded and inference executed successfully.")
        print(f"Inference Results:\n{predictions}")

    except Exception as e:
        print(f"\n[ERROR] Inference pipeline failed: {e}")

if __name__ == "__main__":
    run_batch_inference()