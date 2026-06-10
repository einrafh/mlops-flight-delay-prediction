import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

def register_best_model():
    """
    Evaluates the latest training runs and registers the best performing model.
    Automatically assigns the 'Production' stage to the highest-performing version.
    """
    # Retrieve MLflow tracking URI dynamically
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    model_name = config["mlflow"]["model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError(f"[ERROR] Experiment '{experiment_name}' not found. Ensure training has occurred.")

    print(f"Searching for the best model in experiment: {experiment_name}...")

    # Fetch the top run based on accuracy and F1 score
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC", "metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("[ERROR] No successful training runs found to register.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    accuracy = best_run.data.metrics.get('accuracy', 0.0)
    
    model_uri = f"runs:/{best_run_id}/model"

    print(f"Best Run ID: {best_run_id} (Accuracy: {accuracy:.4f})")
    print(f"Registering model as '{model_name}'...")

    # Register the best model as a new version
    model_version = mlflow.register_model(model_uri, model_name)
    version_number = model_version.version
    
    print(f"[SUCCESS] Model registered successfully as Version {version_number}.")

    # Set the newly registered best model alias to Production
    print(f"Setting Version {version_number} alias to 'Production'...")
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=str(version_number)
    )
    
    print("[SUCCESS] Production transition complete. The API is now serving the latest model.")

if __name__ == "__main__":
    register_best_model()