import mlflow
from mlflow.tracking import MlflowClient

# Connect to local MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment_name = "Flight_Delay_Prediction"
experiment = client.get_experiment_by_name(experiment_name)

if not experiment:
    raise ValueError(f"Experiment '{experiment_name}' not found.")

# Fetch all runs, sorted by best accuracy
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC", "metrics.f1_score DESC"]
)

if len(runs) < 2:
    raise ValueError("Need at least 2 runs to create V1 and V2.")

model_name = "FlightDelayModel"

# Register the second best as Version 1
run_v1 = runs[1].info.run_id
mlflow.register_model(f"runs:/{run_v1}/model", model_name)
print(f"Registered Version 1 from Run ID: {run_v1}")

# Register the best as Version 2
run_v2 = runs[0].info.run_id
mlflow.register_model(f"runs:/{run_v2}/model", model_name)
print(f"Registered Version 2 from Run ID: {run_v2}")