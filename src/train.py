import os
import glob
import argparse
import warnings
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Filter warnings for cleaner terminal output
warnings.filterwarnings("ignore")

def train_model(n_estimators, max_depth):
    """
    Train a Random Forest model, evaluate its performance, and conditionally 
    register it to the MLflow Model Registry based on predefined thresholds.
    """
    mlflow.set_experiment("Flight_Delay_Prediction")

    # 1. Data Loading
    processed_dir = os.path.join("data", "processed")
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not csv_files:
        print("[ERROR] No processed data found. Please run preprocess.py first.")
        return

    latest_file = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)

    # 2. Feature Engineering
    le = LabelEncoder()
    df['airline'] = df['airline'].fillna('Unknown')
    df['airline_encoded'] = le.fit_transform(df['airline'])
    
    X = df[['airline_encoded']] 
    y = df['is_delayed']

    # 3. Stratified Data Splitting
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback to normal split if class count is too low for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 4. MLflow Experiment Tracking
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Configuration: n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"Metrics: Accuracy={acc:.4f}, F1-Score={f1:.4f}")

        # Logging Parameters and Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", os.path.basename(latest_file))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Logging Model Artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print("[INFO] Experiment successfully logged to MLflow.")

        # 5. Automated Evaluation & Model Registry
        THRESHOLD_ACC = 0.80  # 80% accuracy threshold for production readiness

        if acc >= THRESHOLD_ACC:
            print(f"\n[EVALUATION PASSED] Accuracy ({acc:.4f}) meets the {THRESHOLD_ACC} threshold.")
            
            model_name = "FlightDelayModel"
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            print(f"[INFO] Registering model '{model_name}' to MLflow Model Registry...")
            model_version_info = mlflow.register_model(model_uri, model_name)
            
            # Transition the model to 'Staging'
            client = MlflowClient()
            print(f"[INFO] Transitioning Model Version {model_version_info.version} to 'Staging'...")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version_info.version,
                stage="Staging"
            )
            print("[SUCCESS] Automated registry and stage transition completed.")
        else:
            print(f"\n[EVALUATION FAILED] Accuracy ({acc:.4f}) is below the {THRESHOLD_ACC} threshold.")
            print("[WARNING] Model will not be registered or transitioned to Staging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Training Script")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)