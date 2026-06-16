import os
import argparse
import warnings
import pandas as pd
import yaml
import joblib
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Filter warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Initialize environment variables
load_dotenv()

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train_model(n_estimators, max_depth):
    """
    Train a Random Forest model, evaluate its performance, export preprocessing artifacts,
    and conditionally register it to the MLflow Model Registry based on comparative evaluation.
    """
    # Dynamically set tracking URI from environment configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    model_name = config["mlflow"]["model_name"]
    min_accuracy = config["training"]["min_accuracy_threshold"]

    # Initialize MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 1. Data Loading
    processed_file = os.path.join("data", "processed", "processed_flights.csv")
    
    if not os.path.exists(processed_file):
        print(f"[ERROR] Processed data file not found at {processed_file}. Please execute preprocessing first.")
        return

    print(f"[INFO] Loading preprocessed dataset from: {processed_file}")
    df = pd.read_csv(processed_file)

    if df.empty:
        print("[ERROR] The dataset is empty. Cannot proceed with model training.")
        return

    # 2. Feature Engineering & Artifact Serialization
    le = LabelEncoder()
    df['airline'] = df['airline'].fillna('Unknown')
    df['airline_encoded'] = le.fit_transform(df['airline'])
    
    # Save LabelEncoder to a temporary local file
    encoder_path = "label_encoder.pkl"
    joblib.dump(le, encoder_path)
    
    X = df[['airline_encoded']] 
    y = df['is_delayed']

    # 3. Stratified Data Splitting
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("[WARNING] Class distribution too sparse for stratification. Falling back to random split.")
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
        
        print("[INFO] Initiating model training phase...")
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"[INFO] Configuration Evaluated: n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"[INFO] Performance Metrics: Accuracy={acc:.4f}, F1-Score={f1:.4f}")

        # Logging Parameters and Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", "processed_flights.csv")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Logging Model Artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Logging Preprocessing Artifact
        mlflow.log_artifact(encoder_path, artifact_path="preprocessing")
        
        print("[SUCCESS] Experiment and artifacts successfully logged to the MLflow tracking server.")

        # 5. Automated Comparative Evaluation & Model Registry
        if acc >= min_accuracy:
            print(f"\n[EVALUATION PASSED] Model accuracy ({acc:.4f}) meets the baseline threshold ({min_accuracy}).")
            
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            
            print(f"[INFO] Registering model '{model_name}' to MLflow Model Registry...")
            model_version_info = mlflow.register_model(model_uri, model_name)
            client = MlflowClient()
            
            # Comparative Evaluation Logic
            try:
                prod_model = client.get_model_version_by_alias(name=model_name, alias="Production")
                prod_run = mlflow.get_run(prod_model.run_id)
                prod_acc = prod_run.data.metrics.get("accuracy", 0.0)
                
                print(f"[COMPARISON] Current 'Production' model accuracy: {prod_acc:.4f}")
                print(f"[COMPARISON] Newly trained model accuracy: {acc:.4f}")
                
                if acc > prod_acc:
                    print(f"[PROMOTION] New model outperforms current Production ({acc:.4f} > {prod_acc:.4f}).")
                    print(f"[INFO] Setting Model Version {model_version_info.version} alias to 'Production'...")
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="Production",
                        version=str(model_version_info.version)
                    )
                else:
                    print(f"[DEMOTION] New model does not outperform current Production ({acc:.4f} <= {prod_acc:.4f}).")
                    print(f"[INFO] Setting Model Version {model_version_info.version} alias to 'Staging'...")
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="Staging",
                        version=str(model_version_info.version)
                    )
            
            except Exception as e:
                print(f"[INFO] No existing 'Production' model found in registry.")
                print(f"[INFO] Directly promoting Model Version {model_version_info.version} to 'Production'...")
                client.set_registered_model_alias(
                    name=model_name,
                    alias="Production",
                    version=str(model_version_info.version)
                )
                
            print("[SUCCESS] Automated comparative evaluation and stage transition completed.")
        else:
            print(f"\n[EVALUATION FAILED] Model accuracy ({acc:.4f}) is below the required threshold ({min_accuracy}).")
            print("[WARNING] The model will not be registered or transitioned to any deployment stage.")
            
    # Cleaning up local artifact files
    if os.path.exists(encoder_path):
        os.remove(encoder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Delay Prediction Model Training Script")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree.")
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)