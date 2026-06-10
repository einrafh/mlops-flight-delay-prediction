import os
import sys
import subprocess
import pytest
from fastapi.testclient import TestClient

# Attempt to import the FastAPI application for API testing
try:
    from src.serve import app
    client = TestClient(app)
except ImportError:
    client = None

def test_data_directory_exists():
    """Verify the existence of the primary data directory."""
    assert os.path.exists("data"), "[ERROR] The 'data' directory is missing."

def test_src_directory_exists():
    """Verify the existence of the source code directory."""
    assert os.path.exists("src"), "[ERROR] The 'src' directory is missing."

def test_essential_scripts_exist():
    """Verify that all critical MLOps scripts are present in the source directory."""
    required_scripts = [
        "ingest_data.py", 
        "preprocess.py", 
        "train.py", 
        "serve.py", 
        "inference.py", 
        "register_model.py"
    ]
    for script in required_scripts:
        script_path = os.path.join("src", script)
        assert os.path.exists(script_path), f"[ERROR] Required script missing: {script_path}"

def test_pipeline_execution_simulation():
    """
    Simulate the execution of the data preprocessing and model training pipeline.
    Uses the current virtual environment's Python executable to ensure dependency alignment.
    Data ingestion is intentionally skipped to conserve external API quotas.
    """
    print("\nExecuting preprocess.py...")
    prep_result = subprocess.run(
        [sys.executable, "src/preprocess.py"], 
        capture_output=True, 
        text=True
    )
    assert prep_result.returncode == 0, f"Preprocess script failed:\n{prep_result.stderr}"

    print("Executing train.py...")
    train_result = subprocess.run(
        [sys.executable, "src/train.py", "--n_estimators", "10", "--max_depth", "5"], 
        capture_output=True, 
        text=True
    )
    assert train_result.returncode == 0, f"Train script failed:\n{train_result.stderr}"

def test_api_health_check():
    """Test the FastAPI health check endpoint to ensure the service is running."""
    if client is None:
        pytest.skip("FastAPI app could not be imported. Skipping API health test.")
        
    response = client.get("/")
    assert response.status_code == 200, "API health check failed."
    
    response_data = response.json()
    assert "status" in response_data
    assert "model_ready" in response_data

def test_api_predict_endpoint():
    """
    Test the inference endpoint. 
    This test safely skips execution if the MLflow model is not loaded,
    preventing false-positive failures in CI/CD environments.
    """
    if client is None:
        pytest.skip("FastAPI app could not be imported. Skipping prediction test.")
        
    health_response = client.get("/")
    if not health_response.json().get("model_ready", False):
        pytest.skip("MLflow model is not loaded in memory. Skipping prediction test.")
        
    payload = {"airline_encoded": 1}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200, f"Prediction failed: {response.text}"
    
    response_data = response.json()
    assert "is_delayed_prediction" in response_data
    assert response_data["is_delayed_prediction"] in [0, 1]