# Domestic Flight Delay Prediction

This project implements a production oriented Machine Learning system to predict the probability of domestic flight delays. Unlike traditional projects relying on static datasets, this system is designed to handle dynamic data streams, including flight schedules, utilizing Continual Learning principles to mitigate data drift.

## Project Objectives

* ML Task: Binary Classification predicting On Time versus Delay.
* Success Metrics: Maintain high F1 Score and Accuracy thresholds while ensuring low latency for inference API calls.
* Continual Learning System: Implement automated model retraining and data ingestion triggered by daily and tri-daily GitHub Actions cron jobs.

## Directory Structure

The repository follows standard Data Science conventions.

* .github/workflows/ : CI/CD automation pipelines (Data Ingestion & Retraining)
* config/ : Configuration files and model parameters
* data/ : 
  * processed/ : Cleaned data and extracted features
  * raw/ : Raw data extracted via API
* models/ : Trained binary model files
* src/ : Source code for data ingestion, model training, and inference API
* tests/ : Pytest suite for end-to-end pipeline and API validation
* app.py : Streamlit Master Control Panel for pipeline orchestration
* .gitignore : Version control exclusion rules
* README.md : Project documentation
* requirements.txt : Python package dependencies

## Environment Setup

This project uses a standard Python virtual environment for dependency management. Ensure Python 3.10 or higher is installed on your system.

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment by executing `python -m venv .venv`.
4. Activate the virtual environment.
5. Install the necessary packages by executing pip install -r requirements.txt.

## Data Pipeline Execution

The data pipeline consists of automated ingestion and preprocessing scripts designed for continual learning.

1. Data Ingestion:
Execute the ingestion script to fetch real time flight data from the Aviationstack API. The script prevents destructive overwrites by appending timestamps to the JSON outputs.
`python src/ingest_data.py`

2. Data Preprocessing:
Execute the preprocessing script to aggregate and clean all historical raw JSON files. This script filters cancelled flights, calculates delay durations, removes duplicates, and outputs a single continuous `processed_flights.csv` file for model training.
`python src/preprocess.py`

## Data Version Control (DVC)

This project utilizes DVC to manage large datasets and track data lineage without bloating the Git repository. Raw datasets are stored locally or in remote object storage, while only lightweight metadata pointer files (.dvc) are committed to version control.

Data Tracking Workflow:
1. Data addition: Execute dvc add data/raw to calculate the hash of the current dataset state.
2. Versioning: Commit the resulting data/raw.dvc file to Git to lock the dataset version to a specific code commit.
3. Continual Learning Simulation: After executing the ingestion script to append new data, repeat the dvc add and git commit sequence to register the new state.
4. Audit: Execute dvc diff to inspect modifications between data versions across commits.

## Model Inference and Versioning

The active model used for the inference phase is dynamically loaded from the MLflow Model Registry. The training pipeline automatically evaluates new runs and promotes models to the Production or Staging environment only if they pass the minimum 80% accuracy threshold.

## Container Orchestration

This project uses Docker Compose to orchestrate the MLflow Tracking Server and the FastAPI Model Inference Service within a custom network, ensuring persistent storage and service discovery. Ensure that Docker and Docker Desktop are installed and running on your system.

1. Start the services:
Execute the following command in the project root directory to build and start the entire MLOps environment.
`docker compose up -d --build`

2. Access the interfaces:
Once the containers are running, access the services via your web browser.
* MLflow Tracking UI: http://localhost:5000 (Used to view training experiments and Model Registry)
* FastAPI Inference: http://localhost:8000/docs (Interactive Swagger UI to test the /predict endpoint)
* Streamlit Master Dashboard: http://localhost:8501 (Central orchestration UI)
* Prometheus Metrics: http://localhost:8000/metrics (System observability and monitoring)

3. Stop the services:
Execute the following command to safely stop the application and remove the running containers.
`docker compose down`

## Branching Strategy

Development strictly adheres to the GitHub Flow methodology.

* All experiments, fixes, and feature additions must be executed in isolated branches.
* Changes are merged into the main branch exclusively through Pull Requests after code review and validation.
* All code must pass the automated pytest suite before merging to ensure pipeline and API integrity.