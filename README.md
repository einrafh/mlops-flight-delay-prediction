# Domestic Flight Delay Prediction

This project implements a production oriented Machine Learning system to predict the probability of domestic flight delays. Unlike traditional projects relying on static datasets, this system is designed to handle dynamic data streams, including flight schedules and weather conditions, utilizing Continual Learning principles to mitigate data drift.

## Project Objectives

* ML Task: Binary Classification predicting On Time versus Delay.
* Success Metrics: Maintain high F1 Score and Accuracy thresholds while ensuring low latency for inference API calls.
* Continual Learning System: Implement automated model retraining triggered by either weekly schedules or detected performance degradation.

## Directory Structure

The repository follows standard Data Science conventions.

* config/ : Configuration files and model parameters
* data/ : 
  * processed/ : Cleaned data and extracted features
  * raw/ : Raw data extracted via API
* models/ : Trained binary model files
* notebooks/ : Jupyter Notebooks for Exploratory Data Analysis
* src/ : Source code for data ingestion, model training, and inference API
* .gitignore : Version control exclusion rules
* README.md : Project documentation
* requirements.txt : Python package dependencies

## Environment Setup

This project uses a standard Python virtual environment for dependency management. Ensure Python 3.10 or higher is installed on your system.

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment by executing python -m venv venv.
4. Activate the virtual environment.
5. Install the necessary packages by executing pip install -r requirements.txt.

## Data Pipeline Execution

The data pipeline consists of automated ingestion and preprocessing scripts designed for continual learning.

1. Data Ingestion:
Execute the ingestion script to fetch real time flight data from the Aviationstack API. The script prevents destructive overwrites by appending timestamps to the JSON outputs.
`python src/ingest_data.py`

2. Data Preprocessing:
Execute the preprocessing script to clean the latest raw JSON file. This script filters cancelled flights, calculates delay durations, generates the binary classification target (`is_delayed`), and outputs a structured CSV file.
`python src/preprocess.py`

## Data Version Control (DVC)

This project utilizes DVC to manage large datasets and track data lineage without bloating the Git repository. Raw datasets are stored locally or in remote object storage, while only lightweight metadata pointer files (.dvc) are committed to version control.

Data Tracking Workflow:
1. Data addition: Execute dvc add data/raw to calculate the hash of the current dataset state.
2. Versioning: Commit the resulting data/raw.dvc file to Git to lock the dataset version to a specific code commit.
3. Continual Learning Simulation: After executing the ingestion script to append new data, repeat the dvc add and git commit sequence to register the new state.
4. Audit: Execute dvc diff to inspect modifications between data versions across commits.

## Model Inference and Versioning

Currently, the active model used for the inference phase is FlightDelayModel (Version 2), which is set to the Production stage.

Selection Rationale:
This version was selected because it utilizes the optimal hyperparameter configuration (Random Forest with `n_estimators=200` and `max_depth=20`), proven to deliver the most stable classification performance during the MLflow tracking experiments. Furthermore, it has successfully passed programmatic verification for inference readiness using `mlflow.pyfunc.load_model`. Specific metadata ensuring the data lineage of this production model is managed and tracked via DVC in the `model_metadata.yaml.dvc` file.

## Container Orchestration

This project uses Docker Compose to orchestrate the MLflow Tracking Server and the FastAPI Model Inference Service within a custom network, ensuring persistent storage and service discovery. Ensure that Docker and Docker Desktop are installed and running on your system.

1. Start the services:
Execute the following command in the project root directory to build and start the entire MLOps environment.
`docker compose up -d --build`

2. Access the interfaces:
Once the containers are running, access the services via your web browser.
* MLflow Tracking UI: http://localhost:5000 (Used to view training experiments and Model Registry)
* FastAPI Inference: http://localhost:8000/docs (Interactive Swagger UI to test the /predict endpoint)

3. Stop the services:
Execute the following command to safely stop the application and remove the running containers.
`docker compose down`

## Branching Strategy

Development strictly adheres to the GitHub Flow methodology.

* All experiments, fixes, and feature additions must be executed in isolated branches.
* Changes are merged into the main branch exclusively through Pull Requests after code review and validation.