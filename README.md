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

## Branching Strategy

Development strictly adheres to the GitHub Flow methodology.

* All experiments, fixes, and feature additions must be executed in isolated branches.
* Changes are merged into the main branch exclusively through Pull Requests after code review and validation.