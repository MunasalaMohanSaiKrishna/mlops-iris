
# MLOps Pipeline with GitHub Actions and MLflow

This project demonstrates an end-to-end MLOps pipeline, integrating GitHub Actions for CI/CD, MLflow for model management, and a **Streamlit UI** for model prediction.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Running the Application Locally](#running-the-application-locally)
5. [CI/CD Pipeline with GitHub Actions](#cicd-pipeline-with-github-actions)
6. [MLflow Integration](#mlflow-integration)
7. [Deployment](#deployment)
8. [Folder Structure](#folder-structure)
9. [Future Enhancements](#future-enhancements)

---

## Project Overview

This project automates the machine learning workflow using GitHub Actions for CI/CD and MLflow for model management. The model is a **Random Forest classifier** trained on the **Iris dataset** and stored in **MLflow**. The model is deployed via a **Streamlit UI**, allowing users to input features of the Iris flower and get a predicted species.

---

## Technologies Used

- **Python 3.x**: The primary programming language.
- **GitHub Actions**: CI/CD pipeline to automate testing, training, and deployment.
- **MLflow**: Model tracking, versioning, and registry.
- **Streamlit**: Web UI for model interaction.
- **Scikit-learn**: Machine learning model (Random Forest classifier).
- **Docker**: Containerization for deployment.
- **Uvicorn**: ASGI server to run the Streamlit app.

---

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/mlops-project.git
   cd mlops-project
   ```

2. **Create a virtual environment**:
   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   Install all necessary dependencies listed in the `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MLflow**:
   - **Install and configure MLflow**:
     If you're running MLflow locally, ensure you have a local tracking server running.
     ```bash
     mlflow ui
     ```
     This will start the MLflow UI at `http://localhost:5000`.

   - **MLflow Tracking URI**:
     To log models to a specific MLflow server, set the `MLFLOW_TRACKING_URI` environment variable:
     ```bash
     export MLFLOW_TRACKING_URI=http://localhost:5000  # Or your server's URL
     ```

---

## Running the Application Locally

To run the application locally with **Streamlit**:

1. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. This will open the Streamlit interface in your browser at `http://localhost:8501`, where you can input Iris features and get predictions.

---

## CI/CD Pipeline with GitHub Actions

The CI/CD pipeline is set up using **GitHub Actions**. The pipeline automates the following tasks:
- **Install Dependencies**: Installs required Python packages.
- **Run Tests**: Executes unit tests to validate code.
- **Train the Model**: Trains the model on the Iris dataset and logs it to MLflow.
- **Push Model to MLflow**: Pushes the trained model to MLflow for versioning and management.
- **Deploy**: Deploys the model to the production or staging environment (optional).

### GitHub Actions Workflow (`.github/workflows/ci.yml`):

---

## MLflow Integration

**MLflow** is used to track experiments, log models, and manage model versions. Each time a model is trained, it's logged with the associated parameters and metrics.

### MLFLOW (`src\train.py`):

---

## Deployment

1. **Containerize with Docker**:
   This project uses Docker to containerize the Streamlit app for easy deployment.

2. **Deploy to Render**:
   - Push your code to GitHub.
   - Go to [Render](https://render.com/), create a new **Web Service**, and link your GitHub repository.
   - Render will automatically build and deploy the app using Docker.

---

## Folder Structure

```
mlops-project/
├── .github/
│   └── workflows/
│       └── mlops.yml          # GitHub Actions CI/CD pipeline
├── models/
│   └── best_model.pkl         # Trained ML model
├── src/
│   ├── train.py               # Model training script
│   ├── tune.py      # Script to log the model to MLflow
│   ├── evaluate.py
│   ├── predict.py
│   ├── validate.py
├── streamlit_app.py           # Streamlit UI for predictions
├── Dockerfile                 # Dockerfile for containerization
├── requirements.txt           # Python dependencies
├── tests/
│   └── test_model.py          # Unit tests for model code
└── README.md                  # Project documentation
```

---

## Future Enhancements

- **Automated Hyperparameter Tuning**: Implement hyperparameter tuning for the model using tools like `Optuna` or `GridSearchCV`.
- **Model Drift Detection**: Implement model drift detection and automatic retraining.
- **Model Explainability**: Integrate model explainability tools like `SHAP` or `LIME`.
- **Multi-Model Management**: Extend MLflow to support multi-model management for different algorithms or datasets.
- **Monitoring**: Add monitoring to track model performance in production.
