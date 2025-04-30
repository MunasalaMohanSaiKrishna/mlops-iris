import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from loguru import logger
import sys

# Set up logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Log to stderr
logger.add("logs/model_training.log", level="DEBUG", rotation="500 KB", backtrace=True, diagnose=True)  # Optional file logging

# Set MLflow experiment
mlflow.set_experiment("mlops-demo")

try:
    # Log that data is being loaded
    logger.info("Loading data from 'data/iris.csv'")
    df = pd.read_csv("data/iris.csv")
    logger.info("Data loaded successfully")

    # Prepare data
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets")

    # Initialize and train model
    logger.info("Training RandomForest model")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")

    # Evaluate the model
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info("Model accuracy: {}", acc)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    logger.info("Logged accuracy to MLflow")

    # Save the trained model
    joblib.dump(model, "models/best_model.pkl")
    logger.info("Model saved to 'models/best_model.pkl'")

    print("Model trained with accuracy:", acc)

except Exception as e:
    logger.exception("An error occurred during the model training process", e)
    raise
