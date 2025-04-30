import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from loguru import logger
import sys

# Set up logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Log to stderr
logger.add("logs/tune.log", level="DEBUG", rotation="500 KB", backtrace=True, diagnose=True)  # Optional file logging

def objective(trial):
    try:
        df = pd.read_csv("data/iris.csv")  # Adjusted path to be relative to project root
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return acc

    except Exception as e:
        logger.exception("An error occurred during the tuning process", e)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        logger.success(f"Best hyperparameters: {study.best_params}")
    except Exception as e:
        logger.exception("Tuning failed at top level", e)
