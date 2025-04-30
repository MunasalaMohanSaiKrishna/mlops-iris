import joblib
import pandas as pd
from loguru import logger
import sys

# Set up logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Log to stderr
logger.add("logs/prediction.log", level="DEBUG", rotation="500 KB", backtrace=True, diagnose=True)

try:
    # Log that the model is being loaded
    logger.info("Loading the model from 'models/best_model.pkl'")
    model = joblib.load("models/best_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.exception("Failed to load the model", e)
    raise

# Sample input data for prediction
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)", 
        "sepal width (cm)", 
        "petal length (cm)", 
        "petal width (cm)"
    ]
)
logger.debug("Sample data: {}", sample)

try:
    # Log the prediction process
    logger.info("Making prediction for the sample data")
    prediction = model.predict(sample)
    target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_label = int(prediction[0])
    logger.info("Predicted class: {} ({})", predicted_label, target_names[predicted_label])

except Exception as e:
    logger.exception("Prediction failed", e)
    raise
