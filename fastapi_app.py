from fastapi import FastAPI
from pydantic import BaseModel
import joblib
# import numpy as np
import logging

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model = joblib.load("models/best_model.pkl")

# Mapping of class labels to target names
target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Define input data format
class Features(BaseModel):
    features: list[float]

# Define the prediction endpoint
@app.post("/predict")
def predict(f: Features):
    try:
        # Make prediction
        pred = model.predict([f.features])
        
        # Map the prediction to the class name
        predicted_label = int(pred[0])
        predicted_class = target_names[predicted_label]

        # Return the predicted class name
        return {"prediction": predicted_class}

    except Exception as e:
        logging.exception("Prediction failed")
        return {"error": "Prediction failed", "message": str(e)}