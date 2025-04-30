import streamlit as st
import joblib
import pandas as pd
# import numpy as np

# Load the pre-trained model
model = joblib.load("models/best_model.pkl")

# Mapping of class labels to target names
target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Streamlit UI setup
st.title("Iris Species Prediction")
st.write("Enter the features of the iris flower to get the predicted species.")

# Input fields for features
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.0, value=0.2)

# Feature list for prediction
features = [sepal_length, sepal_width, petal_length, petal_width]

# Create a DataFrame with proper column names
input_data = pd.DataFrame([features], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

# Predict button
if st.button("Predict"):
    # Make prediction
    pred = model.predict(input_data)
    
    # Map the prediction to the class name
    predicted_label = int(pred[0])
    predicted_class = target_names[predicted_label]
    
    # Display the result
    st.write(f"Predicted Iris Species: **{predicted_class}**")
