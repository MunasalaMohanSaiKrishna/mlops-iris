import pandas as pd
from pandera import Column, DataFrameSchema
from loguru import logger
import sys

# Set up logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Log to stderr
logger.add(
    "logs/data_validation.log",
    level="DEBUG",
    rotation="500 KB",
    backtrace=True,
    diagnose=True,
)  # Optional file logging

# Define the schema for validation
schema = DataFrameSchema(
    {
        "sepal length (cm)": Column(float),
        "sepal width (cm)": Column(float),
        "petal length (cm)": Column(float),
        "petal width (cm)": Column(float),
        "target": Column(int),
    }
)

try:
    # Log that the data file is being loaded
    logger.info("Loading data from 'data/iris.csv'")
    df = pd.read_csv("data/iris.csv")
    logger.info("Data loaded successfully")

    # Log the data validation process
    logger.info("Validating the data against the schema")
    schema.validate(df)
    logger.info("Data validation passed.")

except Exception as e:
    logger.exception("An error occurred during data loading or validation", e)
    raise
