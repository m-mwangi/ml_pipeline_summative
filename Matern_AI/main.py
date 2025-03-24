from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Load model safely
MODEL_PATH = os.getenv("MODEL_PATH", "xgb_maternal_health.model")  # Allow override via env var

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define class mapping (adjust based on your dataset)
CLASS_MAPPING = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

# Define expected feature names (Adjust based on dataset)
FEATURE_NAMES = [
    "Age", "Systolic BP", "Diastolic BP", "Blood Sugar", 
    "Body Temperature", "Heart Rate"
]

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for input validation
class PredictionInput(BaseModel):
    Age: float
    Systolic_BP: float
    Diastolic_BP: float
    Blood_Sugar: float
    Body_Temperature: float
    Heart_Rate: float

@app.get("/")
def home():
    return {"message": "Welcome to the Maternal Prediction API!"}

@app.get("/features")
def get_features():
    """Returns the names of the features expected in the input."""
    return {
        "feature_names": FEATURE_NAMES,
        "example_input": {
            "Age": 28.0,
            "Systolic_BP": 120.0,
            "Diastolic_BP": 80.0,
            "Blood_Sugar": 90.0,
            "Body_Temperature": 36.5,
            "Heart_Rate": 75.0
        }
    }

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert named features to array format
        features = np.array([
            data.Age, data.Systolic_BP, data.Diastolic_BP, 
            data.Blood_Sugar, data.Body_Temperature, data.Heart_Rate
        ]).reshape(1, -1)

        numeric_prediction = model.predict(features)[0]  # Extract single value
        
        # Convert numeric prediction to human-readable label
        readable_prediction = CLASS_MAPPING.get(numeric_prediction, "Unknown Risk Level")
        
        return {"prediction": readable_prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
