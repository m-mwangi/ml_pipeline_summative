from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import joblib

# Ensure model and scaler paths are correct
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_maternal_health.model")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# Validate model and scaler existence
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🔴 Model file not found at: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"🔴 Scaler file not found at: {SCALER_PATH}")

# Load model and scaler with error handling
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    scaler = joblib.load(SCALER_PATH)  
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    raise RuntimeError(f"🚨 Error loading model or scaler: {e}")

# Define class mapping
CLASS_MAPPING = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# Define expected feature names
FEATURE_NAMES = ["Age", "Systolic_BP", "Diastolic_BP", "Blood_Sugar", "Body_Temperature", "Heart_Rate"]

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
    return {"message": "Welcome to the Maternal Health Risk Prediction API!"}

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

        # Apply scaling
        features_scaled = scaler.transform(features)

        # Make prediction
        numeric_prediction = model.predict(features_scaled)[0]

        # Convert numeric prediction to human-readable label
        readable_prediction = CLASS_MAPPING.get(numeric_prediction, "Unknown Risk Level")

        return {"prediction": readable_prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
