from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib

# Ensure model and scaler paths are correct
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_maternal_health.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

# Load model safely
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model file not found at: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load scaler (optional)
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Warning: Scaler loading failed, using raw inputs. Error: {e}")
        scaler = None
else:
    print("Warning: Scaler file not found. Using raw inputs.")
    scaler = None

# Define class mapping
CLASS_MAPPING = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# Feature names (ensure correct order)
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

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert named features to ordered array
        input_dict = data.dict()
        features = np.array([[input_dict[feature] for feature in FEATURE_NAMES]])

        # Apply scaling if scaler exists
        features_scaled = features if scaler is None else scaler.transform(features)

        # Make prediction
        numeric_prediction = model.predict(features_scaled)[0]

        # Convert numeric prediction to label
        readable_prediction = CLASS_MAPPING.get(numeric_prediction, "Unknown Risk Level")

        return {"prediction": readable_prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
