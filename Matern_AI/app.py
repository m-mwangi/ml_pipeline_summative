from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# Initialize FastAPI app
app = FastAPI()

# Get absolute paths for the model and scaler files
base_dir = os.path.dirname(__file__)  # Get the directory where app.py is located
model_path = os.path.join(base_dir, 'models', 'xgb_maternal_health.pkl')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

# Check if model and scaler files exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    logging.error(f"Model or scaler files are missing! Model path: {model_path}, Scaler path: {scaler_path}")

# Load the saved model and scaler (ensure the model and scaler are in the 'models/' folder)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)  # Load the scaler used during training

# Define input data model for prediction
class PredictionRequest(BaseModel):
    Age: float
    Systolic_BP: float
    Diastolic_BP: float
    Blood_Sugar: float
    Body_Temperature: float
    Heart_Rate: float

# Root route for the welcome message
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Matern_AI Health Prediction app!"}

# Route for model prediction
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data into a DataFrame for the model
        data = pd.DataFrame([request.dict()])

        # Preprocess input data (assuming you used scaling in your pipeline)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)  # Adjust scaling as per your pipeline

        # Make prediction
        prediction = model.predict(scaled_data)

        # Convert the numerical prediction to a human-readable format (if needed)
        risk_labels = {2: "High Risk", 1: "Mid Risk", 0: "Low Risk"}
        predicted_risk = risk_labels[prediction[0]]

        return {"prediction": predicted_risk}

    except Exception as e:
        return {"error": str(e)}

import logging
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from io import StringIO

# Initialize FastAPI app
app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)

# Get absolute paths for the model and scaler files
base_dir = os.path.dirname(__file__)  # Get the directory where app.py is located
model_path = os.path.join(base_dir, 'models', 'xgb_maternal_health.pkl')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

# Check if model and scaler files exist
if not os.path.exists(model_path):
    logging.error(f"Model file not found: {model_path}")

if not os.path.exists(scaler_path):
    logging.error(f"Scaler file not found: {scaler_path}")

# Load the saved model and scaler (ensure the model and scaler are in the 'models/' folder)
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define input data model for prediction
class PredictionRequest(BaseModel):
    Age: float
    Systolic_BP: float
    Diastolic_BP: float
    Blood_Sugar: float
    Body_Temperature: float
    Heart_Rate: float

# Root route for the welcome message
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Matern_AI Health Prediction app!"}

# Route for model prediction
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data into a DataFrame for the model
        data = pd.DataFrame([request.dict()])

        # Preprocess input data using the loaded scaler
        scaled_data = scaler.transform(data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Convert the numerical prediction to a human-readable format (if needed)
        risk_labels = {2: "High Risk", 1: "Mid Risk", 0: "Low Risk"}
        predicted_risk = risk_labels[prediction[0]]

        return {"prediction": predicted_risk}

    except Exception as e:
        return {"error": str(e)}

# Route for retraining the model
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file into a DataFrame
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)

        # Ensure the columns match your original training data
        df = df.dropna()  # Drop missing values

        # Check if 'RiskLevel' column exists
        if 'RiskLevel' in df.columns:
            # Convert the 'RiskLevel' names into numerical labels
            label_encoder = LabelEncoder()
            df['RiskLevel'] = label_encoder.fit_transform(df['RiskLevel'])
        else:
            return {"error": "'RiskLevel' column is missing in the dataset."}

        # Split data into features (X) and target (y)
        X = df.drop('RiskLevel', axis=1)  # Replace 'RiskLevel' with the actual target column name
        y = df['RiskLevel']  # The target variable

        # Apply scaling (use the same scaler that was used in training)
        X_scaled = scaler.transform(X)

        # Retrain the model
        retrained_model = retrain_model(X_scaled, y)

        # Save the retrained model
        joblib.dump(retrained_model, model_path)
        joblib.dump(scaler, scaler_path)  # Save the scaler too

        return {"message": "Model retrained successfully"}

    except Exception as e:
        return {"error": str(e)}

def retrain_model(X, y):
    """Function to retrain the XGBoost model."""
    from xgboost import XGBClassifier
    model = XGBClassifier(eval_metric='mlogloss')
    
    # Fit the model with the provided data
    model.fit(X, y)
    
    return model
