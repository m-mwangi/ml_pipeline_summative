import pandas as pd
import joblib

def make_prediction(model_path, scaler_path, feature_names, sample):
    """Loads the trained model and makes predictions."""
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Convert sample to DataFrame with column names
    sample_df = pd.DataFrame([sample], columns=feature_names)
    
    # Scale the sample
    sample_scaled = scaler.transform(sample_df)

    # Make prediction
    predicted_label = model.predict(sample_scaled)[0]

    # Convert numerical prediction to readable label
    risk_labels = {2: "High Risk", 1: "Mid Risk", 0: "Low Risk"}
    predicted_risk = risk_labels[predicted_label]

    return predicted_risk
