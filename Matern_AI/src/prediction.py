import joblib
import pandas as pd

def make_prediction(model_path, scaler_path, sample):
    """
    Loads the trained model and scaler to make predictions on a new sample.

    Args:
        model_path (str): Path to the saved model file.
        scaler_path (str): Path to the saved scaler file.
        sample (list): Feature values for a single sample.

    Returns:
        str: Predicted risk level.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    sample_df = pd.DataFrame([sample], columns=scaler.feature_names_in_)
    sample_scaled = scaler.transform(sample_df)

    predicted_label = model.predict(sample_scaled)[0]

    risk_labels = {2: "High Risk", 1: "Mid Risk", 0: "Low Risk"}
    return risk_labels[predicted_label]
