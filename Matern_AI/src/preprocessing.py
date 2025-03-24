import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Cleans, encodes, and scales the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        tuple: Scaled training, validation, and test sets (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test).
    """
    df = df.drop_duplicates().copy()

    # Fix incorrect HeartRate value
    df.loc[df["HeartRate"] == 7, "HeartRate"] = df["HeartRate"].mode()[0]

    # Encode categorical labels safely
    df["RiskLevel"] = df["RiskLevel"].replace({"high risk": 2, "mid risk": 1, "low risk": 0})
    df = df.infer_objects(copy=False)  

    # Split features and target
    X = df.drop(columns=["RiskLevel"])
    y = df["RiskLevel"]

    # Train-test split (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Scale data while keeping feature names
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save the scaler
    joblib.dump(scaler, "scaler.pkl")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
