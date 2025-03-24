import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_and_evaluate(X_train, X_val, y_train, y_val):
    """Trains the XGBoost model, tunes hyperparameters, and saves the best model."""
    
    # Define hyperparameter grid
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0]
    }

    # Initialize the model
    xgb_model = XGBClassifier(eval_metric='mlogloss')

    # Perform Grid Search
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)

    # Get the best model
    best_xgb = grid_search_xgb.best_estimator_

    # Evaluate the model
    y_val_pred = best_xgb.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Save the best model
    joblib.dump(best_xgb, "xgb_maternal_health.pkl")

    print(f"✅ Best Validation Accuracy: {val_acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_val, y_val_pred))

    return best_xgb
