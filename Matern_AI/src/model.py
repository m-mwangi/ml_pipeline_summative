import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_and_evaluate(X_train, X_val, y_train, y_val):
    """
    Trains an XGBoost model with hyperparameter tuning and saves the best model.

    Args:
        X_train (pd.DataFrame): Scaled training features.
        X_val (pd.DataFrame): Scaled validation features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.

    Returns:
        XGBClassifier: Best trained model.
    """
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0]
    }

    xgb_model = XGBClassifier(eval_metric='mlogloss')

    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)

    best_xgb = grid_search_xgb.best_estimator_

    y_val_pred = best_xgb.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    joblib.dump(best_xgb, "xgb_maternal_health.model")

    print(f"Best Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

    return best_xgb
