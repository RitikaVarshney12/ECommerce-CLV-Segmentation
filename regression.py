"""
regression.py
--------------
Customer Lifetime Value (CLV) prediction using Linear Regression.

Includes:
    - Model training
    - Evaluation metrics: RMSE, R2 Score, MAPE
    - Helpers for Actual vs Predicted and Residual plots
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Train a Linear Regression model on the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the test set.
    Returns RMSE, R2 Score, and MAPE, along with predictions for plotting.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE is undefined for y_test == 0; guard against that edge case
    mask = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100

    residuals = y_test - y_pred

    return {
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "y_pred": y_pred,
        "y_test": y_test,
        "residuals": residuals,
    }


def predict_clv(model: LinearRegression, scaler, input_features: list) -> float:
    """
    Predict CLV for a single new customer.
    input_features must follow the REGRESSION_FEATURES order defined
    in preprocessing.py: [Age, Total_Orders, Average_Order_Value,
    Purchase_Frequency, Days_Since_Last_Purchase, Customer_Tenure]
    """
    X = np.array(input_features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return max(prediction, 0.0)
