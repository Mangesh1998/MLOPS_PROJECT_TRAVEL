"""
model_training.py
RandomForestModel class for training and evaluating flight price prediction.
"""
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Trains a Random Forest Regressor for flight price prediction."""

    def __init__(self, X: pd.DataFrame, Y: pd.Series):
        self.X = X
        self.Y = Y
        self.model = None
        self.scaler = None

    def random_forest(self):
        """Train the Random Forest model and save artifacts."""
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=0.20, random_state=42
        )

        # Scale features
        logger.info("Scaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter grid
        param_dict = {
            'n_estimators': [300],
            'max_depth': [15],
            'min_samples_split': [10],
            'max_features': ['sqrt'],
            'n_jobs': [2]
        }

        logger.info("Running GridSearchCV for Random Forest...")
        rf_model = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(
            estimator=rf_model,
            param_grid=param_dict,
            cv=3, verbose=2, scoring='r2'
        )
        rf_grid.fit(X_train_scaled, Y_train)
        self.model = rf_grid.best_estimator_

        # Evaluate
        Y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred)

        logger.info(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        # Save model and scaler
        with open('model/rf_model_1.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('model/scaling_1.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info("Model and scaler saved successfully.")
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
