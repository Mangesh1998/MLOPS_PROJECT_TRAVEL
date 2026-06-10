"""
train_models.py
=============================================================
Training script for Travel_ML_System (Flight Price Prediction).
Reads ../Datasets/flights.csv, trains multiple regression models,
and saves all .pkl files into the model/ directory.

Usage:
    cd Travel_ML_System
    python train_models.py
=============================================================
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -- skipping XGBRegressor training.")

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---- Config ------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'flights.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_ORDERING = [
    'from_Florianopolis (SC)', 'from_Sao_Paulo (SP)', 'from_Salvador (BH)',
    'from_Brasilia (DF)', 'from_Rio_de_Janeiro (RJ)', 'from_Campo_Grande (MS)',
    'from_Aracaju (SE)', 'from_Natal (RN)', 'from_Recife (PE)',
    'destination_Florianopolis (SC)', 'destination_Sao_Paulo (SP)',
    'destination_Salvador (BH)', 'destination_Brasilia (DF)',
    'destination_Rio_de_Janeiro (RJ)', 'destination_Campo_Grande (MS)',
    'destination_Aracaju (SE)', 'destination_Natal (RN)', 'destination_Recife (PE)',
    'flightType_economic', 'flightType_firstClass', 'flightType_premium',
    'agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops',
    'week_no', 'week_day', 'day'
]


# ---- Data Loading & Preprocessing --------------------------------------------
def load_and_preprocess(path: str):
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Shape: {df.shape}")

    df['date'] = pd.to_datetime(df['date'])
    df['week_day'] = df['date'].dt.weekday
    df['month']    = df['date'].dt.month
    df['week_no']  = df['date'].dt.isocalendar().week
    df['year']     = df['date'].dt.year
    df['day']      = df['date'].dt.day

    df.rename(columns={"to": "destination"}, inplace=True)
    df['flight_speed'] = round(df['distance'] / df['time'], 2)
    df = pd.get_dummies(df, columns=['from', 'destination', 'flightType', 'agency'])

    drop_cols = ['time', 'flight_speed', 'month', 'year', 'distance', 'date']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    rename_map = {
        'from_Sao Paulo (SP)':              'from_Sao_Paulo (SP)',
        'from_Rio de Janeiro (RJ)':         'from_Rio_de_Janeiro (RJ)',
        'from_Campo Grande (MS)':           'from_Campo_Grande (MS)',
        'destination_Sao Paulo (SP)':       'destination_Sao_Paulo (SP)',
        'destination_Rio de Janeiro (RJ)':  'destination_Rio_de_Janeiro (RJ)',
        'destination_Campo Grande (MS)':    'destination_Campo_Grande (MS)',
    }
    df.rename(columns=rename_map, inplace=True)

    X = df.drop('price', axis=1)
    Y = df['price']
    available = [f for f in FEATURES_ORDERING if f in X.columns]
    X = X[available]

    print(f"[INFO] Features: {X.shape[1]}  |  Samples: {len(Y)}")
    return X, Y


def evaluate(name, model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    mae  = mean_absolute_error(Y_test, Y_pred)
    mse  = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(Y_test, Y_pred)
    print(f"   {name:35s} MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
    return r2


def save_pkl(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"   [SAVED] {filename}  ({size_kb:.1f} KB)")


# ---- Main --------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Flight Price Prediction -- Model Training")
    print("=" * 60)

    X, Y = load_and_preprocess(DATA_PATH)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    print("\n[INFO] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    save_pkl(scaler, 'scaling.pkl')
    save_pkl(scaler, 'scaling_1.pkl')

    # Linear Regression
    print("\n[TRAIN] Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_sc, Y_train)
    evaluate("LinearRegression", lr, X_test_sc, Y_test)
    save_pkl(lr, 'lr_model.pkl')
    save_pkl(lr, 'lr_model_1.pkl')

    # Lasso
    print("\n[TRAIN] Lasso...")
    lasso = Lasso(alpha=1.0, max_iter=5000)
    lasso.fit(X_train_sc, Y_train)
    evaluate("Lasso", lasso, X_test_sc, Y_test)
    save_pkl(lasso, 'lasso_model.pkl')
    save_pkl(lasso, 'lasso_model_1.pkl')

    # Ridge
    print("\n[TRAIN] Ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, Y_train)
    evaluate("Ridge", ridge, X_test_sc, Y_test)
    save_pkl(ridge, 'ridge_model.pkl')
    save_pkl(ridge, 'ridge_model_1.pkl')

    # Decision Tree
    print("\n[TRAIN] Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=15, random_state=42)
    dt.fit(X_train_sc, Y_train)
    evaluate("DecisionTree", dt, X_test_sc, Y_test)
    save_pkl(dt, 'dt_model.pkl')

    # Random Forest (GridSearchCV)
    print("\n[TRAIN] Random Forest (GridSearchCV -- this may take a few minutes)...")
    param_dict = {
        'n_estimators': [300],
        'max_depth': [15],
        'min_samples_split': [10],
        'max_features': ['sqrt'],
        'n_jobs': [2]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(estimator=rf, param_grid=param_dict,
                           cv=3, verbose=1, scoring='r2')
    rf_grid.fit(X_train_sc, Y_train)
    best_rf = rf_grid.best_estimator_
    evaluate("RandomForest (best)", best_rf, X_test_sc, Y_test)
    save_pkl(best_rf, 'rf_model.pkl')
    save_pkl(best_rf, 'rf_model_1.pkl')

    # XGBoost (optional)
    if HAS_XGB:
        print("\n[TRAIN] XGBoost...")
        xgb = XGBRegressor(n_estimators=300, max_depth=10,
                            learning_rate=0.1, random_state=42, verbosity=0)
        xgb.fit(X_train_sc, Y_train)
        evaluate("XGBoost", xgb, X_test_sc, Y_test)
        save_pkl(xgb, 'xgb_model.pkl')

    print("\n" + "=" * 60)
    print("  [DONE] All models trained and saved to model/")
    print("=" * 60)


if __name__ == '__main__':
    main()
