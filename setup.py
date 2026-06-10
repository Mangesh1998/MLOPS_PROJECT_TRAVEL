"""
setup.py — Master Setup Script
=============================================================
Installs all dependencies and trains all 3 ML models.
Run once from the project root:
    python setup.py
=============================================================
"""
import os
import sys
import subprocess


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PACKAGES = [
    "numpy==1.26.4",
    "pandas==2.2.0",
    "scikit-learn==1.3.2",
    "scipy==1.12.0",
    "flask==3.0.2",
    "werkzeug>=3.0.0",
    "joblib==1.3.2",
    "xgboost",
    "mlflow>=2.10.0",
    "streamlit>=1.30.0",
    "sentence-transformers>=2.6.0",
]


def run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result


def install_packages():
    print("\n" + "=" * 60)
    print("  Step 1: Installing Python packages")
    print("=" * 60)
    run([sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES)
    print("✅ All packages installed.")


def train_flight_models():
    print("\n" + "=" * 60)
    print("  Step 2: Training Flight Price Prediction models")
    print("=" * 60)
    script = os.path.join(PROJECT_ROOT, "Travel_ML_System", "train_models.py")
    run([sys.executable, script], cwd=os.path.join(PROJECT_ROOT, "Travel_ML_System"))
    print("✅ Flight price models trained.")


def train_gender_model():
    print("\n" + "=" * 60)
    print("  Step 3: Training Gender Classification model")
    print("  (Requires internet — downloads ~90MB sentence-transformers model)")
    print("=" * 60)
    script = os.path.join(PROJECT_ROOT, "Gender Classification Model", "train_gender_model.py")
    run([sys.executable, script], cwd=os.path.join(PROJECT_ROOT, "Gender Classification Model"))
    print("✅ Gender classification model trained.")


def train_hotel_model():
    print("\n" + "=" * 60)
    print("  Step 4: Training Hotel Recommender model")
    print("=" * 60)
    script = os.path.join(PROJECT_ROOT, "Hotel Recommender System", "train_hotel_model.py")
    run([sys.executable, script], cwd=os.path.join(PROJECT_ROOT, "Hotel Recommender System"))
    print("✅ Hotel recommender model trained.")


def main():
    print("=" * 60)
    print("  MLOPS Project — Master Setup Script")
    print("=" * 60)

    install_packages()
    train_flight_models()
    train_gender_model()
    train_hotel_model()

    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("=" * 60)
    print("""
Next steps — run any of these apps:

  Flight Price (Flask API):
    cd Travel_ML_System
    python app.py
    → Open http://localhost:8000

  Flight Price (Streamlit):
    cd Travel_ML_System
    streamlit run streamlit_app.py

  Gender Classification (Streamlit):
    cd "Gender Classification Model"
    streamlit run streamlit_app.py

  Hotel Recommender (Streamlit):
    cd "Hotel Recommender System"
    streamlit run streamlit_app.py

  MLflow Tracking (for flight price):
    mlflow ui --port 5000
    cd Travel_ML_System
    python flight-price-pred-mlflow.py
""")


if __name__ == "__main__":
    main()
