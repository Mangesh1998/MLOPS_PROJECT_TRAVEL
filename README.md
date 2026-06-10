# ✈️ Travel Recommendation & Price Prediction System (MLOps)

Welcome to the **Travel Recommendation & Price Prediction System**! This repository is an end-to-end MLOps solution containing three distinct Machine Learning sub-systems designed to power next-generation travel portals. The project integrates model training pipelines, experiment tracking, Docker containerization, orchestration, and interactive Streamlit UIs.

---

## 🚀 Project Overview

The project consists of three main sub-systems:

### 1. 📈 Flight Price Prediction & Tracking (`Travel_ML_System`)
- **Objective**: Predicts flight ticket prices based on departure city, destination, agency, date, and flight class.
- **Models Used**: Linear Regression, Lasso, Ridge, Decision Trees, Random Forest, and XGBoost.
- **MLflow Tracking**: Integrated experiment tracking to log runs, parameters, metrics (MAE, RMSE, R²), and model artifacts.
- **Deployment**: Exposes a Flask API and an interactive Streamlit frontend.
- **Orchestration**: Orchestrated using Apache Airflow DAGs (`Airflow_Travel_System`).

### 2. 🏨 Hotel Recommender System (`Hotel Recommender System`)
- **Objective**: Suggests the best hotels for travel users.
- **Model Used**: Collaborative Filtering (CF) recommendation engine trained on user hotel interaction data.
- **Deployment**: Streamlit interface allowing users to enter/select user profiles and view top hotel recommendations.

### 3. 👤 Gender Classification Model (`Gender Classification Model`)
- **Objective**: Classifies gender based on user-supplied text profiles.
- **Model Used**: Uses `sentence-transformers` to generate text embeddings followed by a Tuned Logistic Regression classifier.
- **Deployment**: Streamlit web app for real-time text input and classification.

---

## 🖥️ Streamlit Applications

Here are the visual interfaces for the three Streamlit applications included in this repository:

### 1️⃣ Flight Price Prediction App
Use this app to input travel criteria and get real-time price predictions from the best-performing model.

<img width="1025" height="658" alt="{6E3664E3-87F2-4875-A33F-4478434C58F3}" src="https://github.com/user-attachments/assets/17529658-9728-431c-9b49-53815eff2bb1" />


---

### 2️⃣ Hotel Recommender App
Use this app to generate personalized hotel recommendations using Collaborative Filtering.

<img width="1682" height="667" alt="{B1A08922-5710-41CD-87CC-CC40CDAD5266}" src="https://github.com/user-attachments/assets/18de8cdb-d99c-4b69-8163-da787a89a0a5" />


---

### 3️⃣ Gender Classification App
Use this app to analyze text profiles and predict the gender classification using Sentence Transformer embeddings.

<img width="998" height="727" alt="{11BCF9D1-7E2D-4299-9CCD-452EA88E2683}" src="https://github.com/user-attachments/assets/cc6fe500-943b-420c-b09c-942d86e11abe" />


---

## 🛠️ Technology Stack

- **Core Logic**: Python 3
- **Machine Learning**: Scikit-Learn, XGBoost, Sentence-Transformers
- **Data Manipulation**: Pandas, NumPy
- **Frontend / UIs**: Streamlit
- **Backend API**: Flask
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **Containerization & Devops**: Docker, Docker Compose, Kubernetes (YAML manifests)

---

## 📂 Directory Structure

```text
├── Airflow_Travel_System/      # Airflow DAGs, config, and Docker setup
│   ├── airflow/                 # Airflow configuration and admin password
│   ├── flight_price_prediction_dag.py
│   ├── Dockerfile
│   └── docker-compose.yml
├── Datasets/                   # Flights, Hotels, and Users CSV datasets
├── Gender Classification Model/# Text classification model & Streamlit app
│   ├── model/                  # Serialized embeddings and models
│   ├── app.py
│   ├── train_gender_model.py
│   ├── streamlit_app.py
│   └── Dockerfile
├── Hotel Recommender System/   # Recommendation engine & Streamlit app
│   ├── train_hotel_model.py
│   ├── streamlit_app.py
│   └── cf_recommender_model.pkl
├── Travel_ML_System/           # Flight price prediction & Flask/Streamlit
│   ├── model/                  # Serialized regressor models
│   ├── app.py                  # Flask API
│   ├── streamlit_app.py        # Streamlit App
│   ├── train_models.py         # Training script
│   ├── flight-price-pred-mlflow.py # MLflow integration script
│   └── Dockerfile
├── setup.py                    # Master setup script (Installs deps & trains models)
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Setup

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 1. Install Dependencies & Train Models
You can install all required packages and train the models for all three applications with a single master script:

```bash
python setup.py
```

This script will:
- Install all dependencies (e.g., pandas, scikit-learn, xgboost, sentence-transformers, streamlit, mlflow, flask).
- Train all regression models for Flight Price Prediction.
- Download the sentence-transformer models and train the Gender Classifier.
- Train the Hotel Collaborative Filtering recommender.

### 2. Run the Streamlit Applications

Once training is complete, you can launch the applications:

#### Flight Price Streamlit App:
```bash
cd Travel_ML_System
streamlit run streamlit_app.py
```

#### Hotel Recommender App:
```bash
cd "Hotel Recommender System"
streamlit run streamlit_app.py
```

#### Gender Classification App:
```bash
cd "Gender Classification Model"
streamlit run streamlit_app.py
```

### 3. Run the Flight Price Flask API (Alternative Frontend backend)
```bash
cd Travel_ML_System
python app.py
```
*API will run locally at `http://localhost:8000`.*

### 4. MLflow Experiment Tracking
To log flight price prediction metrics to MLflow:
```bash
# Start MLflow UI on port 5000
mlflow ui --port 5000

# In another terminal, run:
cd Travel_ML_System
python flight-price-pred-mlflow.py
```

---
