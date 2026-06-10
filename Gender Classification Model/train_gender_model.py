"""
train_gender_model.py
=============================================================
Training script for Gender Classification Model.
Reads ../Datasets/users.csv, trains a Logistic Regression
model with sentence embeddings + PCA, and saves .pkl files.

Strategy:
- Primary: sentence-transformers (requires PyTorch)
- Fallback: character n-gram TF-IDF for name features (pure sklearn)

Usage:
    cd "Gender Classification Model"
    python train_gender_model.py
=============================================================
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---- Config ------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'users.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

N_PCA_COMPONENTS = 23   # Must match what app.py expects


# ---- Helpers -----------------------------------------------------------------
def save_pkl(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"   [SAVED] {filename}  ({size_kb:.1f} KB)")


def get_name_embeddings_transformers(names):
    """Use sentence-transformers for rich name embeddings."""
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    embeddings = embed_model.encode(names, show_progress_bar=True)
    print(f"   [INFO] Embedding shape: {embeddings.shape}")
    return embeddings


def get_name_embeddings_tfidf(names, n_features=384):
    """Fallback: character n-gram TF-IDF embeddings for names."""
    print(f"   [INFO] Using TF-IDF character n-gram embeddings (dim={n_features})")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=n_features, strip_accents='unicode'
    )
    embeddings = vectorizer.fit_transform(names).toarray()
    print(f"   [INFO] TF-IDF embedding shape: {embeddings.shape}")
    return embeddings, vectorizer


def main():
    print("=" * 60)
    print("  Gender Classification -- Model Training")
    print("=" * 60)

    # ---- Load data -----------------------------------------------------------
    print(f"\n[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Gender distribution:\n{df['gender'].value_counts()}")

    # Drop rows where gender is 'none' (unknown/undefined)
    df = df[df['gender'].isin(['male', 'female'])].copy()
    print(f"\n   After filtering 'none' gender: {df.shape[0]} rows remain")

    # ---- Encode target -------------------------------------------------------
    # female -> 0, male -> 1
    df['gender_encoded'] = (df['gender'] == 'male').astype(int)

    # ---- Encode company ------------------------------------------------------
    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])

    # ---- Generate name embeddings -------------------------------------------
    use_transformers = False
    tfidf_vectorizer = None

    print("\n[INFO] Attempting to load sentence-transformers...")
    try:
        import torch  # noqa: F401 -- test import
        name_embeddings = get_name_embeddings_transformers(df['name'].tolist())
        use_transformers = True
        print("   [INFO] Using sentence-transformers embeddings.")
    except Exception as e:
        print(f"   [WARN] sentence-transformers unavailable: {e}")
        print("   [INFO] Falling back to TF-IDF character n-gram embeddings.")
        name_embeddings, tfidf_vectorizer = get_name_embeddings_tfidf(df['name'].tolist())

    # ---- PCA on embeddings ---------------------------------------------------
    actual_pca_components = min(N_PCA_COMPONENTS, name_embeddings.shape[1], len(df) - 1)
    print(f"\n[INFO] Applying PCA (n_components={actual_pca_components})...")
    pca = PCA(n_components=actual_pca_components, random_state=42)
    name_embeddings_pca = pca.fit_transform(name_embeddings)
    print(f"   PCA output shape: {name_embeddings_pca.shape}")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # ---- Combine features ----------------------------------------------------
    numerical_features = df[['code', 'company_encoded', 'age']].values
    X = np.hstack((name_embeddings_pca, numerical_features))
    Y = df['gender_encoded'].values
    print(f"\n   Combined feature matrix: {X.shape}")

    # ---- Train/test split ----------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y
    )

    # ---- Scale ---------------------------------------------------------------
    print("\n[INFO] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ---- Logistic Regression (GridSearchCV) ----------------------------------
    print("\n[TRAIN] Logistic Regression (GridSearchCV)...")
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs'],
        'max_iter': [1000],
    }
    lr = LogisticRegression(random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train_sc, Y_train)
    best_lr = grid.best_estimator_

    # ---- Evaluate ------------------------------------------------------------
    Y_pred = best_lr.predict(X_test_sc)
    acc = accuracy_score(Y_test, Y_pred)
    print(f"\n   Best params: {grid.best_params_}")
    print(f"   Test Accuracy: {acc:.4f}")
    print(f"\n{classification_report(Y_test, Y_pred, target_names=['female', 'male'])}")

    # ---- Save artifacts ------------------------------------------------------
    print("\n[INFO] Saving model artifacts...")
    save_pkl(pca,     'pca.pkl')
    save_pkl(pca,     'pca_1.pkl')
    save_pkl(scaler,  'scaler.pkl')
    save_pkl(scaler,  'scaler_1.pkl')
    save_pkl(best_lr, 'tuned_logistic_regression_model.pkl')
    save_pkl(best_lr, 'tuned_logistic_regression_model_1.pkl')

    # Save TF-IDF vectorizer if used (needed by inference apps)
    if tfidf_vectorizer is not None:
        save_pkl(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        # Write a marker so the app knows which embedding strategy was used
        with open(os.path.join(MODEL_DIR, 'embedding_mode.txt'), 'w') as f:
            f.write('tfidf')
        print("   [NOTE] TF-IDF vectorizer saved. App will use TF-IDF for inference.")
    else:
        with open(os.path.join(MODEL_DIR, 'embedding_mode.txt'), 'w') as f:
            f.write('sentence_transformers')

    print("\n" + "=" * 60)
    print("  [DONE] All artifacts saved to model/")
    print("=" * 60)


if __name__ == '__main__':
    main()
