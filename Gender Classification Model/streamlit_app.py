"""
streamlit_app.py -- Gender Classification Model
Supports both sentence-transformers and TF-IDF fallback modes.
Run:  streamlit run streamlit_app.py
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


def get_embedding_mode():
    """Read which embedding mode was used at training time."""
    mode_file = os.path.join(MODEL_DIR, 'embedding_mode.txt')
    if os.path.exists(mode_file):
        with open(mode_file) as f:
            return f.read().strip()
    return 'sentence_transformers'  # default assumption


@st.cache_resource
def load_models():
    mode = get_embedding_mode()
    pca_model    = pickle.load(open(os.path.join(MODEL_DIR, 'pca_1.pkl'), 'rb'))
    scaler_model = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_1.pkl'), 'rb'))
    lr_model     = pickle.load(open(os.path.join(MODEL_DIR, 'tuned_logistic_regression_model_1.pkl'), 'rb'))

    if mode == 'tfidf':
        tfidf = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb'))
        return 'tfidf', tfidf, pca_model, scaler_model, lr_model
    else:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
        return 'transformers', embed_model, pca_model, scaler_model, lr_model


def get_name_embedding(name: str, mode: str, embed_obj) -> np.ndarray:
    if mode == 'tfidf':
        return embed_obj.transform([name]).toarray()
    else:
        return embed_obj.encode([name])


def predict_gender(input_data, mode, embed_obj, pca, scaler, lr_model):
    df = pd.DataFrame([input_data])

    # Encode company
    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])

    # Name embedding
    name_emb = get_name_embedding(input_data['name'], mode, embed_obj)
    name_emb_pca = pca.transform(name_emb)

    # Combine with numerical features
    numerical_features = df[['code', 'company_encoded', 'age']].values
    X = np.hstack((name_emb_pca, numerical_features))

    X = scaler.transform(X)
    y_pred = lr_model.predict(X)
    return y_pred[0]


def main():
    st.set_page_config(page_title="Gender Classification", layout="centered")
    st.title("Gender Classification Model")
    st.markdown(
        "Predicts the gender of a traveller based on their **name**, **age**, "
        "**company**, and **user code**."
    )

    with st.spinner("Loading models..."):
        mode, embed_obj, pca_model, scaler_model, lr_model = load_models()

    if mode == 'tfidf':
        st.info("Running in TF-IDF mode (character n-gram name features).")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        name     = st.text_input("Username", "Charlotte Johnson")
        usercode = st.number_input("Usercode (ID)", min_value=0, max_value=1339, value=100)
    with col2:
        age     = st.slider("Traveller Age", min_value=21, max_value=65, value=35)
        company = st.selectbox(
            "Company Name",
            ["Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA", "4You"]
        )

    data = {'code': usercode, 'company': company, 'name': name, 'age': age}

    if st.button('Predict Gender', use_container_width=True):
        with st.spinner("Generating prediction..."):
            prediction = predict_gender(data, mode, embed_obj, pca_model, scaler_model, lr_model)

        gender = 'Female' if prediction == 0 else 'Male'
        st.success(f"**Predicted Gender: {gender}**")


if __name__ == "__main__":
    main()
