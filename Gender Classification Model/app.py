# Flask endpoint for Gender Classification model inference
import os
import pickle
import warnings

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


def get_embedding_mode():
    """Read which embedding mode was used at training time."""
    mode_file = os.path.join(MODEL_DIR, 'embedding_mode.txt')
    if os.path.exists(mode_file):
        with open(mode_file) as f:
            return f.read().strip()
    return 'sentence_transformers'


EMBEDDING_MODE = get_embedding_mode()

# Load models
scaler_model  = pickle.load(open(os.path.join(MODEL_DIR, 'scaler_1.pkl'), 'rb'))
pca_model     = pickle.load(open(os.path.join(MODEL_DIR, 'pca_1.pkl'), 'rb'))
logistic_model = pickle.load(open(os.path.join(MODEL_DIR, 'tuned_logistic_regression_model_1.pkl'), 'rb'))

if EMBEDDING_MODE == 'tfidf':
    tfidf_vectorizer = pickle.load(open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb'))
    embed_model = None
else:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
    tfidf_vectorizer = None


def get_name_embedding(name: str) -> np.ndarray:
    if EMBEDDING_MODE == 'tfidf':
        return tfidf_vectorizer.transform([name]).toarray()
    else:
        return embed_model.encode([name])


def predict_gender(input_data, lr_model, pca, scaler):
    df = pd.DataFrame([input_data])

    label_encoder = LabelEncoder()
    df['company_encoded'] = label_encoder.fit_transform(df['company'])

    name_emb = get_name_embedding(input_data['name'])
    name_emb_pca = pca.transform(name_emb)

    numerical_features = df[['code', 'company_encoded', 'age']].values
    X = np.hstack((name_emb_pca, numerical_features))
    X = scaler.transform(X)

    y_pred = lr_model.predict(X)
    return y_pred[0]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Gender Classification Model</title>
    <style>
        body { font-family: 'Roboto', sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }
        .container { max-width: 800px; margin: 0 auto; padding: 50px; background-color: #ffffff;
                     border-radius: 12px; box-shadow: 0 20px 30px rgba(0,0,0,0.1); text-align: center; }
        h1 { color: #009688; font-size: 40px; margin-bottom: 20px; }
        form { text-align: left; }
        input[type="text"], input[type="number"], select {
            width: 100%; padding: 14px; margin: 12px 0; border: 2px solid #009688;
            border-radius: 5px; font-size: 18px; background-color: #f9f9f9; color: #555;
            transition: border-color 0.3s ease; box-sizing: border-box; }
        input[type="submit"] { background-color: #009688; color: #ffffff; padding: 16px 32px;
            border: none; border-radius: 6px; cursor: pointer; font-size: 20px;
            margin-top: 25px; transition: background-color 0.3s ease; }
        input[type="submit"]:hover { background-color: #00796b; }
        label { font-size: 18px; font-weight: bold; color: #333; }
        #result { margin-top: 20px; font-size: 24px; color: #00796b; font-weight: bold; }
    </style>
    <script>
        async function submitForm(e) {
            e.preventDefault();
            const form = document.getElementById('predForm');
            const res = await fetch('/predict', { method: 'POST', body: new FormData(form) });
            const data = await res.json();
            document.getElementById('result').textContent = 'Predicted Gender: ' + data.prediction;
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Gender Classification Model</h1>
        <form id="predForm" onsubmit="submitForm(event)">
            <label for="Username">Username:</label>
            <input type="text" name="Username" placeholder="Enter name of traveller" value="Charlotte Johnson">
            <label for="Usercode">Usercode:</label>
            <input type="number" name="Usercode" min="0" max="1339" placeholder="Enter the user id of traveller" value="100">
            <label for="Traveller_Age">Traveller Age:</label>
            <input type="number" name="Traveller_Age" min="21" max="65" placeholder="Enter the age" value="35">
            <label for="company_name">Company Name:</label>
            <select name="company_name">
                <option value="Acme Factory">Acme Factory</option>
                <option value="Wonka Company">Wonka Company</option>
                <option value="Monsters CYA">Monsters CYA</option>
                <option value="Umbrella LTDA">Umbrella LTDA</option>
                <option value="4You">4You</option>
            </select>
            <input type="submit" value="Predict">
        </form>
        <p id="result"></p>
    </div>
</body>
</html>"""


@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        usercode = request.form.get('Usercode')
        company  = request.form.get('company_name')
        name     = request.form.get('Username')
        age      = request.form.get('Traveller_Age')

        data = {'code': int(usercode), 'company': company, 'name': name, 'age': int(age)}
        prediction = predict_gender(data, logistic_model, pca_model, scaler_model)

        gender = 'female' if prediction == 0 else 'male'
        return jsonify({'prediction': gender})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
