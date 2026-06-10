#Import required libraries
import os
import pandas as pd
import pickle

#Import flask library
from flask import Flask, render_template, request, jsonify

# ─── Feature ordering (must match training) ───────────────────────────────────
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')


#function to predict the price
def predict_price(input_data, model, scaler):
    df_input = pd.DataFrame([input_data])
    X = df_input[FEATURES_ORDERING[:-3]]  # all except numeric cols
    # reorder to match full feature list
    df_full = pd.DataFrame([input_data], columns=FEATURES_ORDERING)
    X = scaler.transform(df_full)
    y_prediction = model.predict(X)
    return y_prediction[0]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        boarding    = str(request.form.get('from'))
        destination = str(request.form.get('Destination'))
        flight_type = str(request.form.get('flightType'))
        agency      = str(request.form.get('agency'))
        week_no     = int(request.form.get('week_no'))
        week_day    = int(request.form.get('week_day'))
        day         = int(request.form.get('day'))

        # Build one-hot feature dict using full feature ordering
        travel_dict = {feat: 0 for feat in FEATURES_ORDERING}

        # Map boarding city → feature name (HTML values use underscores e.g. Sao_Paulo)
        boarding_feat    = 'from_' + boarding + ' (SC)' if boarding == 'Florianopolis' else None
        # Generic mapping: value from form is already like "Sao_Paulo", feature is "from_Sao_Paulo (SP)"
        # We find the matching feature by prefix
        for feat in FEATURES_ORDERING:
            if feat.startswith('from_') and feat.replace('from_', '').split(' (')[0] == boarding:
                travel_dict[feat] = 1
            elif feat.startswith('destination_') and feat.replace('destination_', '').split(' (')[0] == destination:
                travel_dict[feat] = 1
            elif feat == f'flightType_{flight_type}':
                travel_dict[feat] = 1
            elif feat == f'agency_{agency}':
                travel_dict[feat] = 1

        travel_dict['week_no']  = week_no
        travel_dict['week_day'] = week_day
        travel_dict['day']      = day

        # Load models
        scaler_model = pickle.load(open(os.path.join(MODEL_DIR, 'scaling_1.pkl'), 'rb'))
        rf_model     = pickle.load(open(os.path.join(MODEL_DIR, 'rf_model_1.pkl'), 'rb'))

        # Prepare DataFrame in correct column order
        df_input = pd.DataFrame([travel_dict], columns=FEATURES_ORDERING)
        X = scaler_model.transform(df_input)
        predicted_price = str(round(rf_model.predict(X)[0], 2))

        return jsonify({'prediction': predicted_price})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)