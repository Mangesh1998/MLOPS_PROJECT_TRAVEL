# Import required libraries
import os
import pandas as pd
import pickle
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# ─── Feature order (must match training) ──────────────────────────────────────
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

CITIES = [
    'Florianopolis (SC)', 'Sao Paulo (SP)', 'Salvador (BH)',
    'Brasilia (DF)', 'Rio de Janeiro (RJ)', 'Campo Grande (MS)',
    'Aracaju (SE)', 'Natal (RN)', 'Recife (PE)'
]

# Map display name → feature column name (spaces → underscores)
def city_to_feature(city: str, prefix: str) -> str:
    return prefix + city.replace(' ', '_')


@st.cache_resource
def load_models():
    scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaling_1.pkl'), 'rb'))
    model  = pickle.load(open(os.path.join(MODEL_DIR, 'rf_model_1.pkl'), 'rb'))
    return scaler, model


def predict_price(input_data, model, scaler):
    df_input = pd.DataFrame([input_data])
    X = scaler.transform(df_input)
    return model.predict(X)[0]


def main():
    st.set_page_config(page_title="✈ Flight Price Prediction", layout="centered")
    st.title('✈ Flight Price Prediction')
    st.markdown("Predict the price of a flight in Brazil using our trained Random Forest model.")

    scaler_model, rf_model = load_models()

    col1, col2 = st.columns(2)
    with col1:
        boarding = st.selectbox('🛫 Boarding City', CITIES)
    with col2:
        destination = st.selectbox('🛬 Destination City', CITIES)

    col3, col4 = st.columns(2)
    with col3:
        flight_class = st.selectbox('💺 Flight Class', ['economic', 'firstClass', 'premium'])
    with col4:
        agency = st.selectbox('🏢 Agency', ['Rainbow', 'CloudFy', 'FlyingDrops'])

    col5, col6, col7 = st.columns(3)
    with col5:
        week_no  = st.number_input('📅 Week Number', min_value=1, max_value=52, value=10)
    with col6:
        week_day = st.number_input('📆 Week Day (0=Mon, 6=Sun)', min_value=0, max_value=6, value=3)
    with col7:
        day = st.number_input('🗓 Day (1–31)', min_value=1, max_value=31, value=15)

    # Build feature dict
    travel_dict = {}

    boarding_feat    = city_to_feature(boarding, 'from_')
    destination_feat = city_to_feature(destination, 'destination_')
    class_feat       = 'flightType_' + flight_class
    agency_feat      = 'agency_' + agency

    for feat in FEATURES_ORDERING:
        if feat.startswith('from_'):
            travel_dict[feat] = 1 if feat == boarding_feat else 0
        elif feat.startswith('destination_'):
            travel_dict[feat] = 1 if feat == destination_feat else 0
        elif feat.startswith('flightType_'):
            travel_dict[feat] = 1 if feat == class_feat else 0
        elif feat.startswith('agency_'):
            travel_dict[feat] = 1 if feat == agency_feat else 0

    travel_dict['week_no']  = week_no
    travel_dict['week_day'] = week_day
    travel_dict['day']      = day

    if st.button('🔮 Predict Flight Price', use_container_width=True):
        predicted_price = predict_price(travel_dict, rf_model, scaler_model)
        st.success(f"💰 Predicted Flight Price Per Person: **${round(predicted_price, 2)}**")
        st.balloons()


if __name__ == '__main__':
    main()
