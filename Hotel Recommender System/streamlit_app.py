"""
streamlit_app.py — Hotel Recommendation System
Uses the pre-trained CFRecommender model from cf_recommender_model.pkl.
Run:  streamlit run streamlit_app.py
"""
import os
import pickle
import math
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds

warnings.filterwarnings("ignore")

# ─── CFRecommender class (must be importable for pickle deserialization) ───────
class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=5, verbose=False):
        if user_id not in self.cf_predictions_df.columns:
            raise KeyError(f"User '{user_id}' not found in prediction data.")

        sorted_user_predictions = (
            self.cf_predictions_df[user_id]
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={user_id: 'recStrength'})
        )

        recommendations_df = (
            sorted_user_predictions[
                ~sorted_user_predictions['name_encoded'].isin(items_to_ignore)
            ]
            .sort_values('recStrength', ascending=False)
            .head(topn)
        )

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            recommendations_df = recommendations_df.merge(
                self.items_df, how='left',
                left_on='name_encoded', right_on='name_encoded'
            )[['name_encoded', 'name', 'recStrength']]
            recommendations_df = pd.DataFrame(
                recommendations_df.groupby('name')
                .max('recStrength')
                .sort_values('recStrength', ascending=False)
            )
        return recommendations_df


# ─── Load model (or build it if pkl not found) ────────────────────────────────
MODEL_PKL  = os.path.join(os.path.dirname(__file__), 'cf_recommender_model.pkl')
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'hotels.csv')
NUMBER_OF_FACTORS_MF = 8


@st.cache_resource
def load_recommender():
    """Load pkl if available, otherwise build from scratch."""
    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        # Load raw data just for the user-code dropdown
        hotel_df = pd.read_csv(DATA_PATH)
        return model, hotel_df

    # ── Build from scratch ────────────────────────────────────
    hotel_df = pd.read_csv(DATA_PATH)

    users_interactions_count_df = (
        hotel_df.groupby(['userCode', 'name']).size()
        .groupby('userCode').size()
    )
    users_with_enough = users_interactions_count_df[
        users_interactions_count_df >= 2
    ].reset_index()[['userCode']]

    interactions_df = hotel_df.merge(
        users_with_enough, how='right',
        left_on='userCode', right_on='userCode'
    )

    label_encoder = LabelEncoder()
    interactions_df['name_encoded'] = label_encoder.fit_transform(
        interactions_df['name']
    )

    interactions_full_df = (
        interactions_df
        .groupby(['name_encoded', 'userCode'])['price']
        .sum()
        .reset_index()
    )

    interactions_train_df, _ = train_test_split(
        interactions_full_df,
        stratify=interactions_full_df['userCode'],
        test_size=0.25,
        random_state=42
    )

    items_users_pivot_df = interactions_train_df.pivot(
        index='userCode', columns='name_encoded', values='price'
    ).fillna(0)

    items_users_pivot_matrix = items_users_pivot_df.values
    user_ids = list(items_users_pivot_df.index)

    U, sigma, Vt = svds(items_users_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    cf_preds_df = pd.DataFrame(
        all_user_predicted_ratings,
        columns=items_users_pivot_df.columns,
        index=user_ids
    ).transpose()

    model = CFRecommender(cf_preds_df, interactions_df)
    return model, hotel_df


# ─── Streamlit App ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Hotel Recommendation System", layout="centered")

    st.title("🏨 Hotel Recommendation System")
    st.markdown(
        """
        Welcome to the Hotel Recommendation App!  
        Select your user code below to get personalised hotel recommendations.
        """
    )

    with st.spinner("Loading recommendation model..."):
        cf_recommender_model, hotel_df = load_recommender()

    # Sidebar input
    st.sidebar.header("User Input")
    usercode_options = sorted(hotel_df['userCode'].unique())
    selected_usercode = st.sidebar.selectbox('Select User Code', usercode_options)

    st.sidebar.markdown("---")
    st.sidebar.header("Get Recommendations")

    if st.sidebar.button("Recommend Hotels"):
        try:
            recommended_hotels = cf_recommender_model.recommend_items(
                selected_usercode, verbose=True
            )
            if recommended_hotels.empty:
                st.warning("No recommendations found for the selected user.")
            else:
                st.success(f"Top recommendations for User **{selected_usercode}**:")
                st.dataframe(recommended_hotels, use_container_width=True)
        except KeyError:
            st.error(
                f"User {selected_usercode} does not have enough interaction "
                "data to generate recommendations."
            )

    st.markdown("---")
    st.markdown("*Powered by Collaborative Filtering (SVD Matrix Factorisation).*")


if __name__ == '__main__':
    main()
