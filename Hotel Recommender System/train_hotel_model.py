"""
train_hotel_model.py
=============================================================
Training script for Hotel Recommender System.
Reads hotels.csv (same directory), trains collaborative
filtering model (SVD/matrix factorization) and saves it.

Usage:
    cd "Hotel Recommender System"
    python train_hotel_model.py
=============================================================
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'hotels.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cf_recommender_model.pkl')
NUMBER_OF_FACTORS_MF = 8


# ─── CFRecommender class ──────────────────────────────────────────────────────
class CFRecommender:
    """Collaborative Filtering recommender using SVD matrix factorisation."""

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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Hotel Recommender - Model Training")
    print("=" * 60)

    print(f"[INFO] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    hotel_df = df.copy()
    print(f"   Shape: {hotel_df.shape}")
    print(f"   Columns: {list(hotel_df.columns)}")

    # ── Filter users with >= 2 interactions ───────────────────
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
    print(f"[INFO] After filtering: {len(interactions_df)} interactions "
          f"from {interactions_df['userCode'].nunique()} users")

    # ── Encode hotel names ────────────────────────────────────
    label_encoder = LabelEncoder()
    interactions_df['name_encoded'] = label_encoder.fit_transform(
        interactions_df['name']
    )

    # ── Aggregate interactions ────────────────────────────────
    interactions_full_df = (
        interactions_df
        .groupby(['name_encoded', 'userCode'])['price']
        .sum()
        .reset_index()
    )

    # ── Train / test split ────────────────────────────────────
    interactions_train_df, interactions_test_df = train_test_split(
        interactions_full_df,
        stratify=interactions_full_df['userCode'],
        test_size=0.25,
        random_state=42
    )
    print(f"   Train interactions: {len(interactions_train_df)}")
    print(f"   Test  interactions: {len(interactions_test_df)}")

    # ── Build pivot matrix ────────────────────────────────────
    items_users_pivot_df = interactions_train_df.pivot(
        index='userCode', columns='name_encoded', values='price'
    ).fillna(0)

    items_users_pivot_matrix = items_users_pivot_df.values
    user_ids = list(items_users_pivot_df.index)
    print(f"\n   Pivot matrix shape: {items_users_pivot_matrix.shape}")

    # ── SVD matrix factorisation ──────────────────────────────
    print(f"   Running SVD (k={NUMBER_OF_FACTORS_MF})...")
    U, sigma, Vt = svds(items_users_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    cf_preds_df = pd.DataFrame(
        all_user_predicted_ratings,
        columns=items_users_pivot_df.columns,
        index=user_ids
    ).transpose()

    # ── Build and save model ──────────────────────────────────
    cf_model = CFRecommender(cf_preds_df, interactions_df)

    print(f"\n💾 Saving model to: {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(cf_model, f)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"   ✅ Saved cf_recommender_model.pkl  ({size_kb:.1f} KB)")

    # ── Quick sanity check ────────────────────────────────────
    print("\n🔍 Sanity check — recommending for first user...")
    first_user = user_ids[0]
    recs = cf_model.recommend_items(first_user, verbose=True)
    print(f"   Recommendations for user {first_user}:")
    print(recs.to_string())

    print("\n" + "=" * 60)
    print("  ✅ Hotel recommender model saved successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
