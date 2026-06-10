"""
data_transformation.py
DataTransformer class for feature engineering on flights dataset.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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


class DataTransformer:
    """Applies feature engineering and preprocessing to raw flights data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transform(self):
        """
        Perform all feature engineering steps.
        Returns X (features DataFrame) and Y (target Series).
        """
        logger.info("Starting data transformation...")
        df = self.df.copy()

        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        df['week_day'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['week_no'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['day'] = df['date'].dt.day

        # Rename destination column
        df.rename(columns={"to": "destination"}, inplace=True)

        # New feature: flight speed
        df['flight_speed'] = round(df['distance'] / df['time'], 2)

        # One-hot encoding
        df = pd.get_dummies(df, columns=['from', 'destination', 'flightType', 'agency'])

        # Drop irrelevant features
        df.drop(columns=['time', 'flight_speed', 'month', 'year', 'distance', 'date'],
                axis=1, inplace=True, errors='ignore')

        # Rename columns with spaces to use underscores (for Flask/API compatibility)
        rename_map = {
            'from_Sao Paulo (SP)': 'from_Sao_Paulo (SP)',
            'from_Rio de Janeiro (RJ)': 'from_Rio_de_Janeiro (RJ)',
            'from_Campo Grande (MS)': 'from_Campo_Grande (MS)',
            'destination_Sao Paulo (SP)': 'destination_Sao_Paulo (SP)',
            'destination_Rio de Janeiro (RJ)': 'destination_Rio_de_Janeiro (RJ)',
            'destination_Campo Grande (MS)': 'destination_Campo_Grande (MS)',
        }
        df.rename(columns=rename_map, inplace=True)

        # Separate features and target
        X = df.drop('price', axis=1)
        Y = df['price']

        # Keep only columns we need, in correct order
        available_features = [f for f in FEATURES_ORDERING if f in X.columns]
        X = X[available_features]

        logger.info(f"Transformation complete. X shape: {X.shape}")
        return X, Y
