"""
data_ingestion.py
DataLoader class for loading flights.csv dataset.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of flight data from a CSV file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load the CSV file and return a DataFrame."""
        logger.info(f"Loading data from: {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
