# src/features/temporal.py
"""
Temporal feature engineering without data leakage
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TemporalFeatures:
    """Temporal feature engineering without data leakage"""

    def __init__(self,
                 temporal_column: str = "timestamp"):
        self.temporal_column = temporal_column

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic temporal features from datetime column

        Args:
            data: Input DataFrame with temporal column

        Returns:
            DataFrame with temporal features added
        """
        if self.temporal_column not in data.columns:
            logger.warning(f"Temporal column '{self.temporal_column}' not found in data")
            return data

        df = data.copy()

        # Ensure temporal column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.temporal_column]):
            try:
                df[self.temporal_column] = pd.to_datetime(df[self.temporal_column])
                logger.info(f"Converted {self.temporal_column} to datetime")
            except Exception as e:
                logger.error(f"Failed to convert {self.temporal_column} to datetime: {e}")
                return df

        # Create basic temporal features
        df[f'{self.temporal_column}_hour'] = df[self.temporal_column].dt.hour
        df[f'{self.temporal_column}_day'] = df[self.temporal_column].dt.day
        df[f'{self.temporal_column}_month'] = df[self.temporal_column].dt.month
        df[f'{self.temporal_column}_weekday'] = df[self.temporal_column].dt.weekday

        logger.info("Created basic temporal features")
        return df

def aggregate_by_entity(df: pd.DataFrame, entity_col: str, windows: List[int]) -> pd.DataFrame:
    """Aggregate features by entity over time windows"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for window in windows:
        # Rolling aggregations
        df[f'{entity_col}_amount_sum_{window}d'] = df.groupby(entity_col)['amount'].rolling(f'{window}D').sum().reset_index(0, drop=True)
        df[f'{entity_col}_amount_mean_{window}d'] = df.groupby(entity_col)['amount'].rolling(f'{window}D').mean().reset_index(0, drop=True)
        df[f'{entity_col}_count_{window}d'] = df.groupby(entity_col)['amount'].rolling(f'{window}D').count().reset_index(0, drop=True)

    return df.fillna(0)

def compute_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic network features"""
    # Simple network features: degree, centrality approximation
    df = df.copy()

    # Node degrees
    source_degrees = df['source'].value_counts()
    target_degrees = df['target'].value_counts()

    df['source_degree'] = df['source'].map(source_degrees).fillna(0)
    df['target_degree'] = df['target'].map(target_degrees).fillna(0)

    # Transaction frequency between pairs
    pair_counts = df.groupby(['source', 'target']).size()
    df['pair_frequency'] = df.set_index(['source', 'target']).index.map(pair_counts).fillna(0)

    return df