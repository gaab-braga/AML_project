"""
Feature engineering utilities for AML project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

def load_raw_transactions(file_path: str) -> pd.DataFrame:
    """
    Load raw transaction data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with raw transactions
    """
    logger.info(f"Loading raw transactions from {file_path}")
    df = pd.read_csv(file_path)

    # Ensure timestamp is datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    logger.info(f"Loaded {len(df)} transactions")
    return df

def validate_data_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data validation and compliance checks.

    Args:
        df: Input DataFrame

    Returns:
        Validated DataFrame
    """
    logger.info("Validating data compliance...")

    # Check required columns
    required_cols = ['Timestamp', 'From Account', 'To Account', 'Amount Paid']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Basic validations
    df = df.copy()
    df = df.dropna(subset=['Amount Paid', 'Timestamp'])
    df = df[df['Amount Paid'] > 0]

    logger.info(f"Data validation complete. Remaining transactions: {len(df)}")
    return df

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data - remove duplicates, handle missing values.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning transaction data...")

    # Remove duplicates
    df_clean = df.drop_duplicates()

    # Handle missing values
    df_clean = df_clean.dropna(subset=['Amount Paid', 'Timestamp'])

    # Ensure positive amounts
    df_clean = df_clean[df_clean['Amount Paid'] > 0]

    logger.info(f"Cleaned data: {len(df_clean)} transactions")
    return df_clean

def impute_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic imputation and encoding.

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame
    """
    logger.info("Imputing and encoding...")

    df_processed = df.copy()

    # Basic imputation for numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    logger.info("Imputation and encoding complete")
    return df_processed

def aggregate_by_entity(df: pd.DataFrame, entity_col: str = 'From Account') -> pd.DataFrame:
    """
    Basic aggregation by entity.

    Args:
        df: Input DataFrame
        entity_col: Column to aggregate by

    Returns:
        Aggregated DataFrame
    """
    logger.info(f"Aggregating by {entity_col}...")

    agg_df = df.groupby(entity_col).agg({
        'Amount Paid': ['count', 'sum', 'mean', 'std'],
        'is_fraud': 'mean'
    }).round(4)

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]

    logger.info(f"Aggregated data for {len(agg_df)} entities")
    return agg_df.reset_index()

def create_temporal_features(df: pd.DataFrame, windows: list = [7, 30]) -> pd.DataFrame:
    """
    Create temporal aggregation features.

    Args:
        df: Transaction DataFrame with 'source', 'timestamp', 'amount'
        windows: Rolling window sizes in days

    Returns:
        DataFrame with temporal features
    """
    logger.info("Creating temporal features...")

    df_temp = df.copy()

    # Ensure timestamp is datetime
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])

    # Create temporal features for each window
    for window in windows:
        # Group by source and create rolling features
        grouped = df_temp.groupby('source')

        # Initialize new columns
        df_temp[f'source_amount_sum_{window}d'] = 0.0
        df_temp[f'source_amount_mean_{window}d'] = 0.0
        df_temp[f'source_transaction_count_{window}d'] = 0

        # Process each group
        for name, group in grouped:
            if len(group) > 0:
                group_sorted = group.sort_values('timestamp').copy()

                # Create rolling window aggregations
                rolling = group_sorted.set_index('timestamp')['amount'].rolling(
                    window=f'{window}D',
                    min_periods=1
                )

                # Calculate aggregations
                sum_vals = rolling.sum()
                mean_vals = rolling.mean()
                count_vals = rolling.count()

                # Update the main dataframe
                df_temp.loc[group_sorted.index, f'source_amount_sum_{window}d'] = sum_vals.values
                df_temp.loc[group_sorted.index, f'source_amount_mean_{window}d'] = mean_vals.values
                df_temp.loc[group_sorted.index, f'source_transaction_count_{window}d'] = count_vals.values

    # Fill NaN values
    temporal_cols = [col for col in df_temp.columns if any(s in col for s in ['source_amount_', 'source_transaction_count_'])]
    df_temp[temporal_cols] = df_temp[temporal_cols].fillna(0)

    # Add time-based features
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['day_of_week'] = df_temp['timestamp'].dt.weekday
    df_temp['is_business_hours'] = df_temp['timestamp'].dt.hour.between(9, 17).astype(int)
    df_temp['is_weekend'] = df_temp['timestamp'].dt.weekday.isin([5, 6]).astype(int)

    logger.info(f"Created temporal features: {len(temporal_cols) + 4} columns")
    return df_temp

def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create network-based features.

    Args:
        df: Transaction DataFrame with 'source', 'target'

    Returns:
        DataFrame with network features
    """
    logger.info("Creating network features...")

    try:
        import networkx as nx

        # Create directed graph
        G = nx.from_pandas_edgelist(
            df, 'source', 'target',
            create_using=nx.DiGraph()
        )

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)

        # Create features DataFrame
        network_features = []
        for node in G.nodes():
            network_features.append({
                'node': node,
                'degree_centrality': degree_centrality.get(node, 0),
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'degree': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node)
            })

        network_df = pd.DataFrame(network_features)

    except ImportError:
        logger.warning("NetworkX not available, creating basic degree features...")

        # Basic degree calculation without NetworkX
        source_degrees = df.groupby('source').size().rename('out_degree')
        target_degrees = df.groupby('target').size().rename('in_degree')

        # Combine all unique nodes
        all_nodes = set(df['source'].unique()) | set(df['target'].unique())

        network_features = []
        for node in all_nodes:
            network_features.append({
                'node': node,
                'degree': source_degrees.get(node, 0) + target_degrees.get(node, 0),
                'in_degree': target_degrees.get(node, 0),
                'out_degree': source_degrees.get(node, 0),
                'degree_centrality': (source_degrees.get(node, 0) + target_degrees.get(node, 0)) / len(all_nodes),
                'in_degree_centrality': target_degrees.get(node, 0) / len(all_nodes),
                'out_degree_centrality': source_degrees.get(node, 0) / len(all_nodes)
            })

        network_df = pd.DataFrame(network_features)

    logger.info(f"Created network features for {len(network_df)} nodes")
    return network_df

def encode_categorical_features(df: pd.DataFrame, target_col: str = 'is_fraud') -> tuple:
    """
    Encode categorical features.

    Args:
        df: DataFrame with categorical columns
        target_col: Target column name

    Returns:
        Tuple of (encoded DataFrame, encoding mappings)
    """
    logger.info("Encoding categorical features...")

    df_encoded = df.copy()
    encoders = {}

    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]

    for col in categorical_cols:
        # Frequency encoding
        freq_encoding = df_encoded[col].value_counts(normalize=True)
        df_encoded[col] = df_encoded[col].map(freq_encoding)
        encoders[col] = freq_encoding.to_dict()

        # Fill NaN with global mean
        df_encoded[col] = df_encoded[col].fillna(freq_encoding.mean())

    logger.info(f"Encoded {len(encoders)} categorical columns")
    return df_encoded, encoders

class AMLFeaturePipeline:
    """Basic AML feature engineering pipeline."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform features."""
        logger.info("Fitting and transforming features...")

        X_transformed = X.copy()

        # Handle categorical encoding
        if self.config.get('categorical_encoding'):
            X_transformed, self.encoders = encode_categorical_features(X_transformed)

        # Basic scaling
        if self.config.get('scaler_type') == 'robust':
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'is_fraud']

            for col in numeric_cols:
                median = X_transformed[col].median()
                q75, q25 = np.percentile(X_transformed[col].dropna(), [75, 25])
                iqr = q75 - q25

                if iqr == 0:
                    iqr = 1

                X_transformed[col] = (X_transformed[col] - median) / iqr
                self.scalers[col] = {'median': median, 'iqr': iqr}

        logger.info(f"Pipeline fitted with {len(X_transformed.columns)} features")
        return X_transformed.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted pipeline."""
        X_transformed = X.copy()

        # Apply categorical encoding
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                freq_map = pd.Series(encoder)
                X_transformed[col] = X_transformed[col].map(freq_map)
                X_transformed[col] = X_transformed[col].fillna(freq_map.mean())

        # Apply scaling
        for col, params in self.scalers.items():
            if col in X_transformed.columns:
                X_transformed[col] = (X_transformed[col] - params['median']) / params['iqr']

        return X_transformed.values

    def save(self, path: str):
        """Save pipeline to disk."""
        import joblib
        joblib.dump({
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders
        }, path)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load pipeline from disk."""
        import joblib
        data = joblib.load(path)

        pipeline = cls(data['config'])
        pipeline.scalers = data['scalers']
        pipeline.encoders = data['encoders']

        logger.info(f"Pipeline loaded from {path}")
        return pipeline