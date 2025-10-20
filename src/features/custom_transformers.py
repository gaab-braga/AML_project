"""
Custom Transformers for AML Feature Engineering Pipeline

This module contains custom transformer classes that inherit from sklearn's
BaseEstimator and TransformerMixin, allowing them to be used in sklearn pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.aml_features import (
    clean_transactions,
    encode_categorical_features,
    create_temporal_features,
    create_network_features
)
from src.features.pattern_engineering import PatternFeatureEngineer


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for data cleaning operations.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Clean the transaction data.

        Parameters:
        X (pd.DataFrame): Raw transaction data

        Returns:
        pd.DataFrame: Cleaned transaction data
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Use the existing cleaning function
        X_clean = clean_transactions(X.copy())
        return X_clean


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for categorical encoding.
    """

    def __init__(self, target_col='is_fraud'):
        self.target_col = target_col
        self.encoders_ = {}

    def fit(self, X, y=None):
        """
        Fit the categorical encoders.

        Parameters:
        X (pd.DataFrame): Input data with categorical columns
        y: Ignored

        Returns:
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Use the existing encoding function
        _, self.encoders_ = encode_categorical_features(X.copy(), target_col=self.target_col)
        return self

    def transform(self, X):
        """
        Transform categorical columns using fitted encoders.

        Parameters:
        X (pd.DataFrame): Input data

        Returns:
        pd.DataFrame: Data with encoded categorical columns
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        X_encoded, _ = encode_categorical_features(X.copy(), target_col=self.target_col)
        return X_encoded


class TemporalFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for temporal feature generation.
    """

    def __init__(self, windows=[7, 30]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Generate temporal features.

        Parameters:
        X (pd.DataFrame): Clean transaction data

        Returns:
        pd.DataFrame: Data with temporal features added
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Use the existing temporal feature function
        X_with_temporal = create_temporal_features(X.copy(), windows=self.windows)
        return X_with_temporal


class NetworkFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for network feature generation.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Generate network features.

        Parameters:
        X (pd.DataFrame): Transaction data

        Returns:
        pd.DataFrame: Data with network features added
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Create network features
        network_features_df = create_network_features(X.copy())

        # Merge network features back to transactions
        X_with_network = X.copy()

        # Merge for source accounts
        X_with_network = X_with_network.merge(
            network_features_df[['node', 'degree', 'in_degree', 'out_degree',
                               'degree_centrality', 'in_degree_centrality', 'out_degree_centrality']],
            left_on='source',
            right_on='node',
            how='left'
        ).rename(columns={
            'degree': 'source_degree',
            'in_degree': 'source_in_degree',
            'out_degree': 'source_out_degree',
            'degree_centrality': 'source_degree_centrality',
            'in_degree_centrality': 'source_in_degree_centrality',
            'out_degree_centrality': 'source_out_degree_centrality'
        }).drop('node', axis=1, errors='ignore')

        # Merge for target accounts
        X_with_network = X_with_network.merge(
            network_features_df[['node', 'degree', 'in_degree', 'out_degree',
                               'degree_centrality', 'in_degree_centrality', 'out_degree_centrality']],
            left_on='target',
            right_on='node',
            how='left'
        ).rename(columns={
            'degree': 'target_degree',
            'in_degree': 'target_in_degree',
            'out_degree': 'target_out_degree',
            'degree_centrality': 'target_degree_centrality',
            'in_degree_centrality': 'target_in_degree_centrality',
            'out_degree_centrality': 'target_out_degree_centrality'
        }).drop('node', axis=1, errors='ignore')

        # Fill NaN values
        network_cols = [col for col in X_with_network.columns if col.startswith(('source_', 'target_')) and
                       ('degree' in col or 'centrality' in col)]
        for col in network_cols:
            X_with_network[col] = X_with_network[col].fillna(0)

        return X_with_network


class PatternFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer for pattern-based feature generation.
    """

    def __init__(self):
        self.pattern_engineer_ = None

    def fit(self, X, y=None):
        """
        Fit the pattern feature generator.

        Parameters:
        X (pd.DataFrame): Input data
        y: Ignored

        Returns:
        self
        """
        self.pattern_engineer_ = PatternFeatureEngineer()
        return self

    def transform(self, X):
        """
        Generate pattern-based features.

        Parameters:
        X (pd.DataFrame): Input data

        Returns:
        pd.DataFrame: Data with pattern features added
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if self.pattern_engineer_ is None:
            raise ValueError("Transformer must be fitted before transform")

        X_with_patterns = self.pattern_engineer_.create_pattern_similarity_features(X.copy())
        return X_with_patterns


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature selection based on IV or other criteria.
    """

    def __init__(self, min_iv=0.02, target_col='is_fraud'):
        self.min_iv = min_iv
        self.target_col = target_col
        self.selected_features_ = []

    def fit(self, X, y=None):
        """
        Fit the feature selector by calculating IV and selecting features.

        Parameters:
        X (pd.DataFrame): Input data with features
        y: Ignored

        Returns:
        self
        """
        from src.features.iv_calculator import calculate_iv, get_predictive_features

        # Calculate IV for all numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]

        if len(numeric_cols) == 0:
            self.selected_features_ = []
            return self

        iv_results = calculate_iv(
            X,
            target_col=self.target_col,
            bins=10,
            max_iv=10.0,
            min_samples=1,
            max_unique_values=100000
        )

        # Get predictive features
        predictive_features = get_predictive_features(iv_results, min_iv=self.min_iv, exclude_suspect=True)
        self.selected_features_ = predictive_features['variable'].tolist() if not predictive_features.empty else []

        # Always include target column
        if self.target_col not in self.selected_features_:
            self.selected_features_.append(self.target_col)

        return self

    def transform(self, X):
        """
        Select features based on fitted selection.

        Parameters:
        X (pd.DataFrame): Input data

        Returns:
        pd.DataFrame: Data with selected features only
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Ensure selected features exist in the data
        available_features = [col for col in self.selected_features_ if col in X.columns]

        if len(available_features) == 0:
            raise ValueError("No selected features available in the data")

        return X[available_features].copy()