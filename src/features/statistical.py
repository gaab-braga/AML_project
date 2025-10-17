# src/features/statistical.py
"""
Statistical feature engineering
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, f_classif

from .base import FeatureEngineer
from ..utils import logger


class StatisticalFeatures(FeatureEngineer):
    """Statistical feature engineering"""

    def __init__(self,
                 name: str = "statistical_features",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.numeric_features = []
        self.categorical_features = []
        self.statistical_features = []
        self.scalers = {}
        self.feature_stats = {}

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from numeric and categorical columns

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with statistical features added
        """
        df = data.copy()

        # Identify column types if not already done
        if not self.numeric_features and not self.categorical_features:
            self._identify_column_types(df)

        # Create numeric statistical features
        if self.numeric_features:
            df = self._create_numeric_features(df)

        # Create categorical statistical features
        if self.categorical_features:
            df = self._create_categorical_features(df)

        # Create interaction features
        if len(self.numeric_features) > 1:
            df = self._create_interaction_features(df)

        # Create distribution-based features
        df = self._create_distribution_features(df)

        self.logger.info(f"Created {len(self.created_features) - len(self.numeric_features) - len(self.categorical_features)} statistical features")
        return df

    def _identify_column_types(self, data: pd.DataFrame) -> None:
        """Identify numeric and categorical columns"""
        self.numeric_features = []
        self.categorical_features = []

        exclude_cols = self.config.get('exclude_columns', [])
        # Add common target column names
        exclude_cols.extend(['target', 'label', 'y', 'outcome'])
        max_categories = self.config.get('max_categories', 50)

        for col in data.columns:
            if col in exclude_cols:
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if it's truly numeric (not ID-like)
                if self._is_numeric_feature(data[col]):
                    self.numeric_features.append(col)
                else:
                    self.categorical_features.append(col)
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                if data[col].nunique() <= max_categories:
                    self.categorical_features.append(col)

        self.logger.info(f"Identified {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features")

    def _is_numeric_feature(self, series: pd.Series) -> bool:
        """Check if series is a true numeric feature (not ID-like)"""
        # Only exclude clear ID patterns
        if series.dtype in ['int64', 'int32']:
            # Check if it's a sequential ID (1, 2, 3, ...)
            sorted_vals = series.sort_values().values
            if len(sorted_vals) > 1:
                expected = np.arange(sorted_vals[0], sorted_vals[0] + len(sorted_vals))
                if np.array_equal(sorted_vals, expected):
                    return False

            # Check if it's mostly sequential with small gaps
            diff = np.diff(sorted_vals)
            if len(diff) > 0 and np.mean(diff) <= 2:  # Average step <= 2
                return False

        # For float columns, assume they're numeric features
        return True

    def _create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from numeric columns"""
        numeric_data = df[self.numeric_features]

        # Basic statistical features for each numeric column
        for col in self.numeric_features:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            # Percentiles
            for percentile in [10, 25, 50, 75, 90]:
                feat_name = f'{col}_p{percentile}'
                df[feat_name] = df[col].rank(pct=True) * 100  # Percentile rank
                self.statistical_features.append(feat_name)
                self.log_feature_creation(feat_name, "percentile", percentile=percentile)

            # Z-score
            if len(series) > 1:
                zscore_col = f'{col}_zscore'
                df[zscore_col] = (df[col] - series.mean()) / series.std()
                self.statistical_features.append(zscore_col)
                self.log_feature_creation(zscore_col, "zscore")

            # Robust z-score (using median)
            if len(series) > 1:
                median = series.median()
                mad = (series - median).abs().median()
                if mad > 0:
                    robust_z_col = f'{col}_robust_zscore'
                    df[robust_z_col] = (df[col] - median) / mad
                    self.statistical_features.append(robust_z_col)
                    self.log_feature_creation(robust_z_col, "robust_zscore")

            # Log transform (if positive values)
            if (df[col].dropna() > 0).all() and (df[col] > 0).any():
                log_col = f'{col}_log'
                df[log_col] = np.log1p(df[col])
                self.statistical_features.append(log_col)
                self.log_feature_creation(log_col, "log_transform")

        # Cross-column statistical features
        if len(self.numeric_features) > 1:
            df = self._create_cross_numeric_features(df)

        return df

    def _create_cross_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that combine multiple numeric columns"""
        numeric_data = df[self.numeric_features]

        # Sum, mean, std of all numeric features
        df['numeric_sum'] = numeric_data.sum(axis=1)
        df['numeric_mean'] = numeric_data.mean(axis=1)
        df['numeric_std'] = numeric_data.std(axis=1)
        df['numeric_min'] = numeric_data.min(axis=1)
        df['numeric_max'] = numeric_data.max(axis=1)
        df['numeric_range'] = numeric_data.max(axis=1) - numeric_data.min(axis=1)

        cross_features = ['numeric_sum', 'numeric_mean', 'numeric_std',
                         'numeric_min', 'numeric_max', 'numeric_range']
        self.statistical_features.extend(cross_features)

        for feat in cross_features:
            self.log_feature_creation(feat, "cross_numeric")

        # Coefficient of variation
        mean_val = numeric_data.mean(axis=1)
        std_val = numeric_data.std(axis=1)
        df['numeric_cv'] = np.where(mean_val != 0, std_val / mean_val.abs(), 0)
        self.statistical_features.append('numeric_cv')
        self.log_feature_creation('numeric_cv', "coefficient_variation")

        return df

    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from categorical columns"""
        for col in self.categorical_features:
            value_counts = df[col].value_counts()

            # Frequency encoding
            freq_map = value_counts / len(df)
            freq_col = f'{col}_freq'
            df[freq_col] = df[col].map(freq_map)
            self.statistical_features.append(freq_col)
            self.log_feature_creation(freq_col, "frequency_encoding")

            # Count encoding
            count_col = f'{col}_count'
            df[count_col] = df[col].map(value_counts)
            self.statistical_features.append(count_col)
            self.log_feature_creation(count_col, "count_encoding")

            # Is rare category
            rare_threshold = self.config.get('rare_threshold', 0.05)
            rare_col = f'{col}_is_rare'
            df[rare_col] = (df[count_col] / len(df)) < rare_threshold
            self.statistical_features.append(rare_col)
            self.log_feature_creation(rare_col, "rare_category", threshold=rare_threshold)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        numeric_data = df[self.numeric_features]

        # Pairwise interactions (limit to avoid explosion)
        max_interactions = self.config.get('max_interactions', 5)
        interaction_count = 0

        for i, col1 in enumerate(self.numeric_features):
            for col2 in self.numeric_features[i+1:]:
                if interaction_count >= max_interactions:
                    break

                # Ratio
                ratio_col = f'{col1}_{col2}_ratio'
                df[ratio_col] = np.where(df[col2] != 0, df[col1] / df[col2], 0)
                self.statistical_features.append(ratio_col)
                self.log_feature_creation(ratio_col, "ratio_interaction")

                # Difference
                diff_col = f'{col1}_{col2}_diff'
                df[diff_col] = df[col1] - df[col2]
                self.statistical_features.append(diff_col)
                self.log_feature_creation(diff_col, "difference_interaction")

                # Product
                prod_col = f'{col1}_{col2}_prod'
                df[prod_col] = df[col1] * df[col2]
                self.statistical_features.append(prod_col)
                self.log_feature_creation(prod_col, "product_interaction")

                interaction_count += 3

            if interaction_count >= max_interactions:
                break

        return df

    def _create_distribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distribution-based features"""
        # Only use truly numeric columns for distribution features
        numeric_cols = [col for col in self.numeric_features if col in df.columns and
                       pd.api.types.is_numeric_dtype(df[col]) and
                       not pd.api.types.is_bool_dtype(df[col])]

        if not numeric_cols:
            return df

        numeric_data = df[numeric_cols]

        # Skewness and kurtosis - only for numeric data
        try:
            df['numeric_skewness'] = stats.skew(numeric_data, axis=1, nan_policy='omit')
            df['numeric_kurtosis'] = stats.kurtosis(numeric_data, axis=1, nan_policy='omit')
        except (TypeError, ValueError):
            # Fallback if scipy can't handle the data
            df['numeric_skewness'] = 0.0
            df['numeric_kurtosis'] = 0.0

        # Entropy-like measure
        df['numeric_entropy'] = -np.sum(numeric_data * np.log1p(numeric_data.abs()), axis=1)

        # Number of zero values
        df['numeric_zero_count'] = (numeric_data == 0).sum(axis=1)

        # Number of negative values
        df['numeric_negative_count'] = (numeric_data < 0).sum(axis=1)

        dist_features = ['numeric_skewness', 'numeric_kurtosis', 'numeric_entropy',
                        'numeric_zero_count', 'numeric_negative_count']
        self.statistical_features.extend(dist_features)

        for feat in dist_features:
            self.log_feature_creation(feat, "distribution")

        return df

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit statistical feature engineer"""
        self.logger.info("Fitting statistical feature engineer")

        # Identify column types
        self._identify_column_types(data)

        # Compute statistics for scaling/normalization
        if self.numeric_features:
            self._compute_feature_stats(data)

        # Fit scalers if configured
        if self.config.get('scale_features', False):
            self._fit_scalers(data)

        # Compute feature importance if target available
        if target is not None:
            self._compute_feature_importance(data, target)

    def _compute_feature_stats(self, data: pd.DataFrame) -> None:
        """Compute statistics for numeric features"""
        for col in self.numeric_features:
            series = data[col].dropna()
            if len(series) > 0:
                self.feature_stats[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'median': series.median(),
                    'min': series.min(),
                    'max': series.max(),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series)
                }

        self.logger.info(f"Computed statistics for {len(self.feature_stats)} numeric features")

    def _fit_scalers(self, data: pd.DataFrame) -> None:
        """Fit feature scalers"""
        scaler_type = self.config.get('scaler_type', 'standard')

        for col in self.numeric_features:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                continue

            # Fit on non-null values
            non_null_data = data[col].dropna().values.reshape(-1, 1)
            if len(non_null_data) > 1:
                scaler.fit(non_null_data)
                self.scalers[col] = scaler

        self.logger.info(f"Fitted {len(self.scalers)} feature scalers")

    def _compute_feature_importance(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Compute feature importance scores"""
        if not self.numeric_features:
            return

        try:
            # Use mutual information for regression, ANOVA F-test for classification
            X = data[self.numeric_features].fillna(0)

            if target.dtype in ['int64', 'int32', 'category'] and target.nunique() <= 20:
                # Classification
                f_scores, _ = f_classif(X, target)
                importance_scores = dict(zip(self.numeric_features, f_scores))
            else:
                # Regression
                mi_scores = mutual_info_regression(X, target)
                importance_scores = dict(zip(self.numeric_features, mi_scores))

            # Store top features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            self._feature_importance = dict(sorted_features[:10])  # Top 10

            self.logger.info(f"Computed feature importance for {len(self.numeric_features)} features")

        except Exception as e:
            self.logger.warning(f"Failed to compute feature importance: {e}")

    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical features"""
        results = {
            "valid": True,
            "numeric_features": len(self.numeric_features),
            "categorical_features": len(self.categorical_features),
            "statistical_features": len(self.statistical_features),
            "total_new_features": len(self.created_features)
        }

        # Check for NaN values in statistical features
        if self.statistical_features:
            stat_features_in_data = [f for f in self.statistical_features if f in data.columns]
            if stat_features_in_data:
                nan_counts = data[stat_features_in_data].isnull().sum()
                total_nans = nan_counts.sum()

                if total_nans > 0:
                    results["warnings"] = [f"Found {total_nans} NaN values in statistical features"]
                    results["nan_features"] = nan_counts[nan_counts > 0].to_dict()

        # Check feature statistics
        if hasattr(self, 'feature_stats') and self.feature_stats:
            results["feature_stats_computed"] = len(self.feature_stats)

        # Check scalers
        if self.scalers:
            results["scalers_fitted"] = len(self.scalers)

        return results