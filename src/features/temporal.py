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
from sklearn.base import clone
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


class TemporalFeatures:
    """Temporal feature engineering without data leakage"""

    def __init__(self,
                 temporal_column: str = "timestamp"):
        self.temporal_column = temporal_column

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime column

        Args:
            data: Input DataFrame with temporal column

        Returns:
            DataFrame with temporal features added
        """
        if self.temporal_column not in data.columns:
            logging.warning(f"Temporal column '{self.temporal_column}' not found in data")
            return data

        df = data.copy()

        # Ensure temporal column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.temporal_column]):
            try:
                df[self.temporal_column] = pd.to_datetime(df[self.temporal_column])
                logging.info(f"Converted {self.temporal_column} to datetime")
            except Exception as e:
                logging.error(f"Failed to convert {self.temporal_column} to datetime: {e}")
                return df

        # Create datetime features
        df = self._create_datetime_features(df)

        # Create cyclical features
        df = self._create_cyclical_features(df)

        # Create lag features (if target column exists for training)
        if 'target' in df.columns or hasattr(self, '_target_column'):
            df = self._create_lag_features(df)

        # Create rolling statistics (if target column exists for training)
        if 'target' in df.columns or hasattr(self, '_target_column'):
            df = self._create_rolling_features(df)

        logging.info(f"Created {len(self.created_features)} temporal features")
        return df

    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic datetime features"""
        dt_col = df[self.temporal_column]

        # Year, month, day
        df[f'{self.temporal_column}_year'] = dt_col.dt.year
        df[f'{self.temporal_column}_month'] = dt_col.dt.month
        df[f'{self.temporal_column}_day'] = dt_col.dt.day
        df[f'{self.temporal_column}_dayofyear'] = dt_col.dt.dayofyear
        df[f'{self.temporal_column}_weekday'] = dt_col.dt.weekday

        # Hour, minute, second (if available)
        if hasattr(dt_col.dt, 'hour'):
            df[f'{self.temporal_column}_hour'] = dt_col.dt.hour
            df[f'{self.temporal_column}_minute'] = dt_col.dt.minute

        # Quarter and week of year
        df[f'{self.temporal_column}_quarter'] = dt_col.dt.quarter
        df[f'{self.temporal_column}_weekofyear'] = dt_col.dt.isocalendar().week

        # Is weekend, is month start/end
        df[f'{self.temporal_column}_is_weekend'] = dt_col.dt.weekday >= 5
        df[f'{self.temporal_column}_is_month_start'] = dt_col.dt.is_month_start
        df[f'{self.temporal_column}_is_month_end'] = dt_col.dt.is_month_end

        # Season (meteorological)
        month_to_season = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        df[f'{self.temporal_column}_season'] = dt_col.dt.month.map(month_to_season)

        # Log created features
        datetime_cols = [col for col in df.columns if col.startswith(f'{self.temporal_column}_') and col != self.temporal_column]
        self.datetime_features.extend(datetime_cols)

        for col in datetime_cols:
            self.log_feature_creation(col, "categorical" if "season" in col else "numeric")

        return df

    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical features using sine/cosine encoding"""
        dt_col = df[self.temporal_column]

        # Month cyclical encoding
        month_sin = np.sin(2 * np.pi * dt_col.dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt_col.dt.month / 12)
        df[f'{self.temporal_column}_month_sin'] = month_sin
        df[f'{self.temporal_column}_month_cos'] = month_cos

        # Day of year cyclical encoding
        dayofyear_sin = np.sin(2 * np.pi * dt_col.dt.dayofyear / 365.25)
        dayofyear_cos = np.cos(2 * np.pi * dt_col.dt.dayofyear / 365.25)
        df[f'{self.temporal_column}_dayofyear_sin'] = dayofyear_sin
        df[f'{self.temporal_column}_dayofyear_cos'] = dayofyear_cos

        # Weekday cyclical encoding
        weekday_sin = np.sin(2 * np.pi * dt_col.dt.weekday / 7)
        weekday_cos = np.cos(2 * np.pi * dt_col.dt.weekday / 7)
        df[f'{self.temporal_column}_weekday_sin'] = weekday_sin
        df[f'{self.temporal_column}_weekday_cos'] = weekday_cos

        # Hour cyclical encoding (if available)
        if hasattr(dt_col.dt, 'hour'):
            hour_sin = np.sin(2 * np.pi * dt_col.dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * dt_col.dt.hour / 24)
            df[f'{self.temporal_column}_hour_sin'] = hour_sin
            df[f'{self.temporal_column}_hour_cos'] = hour_cos

        # Log created features
        cyclical_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]
        self.cyclical_features.extend(cyclical_cols)

        for col in cyclical_cols:
            self.log_feature_creation(col, "cyclical")

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features (only for training data with proper temporal ordering)"""
        if not self._is_temporal_ordered(df):
            logging.warning("Data not properly temporally ordered, skipping lag features")
            return df

        target_col = 'target' if 'target' in df.columns else getattr(self, '_target_column', None)
        if not target_col or target_col not in df.columns:
            return df

        # Sort by temporal column to ensure proper ordering
        df = df.sort_values(self.temporal_column).copy()

        # Create lag features
        lag_periods = self.config.get('lag_periods', [1, 7, 14, 30])

        for lag in lag_periods:
            lag_col = f'{target_col}_lag_{lag}'
            df[lag_col] = df[target_col].shift(lag)
            self.lag_features.append(lag_col)
            self.log_feature_creation(lag_col, "lag", lag_period=lag)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features (only for training data)"""
        if not self._is_temporal_ordered(df):
            logging.warning("Data not properly temporally ordered, skipping rolling features")
            return df

        target_col = 'target' if 'target' in df.columns else getattr(self, '_target_column', None)
        if not target_col or target_col not in df.columns:
            return df

        # Sort by temporal column
        df = df.sort_values(self.temporal_column).copy()

        # Create rolling features
        windows = self.config.get('rolling_windows', [7, 14, 30])

        for window in windows:
            # Rolling mean
            mean_col = f'{target_col}_rolling_mean_{window}'
            df[mean_col] = df[target_col].rolling(window=window, min_periods=1).mean()
            self.rolling_features.append(mean_col)
            self.log_feature_creation(mean_col, "rolling", window=window, statistic="mean")

            # Rolling std
            std_col = f'{target_col}_rolling_std_{window}'
            df[std_col] = df[target_col].rolling(window=window, min_periods=1).std()
            self.rolling_features.append(std_col)
            self.log_feature_creation(std_col, "rolling", window=window, statistic="std")

        return df

    def _is_temporal_ordered(self, df: pd.DataFrame) -> bool:
        """Check if data is properly temporally ordered"""
        return df[self.temporal_column].is_monotonic_increasing

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit temporal feature engineer"""
        logging.info("Fitting temporal feature engineer")

        # Store target column for lag/rolling features
        if target is not None:
            self._target_column = target.name if hasattr(target, 'name') else 'target'

        # Analyze temporal patterns in training data
        if self.temporal_column in data.columns:
            temp_data = data.copy()
            if not pd.api.types.is_datetime64_any_dtype(temp_data[self.temporal_column]):
                temp_data[self.temporal_column] = pd.to_datetime(temp_data[self.temporal_column])

            # Analyze temporal distribution
            self._temporal_stats = {
                'date_range': (temp_data[self.temporal_column].min(), temp_data[self.temporal_column].max()),
                'total_periods': len(temp_data),
                'frequency': self._infer_frequency(temp_data[self.temporal_column])
            }

            logging.info("Temporal analysis completed",
                           extra={"extra_fields": {
                               "date_range": self._temporal_stats['date_range'],
                               "total_periods": self._temporal_stats['total_periods']
                           }})

    def _infer_frequency(self, dates: pd.Series) -> str:
        """Infer temporal frequency"""
        try:
            freq = pd.infer_freq(dates.sort_values())
            return freq if freq else 'irregular'
        except:
            return 'irregular'

    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal features"""
        results = {
            "valid": True,
            "temporal_column": self.temporal_column,
            "datetime_features": len(self.datetime_features),
            "cyclical_features": len(self.cyclical_features),
            "lag_features": len(self.lag_features),
            "rolling_features": len(self.rolling_features)
        }

        # Check for missing temporal column
        if self.temporal_column not in data.columns:
            results["valid"] = False
            results["errors"] = [f"Temporal column '{self.temporal_column}' not found"]
            return results

        # Check datetime conversion
        try:
            pd.to_datetime(data[self.temporal_column])
        except Exception as e:
            results["valid"] = False
            results["errors"] = [f"Cannot convert temporal column to datetime: {e}"]
            return results

        # Check for NaN values in created features
        created_features = (self.datetime_features + self.cyclical_features +
                          self.lag_features + self.rolling_features)
        nan_counts = data[created_features].isnull().sum()

        if nan_counts.sum() > 0:
            results["warnings"] = [f"NaN values in features: {nan_counts[nan_counts > 0].to_dict()}"]

        return results
def create_temporal_features_safe(df: pd.DataFrame,
                                 time_col: str = 'Timestamp',
                                 group_cols: Optional[List[str]] = None,
                                 windows: Optional[List[int]] = None,
                                 logger: Optional[Any] = None) -> pd.DataFrame:
    """
    Create temporal features that avoid data leakage by using only past data.

    This function creates rolling window aggregations for specified groups and time windows,
    ensuring that only historical data is used to prevent information leakage from future
    observations. All aggregations are calculated using expanding or rolling windows that
    respect temporal ordering.

    Args:
        df: Input dataframe with temporal ordering. Must contain the time column
            and numeric columns for aggregation.
        time_col: Name of timestamp column. Defaults to 'Timestamp'.
        group_cols: Columns to group by for aggregations. If None, defaults to
            ['Account', 'From Bank', 'To Bank'].
        windows: Time windows in days for rolling calculations. If None, defaults to
            [7, 30, 90].
        logger: Optional logger instance for logging operations.

    Returns:
        DataFrame with temporal features added. Original columns are preserved,
        new features are added with suffixes indicating the aggregation type and window.

    Raises:
        ValueError: If time_col is not found in dataframe or if no numeric columns
            are available for aggregation.
        TypeError: If input parameters have incorrect types.

    Examples:
        >>> df = pd.DataFrame({
        ...     'Timestamp': pd.date_range('2023-01-01', periods=100),
        ...     'Account': np.random.randint(1, 5, 100),
        ...     'Amount': np.random.randn(100)
        ... })
        >>> result = create_temporal_features_safe(df)
        >>> # Creates features like: Account_Amount_sum_7d, Account_Amount_mean_30d, etc.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if not isinstance(time_col, str):
        raise TypeError("Parameter 'time_col' must be a string")

    if group_cols is not None and not isinstance(group_cols, list):
        raise TypeError("Parameter 'group_cols' must be a list of strings or None")

    if windows is not None and not isinstance(windows, list):
        raise TypeError("Parameter 'windows' must be a list of integers or None")

    # Set defaults
    if group_cols is None:
        group_cols = ['Account', 'From Bank', 'To Bank']

    if windows is None:
        windows = [7, 30, 90]

    # Validate inputs
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe columns: {list(df.columns)}")

    if len(df) == 0:
        if logger:
            logger.warning("Input dataframe is empty, returning unchanged")
        return df.copy()

    df_temp = df.copy()

    # Ensure timestamp is datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_temp[time_col]):
            df_temp[time_col] = pd.to_datetime(df_temp[time_col])
            if logger:
                logger.info(f"Converted '{time_col}' column to datetime")
    except Exception as e:
        raise ValueError(f"Failed to convert '{time_col}' to datetime: {e}")

    # Sort by time to ensure proper temporal ordering
    df_temp = df_temp.sort_values(time_col).reset_index(drop=True)

    # Create temporal features for each group
    temporal_features_created = []

    for group_col in group_cols:
        if group_col not in df_temp.columns:
            if logger:
                logger.warning(f"Group column '{group_col}' not found in dataframe, skipping")
            continue

        if logger:
            logger.info(f"Creating temporal features for group: {group_col}")

        # Group by the column
        grouped = df_temp.groupby(group_col)

        for window in windows:
            # Find amount columns (columns containing 'amount' in name)
            amount_cols = [col for col in df_temp.columns
                          if 'amount' in col.lower() and col != time_col]

            if not amount_cols:
                if logger:
                    logger.warning(f"No amount columns found for aggregation in group '{group_col}'")
                continue

            for amount_col in amount_cols:
                if amount_col not in df_temp.columns:
                    continue

                try:
                    # Rolling sum
                    roll_sum = grouped[amount_col].rolling(window=window, min_periods=1).sum()
                    sum_col = f'{group_col}_{amount_col}_sum_{window}d'
                    df_temp[sum_col] = roll_sum.reset_index(level=0, drop=True)
                    temporal_features_created.append(sum_col)

                    # Rolling mean
                    roll_mean = grouped[amount_col].rolling(window=window, min_periods=1).mean()
                    mean_col = f'{group_col}_{amount_col}_mean_{window}d'
                    df_temp[mean_col] = roll_mean.reset_index(level=0, drop=True)
                    temporal_features_created.append(mean_col)

                    # Rolling count
                    roll_count = grouped[amount_col].rolling(window=window, min_periods=1).count()
                    count_col = f'{group_col}_{amount_col}_count_{window}d'
                    df_temp[count_col] = roll_count.reset_index(level=0, drop=True)
                    temporal_features_created.append(count_col)

                    # Rolling std
                    roll_std = grouped[amount_col].rolling(window=window, min_periods=1).std()
                    std_col = f'{group_col}_{amount_col}_std_{window}d'
                    df_temp[std_col] = roll_std.reset_index(level=0, drop=True)
                    temporal_features_created.append(std_col)

                except Exception as e:
                    if logger:
                        logger.error(f"Error creating temporal features for {group_col}_{amount_col}_window_{window}: {e}")
                    continue

    # Fill NaN values with 0 (for periods with insufficient history)
    temporal_cols = [col for col in df_temp.columns
                    if any(suffix in col for suffix in ['_sum_', '_mean_', '_count_', '_std_'])]

    if temporal_cols:
        df_temp[temporal_cols] = df_temp[temporal_cols].fillna(0)
        if logger:
            logger.info(f"Created {len(temporal_cols)} temporal features, filled NaN values with 0")
    else:
        if logger:
            logger.warning("No temporal features were created")

    return df_temp

def get_temporal_cv_splits(X: pd.DataFrame,
                          y: pd.Series,
                          n_splits: int = 5,
                          test_size: int = 30,
                          gap: int = 0,
                          logger: Optional[Any] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create temporal cross-validation splits that respect time ordering.

    This function generates cross-validation splits that maintain temporal ordering,
    ensuring that training data always comes before test data chronologically.
    This prevents data leakage that can occur with random splits in time series data.

    Args:
        X: Feature matrix with temporal ordering. Data should be sorted by time.
        y: Target variable corresponding to X.
        n_splits: Number of CV splits to create. Defaults to 5.
        test_size: Size of test set in each fold (number of samples). Defaults to 30.
        gap: Number of samples to skip between train and test sets. Defaults to 0.
        logger: Optional logger instance for logging operations.

    Returns:
        List of (train_idx, test_idx) tuples, where each tuple contains numpy arrays
        of indices for training and testing in that fold.

    Raises:
        ValueError: If n_splits or test_size are invalid, or if data is insufficient
            for the requested splits.
        TypeError: If input parameters have incorrect types.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create sample time-ordered data
        >>> X = pd.DataFrame({'feature': range(100)})
        >>> y = pd.Series(range(100))
        >>> splits = get_temporal_cv_splits(X, y, n_splits=3, test_size=20)
        >>> len(splits)
        3
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Parameter 'X' must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("Parameter 'y' must be a pandas Series")

    if not isinstance(n_splits, int) or n_splits <= 0:
        raise ValueError("Parameter 'n_splits' must be a positive integer")

    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError("Parameter 'test_size' must be a positive integer")

    if not isinstance(gap, int) or gap < 0:
        raise ValueError("Parameter 'gap' must be a non-negative integer")

    # Assume data is sorted by time (index represents time)
    n_samples = len(X)

    if n_samples == 0:
        raise ValueError("Input data is empty")

    # Validate sufficient data for splits
    min_required_samples = n_splits * test_size + gap
    if n_samples < min_required_samples:
        raise ValueError(f"Insufficient data for {n_splits} splits with test_size={test_size} and gap={gap}. "
                        f"Need at least {min_required_samples} samples, got {n_samples}")

    if logger:
        logger.info(f"Creating {n_splits} temporal CV splits with test_size={test_size}, gap={gap}")

    # Calculate split points
    splits = []

    for i in range(n_splits):
        # Calculate test set end point
        test_end = n_samples - (n_splits - i) * test_size

        # Ensure we don't go below 0
        test_end = max(test_size, test_end)

        # Calculate test set start point
        test_start = max(0, test_end - test_size)

        # Calculate train set end point (with gap)
        train_end = test_start - gap

        if train_end <= 0:
            if logger:
                logger.warning(f"Insufficient training data for fold {i+1}, skipping")
            continue

        # Get indices
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, min(test_end, n_samples), dtype=int)

        # Validate split
        if len(train_idx) == 0 or len(test_idx) == 0:
            if logger:
                logger.warning(f"Empty train or test set for fold {i+1}, skipping")
            continue

        splits.append((train_idx, test_idx))

        if logger:
            logger.debug(f"Fold {i+1}: train_size={len(train_idx)}, test_size={len(test_idx)}")

    if not splits:
        raise ValueError("No valid splits could be created with the given parameters")

    if logger:
        logger.info(f"Successfully created {len(splits)} temporal CV splits")

    return splits

def evaluate_temporal_cv(model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
                        metrics: Optional[List[str]] = None,
                        logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate model using temporal cross-validation.

    This function performs cross-validation using pre-defined temporal splits,
    ensuring that temporal ordering is respected. It computes multiple metrics
    for each fold and provides summary statistics.

    Args:
        model: Scikit-learn compatible model with fit/predict methods.
        X: Feature matrix.
        y: Target variable.
        cv_splits: List of (train_idx, test_idx) tuples from get_temporal_cv_splits.
        metrics: List of metrics to compute. Defaults to ['recall', 'precision', 'f1', 'accuracy'].
        logger: Optional logger instance for logging operations.

    Returns:
        Dictionary containing:
        - Individual fold results for each metric
        - Summary statistics (mean, std) for each metric
        - Train/test scores for overfitting analysis

    Raises:
        ValueError: If cv_splits is empty or contains invalid indices.
        TypeError: If input parameters have incorrect types.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> splits = get_temporal_cv_splits(X, y, n_splits=3)
        >>> results = evaluate_temporal_cv(model, X, y, splits)
        >>> print(f"Mean F1: {results['f1_summary']['test_mean']:.3f}")
    """
    if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
        raise TypeError("Parameter 'model' must be a scikit-learn compatible estimator")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Parameter 'X' must be a pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("Parameter 'y' must be a pandas Series")

    if not isinstance(cv_splits, list) or len(cv_splits) == 0:
        raise ValueError("Parameter 'cv_splits' must be a non-empty list of (train_idx, test_idx) tuples")

    if metrics is not None and not isinstance(metrics, list):
        raise TypeError("Parameter 'metrics' must be a list of strings or None")

    # Set default metrics
    if metrics is None:
        metrics = ['recall', 'precision', 'f1', 'accuracy']

    # Validate metrics
    valid_metrics = ['recall', 'precision', 'f1', 'accuracy']
    invalid_metrics = [m for m in metrics if m not in valid_metrics]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid options: {valid_metrics}")

    if logger:
        logger.info(f"Evaluating model with {len(cv_splits)} temporal CV folds, metrics: {metrics}")

    results = {metric: [] for metric in metrics}
    results['train_scores'] = []
    results['test_scores'] = []
    results['fold_details'] = []

    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        try:
            if logger:
                logger.debug(f"Evaluating fold {fold + 1}/{len(cv_splits)}...")

            # Validate indices
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(f"Empty train or test indices in fold {fold + 1}")

            max_train_idx = max(train_idx)
            max_test_idx = max(test_idx)
            max_idx = max(max_train_idx, max_test_idx)

            if max_idx >= len(X):
                raise ValueError(f"Indices out of bounds in fold {fold + 1}: max index {max_idx} >= {len(X)}")

            # Split data
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)

            # Get predictions
            y_pred_train = model_clone.predict(X_train_fold)
            y_pred_test = model_clone.predict(X_test_fold)

            # Calculate probabilities for scoring (if available)
            y_proba_train = None
            y_proba_test = None
            if hasattr(model_clone, 'predict_proba'):
                try:
                    y_proba_train = model_clone.predict_proba(X_train_fold)[:, 1]
                    y_proba_test = model_clone.predict_proba(X_test_fold)[:, 1]
                except (AttributeError, IndexError):
                    pass  # Some models might not support predict_proba

            # Calculate metrics for this fold
            fold_results = {}

            for metric in metrics:
                try:
                    if metric == 'recall':
                        train_score = recall_score(y_train_fold, y_pred_train)
                        test_score = recall_score(y_test_fold, y_pred_test)
                    elif metric == 'precision':
                        train_score = precision_score(y_train_fold, y_pred_train)
                        test_score = precision_score(y_test_fold, y_pred_test)
                    elif metric == 'f1':
                        train_score = f1_score(y_train_fold, y_pred_train)
                        test_score = f1_score(y_test_fold, y_pred_test)
                    elif metric == 'accuracy':
                        train_score = accuracy_score(y_train_fold, y_pred_train)
                        test_score = accuracy_score(y_test_fold, y_pred_test)

                    fold_result = {
                        'train': float(train_score),
                        'test': float(test_score),
                        'overfitting': float(train_score - test_score)
                    }

                    results[metric].append(fold_result)
                    fold_results[metric] = fold_result

                except Exception as e:
                    if logger:
                        logger.warning(f"Error calculating {metric} for fold {fold + 1}: {e}")
                    # Add NaN values for failed metrics
                    fold_result = {'train': np.nan, 'test': np.nan, 'overfitting': np.nan}
                    results[metric].append(fold_result)
                    fold_results[metric] = fold_result

            # Store overall scores for overfitting analysis
            valid_train_scores = [fold_results[m]['train'] for m in metrics if not np.isnan(fold_results[m]['train'])]
            valid_test_scores = [fold_results[m]['test'] for m in metrics if not np.isnan(fold_results[m]['test'])]

            results['train_scores'].append(np.mean(valid_train_scores) if valid_train_scores else np.nan)
            results['test_scores'].append(np.mean(valid_test_scores) if valid_test_scores else np.nan)

            # Store fold details
            fold_detail = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': fold_results
            }
            results['fold_details'].append(fold_detail)

        except Exception as e:
            if logger:
                logger.error(f"Error in fold {fold + 1}: {e}")
            # Add empty results for failed fold
            for metric in metrics:
                results[metric].append({'train': np.nan, 'test': np.nan, 'overfitting': np.nan})
            results['train_scores'].append(np.nan)
            results['test_scores'].append(np.nan)

    # Calculate summary statistics
    for metric in metrics:
        train_scores = [fold['train'] for fold in results[metric] if not np.isnan(fold['train'])]
        test_scores = [fold['test'] for fold in results[metric] if not np.isnan(fold['test'])]
        overfitting_scores = [fold['overfitting'] for fold in results[metric] if not np.isnan(fold['overfitting'])]

        results[f'{metric}_summary'] = {
            'train_mean': float(np.mean(train_scores)) if train_scores else np.nan,
            'train_std': float(np.std(train_scores)) if train_scores else np.nan,
            'test_mean': float(np.mean(test_scores)) if test_scores else np.nan,
            'test_std': float(np.std(test_scores)) if test_scores else np.nan,
            'overfitting_mean': float(np.mean(overfitting_scores)) if overfitting_scores else np.nan,
            'overfitting_std': float(np.std(overfitting_scores)) if overfitting_scores else np.nan,
            'n_folds': len([s for s in results[metric] if not np.isnan(s['train'])])
        }

    # Overall summary
    valid_train_scores = [s for s in results['train_scores'] if not np.isnan(s)]
    valid_test_scores = [s for s in results['test_scores'] if not np.isnan(s)]

    results['overall_summary'] = {
        'total_folds': len(cv_splits),
        'successful_folds': len(valid_train_scores),
        'avg_train_score': float(np.mean(valid_train_scores)) if valid_train_scores else np.nan,
        'avg_test_score': float(np.mean(valid_test_scores)) if valid_test_scores else np.nan,
        'avg_overfitting': float(np.mean(valid_train_scores) - np.mean(valid_test_scores)) if valid_train_scores and valid_test_scores else np.nan
    }

    if logger:
        logger.info(f"Temporal CV evaluation completed: {results['overall_summary']['successful_folds']}/{results['overall_summary']['total_folds']} folds successful")

    return results

# Alias for backward compatibility
create_temporal_aggregations_safe = create_temporal_features_safe

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

def create_temporal_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap_days=1):
    """
    Cria splits temporais adequados para dados de séries temporais.

    Args:
        df: DataFrame ordenado por timestamp
        train_ratio: proporção para treino
        val_ratio: proporção para validação
        test_ratio: proporção para teste
        gap_days: dias de gap entre splits para evitar data leakage

    Returns:
        train_df, val_df, test_df: DataFrames divididos temporalmente
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios devem somar 1.0"

    # Para dados com período muito curto, usar divisão por tempo absoluto
    total_period = df['Timestamp'].max() - df['Timestamp'].min()

    if total_period.days < 1:
        # Se período < 1 dia, dividir por tempo absoluto (evita gaps desnecessários)
        print("   ⚠️ Período muito curto detectado. Usando divisão por tempo absoluto...")

        # Calcular pontos de corte baseados em tempo absoluto
        train_end = df['Timestamp'].min() + total_period * train_ratio
        val_end = train_end + (total_period * val_ratio)

        # Criar splits sem gap (dados são muito próximos temporalmente)
        train_df = df[df['Timestamp'] <= train_end].copy()
        val_df = df[(df['Timestamp'] > train_end) & (df['Timestamp'] <= val_end)].copy()
        test_df = df[df['Timestamp'] > val_end].copy()
    else:
        # Período normal - usar abordagem com gaps
        train_end = df['Timestamp'].min() + total_period * train_ratio
        val_end = train_end + (total_period * val_ratio)

        # Adicionar gap de segurança
        gap = timedelta(days=gap_days)
        val_start = train_end + gap
        test_start = val_end + gap

        # Criar splits
        train_df = df[df['Timestamp'] <= train_end].copy()
        val_df = df[(df['Timestamp'] >= val_start) & (df['Timestamp'] <= val_end)].copy()
        test_df = df[df['Timestamp'] >= test_start].copy()

    return train_df, val_df, test_df


def validate_temporal_splits(train_df, val_df, test_df):
    """
    Valida a integridade dos splits temporais.
    """
    print("🔍 VALIDANDO SPLITS TEMPORAIS...")

    # 1. Verificar ordenação temporal
    train_max = train_df['Timestamp'].max()
    val_min = val_df['Timestamp'].min()
    val_max = val_df['Timestamp'].max()
    test_min = test_df['Timestamp'].min()

    print("   ⏰ Verificando ordenação temporal...")
    if train_max < val_min and val_max < test_min:
        print("   ✅ Ordenação temporal correta!")
    else:
        print("   ❌ ERRO: Ordenação temporal violada!")
        return False

    # 2. Verificar ausência de sobreposição
    print("   🚫 Verificando ausência de sobreposição...")
    overlap_val = len(val_df[val_df['Timestamp'] <= train_max])
    overlap_test = len(test_df[test_df['Timestamp'] <= val_max])

    if overlap_val == 0 and overlap_test == 0:
        print("   ✅ Sem sobreposição entre splits!")
    else:
        print(f"   ❌ ERRO: {overlap_val} sobreposições val, {overlap_test} sobreposições test!")
        return False

    # 3. Verificar distribuição temporal
    print("   📈 Verificando distribuição temporal...")
    total_period = test_df['Timestamp'].max() - train_df['Timestamp'].min()
    train_period = train_df['Timestamp'].max() - train_df['Timestamp'].min()
    val_period = val_df['Timestamp'].max() - val_df['Timestamp'].min()
    test_period = test_df['Timestamp'].max() - test_df['Timestamp'].min()

    print(f"   Período total: {total_period}")
    print(f"   Train: {train_period} ({train_period/total_period*100:.1f}%)")
    print(f"   Validation: {val_period} ({val_period/total_period*100:.1f}%)")
    print(f"   Test: {test_period} ({test_period/total_period*100:.1f}%)")

    # 4. Verificar distribuição do target
    print("   🎯 Verificando distribuição do target...")
    train_fraud = train_df['Is Laundering'].mean()
    val_fraud = val_df['Is Laundering'].mean()
    test_fraud = test_df['Is Laundering'].mean()

    print(f"   Taxa de fraude - Train: {train_fraud:.3%}")
    print(f"   Taxa de fraude - Validation: {val_fraud:.3%}")
    print(f"   Taxa de fraude - Test: {test_fraud:.3%}")

    # Verificar se distribuições são similares (diferença < 5%)
    max_diff = max(abs(train_fraud - val_fraud), abs(val_fraud - test_fraud), abs(train_fraud - test_fraud))
    if max_diff < 0.05:
        print("   ✅ Distribuições do target similares!")
    else:
        print(f"   ⚠️ Diferença máxima na taxa de fraude: {max_diff:.3%}")

    print("✅ VALIDAÇÃO CONCLUÍDA!")
    return True
