# src/features/leakage_detector.py
"""
Leakage detection for temporal features
"""

from typing import Dict, Any, Optional, List, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

from .base import LeakageDetector
from ..utils import logger


class TemporalLeakageDetector(LeakageDetector):
    """Detects temporal data leakage"""

    def __init__(self,
                 name: str = "temporal_leakage_detector",
                 temporal_column: str = "timestamp",
                 target_column: str = "target",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, temporal_column, config or {})
        self.target_column = target_column
        self.leakage_types = {
            'future_leakage': [],
            'data_drift': [],
            'label_leakage': [],
            'temporal_overlap': []
        }

    def detect_leakage(self, data: pd.DataFrame) -> List[str]:
        """
        Detect various types of temporal leakage

        Args:
            data: DataFrame to analyze

        Returns:
            List of detected leaky features
        """
        leaky_features = []

        if not self.validate_temporal_ordering(data):
            self.logger.warning("Data is not properly temporally ordered, leakage detection may be unreliable")
            return leaky_features

        # Detect future information leakage
        future_leaky = self._detect_future_leakage(data)
        leaky_features.extend(future_leaky)
        self.leakage_types['future_leakage'] = future_leaky

        # Detect data drift indicators
        drift_features = self._detect_data_drift(data)
        leaky_features.extend(drift_features)
        self.leakage_types['data_drift'] = drift_features

        # Detect label leakage through correlations
        label_leaky = self._detect_label_leakage(data)
        leaky_features.extend(label_leaky)
        self.leakage_types['label_leakage'] = label_leaky

        # Detect temporal overlap issues
        overlap_features = self._detect_temporal_overlap(data)
        leaky_features.extend(overlap_features)
        self.leakage_types['temporal_overlap'] = overlap_features

        # Remove duplicates
        leaky_features = list(set(leaky_features))

        self.logger.info(f"Detected {len(leaky_features)} potentially leaky features across {sum(len(v) for v in self.leakage_types.values())} leakage types")

        return leaky_features

    def _detect_future_leakage(self, data: pd.DataFrame) -> List[str]:
        """Detect features that contain future information"""
        leaky_features = []

        if self.temporal_column not in data.columns or self.target_column not in data.columns:
            return leaky_features

        # Sort data by time
        sorted_data = data.sort_values(self.temporal_column).copy()
        target_values = sorted_data[self.target_column].values

        # Check for features that correlate too strongly with future target values
        lookaheads = self.config.get('future_leakage_lookaheads', [1, 7, 14, 30])

        for col in sorted_data.columns:
            if col in [self.temporal_column, self.target_column]:
                continue

            if not pd.api.types.is_numeric_dtype(sorted_data[col]):
                continue

            series = sorted_data[col].fillna(sorted_data[col].median()).values

            for lookahead in lookaheads:
                if lookahead >= len(target_values):
                    continue

                # Calculate correlation with future target
                future_target = target_values[lookahead:]
                current_series = series[:-lookahead]

                if len(current_series) > 10:  # Minimum sample size
                    try:
                        corr = abs(np.corrcoef(current_series, future_target)[0, 1])
                        if not np.isnan(corr) and corr > self.config.get('future_leakage_threshold', 0.8):
                            leaky_features.append(f"{col}_future_leakage_{lookahead}")
                            self.logger.warning(f"High correlation ({corr:.3f}) between {col} and {lookahead}-step future target")
                            break  # Only flag once per feature
                    except:
                        continue

        return list(set(leaky_features))

    def _detect_data_drift(self, data: pd.DataFrame) -> List[str]:
        """Detect features that show significant drift over time"""
        drift_features = []

        if self.temporal_column not in data.columns:
            return drift_features

        # Sort by time
        sorted_data = data.sort_values(self.temporal_column).copy()

        # Split data into temporal windows
        n_windows = self.config.get('drift_windows', 5)
        window_size = len(sorted_data) // n_windows

        if window_size < 10:  # Not enough data for drift detection
            return drift_features

        for col in sorted_data.columns:
            if col == self.temporal_column or not pd.api.types.is_numeric_dtype(sorted_data[col]):
                continue

            # Calculate statistics for each window
            window_stats = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(sorted_data))
                window_data = sorted_data[col].iloc[start_idx:end_idx]

                if len(window_data.dropna()) > 5:
                    window_stats.append({
                        'mean': window_data.mean(),
                        'std': window_data.std(),
                        'median': window_data.median()
                    })

            if len(window_stats) >= 3:
                # Check for significant changes in distribution
                means = [stat['mean'] for stat in window_stats]
                stds = [stat['std'] for stat in window_stats]

                # Calculate coefficient of variation of means
                if np.mean(means) != 0:
                    cv_means = np.std(means) / abs(np.mean(means))
                    if cv_means > self.config.get('drift_threshold', 0.5):
                        drift_features.append(f"{col}_data_drift")
                        self.logger.warning(f"Significant drift detected in {col} (CV of means: {cv_means:.3f})")

        return drift_features

    def _detect_label_leakage(self, data: pd.DataFrame) -> List[str]:
        """Detect features that leak label information"""
        leaky_features = []

        if self.target_column not in data.columns:
            return leaky_features

        target = data[self.target_column]

        for col in data.columns:
            if col in [self.temporal_column, self.target_column]:
                continue

            # Skip non-numeric columns for correlation-based detection
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue

            series = data[col].fillna(data[col].median())

            # Calculate correlation with target
            try:
                if target.dtype in ['int64', 'int32'] and target.nunique() <= 20:
                    # Classification: use point-biserial correlation or ANOVA
                    from scipy.stats import f_oneway

                    groups = [series[target == val] for val in target.unique()]
                    if all(len(g) > 1 for g in groups):
                        f_stat, p_val = f_oneway(*groups)
                        if p_val < 0.001:  # Very strong relationship
                            leaky_features.append(f"{col}_label_leakage")
                            self.logger.warning(f"Strong relationship between {col} and target (p={p_val:.6f})")
                else:
                    # Regression: use Pearson correlation
                    corr = abs(series.corr(target))
                    if not np.isnan(corr) and corr > self.config.get('label_leakage_threshold', 0.9):
                        leaky_features.append(f"{col}_label_leakage")
                        self.logger.warning(f"Very high correlation ({corr:.3f}) between {col} and target")

            except Exception as e:
                self.logger.debug(f"Could not compute correlation for {col}: {e}")
                continue

        return list(set(leaky_features))

    def _detect_temporal_overlap(self, data: pd.DataFrame) -> List[str]:
        """Detect features that might have temporal overlap issues"""
        overlap_features = []

        if self.temporal_column not in data.columns:
            return overlap_features

        # Check for features that are constant within short time windows
        # This might indicate data processing artifacts
        sorted_data = data.sort_values(self.temporal_column).copy()

        # Group by time windows
        time_windows = pd.cut(sorted_data[self.temporal_column],
                            bins=self.config.get('overlap_bins', 10))

        for col in sorted_data.columns:
            if col == self.temporal_column or not pd.api.types.is_numeric_dtype(sorted_data[col]):
                continue

            # Check if feature is nearly constant within windows
            window_std = sorted_data.groupby(time_windows)[col].std()

            # If many windows have very low standard deviation, might be problematic
            low_std_windows = (window_std < sorted_data[col].std() * 0.01).sum()
            total_windows = len(window_std)

            if total_windows > 3 and low_std_windows / total_windows > 0.5:
                overlap_features.append(f"{col}_temporal_overlap")
                self.logger.warning(f"Potential temporal overlap in {col}: {low_std_windows}/{total_windows} windows have very low variance")

        return overlap_features

    def get_leakage_report(self) -> Dict[str, Any]:
        """Get detailed leakage detection report"""
        return {
            'total_leaky_features': len(self.leaky_features),
            'leakage_types': {k: len(v) for k, v in self.leakage_types.items()},
            'detailed_findings': self.leakage_types,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []

        if self.leakage_types['future_leakage']:
            recommendations.append("Remove or delay features with future leakage to prevent overfitting")

        if self.leakage_types['data_drift']:
            recommendations.append("Monitor and handle data drift in production - consider retraining strategies")

        if self.leakage_types['label_leakage']:
            recommendations.append("Review feature engineering process to prevent label leakage")

        if self.leakage_types['temporal_overlap']:
            recommendations.append("Investigate data collection process for potential temporal artifacts")

        if not any(self.leakage_types.values()):
            recommendations.append("No significant leakage detected - proceed with caution and monitor in production")

        return recommendations

    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal features for leakage"""
        results = super().validate_features(data)

        # Add leakage-specific validation
        leakage_report = self.get_leakage_report()
        results.update({
            'leakage_detected': leakage_report['total_leaky_features'] > 0,
            'leakage_types': leakage_report['leakage_types'],
            'leakage_recommendations': leakage_report['recommendations']
        })

        return results


class RollingStatisticsLeakageDetector(LeakageDetector):
    """Detects leakage in rolling statistics features"""

    def __init__(self,
                 name: str = "rolling_leakage_detector",
                 temporal_column: str = "timestamp",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, temporal_column, config or {})
        self.rolling_features = []

    def detect_leakage(self, data: pd.DataFrame) -> List[str]:
        """Detect rolling statistics that might leak future information"""
        leaky_features = []

        if not self.validate_temporal_ordering(data):
            return leaky_features

        # Find rolling features
        rolling_patterns = ['rolling', 'moving', 'ma_', 'ewm']
        for col in data.columns:
            if any(pattern in col.lower() for pattern in rolling_patterns):
                self.rolling_features.append(col)

        # Check for look-ahead bias in rolling features
        for feature in self.rolling_features:
            if self._has_lookahead_bias(data, feature):
                leaky_features.append(feature)
                self.logger.warning(f"Rolling feature {feature} may have look-ahead bias")

        return leaky_features

    def _has_lookahead_bias(self, data: pd.DataFrame, feature: str) -> bool:
        """Check if rolling feature has look-ahead bias"""
        if feature not in data.columns:
            return False

        # For rolling features, check if they correlate with future values
        # This is a simplified check - in practice, you'd need domain knowledge
        sorted_data = data.sort_values(self.temporal_column)

        # Look for suspiciously smooth or predictive patterns
        series = sorted_data[feature].fillna(method='bfill').fillna(method='ffill')

        # Check if the feature changes too abruptly (might indicate data leakage)
        diff = series.diff().abs()
        abrupt_changes = (diff > series.std() * 3).sum()

        # If too many abrupt changes, might be problematic
        return abrupt_changes > len(series) * 0.1


class CrossValidationLeakageDetector(LeakageDetector):
    """Detects leakage that occurs during cross-validation"""

    def __init__(self,
                 name: str = "cv_leakage_detector",
                 temporal_column: str = "timestamp",
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, temporal_column, config or {})

    def detect_leakage(self, data: pd.DataFrame) -> List[str]:
        """Detect features that might cause leakage in cross-validation"""
        leaky_features = []

        # Check for features that are too predictive (potential overfitting)
        # This is a simplified implementation

        # Look for features with suspiciously high importance
        # In a real implementation, this would use feature importance from a model

        suspicious_patterns = [
            'id', 'index', 'row', 'sequential',
            'timestamp', 'date', 'time'  # These might be indirectly leaky
        ]

        for col in data.columns:
            if col == self.temporal_column:
                continue

            col_lower = col.lower()
            if any(pattern in col_lower for pattern in suspicious_patterns):
                if data[col].nunique() / len(data) > 0.9:  # High uniqueness
                    leaky_features.append(col)
                    self.logger.warning(f"Potentially leaky feature in CV: {col} (high uniqueness)")

        return leaky_features


def detect_data_leakage_features(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                threshold_ks: float = 0.05,
                                threshold_mi: float = 0.1) -> Dict[str, Any]:
    """
    Detect potential data leakage in features by comparing train/test distributions
    and analyzing feature-target relationships.

    This function performs comprehensive leakage detection by:
    1. Comparing feature distributions between train and test sets using KS test
    2. Analyzing mutual information between features and target variable
    3. Flagging features that show suspicious patterns

    Args:
        X_train: Training features DataFrame. Must contain numeric columns for analysis.
        X_test: Test features DataFrame. Must have same columns as X_train.
        y_train: Training target series. Used for mutual information calculations.
        threshold_ks: P-value threshold for KS test distribution difference.
            Lower values (e.g., 0.05) are more conservative. Defaults to 0.05.
        threshold_mi: Mutual information threshold for suspicious features.
            Higher values (e.g., 0.1) reduce false positives. Defaults to 0.1.

    Returns:
        Dictionary containing leakage analysis results with the following keys:
        - 'leakage_candidates': List of suspicious feature dictionaries
        - 'distribution_shifts': List of features with distribution shifts
        - 'total_features_analyzed': Total number of features analyzed
        - 'suspicious_features_count': Number of suspicious features found
        - 'distribution_shift_count': Number of features with distribution shifts

    Raises:
        TypeError: If inputs are not of expected types (DataFrame, Series).
        ValueError: If DataFrames have incompatible shapes or missing columns.
        RuntimeError: If statistical calculations fail unexpectedly.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> X_train = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
        >>> X_test = pd.DataFrame({'feature1': np.random.randn(50), 'feature2': np.random.randn(50)})
        >>> y_train = pd.Series(np.random.randint(0, 2, 100))
        >>> results = detect_data_leakage_features(X_train, X_test, y_train)
        >>> print(f"Found {results['suspicious_features_count']} suspicious features")

    Notes:
        - Features are flagged as suspicious if they show distribution shifts OR high mutual information
        - Risk levels are assigned: 'HIGH' (both conditions), 'MEDIUM' (one condition)
        - Results are sorted by risk level and mutual information score
        - Categorical features use chi-square test instead of mutual information
    """
    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"X_train and X_test must have same number of columns. Got {X_train.shape[1]} vs {X_test.shape[1]}")

    if len(X_train) != len(y_train):
        raise ValueError(f"X_train and y_train must have same length. Got {len(X_train)} vs {len(y_train)}")

    if not set(X_train.columns).issubset(set(X_test.columns)):
        missing_cols = set(X_train.columns) - set(X_test.columns)
        raise ValueError(f"X_test is missing columns present in X_train: {missing_cols}")

    # Validate thresholds
    if not (0 < threshold_ks <= 1):
        raise ValueError("threshold_ks must be between 0 and 1")

    if not (0 <= threshold_mi <= 1):
        raise ValueError("threshold_mi must be between 0 and 1")

    logger.info(f"🔍 Starting leakage detection analysis on {len(X_train.columns)} features")
    logger.info(f"   Train set: {X_train.shape}, Test set: {X_test.shape}")

    leakage_candidates = []
    distribution_shifts = []

    print("🔍 Analyzing feature distributions for data leakage...")

    for feature in X_train.columns:
        try:
            # Skip non-analyzing columns if needed
            if feature in ['index', 'level_0']:
                continue

            # KS test for distribution difference
            try:
                train_values = X_train[feature].dropna()
                test_values = X_test[feature].dropna()

                if len(train_values) < 10 or len(test_values) < 10:
                    logger.warning(f"Skipping {feature}: insufficient data for KS test")
                    continue

                ks_stat, ks_pvalue = stats.ks_2samp(train_values, test_values)

                # Validate KS test results
                if np.isnan(ks_stat) or np.isnan(ks_pvalue):
                    logger.warning(f"KS test failed for {feature}: invalid results")
                    continue

            except Exception as e:
                logger.warning(f"KS test failed for {feature}: {e}")
                continue

            # Mutual information with target (only for train set)
            try:
                if X_train[feature].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numeric features: use mutual information
                    mi_score = mutual_info_classif(X_train[[feature]].fillna(0), y_train)[0]

                    # Validate MI score
                    if np.isnan(mi_score) or mi_score < 0:
                        mi_score = 0

                else:
                    # Categorical features: use chi-square test approximation
                    contingency_table = pd.crosstab(X_train[feature].fillna('missing'), y_train)
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        try:
                            chi2, p_val, _, _ = chi2_contingency(contingency_table)
                            # Normalize chi-square to approximate mutual information range
                            mi_score = min(chi2 / len(X_train), 1.0)
                        except Exception:
                            mi_score = 0
                    else:
                        mi_score = 0

            except Exception as e:
                logger.warning(f"Mutual information calculation failed for {feature}: {e}")
                mi_score = 0

            # Flag suspicious features
            is_distribution_shift = ks_pvalue < threshold_ks
            is_high_mi = mi_score > threshold_mi

            if is_distribution_shift or is_high_mi:
                risk_level = 'HIGH' if (is_distribution_shift and is_high_mi) else 'MEDIUM'

                candidate = {
                    'feature': feature,
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'mutual_info': float(mi_score),
                    'distribution_shift': bool(is_distribution_shift),
                    'high_mutual_info': bool(is_high_mi),
                    'risk_level': risk_level
                }

                leakage_candidates.append(candidate)

                # Log findings
                if is_distribution_shift:
                    logger.warning(f"Distribution shift in {feature} (KS p-value: {ks_pvalue:.4f})")
                if is_high_mi:
                    logger.warning(f"High mutual information in {feature} (MI: {mi_score:.4f})")

            if is_distribution_shift:
                distribution_shifts.append(feature)

        except Exception as e:
            logger.error(f"Unexpected error analyzing feature {feature}: {e}")
            continue

    # Sort by risk level and MI score
    def sort_key(candidate):
        risk_order = {'HIGH': 0, 'MEDIUM': 1, 'UNKNOWN': 2}
        return (risk_order.get(candidate['risk_level'], 2), -candidate['mutual_info'])

    leakage_candidates.sort(key=sort_key)

    results = {
        'leakage_candidates': leakage_candidates,
        'distribution_shifts': distribution_shifts,
        'total_features_analyzed': len(X_train.columns),
        'suspicious_features_count': len(leakage_candidates),
        'distribution_shift_count': len(distribution_shifts)
    }

    # Log summary
    logger.info(f"✅ Leakage detection complete: {len(leakage_candidates)} suspicious features found")
    logger.info(f"   Distribution shifts: {len(distribution_shifts)} features")

    if leakage_candidates:
        print(f"✅ Analysis complete: {len(leakage_candidates)} suspicious features found")
        print(f"   Distribution shifts: {len(distribution_shifts)}")

        # Display top suspicious features
        print("\n🔴 TOP SUSPICIOUS FEATURES:")
        for i, candidate in enumerate(leakage_candidates[:5]):
            print(f"  {i+1}. {candidate['feature']}")
            print(f"     Risk: {candidate['risk_level']}, KS p-value: {candidate['ks_pvalue']:.4f}, MI: {candidate['mutual_info']:.4f}")
    else:
        print("✅ No suspicious features detected")

    return results


def remove_leaky_features(X: pd.DataFrame,
                         leakage_candidates: List[Dict[str, Any]],
                         conservative: bool = True,
                         additional_patterns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove potentially leaky features from the dataset based on leakage analysis results.

    This function removes features identified as potentially leaky through various detection
    methods, with options for conservative or selective removal strategies.

    Args:
        X: Feature matrix (DataFrame) to clean.
        leakage_candidates: List of dictionaries containing leakage analysis results.
            Each dict should have at least 'feature' and 'risk_level' keys.
        conservative: If True, remove all suspicious features. If False, only high-risk ones.
            Defaults to True for safety.
        additional_patterns: Additional column name patterns to remove (e.g., ['_temp', '_debug']).
            Defaults to common aggregation patterns.

    Returns:
        Tuple of (cleaned DataFrame, list of removed feature names).

    Raises:
        TypeError: If X is not a pandas DataFrame or leakage_candidates is not a list.
        ValueError: If leakage_candidates contains invalid entries.

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feature1': [1, 2], 'feature2_mean': [3, 4], 'safe_feature': [5, 6]})
        >>> candidates = [{'feature': 'feature1', 'risk_level': 'HIGH'}]
        >>> X_clean, removed = remove_leaky_features(X, candidates)
        >>> print(removed)
        ['feature1', 'feature2_mean']
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(leakage_candidates, list):
        raise TypeError("leakage_candidates must be a list")

    # Validate leakage_candidates structure
    for i, candidate in enumerate(leakage_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"leakage_candidates[{i}] must be a dictionary")
        if 'feature' not in candidate:
            raise ValueError(f"leakage_candidates[{i}] must contain 'feature' key")

    # Default patterns for potentially leaky aggregation features
    if additional_patterns is None:
        additional_patterns = ['_mean', '_sum', '_count', '_std', '_min', '_max', '_median']

    features_to_remove = set()

    # Process leakage candidates
    for candidate in leakage_candidates:
        feature_name = candidate['feature']

        if conservative:
            # Remove all suspicious features
            features_to_remove.add(feature_name)
        else:
            # Remove only high-risk features
            risk_level = candidate.get('risk_level', 'UNKNOWN')
            if risk_level == 'HIGH':
                features_to_remove.add(feature_name)

    # Remove features matching aggregation patterns
    for col in X.columns:
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in additional_patterns):
            features_to_remove.add(col)

    # Convert to sorted list for consistency
    features_to_remove = sorted(list(features_to_remove))

    # Log removal information
    print(f"Removing {len(features_to_remove)} potentially leaky features...")
    if features_to_remove:
        print(f"Features to remove: {features_to_remove}")

    # Remove features (errors='ignore' handles cases where feature doesn't exist)
    X_clean = X.drop(columns=features_to_remove, errors='ignore')

    remaining_features = X_clean.shape[1]
    print(f"✅ Removed {len(features_to_remove)} features. Remaining: {remaining_features}")

    return X_clean, features_to_remove