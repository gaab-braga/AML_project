"""
AML Pipeline Automated Retraining System

This module implements automated model retraining pipelines with
data drift detection, performance monitoring, and continuous learning.
"""

import os
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
import configparser
import hashlib
import threading
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry.mlflow_integration import AMLPipelineMLflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient system operation.

    Prevents cascade failures by temporarily stopping operations
    when failure rate exceeds threshold.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 3600,
                 expected_exception: Exception = Exception):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Type of exception to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: When circuit is open
            Exception: Original function exception
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenException("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker reset to CLOSED after successful operation")

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_attempt_time': (self.last_failure_time + timedelta(seconds=self.recovery_timeout)).isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class DataValidator:
    """
    Comprehensive data validation for ML pipelines.

    Ensures data quality and prevents training on corrupted datasets.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        Initialize data validator.

        Args:
            config: MLOps configuration
        """
        self.config = config
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration."""
        return {
            'missing_value_threshold': self.config.getfloat('data_validation', 'missing_value_threshold', fallback=0.1),
            'duplicate_threshold': self.config.getfloat('data_validation', 'duplicate_threshold', fallback=0.05),
            'outlier_method': self.config.get('data_validation', 'outlier_method', fallback='iqr'),
            'outlier_threshold': self.config.getfloat('data_validation', 'outlier_threshold', fallback=1.5),
            'schema_validation': self.config.getboolean('data_validation', 'schema_validation', fallback=True)
        }

    def validate_dataset(self, data: pd.DataFrame, reference_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive data validation.

        Args:
            data: Dataset to validate
            reference_schema: Expected schema for validation

        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'checks': {},
            'issues': [],
            'recommendations': []
        }

        # Schema validation
        if self.validation_rules['schema_validation'] and reference_schema:
            schema_check = self._validate_schema(data, reference_schema)
            validation_results['checks']['schema'] = schema_check
            if not schema_check['valid']:
                validation_results['is_valid'] = False
                validation_results['issues'].extend(schema_check['issues'])

        # Missing values check
        missing_check = self._check_missing_values(data)
        validation_results['checks']['missing_values'] = missing_check
        if not missing_check['acceptable']:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing values exceed threshold: {missing_check['missing_percentage']:.2%}")

        # Duplicate check
        duplicate_check = self._check_duplicates(data)
        validation_results['checks']['duplicates'] = duplicate_check
        if not duplicate_check['acceptable']:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Duplicates exceed threshold: {duplicate_check['duplicate_percentage']:.2%}")

        # Outlier check
        outlier_check = self._check_outliers(data)
        validation_results['checks']['outliers'] = outlier_check

        # Data type consistency
        type_check = self._check_data_types(data)
        validation_results['checks']['data_types'] = type_check

        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)

        return validation_results

    def _validate_schema(self, data: pd.DataFrame, reference_schema: Dict) -> Dict[str, Any]:
        """Validate data schema against reference."""
        issues = []
        valid = True

        expected_columns = set(reference_schema.get('columns', []))
        actual_columns = set(data.columns)

        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
            valid = False

        if extra_columns:
            issues.append(f"Extra columns: {extra_columns}")
            # Not necessarily invalid, but worth noting

        # Check data types
        for col in expected_columns & actual_columns:
            expected_type = reference_schema.get('dtypes', {}).get(col)
            actual_type = str(data[col].dtype)

            if expected_type and expected_type != actual_type:
                issues.append(f"Type mismatch for {col}: expected {expected_type}, got {actual_type}")
                valid = False

        return {
            'valid': valid,
            'issues': issues,
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns)
        }

    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values."""
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        total_cells = data.shape[0] * data.shape[1]
        missing_percentage = total_missing / total_cells if total_cells > 0 else 0

        acceptable = missing_percentage <= self.validation_rules['missing_value_threshold']

        return {
            'acceptable': acceptable,
            'missing_percentage': missing_percentage,
            'missing_by_column': missing_counts.to_dict(),
            'columns_with_missing': (missing_counts > 0).sum()
        }

    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows."""
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = duplicate_count / len(data) if len(data) > 0 else 0

        acceptable = duplicate_percentage <= self.validation_rules['duplicate_threshold']

        return {
            'acceptable': acceptable,
            'duplicate_percentage': duplicate_percentage,
            'duplicate_count': duplicate_count
        }

    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for outliers in numerical columns."""
        outlier_info = {}

        for column in data.select_dtypes(include=[np.number]).columns:
            if self.validation_rules['outlier_method'] == 'iqr':
                outliers = self._detect_outliers_iqr(data[column])
            elif self.validation_rules['outlier_method'] == 'zscore':
                outliers = self._detect_outliers_zscore(data[column])
            else:
                outliers = []

            outlier_percentage = len(outliers) / len(data) if len(data) > 0 else 0

            outlier_info[column] = {
                'outlier_count': len(outliers),
                'outlier_percentage': outlier_percentage,
                'outlier_indices': outliers[:10]  # First 10 for reporting
            }

        return outlier_info

    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (self.validation_rules['outlier_threshold'] * IQR)
        upper_bound = Q3 + (self.validation_rules['outlier_threshold'] * IQR)

        return series[(series < lower_bound) | (series > upper_bound)].index.tolist()

    def _detect_outliers_zscore(self, series: pd.Series) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series[z_scores > self.validation_rules['outlier_threshold']].index.tolist()

    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency."""
        type_info = {}

        for column in data.columns:
            dtype = data[column].dtype
            type_info[column] = {
                'dtype': str(dtype),
                'unique_values': data[column].nunique() if dtype == 'object' else None,
                'sample_values': data[column].head(3).tolist()
            }

        return type_info

    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if not validation_results['checks']['missing_values']['acceptable']:
            recommendations.append("Consider imputation strategies for missing values (mean/median for numerical, mode for categorical)")

        if not validation_results['checks']['duplicates']['acceptable']:
            recommendations.append("Remove duplicate rows to prevent data leakage")

        outlier_columns = [col for col, info in validation_results['checks']['outliers'].items()
                          if info['outlier_percentage'] > 0.05]  # More than 5% outliers
        if outlier_columns:
            recommendations.append(f"Consider outlier treatment for columns: {outlier_columns}")

        return recommendations

    def clean_dataset(self, data: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """
        Clean dataset based on validation results.

        Args:
            data: Original dataset
            validation_results: Results from validate_dataset

        Returns:
            Cleaned dataset
        """
        cleaned_data = data.copy()

        # Remove duplicates if excessive
        if not validation_results['checks']['duplicates']['acceptable']:
            cleaned_data = cleaned_data.drop_duplicates()
            logger.info(f"Removed {validation_results['checks']['duplicates']['duplicate_count']} duplicate rows")

        # Handle missing values (simple imputation)
        missing_check = validation_results['checks']['missing_values']
        for col, count in missing_check['missing_by_column'].items():
            if count > 0:
                if cleaned_data[col].dtype in ['int64', 'float64']:
                    # Numerical: fill with median
                    median_val = cleaned_data[col].median()
                    cleaned_data[col] = cleaned_data[col].fillna(median_val)
                else:
                    # Categorical: fill with mode
                    mode_val = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown'
                    cleaned_data[col] = cleaned_data[col].fillna(mode_val)

        return cleaned_data


class RobustDriftDetector:
    """
    Robust statistical drift detection using multiple methods.

    Implements Kolmogorov-Smirnov test, Wasserstein distance, and
    population stability index for comprehensive drift detection.
    """

    def __init__(self, reference_data: pd.DataFrame, config: configparser.ConfigParser):
        """
        Initialize robust drift detector.

        Args:
            reference_data: Reference dataset for comparison
            config: MLOps configuration
        """
        self.reference_data = reference_data.copy()
        self.config = config
        self.drift_threshold = config.getfloat('model_retraining', 'drift_threshold', fallback=0.05)
        self.significance_level = config.getfloat('model_retraining', 'significance_level', fallback=0.05)

        # Calculate reference statistics
        self.reference_stats = self._calculate_reference_stats()

        # Initialize circuit breaker for drift detection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.getint('model_retraining', 'drift_circuit_failure_threshold', fallback=3),
            recovery_timeout=config.getint('model_retraining', 'drift_circuit_recovery_timeout', fallback=1800)
        )

        logger.info("Robust drift detector initialized")

    def _calculate_reference_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive reference statistics."""
        stats = {}

        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['int64', 'float64']:
                series = self.reference_data[column].dropna()
                stats[column] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'median': series.median(),
                    'quartiles': series.quantile([0.25, 0.5, 0.75]).to_dict(),
                    'min': series.min(),
                    'max': series.max(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'distribution': self._estimate_distribution(series)
                }
            else:
                # Categorical columns
                value_counts = self.reference_data[column].value_counts()
                stats[column] = {
                    'value_counts': value_counts.to_dict(),
                    'unique_count': self.reference_data[column].nunique(),
                    'mode': value_counts.index[0] if not value_counts.empty else None,
                    'entropy': self._calculate_entropy(value_counts)
                }

        return stats

    def _estimate_distribution(self, series: pd.Series) -> str:
        """Estimate the distribution type of a numerical series."""
        # Simple distribution estimation based on skewness and kurtosis
        skew = series.skew()
        kurt = series.kurtosis()

        if abs(skew) < 0.5 and abs(kurt) < 0.5:
            return 'normal'
        elif skew > 1:
            return 'right_skewed'
        elif skew < -1:
            return 'left_skewed'
        else:
            return 'unknown'

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        probabilities = value_counts / value_counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift using multiple statistical methods.

        Args:
            current_data: Current dataset to compare

        Returns:
            Comprehensive drift detection results
        """
        def _drift_detection_core():
            return self._detect_drift_internal(current_data)

        try:
            return self.circuit_breaker.call(_drift_detection_core)
        except CircuitBreakerOpenException:
            logger.warning("Drift detection circuit breaker is OPEN, returning no drift")
            return {
                'overall_drift_score': 0.0,
                'drift_detected': False,
                'circuit_breaker_open': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            # Return conservative result on failure
            return {
                'overall_drift_score': 0.0,
                'drift_detected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _detect_drift_internal(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Internal drift detection implementation.
        """
        drift_results = {
            'overall_drift_score': 0.0,
            'feature_drift': {},
            'drift_detected': False,
            'methods_used': [],
            'timestamp': datetime.now().isoformat(),
            'confidence_level': self.significance_level
        }

        total_drift_score = 0.0
        feature_count = 0

        for column in self.reference_data.columns:
            if column not in current_data.columns:
                logger.warning(f"Column {column} missing in current data")
                continue

            ref_series = self.reference_data[column].dropna()
            curr_series = current_data[column].dropna()

            if len(curr_series) == 0:
                logger.warning(f"No valid data for column {column}")
                continue

            if self.reference_data[column].dtype in ['int64', 'float64']:
                # Numerical drift detection
                drift_score, methods = self._calculate_numerical_drift(ref_series, curr_series)
            else:
                # Categorical drift detection
                drift_score, methods = self._calculate_categorical_drift(
                    self.reference_stats[column], curr_series
                )

            drift_results['feature_drift'][column] = {
                'drift_score': drift_score,
                'significant_drift': drift_score > self.drift_threshold,
                'methods': methods
            }

            drift_results['methods_used'].extend(methods)
            total_drift_score += drift_score
            feature_count += 1

        # Remove duplicate methods
        drift_results['methods_used'] = list(set(drift_results['methods_used']))

        # Calculate overall drift score
        drift_results['overall_drift_score'] = total_drift_score / feature_count if feature_count > 0 else 0.0
        drift_results['drift_detected'] = drift_results['overall_drift_score'] > self.drift_threshold

        # Add severity classification
        drift_results['severity'] = self._classify_drift_severity(drift_results['overall_drift_score'])

        logger.info(f"Drift detected: {drift_results['drift_detected']}, score: {drift_results['overall_drift_score']:.3f}")
        return drift_results

    def _calculate_numerical_drift(self, ref_series: pd.Series, curr_series: pd.Series) -> Tuple[float, List[str]]:
        """
        Calculate drift score for numerical features using multiple methods.

        Returns:
            Tuple of (drift_score, methods_used)
        """
        methods_used = []
        drift_scores = []

        # Method 1: Kolmogorov-Smirnov test
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(ref_series, curr_series)
            if ks_pvalue < self.significance_level:
                drift_scores.append(ks_stat)
                methods_used.append('kolmogorov_smirnov')
        except ImportError:
            logger.warning("scipy not available for KS test")

        # Method 2: Wasserstein distance (Earth Mover's Distance)
        try:
            from scipy.stats import wasserstein_distance
            wasserstein = wasserstein_distance(ref_series, curr_series)
            # Normalize by reference distribution scale
            normalized_wasserstein = wasserstein / (ref_series.std() + 1e-6)
            drift_scores.append(min(normalized_wasserstein, 1.0))  # Cap at 1.0
            methods_used.append('wasserstein_distance')
        except ImportError:
            logger.warning("scipy not available for Wasserstein distance")

        # Method 3: Population Stability Index (PSI)
        psi_score = self._calculate_psi(ref_series, curr_series)
        drift_scores.append(psi_score)
        methods_used.append('population_stability_index')

        # Method 4: Simple statistical comparison (fallback)
        if not drift_scores:
            mean_diff = abs(ref_series.mean() - curr_series.mean()) / (ref_series.std() + 1e-6)
            std_diff = abs(ref_series.std() - curr_series.std()) / (ref_series.std() + 1e-6)
            simple_score = (mean_diff + std_diff) / 2.0
            drift_scores.append(simple_score)
            methods_used.append('statistical_comparison')

        # Combine scores using maximum (conservative approach)
        final_score = max(drift_scores) if drift_scores else 0.0

        return final_score, methods_used

    def _calculate_categorical_drift(self, ref_stats: Dict, curr_series: pd.Series) -> Tuple[float, List[str]]:
        """
        Calculate drift score for categorical features.

        Returns:
            Tuple of (drift_score, methods_used)
        """
        methods_used = []
        drift_scores = []

        # Method 1: Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(ref_stats['value_counts'], curr_series.value_counts())
        drift_scores.append(js_divergence)
        methods_used.append('jensen_shannon_divergence')

        # Method 2: Chi-square test
        try:
            from scipy.stats import chi2_contingency
            # Create contingency table
            all_categories = set(ref_stats['value_counts'].keys()) | set(curr_series.value_counts().keys())

            ref_counts = [ref_stats['value_counts'].get(cat, 0) for cat in all_categories]
            curr_counts = [curr_series.value_counts().get(cat, 0) for cat in all_categories]

            contingency_table = np.array([ref_counts, curr_counts])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < self.significance_level:
                # Normalize chi-square statistic
                expected = contingency_table.sum() / 2
                normalized_chi2 = min(chi2 / (expected * len(all_categories)), 1.0)
                drift_scores.append(normalized_chi2)
                methods_used.append('chi_square_test')
        except ImportError:
            logger.warning("scipy not available for chi-square test")

        # Method 3: Distribution difference
        ref_dist = np.array(list(ref_stats['value_counts'].values()))
        curr_dist = np.array(list(curr_series.value_counts()))

        # Normalize distributions
        ref_dist = ref_dist / ref_dist.sum() if ref_dist.sum() > 0 else ref_dist
        curr_dist = curr_dist / curr_dist.sum() if curr_dist.sum() > 0 else curr_dist

        # Calculate L1 distance
        l1_distance = np.sum(np.abs(ref_dist - curr_dist)) / 2.0  # Normalize to [0, 1]
        drift_scores.append(l1_distance)
        methods_used.append('distribution_difference')

        # Combine scores using maximum
        final_score = max(drift_scores) if drift_scores else 0.0

        return final_score, methods_used

    def _calculate_psi(self, ref_series: pd.Series, curr_series: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change
        """
        try:
            # Create bins
            combined = pd.concat([ref_series, curr_series])
            bin_edges = pd.qcut(combined, q=bins, duplicates='drop', retbins=True)[1]

            # Calculate distributions
            ref_hist, _ = np.histogram(ref_series, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_series, bins=bin_edges)

            # Convert to percentages
            ref_pct = ref_hist / len(ref_series)
            curr_pct = curr_hist / len(curr_series)

            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
            curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

            return psi

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def _calculate_js_divergence(self, ref_counts: Dict, curr_counts: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence for categorical distributions."""
        # Get all categories
        all_categories = set(ref_counts.keys()) | set(curr_counts.index)

        # Create probability distributions
        ref_total = sum(ref_counts.values())
        curr_total = curr_counts.sum()

        ref_probs = np.array([ref_counts.get(cat, 0) / ref_total for cat in all_categories])
        curr_probs = np.array([curr_counts.get(cat, 0) / curr_total for cat in all_categories])

        # Calculate Jensen-Shannon divergence
        m = (ref_probs + curr_probs) / 2

        # Avoid log(0)
        ref_probs = np.where(ref_probs == 0, 1e-10, ref_probs)
        curr_probs = np.where(curr_probs == 0, 1e-10, curr_probs)
        m = np.where(m == 0, 1e-10, m)

        js_div = 0.5 * np.sum(ref_probs * np.log(ref_probs / m)) + 0.5 * np.sum(curr_probs * np.log(curr_probs / m))

        return js_div / np.log(2)  # Convert to base 2

    def _classify_drift_severity(self, drift_score: float) -> str:
        """Classify drift severity based on score."""
        if drift_score < 0.1:
            return 'low'
        elif drift_score < 0.25:
            return 'moderate'
        elif drift_score < 0.5:
            return 'high'
        else:
            return 'critical'

    def get_health_status(self) -> Dict[str, Any]:
        """Get detector health status."""
        return {
            'circuit_breaker': self.circuit_breaker.get_status(),
            'reference_data_shape': self.reference_data.shape,
            'reference_stats_calculated': len(self.reference_stats) > 0,
            'last_detection_time': getattr(self, '_last_detection_time', None)
        }


class DataDriftDetector:
    """
    Legacy wrapper for RobustDriftDetector for backward compatibility.
    """

    def __init__(self, reference_data: pd.DataFrame, config: configparser.ConfigParser):
        """
        Initialize drift detector with robust backend.

        Args:
            reference_data: Reference dataset for comparison
            config: MLOps configuration
        """
        self.robust_detector = RobustDriftDetector(reference_data, config)
        logger.info("Data drift detector initialized with robust backend")

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift using robust methods.

        Args:
            current_data: Current dataset to compare

        Returns:
            Drift detection results
        """
        return self.robust_detector.detect_drift(current_data)


class ModelPerformanceMonitor:
    """
    Enhanced performance monitoring with statistical significance testing.

    Provides robust performance degradation detection with confidence intervals
    and adaptive baselines.
    """

    def __init__(self, config: configparser.ConfigParser):
        """
        Initialize enhanced performance monitor.

        Args:
            config: MLOps configuration
        """
        self.config = config
        self.performance_history = []
        self.max_history_size = config.getint('model_retraining', 'max_performance_history', fallback=1000)

        # Adaptive baseline settings
        self.baseline_window = config.getint('model_retraining', 'baseline_window_days', fallback=30)
        self.min_baseline_size = config.getint('model_retraining', 'min_baseline_size', fallback=10)
        self.confidence_level = config.getfloat('model_retraining', 'confidence_level', fallback=0.95)

        # Performance thresholds with confidence intervals
        self.performance_thresholds = {
            'accuracy_drop': config.getfloat('model_retraining', 'performance_thresholds.accuracy_drop', fallback=0.05),
            'precision_drop': config.getfloat('model_retraining', 'performance_thresholds.precision_drop', fallback=0.10),
            'recall_drop': config.getfloat('model_retraining', 'performance_thresholds.recall_drop', fallback=0.10),
            'f1_drop': config.getfloat('model_retraining', 'performance_thresholds.f1_drop', fallback=0.08)
        }

        # Statistical test settings
        self.enable_statistical_tests = config.getboolean('model_retraining', 'enable_statistical_tests', fallback=True)
        self.minimum_sample_size = config.getint('model_retraining', 'minimum_sample_size', fallback=30)

        logger.info("Enhanced performance monitor initialized")

    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate model performance with enhanced metrics and metadata.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            metadata: Additional evaluation metadata

        Returns:
            Comprehensive performance metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true)
        }

        # Add probability-based metrics if available
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                except ImportError:
                    pass

        # Add class-specific metrics for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                from sklearn.metrics import classification_report
                report = classification_report(y_true, y_pred, output_dict=True)

                metrics.update({
                    'precision_class_0': report['0']['precision'],
                    'precision_class_1': report['1']['precision'],
                    'recall_class_0': report['0']['recall'],
                    'recall_class_1': report['1']['recall'],
                    'f1_class_0': report['0']['f1-score'],
                    'f1_class_1': report['1']['f1-score']
                })
            except ImportError:
                pass

        # Add metadata
        if metadata:
            metrics['metadata'] = metadata

        # Store in history
        self.performance_history.append(metrics)

        # Maintain history size limit
        if len(self.performance_history) > self.max_history_size:
            # Remove oldest entries, but keep at least minimum baseline size
            keep_size = max(self.min_baseline_size, self.max_history_size - 10)
            self.performance_history = self.performance_history[-keep_size:]

        return metrics

    def check_performance_degradation(self, current_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check for performance degradation using statistical methods.

        Args:
            current_metrics: Current performance metrics (optional, uses last if not provided)

        Returns:
            Comprehensive degradation analysis
        """
        if len(self.performance_history) < self.min_baseline_size:
            return {
                'degradation_detected': False,
                'reason': f'Insufficient performance history: {len(self.performance_history)} < {self.min_baseline_size}',
                'confidence': 0.0
            }

        # Use provided metrics or get latest
        if current_metrics is None:
            current_metrics = self.performance_history[-1]

        # Calculate adaptive baseline
        baseline_metrics = self._calculate_adaptive_baseline(current_metrics['timestamp'])

        if not baseline_metrics:
            return {
                'degradation_detected': False,
                'reason': 'Could not calculate baseline metrics',
                'confidence': 0.0
            }

        # Perform degradation analysis
        degradation_results = self._analyze_degradation(current_metrics, baseline_metrics)

        # Add statistical significance testing if enabled
        if self.enable_statistical_tests and len(self.performance_history) >= self.minimum_sample_size:
            statistical_results = self._perform_statistical_tests(current_metrics, baseline_metrics)
            degradation_results['statistical_tests'] = statistical_results

            # Adjust confidence based on statistical significance
            if statistical_results.get('significant_degradation', False):
                degradation_results['confidence'] = min(degradation_results['confidence'] * 1.5, 1.0)

        return degradation_results

    def _calculate_adaptive_baseline(self, current_timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Calculate adaptive baseline based on recent performance history.

        Considers temporal patterns and removes outliers.
        """
        try:
            current_time = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))

            # Filter recent history within baseline window
            recent_history = []
            for entry in reversed(self.performance_history[:-1]):  # Exclude current
                entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                if (current_time - entry_time).days <= self.baseline_window:
                    recent_history.append(entry)
                else:
                    break

            if len(recent_history) < self.min_baseline_size:
                return None

            # Calculate robust statistics (remove outliers)
            baseline = {}
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

            for metric in key_metrics:
                values = [entry.get(metric) for entry in recent_history if metric in entry]
                if len(values) >= self.min_baseline_size:
                    # Remove outliers using IQR
                    q75, q25 = np.percentile(values, [75, 25])
                    iqr = q75 - q25
                    lower_bound = q25 - (1.5 * iqr)
                    upper_bound = q75 + (1.5 * iqr)

                    filtered_values = [v for v in values if lower_bound <= v <= upper_bound]

                    if filtered_values:
                        baseline[metric] = {
                            'mean': np.mean(filtered_values),
                            'std': np.std(filtered_values),
                            'median': np.median(filtered_values),
                            'q25': np.percentile(filtered_values, 25),
                            'q75': np.percentile(filtered_values, 75),
                            'sample_size': len(filtered_values)
                        }

            return baseline if baseline else None

        except Exception as e:
            logger.error(f"Error calculating adaptive baseline: {e}")
            return None

    def _analyze_degradation(self, current: Dict, baseline: Dict) -> Dict[str, Any]:
        """
        Analyze performance degradation with confidence intervals.
        """
        degradation_detected = False
        degradation_details = {}
        max_drop = 0.0

        for metric, threshold in self.performance_thresholds.items():
            if metric in baseline and metric in current:
                baseline_stats = baseline[metric]
                current_value = current[metric]

                # Calculate drop with confidence interval
                baseline_mean = baseline_stats['mean']
                baseline_std = baseline_stats['std']
                sample_size = baseline_stats['sample_size']

                # Standard error of the mean
                se_mean = baseline_std / np.sqrt(sample_size) if sample_size > 1 else 0

                # Confidence interval (t-distribution approximation)
                z_score = 1.96  # 95% confidence
                margin_error = z_score * se_mean

                # Expected baseline range
                baseline_lower = baseline_mean - margin_error
                baseline_upper = baseline_mean + margin_error

                # Calculate drop
                drop = baseline_mean - current_value
                relative_drop = drop / baseline_mean if baseline_mean != 0 else 0

                # Check if current value is below baseline confidence interval
                significant_drop = current_value < baseline_lower

                if significant_drop and relative_drop > threshold:
                    degradation_detected = True
                    max_drop = max(max_drop, relative_drop)

                    degradation_details[metric] = {
                        'baseline_mean': baseline_mean,
                        'baseline_ci_lower': baseline_lower,
                        'baseline_ci_upper': baseline_upper,
                        'current_value': current_value,
                        'absolute_drop': drop,
                        'relative_drop': relative_drop,
                        'threshold': threshold,
                        'significant': True
                    }

        confidence = min(max_drop * 2, 1.0) if degradation_detected else 0.0

        return {
            'degradation_detected': degradation_detected,
            'details': degradation_details,
            'confidence': confidence,
            'baseline_size': len(self.performance_history) - 1,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _perform_statistical_tests(self, current: Dict, baseline: Dict) -> Dict[str, Any]:
        """
        Perform statistical significance tests for performance changes.
        """
        statistical_results = {
            'significant_degradation': False,
            'tests_performed': [],
            'p_values': {}
        }

        try:
            # Collect historical values for each metric
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']

            for metric in key_metrics:
                if metric in baseline:
                    # Get historical values
                    historical_values = []
                    for entry in self.performance_history[:-1]:  # Exclude current
                        if metric in entry:
                            historical_values.append(entry[metric])

                    if len(historical_values) >= self.minimum_sample_size:
                        current_value = current.get(metric)
                        if current_value is not None:
                            # Perform one-sample t-test
                            t_stat, p_value = self._one_sample_ttest(historical_values, current_value)

                            statistical_results['tests_performed'].append(metric)
                            statistical_results['p_values'][metric] = p_value

                            # Check if degradation is statistically significant
                            if p_value < (1 - self.confidence_level) and current_value < np.mean(historical_values):
                                statistical_results['significant_degradation'] = True

        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")

        return statistical_results

    def _one_sample_ttest(self, sample: List[float], test_value: float) -> Tuple[float, float]:
        """
        Perform one-sample t-test.

        Returns:
            Tuple of (t_statistic, p_value)
        """
        try:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(sample, test_value)
            return t_stat, p_value
        except ImportError:
            # Fallback: simple z-test approximation
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            sample_size = len(sample)

            if sample_std == 0:
                return 0.0, 1.0

            z_stat = (test_value - sample_mean) / (sample_std / np.sqrt(sample_size))

            # Approximate p-value for two-tailed test
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))

            return z_stat, p_value

    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze performance trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Trend analysis results
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) >= cutoff_date
        ]

        if len(recent_history) < 5:
            return {'error': 'Insufficient data for trend analysis'}

        trends = {}
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for metric in key_metrics:
            values = [entry.get(metric) for entry in recent_history if metric in entry]
            timestamps = [entry['timestamp'] for entry in recent_history if metric in entry]

            if len(values) >= 5:
                # Simple linear trend
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)

                # Calculate trend strength
                r_squared = np.corrcoef(x, values)[0, 1] ** 2

                trends[metric] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'trend_direction': 'improving' if slope > 0 else 'degrading',
                    'trend_strength': 'strong' if r_squared > 0.7 else 'weak' if r_squared > 0.3 else 'none',
                    'data_points': len(values)
                }

        return {
            'trends': trends,
            'analysis_period_days': days,
            'total_evaluations': len(recent_history)
        }


class AutomatedRetrainingPipeline:
    """
    Enterprise-grade automated model retraining pipeline with comprehensive monitoring.

    Features:
    - Circuit breaker pattern for resilience
    - Data validation and cleaning
    - Health checks and monitoring
    - Statistical significance testing
    - Comprehensive error handling
    """

    def __init__(self, config_path: str = "./mlops/mlops-config.ini"):
        """
        Initialize enterprise retraining pipeline.

        Args:
            config_path: Path to MLOps configuration file
        """
        self.config = self._load_config(config_path)
        self.mlflow_integration = AMLPipelineMLflow(config_path)

        # Initialize components with circuit breakers
        self.performance_monitor = ModelPerformanceMonitor(self.config)
        self.data_validator = DataValidator(self.config)

        # Circuit breakers for different operations
        self.retraining_circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.getint('model_retraining', 'retraining_circuit_failure_threshold', fallback=3),
            recovery_timeout=self.config.getint('model_retraining', 'retraining_circuit_recovery_timeout', fallback=3600)
        )

        self.mlflow_circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.getint('model_retraining', 'mlflow_circuit_failure_threshold', fallback=5),
            recovery_timeout=self.config.getint('model_retraining', 'mlflow_circuit_recovery_timeout', fallback=1800)
        )

        # Retraining state with thread safety
        self.last_retraining = None
        self.retraining_triggers = self.config.get('model_retraining', 'retraining_triggers').strip('[]').replace(' ', '').split(',')
        self.time_based_interval = self.config.getint('model_retraining', 'time_based_retraining_days', fallback=30)

        # Data validation settings
        self.enable_data_validation = self.config.getboolean('model_retraining', 'enable_data_validation', fallback=True)
        self.enable_data_cleaning = self.config.getboolean('model_retraining', 'enable_data_cleaning', fallback=True)

        # Reference data and drift detector
        self.drift_detector = None
        self.reference_data_schema = None

        # Callbacks and monitoring
        self.retraining_callbacks: List[Callable] = []
        self.health_check_interval = self.config.getint('model_retraining', 'health_check_interval_seconds', fallback=300)
        self.last_health_check = None

        logger.info("Enterprise automated retraining pipeline initialized")

    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load MLOps configuration with validation."""
        config = configparser.ConfigParser()
        config.read(config_path)

        # Validate required sections
        required_sections = ['model_retraining', 'mlflow']
        for section in required_sections:
            if not config.has_section(section):
                raise ValueError(f"Required configuration section '{section}' not found")

        return config

    def register_retraining_callback(self, callback: Callable):
        """
        Register a callback function to be called when retraining is triggered.

        Args:
            callback: Function to call with retraining results
        """
        self.retraining_callbacks.append(callback)

    def initialize_baseline(self, reference_data: pd.DataFrame, model_name: str = "fraud_detection_model"):
        """
        Initialize baseline data and model for drift detection with validation.

        Args:
            reference_data: Reference dataset
            model_name: Name of the model to monitor
        """
        logger.info("Initializing retraining baseline with validation...")

        # Validate reference data
        if self.enable_data_validation:
            validation_results = self.data_validator.validate_dataset(reference_data)
            if not validation_results['is_valid']:
                logger.warning("Reference data validation failed:")
                for issue in validation_results['issues']:
                    logger.warning(f"  - {issue}")

                if self.enable_data_cleaning:
                    logger.info("Attempting to clean reference data...")
                    reference_data = self.data_validator.clean_dataset(reference_data, validation_results)
                    logger.info("Reference data cleaned")
                else:
                    logger.warning("Data cleaning disabled, proceeding with potentially invalid data")

        # Store reference schema
        self.reference_data_schema = {
            'columns': list(reference_data.columns),
            'dtypes': {col: str(reference_data[col].dtype) for col in reference_data.columns},
            'shape': reference_data.shape
        }

        # Initialize drift detector
        self.drift_detector = RobustDriftDetector(reference_data, self.config)
        self.model_name = model_name

        # Load current production model with circuit breaker
        def load_model():
            model_uri = self.mlflow_integration.get_model_uri(model_name, stage="production")
            return self.mlflow_integration.load_model(model_uri)

        try:
            self.baseline_model = self.mlflow_circuit_breaker.call(load_model)
            logger.info(f"Baseline model loaded from: {self.mlflow_integration.get_model_uri(model_name, stage='production')}")
        except CircuitBreakerOpenException:
            logger.warning("MLflow circuit breaker open, skipping baseline model load")
            self.baseline_model = None
        except Exception as e:
            logger.warning(f"Could not load production model: {e}")
            self.baseline_model = None

        logger.info("Retraining baseline initialized successfully")

    def check_retraining_needed(self, current_data: pd.DataFrame,
                              y_true: Optional[np.ndarray] = None,
                              y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Check if retraining is needed with comprehensive validation and error handling.

        Args:
            current_data: Current dataset
            y_true: True labels (optional)
            y_pred: Predicted labels (optional)

        Returns:
            Retraining decision with detailed reasoning
        """
        retraining_decision = {
            'retraining_needed': False,
            'triggers': [],
            'confidence': 0.0,
            'details': {},
            'validation_results': None,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Data validation
            if self.enable_data_validation:
                validation_results = self.data_validator.validate_dataset(
                    current_data, self.reference_data_schema
                )
                retraining_decision['validation_results'] = validation_results

                if not validation_results['is_valid']:
                    logger.warning("Current data validation failed, triggering retraining")
                    retraining_decision['triggers'].append('data_validation_failure')
                    retraining_decision['confidence'] = max(retraining_decision['confidence'], 0.9)
                    retraining_decision['details']['validation_issues'] = validation_results['issues']

            # Data drift detection
            if self.drift_detector and 'data_drift' in self.retraining_triggers:
                drift_results = self.drift_detector.detect_drift(current_data)
                retraining_decision['details']['data_drift'] = drift_results

                if drift_results.get('drift_detected', False):
                    retraining_decision['triggers'].append('data_drift')
                    severity_multiplier = {'low': 1.0, 'moderate': 1.2, 'high': 1.5, 'critical': 2.0}
                    confidence_boost = severity_multiplier.get(drift_results.get('severity', 'low'), 1.0)
                    retraining_decision['confidence'] = max(retraining_decision['confidence'],
                                                          drift_results['overall_drift_score'] * confidence_boost)

            # Performance degradation
            if y_true is not None and y_pred is not None and 'performance_degradation' in self.retraining_triggers:
                perf_results = self.performance_monitor.check_performance_degradation()
                retraining_decision['details']['performance'] = perf_results

                if perf_results.get('degradation_detected', False):
                    retraining_decision['triggers'].append('performance_degradation')
                    retraining_decision['confidence'] = max(retraining_decision['confidence'],
                                                          perf_results.get('confidence', 0.8))

            # Time-based retraining
            if 'time_based' in self.retraining_triggers:
                if self.last_retraining is None:
                    retraining_decision['triggers'].append('time_based')
                    retraining_decision['confidence'] = max(retraining_decision['confidence'], 0.6)
                else:
                    days_since_retraining = (datetime.now() - self.last_retraining).days
                    if days_since_retraining >= self.time_based_interval:
                        retraining_decision['triggers'].append('time_based')
                        retraining_decision['confidence'] = max(retraining_decision['confidence'], 0.7)

            # Make final decision
            retraining_decision['retraining_needed'] = len(retraining_decision['triggers']) > 0

            logger.info(f"Retraining check: {retraining_decision['retraining_needed']} "
                       f"(triggers: {retraining_decision['triggers']}, "
                       ".2f")

        except Exception as e:
            logger.error(f"Retraining check failed: {e}")
            retraining_decision['error'] = str(e)
            # Conservative approach: assume retraining needed on error
            retraining_decision['retraining_needed'] = True
            retraining_decision['triggers'] = ['error_fallback']
            retraining_decision['confidence'] = 0.5

        return retraining_decision

    def trigger_retraining(self, training_data: pd.DataFrame,
                          target_column: str = 'is_fraud',
                          model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trigger automated model retraining with enterprise-grade error handling.

        Args:
            training_data: Training dataset
            target_column: Name of target column
            model_params: Model parameters (optional)

        Returns:
            Comprehensive retraining results
        """
        def retraining_operation():
            return self._execute_retraining(training_data, target_column, model_params)

        try:
            return self.retraining_circuit_breaker.call(retraining_operation)
        except CircuitBreakerOpenException:
            logger.error("Retraining circuit breaker is OPEN, skipping retraining")
            return {
                'success': False,
                'error': 'Circuit breaker open - too many recent failures',
                'circuit_breaker_open': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Retraining failed with unhandled error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _execute_retraining(self, training_data: pd.DataFrame,
                           target_column: str, model_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the actual retraining process.
        """
        logger.info("Starting enterprise automated model retraining...")

        start_time = datetime.now()
        retraining_results = {
            'success': False,
            'model_version': None,
            'metrics': {},
            'artifacts': {},
            'validation_results': None,
            'timestamp': start_time.isoformat(),
            'duration': None
        }

        try:
            # Data validation and cleaning
            if self.enable_data_validation:
                validation_results = self.data_validator.validate_dataset(training_data)
                retraining_results['validation_results'] = validation_results

                if not validation_results['is_valid']:
                    logger.warning("Training data validation failed, but proceeding...")
                    if self.enable_data_cleaning:
                        training_data = self.data_validator.clean_dataset(training_data, validation_results)
                        logger.info("Training data cleaned")

            # Prepare data
            if target_column not in training_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in training data")

            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]

            # Generate unique seed to prevent data leakage
            seed = int(hashlib.md5(f"{start_time.isoformat()}_{len(training_data)}".encode()).hexdigest(), 16) % (2**32)

            # Default model parameters
            if model_params is None:
                model_params = {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': seed
                }

            # Start MLflow experiment with circuit breaker protection
            def mlflow_operations():
                with self.mlflow_integration.start_run("automated_retraining",
                                                     f"retraining_{start_time.strftime('%Y%m%d_%H%M%S')}"):

                    # Log retraining metadata
                    self.mlflow_integration.log_parameters({
                        'retraining_type': 'automated',
                        'data_size': len(training_data),
                        'feature_count': len(X.columns),
                        'trigger_time': start_time.isoformat(),
                        'data_validation_performed': self.enable_data_validation,
                        'seed': seed,
                        **model_params
                    })

                    # Split data with stratification
                    from sklearn.model_selection import train_test_split

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=seed, stratify=y
                    )

                    # Train model
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**model_params)
                    model.fit(X_train, y_train)

                    # Evaluate model
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    # Calculate comprehensive metrics
                    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'auc_roc': roc_auc_score(y_test, y_pred_proba)
                    }

                    # Add confusion matrix metrics
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    metrics.update({
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_positives': int(tp),
                        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
                    })

                    # Log metrics
                    self.mlflow_integration.log_metrics(metrics)

                    # Log model
                    self.mlflow_integration.log_model(model, "model", "sklearn")

                    # Log feature importance
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    feature_importance.to_csv("feature_importance.csv", index=False)
                    self.mlflow_integration.log_artifact("feature_importance.csv")

                    # Log classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    report_df.to_csv("classification_report.csv")
                    self.mlflow_integration.log_artifact("classification_report.csv")

                    # Register new model
                    model_version = self.mlflow_integration.register_model(
                        f"{self.model_name}_retrained",
                        f"runs:/{mlflow.active_run().info.run_id}/model"
                    )

                    return model_version, metrics, {
                        'feature_importance': feature_importance.to_dict('records'),
                        'classification_report': report,
                        'model_params': model_params,
                        'data_split_seed': seed
                    }

            # Execute MLflow operations with circuit breaker
            model_version, metrics, artifacts = self.mlflow_circuit_breaker.call(mlflow_operations)

            # Update results
            retraining_results.update({
                'success': True,
                'model_version': model_version,
                'metrics': metrics,
                'artifacts': artifacts
            })

            # Update baseline if successful
            self.last_retraining = datetime.now()
            retraining_results['duration'] = (datetime.now() - start_time).total_seconds()

            logger.info(f"Enterprise retraining completed successfully in {retraining_results['duration']:.1f}s")

        except Exception as e:
            logger.error(f"Retraining execution failed: {e}")
            retraining_results['error'] = str(e)

        # Call callbacks
        for callback in self.retraining_callbacks:
            try:
                callback(retraining_results)
            except Exception as e:
                logger.error(f"Retraining callback failed: {e}")

        return retraining_results

    def perform_health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the retraining system.

        Returns:
            Health check results
        """
        health_status = {
            'overall_health': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            }
        }

        try:
            # Check configuration
            health_status['checks']['configuration'] = {
                'status': 'healthy',
                'details': {
                    'config_loaded': self.config is not None,
                    'required_sections': ['model_retraining', 'mlflow'],
                    'sections_present': list(self.config.sections())
                }
            }

            # Check MLflow integration
            try:
                experiments = self.mlflow_circuit_breaker.call(lambda: self.mlflow_integration.list_experiments())
                health_status['checks']['mlflow'] = {
                    'status': 'healthy',
                    'details': {
                        'experiments_count': len(experiments),
                        'circuit_breaker': self.mlflow_circuit_breaker.get_status()
                    }
                }
            except CircuitBreakerOpenException:
                health_status['checks']['mlflow'] = {
                    'status': 'degraded',
                    'details': {'circuit_breaker_open': True}
                }
                health_status['overall_health'] = 'degraded'
            except Exception as e:
                health_status['checks']['mlflow'] = {
                    'status': 'unhealthy',
                    'details': {'error': str(e)}
                }
                health_status['overall_health'] = 'unhealthy'

            # Check drift detector
            if self.drift_detector:
                drift_health = self.drift_detector.get_health_status()
                health_status['checks']['drift_detector'] = {
                    'status': 'healthy' if drift_health.get('circuit_breaker', {}).get('state') == 'closed' else 'degraded',
                    'details': drift_health
                }
            else:
                health_status['checks']['drift_detector'] = {
                    'status': 'not_initialized',
                    'details': {'message': 'Baseline not set'}
                }

            # Check performance monitor
            perf_history_count = len(self.performance_monitor.performance_history)
            health_status['checks']['performance_monitor'] = {
                'status': 'healthy' if perf_history_count > 0 else 'warning',
                'details': {
                    'history_count': perf_history_count,
                    'max_history': self.performance_monitor.max_history_size
                }
            }

            # Check retraining circuit breaker
            retraining_cb_status = self.retraining_circuit_breaker.get_status()
            cb_healthy = retraining_cb_status['state'] == 'closed'
            health_status['checks']['retraining_circuit_breaker'] = {
                'status': 'healthy' if cb_healthy else 'degraded',
                'details': retraining_cb_status
            }
            if not cb_healthy:
                health_status['overall_health'] = 'degraded'

            # Check data validator
            health_status['checks']['data_validator'] = {
                'status': 'healthy',
                'details': {
                    'validation_enabled': self.enable_data_validation,
                    'cleaning_enabled': self.enable_data_cleaning
                }
            }

            # Check retraining state
            days_since_retraining = None
            if self.last_retraining:
                days_since_retraining = (datetime.now() - self.last_retraining).days

            health_status['checks']['retraining_state'] = {
                'status': 'healthy',
                'details': {
                    'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
                    'days_since_retraining': days_since_retraining,
                    'triggers_enabled': self.retraining_triggers
                }
            }

            # Overall assessment
            unhealthy_checks = [k for k, v in health_status['checks'].items() if v['status'] == 'unhealthy']
            degraded_checks = [k for k, v in health_status['checks'].items() if v['status'] == 'degraded']

            if unhealthy_checks:
                health_status['overall_health'] = 'unhealthy'
                health_status['critical_issues'] = unhealthy_checks
            elif degraded_checks:
                health_status['overall_health'] = 'degraded'
                health_status['issues'] = degraded_checks

            self.last_health_check = datetime.now()

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_health'] = 'unhealthy'
            health_status['error'] = str(e)

        return health_status

    def schedule_monitoring(self, check_interval_hours: int = 24):
        """
        Schedule periodic monitoring and retraining checks with health monitoring.

        Args:
            check_interval_hours: Interval between checks in hours
        """
        def monitoring_job():
            logger.info("Running enterprise monitoring check...")

            try:
                # Perform health check
                if (self.last_health_check is None or
                    (datetime.now() - self.last_health_check).total_seconds() > self.health_check_interval):
                    health_status = self.perform_health_check()
                    if health_status['overall_health'] == 'unhealthy':
                        logger.error("System health check failed, skipping monitoring")
                        return

                # Load current production data (this would be implemented based on your data source)
                # current_data = load_current_production_data()

                # For demo purposes, we'll simulate
                current_data = pd.DataFrame({
                    'amount': np.random.exponential(100, 1000),
                    'customer_age': np.random.normal(35, 10, 1000).clip(18, 80),
                    'transaction_frequency': np.random.poisson(5, 1000),
                    'is_international': np.random.choice([0, 1], 1000),
                    'is_night_transaction': np.random.choice([0, 1], 1000),
                    'risk_score': np.random.beta(2, 5, 1000),
                })

                # Check if retraining is needed
                decision = self.check_retraining_needed(current_data)

                if decision['retraining_needed']:
                    logger.info(f"Retraining triggered by: {decision['triggers']}")

                    # Generate training data (in real scenario, this would be your full dataset)
                    training_data = current_data.copy()
                    training_data['is_fraud'] = np.random.binomial(1,
                        training_data['risk_score'] * 0.3 + training_data['is_international'] * 0.2)

                    # Trigger retraining
                    results = self.trigger_retraining(training_data)

                    if results['success']:
                        logger.info("Enterprise automated retraining completed successfully")
                    else:
                        logger.error(f"Enterprise automated retraining failed: {results.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Enterprise monitoring job failed: {e}")

        # Schedule the job
        schedule.every(check_interval_hours).hours.do(monitoring_job)

        logger.info(f"Enterprise monitoring scheduled every {check_interval_hours} hours")

    def start_monitoring_daemon(self):
        """Start the monitoring daemon with comprehensive error handling."""
        logger.info("Starting enterprise automated retraining monitoring daemon...")

        # Initial health check
        health_status = self.perform_health_check()
        if health_status['overall_health'] == 'unhealthy':
            logger.error("Initial health check failed, not starting daemon")
            return

        # Initial check
        schedule.run_all()

        # Main loop
        try:
            while True:
                schedule.run_pending()

                # Periodic health check
                if (datetime.now() - self.last_health_check).total_seconds() > self.health_check_interval:
                    health_status = self.perform_health_check()
                    if health_status['overall_health'] == 'unhealthy':
                        logger.error("Health check failed, pausing daemon for recovery")
                        time.sleep(300)  # Wait 5 minutes before retrying
                        continue

                time.sleep(3600)  # Check every hour
        except KeyboardInterrupt:
            logger.info("Enterprise monitoring daemon stopped by user")
        except Exception as e:
            logger.error(f"Enterprise monitoring daemon error: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring status.

        Returns:
            Detailed monitoring status information
        """
        health_status = self.perform_health_check()

        return {
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'drift_detector_active': self.drift_detector is not None,
            'performance_history_count': len(self.performance_monitor.performance_history),
            'retraining_triggers': self.retraining_triggers,
            'scheduled_checks': len(schedule.jobs),
            'model_name': getattr(self, 'model_name', None),
            'health_status': health_status,
            'circuit_breakers': {
                'retraining': self.retraining_circuit_breaker.get_status(),
                'mlflow': self.mlflow_circuit_breaker.get_status()
            },
            'data_validation_enabled': self.enable_data_validation,
            'data_cleaning_enabled': self.enable_data_cleaning,
            'reference_data_schema': self.reference_data_schema
        }


# Convenience functions
def create_retraining_pipeline(config_path: str = "mlops-config.ini") -> AutomatedRetrainingPipeline:
    """
    Create and configure automated retraining pipeline.

    Args:
        config_path: Path to MLOps configuration

    Returns:
        Configured retraining pipeline
    """
    return AutomatedRetrainingPipeline(config_path)


def quick_retraining_check(pipeline: AutomatedRetrainingPipeline,
                          current_data: pd.DataFrame) -> bool:
    """
    Quick check if retraining is needed.

    Args:
        pipeline: Retraining pipeline instance
        current_data: Current dataset

    Returns:
        True if retraining is needed
    """
    decision = pipeline.check_retraining_needed(current_data)
    return decision['retraining_needed']


# Export main classes
__all__ = [
    'DataDriftDetector',
    'ModelPerformanceMonitor',
    'AutomatedRetrainingPipeline',
    'create_retraining_pipeline',
    'quick_retraining_check'
]