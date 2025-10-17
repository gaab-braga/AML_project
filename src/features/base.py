# src/features/base.py
"""
Base classes and interfaces for feature engineering
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils import logger
from ..config import settings
from .metrics import feature_monitor, monitor_feature_operation
from .versioning import version_manager, FeatureVersion


class FeatureEngineer(ABC):
    """Abstract base class for feature engineers"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        if self.__class__ == FeatureEngineer:
            raise TypeError("Cannot instantiate abstract class FeatureEngineer directly")
        self.name = name
        self.config = config or {}
        self.logger = logger.getChild(f"features.{name}")
        self.created_features = []

    @monitor_feature_operation("create_features")
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from input data

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with new features
        """
        pass

    @monitor_feature_operation("validate_features")
    @abstractmethod
    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate created features

        Args:
            data: DataFrame with features

        Returns:
            Validation results
        """
        pass

    @monitor_feature_operation("fit")
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit feature engineering on training data

        Args:
            data: Training data
        """
        pass

    @monitor_feature_operation("transform")
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        return self.create_features(data)

    @monitor_feature_operation("fit_transform")
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            data: Training data

        Returns:
            Transformed training data
        """
        self.fit(data)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get names of created features"""
        return self.created_features.copy()

    def create_version(self, name: Optional[str] = None) -> FeatureVersion:
        """Create a version of this feature engineer"""
        if name is None:
            name = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version = version_manager.create_version(
            name=name,
            config=self.config,
            feature_names=self.created_features,
            metadata={
                'engineer_class': self.__class__.__name__,
                'created_features_count': len(self.created_features)
            }
        )
        self.logger.info(f"Created version: {name} ({version.version_hash})")
        return version

    def load_version(self, name: str) -> bool:
        """Load a version configuration"""
        version = version_manager.get_version(name)
        if version:
            self.config = version.config.copy()
            self.created_features = version.feature_names.copy()
            self.logger.info(f"Loaded version: {name}")
            return True
        else:
            self.logger.error(f"Version not found: {name}")
            return False

    def log_feature_creation(self, feature_name: str, feature_type: str = "numeric", **metadata):
        """Log feature creation"""
        log_data = {
            "feature_name": feature_name,
            "feature_type": feature_type,
            "operation": "feature_creation"
        }
        log_data.update(metadata)

        self.logger.info(f"Created feature: {feature_name} (type: {feature_type})",
                        extra={"extra_fields": log_data})
        self.created_features.append(feature_name)


class LeakageDetector(FeatureEngineer):
    """Base class for leakage detection"""

    def __init__(self, name: str, temporal_column: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.temporal_column = temporal_column
        self.leaky_features = []

    @monitor_feature_operation("detect_leakage")
    @abstractmethod
    def detect_leakage(self, data: pd.DataFrame) -> List[str]:
        """
        Detect potentially leaky features

        Args:
            data: DataFrame to analyze

        Returns:
            List of potentially leaky feature names
        """
        pass

    def validate_temporal_ordering(self, data: pd.DataFrame) -> bool:
        """
        Validate temporal ordering of data

        Args:
            data: DataFrame to validate

        Returns:
            True if properly ordered
        """
        if self.temporal_column not in data.columns:
            self.logger.error(f"Temporal column {self.temporal_column} not found")
            return False

        is_sorted = data[self.temporal_column].is_monotonic_increasing
        if not is_sorted:
            self.logger.warning("Data is not sorted by temporal column")

        return is_sorted

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features by detecting and handling leakage"""
        # Detect leaky features
        leaky_features = self.detect_leakage(data)

        if leaky_features:
            self.logger.warning(f"Detected {len(leaky_features)} potentially leaky features",
                              leaky_features=leaky_features[:10])  # Log first 10

            self.leaky_features = leaky_features

            # Remove leaky features if configured
            if self.config.get("remove_leakyFeatures", False):
                data = data.drop(columns=leaky_features, errors='ignore')
                self.logger.info(f"Removed {len(leaky_features)} leaky features")

        return data

    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature leakage"""
        results = {
            "valid": True,
            "leaky_features": self.leaky_features,
            "total_features": len(data.columns),
            "leakage_detected": len(self.leaky_features) > 0
        }

        if self.leaky_features:
            results["valid"] = False
            results["warnings"] = [f"Detected {len(self.leaky_features)} leaky features"]

        return results


class FeatureSelector(FeatureEngineer):
    """Base class for feature selection"""

    def __init__(self, name: str, method: str = "importance", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.method = method
        self.selected_features = []
        self.feature_importance = {}

    @monitor_feature_operation("select_features")
    @abstractmethod
    def select_features(self, data: pd.DataFrame, target: pd.Series) -> List[str]:
        """
        Select important features

        Args:
            data: Feature data
            target: Target variable

        Returns:
            List of selected feature names
        """
        pass

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """Fit feature selector"""
        if target is not None:
            self.selected_features = self.select_features(data, target)
            self.logger.info(f"Selected {len(self.selected_features)} features using {self.method}")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features"""
        if not self.selected_features:
            self.logger.warning("No features selected, returning original data")
            return data

        available_features = [f for f in self.selected_features if f in data.columns]

        if len(available_features) != len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            self.logger.warning(f"Missing features in data: {missing}")

        if available_features:
            return data[available_features].copy()
        else:
            self.logger.error("No selected features available in data")
            return data.iloc[:, :0].copy()  # Return empty DataFrame with same index

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()

    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate selected features"""
        results = {
            "valid": True,
            "selected_features": self.selected_features,
            "total_selected": len(self.selected_features),
            "available_in_data": sum(1 for f in self.selected_features if f in data.columns)
        }

        missing_features = [f for f in self.selected_features if f not in data.columns]
        if missing_features:
            results["valid"] = False
            results["errors"] = [f"Missing features: {missing_features}"]

        return results


class FeaturePipeline:
    """Feature engineering pipeline"""

    def __init__(self, name: str, engineers: List[FeatureEngineer]):
        self.name = name
        self.engineers = engineers
        self.logger = logger.getChild(f"feature_pipeline.{name}")
        self.is_fitted = False

    @monitor_feature_operation("pipeline_fit")
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> None:
        """
        Fit all feature engineers

        Args:
            data: Training data
            target: Target variable (optional)
        """
        self.logger.info(f"Fitting feature pipeline: {self.name}")

        current_data = data.copy()

        for i, engineer in enumerate(self.engineers):
            try:
                self.logger.info(f"Fitting engineer {i+1}/{len(self.engineers)}: {engineer.name}")

                if hasattr(engineer, 'fit'):
                    engineer.fit(current_data, target)

                # Update data for next engineer
                if hasattr(engineer, 'transform'):
                    current_data = engineer.transform(current_data)

            except Exception as e:
                error_msg = f"Pipeline fit failed at engineer {engineer.name}: {str(e)}"
                self.logger.error(error_msg,
                                extra={"extra_fields": {
                                    "engineer": engineer.name,
                                    "step": i+1
                                }})
                raise

        self.is_fitted = True
        self.logger.info(f"Feature pipeline fitted: {self.name}")

    @monitor_feature_operation("pipeline_transform")
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through all engineers

        Args:
            data: Data to transform

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        self.logger.info(f"Transforming data through pipeline: {self.name}")

        current_data = data.copy()

        for i, engineer in enumerate(self.engineers):
            try:
                self.logger.info(f"Transforming with engineer {i+1}/{len(self.engineers)}: {engineer.name}")

                if hasattr(engineer, 'transform'):
                    current_data = engineer.transform(current_data)

            except Exception as e:
                error_msg = f"Pipeline transform failed at engineer {engineer.name}: {str(e)}"
                self.logger.error(error_msg,
                                extra={"extra_fields": {
                                    "engineer": engineer.name,
                                    "step": i+1
                                }})
                raise

        self.logger.info(f"Data transformation completed: {self.name}")
        return current_data

    @monitor_feature_operation("pipeline_fit_transform")
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(data, target)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get all created feature names"""
        all_features = []
        for engineer in self.engineers:
            if hasattr(engineer, 'get_feature_names'):
                all_features.extend(engineer.get_feature_names())
        return all_features

    def validate_pipeline(self) -> List[str]:
        """Validate pipeline configuration"""
        errors = []

        if not self.engineers:
            errors.append("Feature pipeline has no engineers")
            return errors

        # Check for required methods
        for engineer in self.engineers:
            if not hasattr(engineer, 'create_features'):
                errors.append(f"Engineer {engineer.name} missing create_features method")

        return errors

    def create_version(self, name: Optional[str] = None) -> FeatureVersion:
        """Create a version of this pipeline"""
        if name is None:
            name = f"pipeline_{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"