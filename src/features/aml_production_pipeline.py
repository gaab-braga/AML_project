"""
AML Feature Engineering Production Pipeline

This module provides a production-ready pipeline for AML feature engineering
that can be imported and used in notebooks or other scripts.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.features.custom_transformers import (
    DataCleaner,
    CategoricalEncoder,
    TemporalFeatureGenerator,
    NetworkFeatureGenerator,
    PatternFeatureGenerator,
    FeatureSelector
)


class AMLFeaturePipeline:
    """
    Production-ready pipeline for AML feature engineering.

    This class encapsulates the entire feature engineering process in a
    sklearn-compatible pipeline that prevents data leakage and ensures
    reproducibility.
    """

    def __init__(self, config_path=None):
        """
        Initialize the AML feature pipeline.

        Parameters:
        config_path (str): Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'features.yaml'

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pipeline = None
        self.is_fitted = False

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _build_pipeline(self):
        """Build the sklearn pipeline based on configuration."""
        steps = []

        # Data cleaning
        if self.config['pipeline']['steps'][0]['enabled']:
            steps.append(('data_cleaning', DataCleaner()))

        # Categorical encoding
        if self.config['pipeline']['steps'][1]['enabled']:
            steps.append(('categorical_encoding', CategoricalEncoder(
                target_col=self.config['features']['target_col']
            )))

        # Temporal features (disabled based on IV analysis)
        if self.config['pipeline']['steps'][2]['enabled']:
            steps.append(('temporal_features', TemporalFeatureGenerator(
                windows=self.config['features']['temporal']['windows']
            )))

        # Network features (disabled based on IV analysis)
        if self.config['pipeline']['steps'][3]['enabled']:
            steps.append(('network_features', NetworkFeatureGenerator()))

        # Pattern features
        if self.config['pipeline']['steps'][4]['enabled']:
            steps.append(('pattern_features', PatternFeatureGenerator()))

        # Feature selection
        if self.config['pipeline']['steps'][5]['enabled']:
            steps.append(('feature_selection', FeatureSelector(
                min_iv=self.config['features']['selection']['min_iv'],
                target_col=self.config['features']['target_col']
            )))

        # Scaling (optional)
        if self.config['pipeline']['steps'][6]['enabled']:
            steps.append(('scaler', StandardScaler()))

        self.pipeline = Pipeline(steps=steps)
        return self.pipeline

    def fit(self, X, y=None):
        """
        Fit the feature engineering pipeline.

        Parameters:
        X (pd.DataFrame): Raw input data
        y: Ignored (for sklearn compatibility)

        Returns:
        self
        """
        if self.pipeline is None:
            self._build_pipeline()

        print("Fitting AML feature engineering pipeline...")
        self.pipeline.fit(X, y)
        self.is_fitted = True
        print("Pipeline fitted successfully!")
        return self

    def transform(self, X):
        """
        Transform data using the fitted pipeline.

        Parameters:
        X (pd.DataFrame): Input data to transform

        Returns:
        pd.DataFrame: Transformed data with engineered features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        print("Transforming data through feature engineering pipeline...")
        X_transformed = self.pipeline.transform(X)
        print(f"Transformation complete! Output shape: {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform data in one step.

        Parameters:
        X (pd.DataFrame): Raw input data
        y: Ignored

        Returns:
        pd.DataFrame: Transformed data
        """
        return self.fit(X, y).transform(X)

    def save_pipeline(self, filepath=None):
        """
        Save the fitted pipeline to disk.

        Parameters:
        filepath (str): Path to save the pipeline. If None, uses config default.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        if filepath is None:
            artifacts_path = Path(self.config['data']['artifacts_path'])
            artifacts_path.mkdir(parents=True, exist_ok=True)
            filepath = artifacts_path / self.config['output']['pipeline_file']

        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to: {filepath}")

    @classmethod
    def load_pipeline(cls, filepath):
        """
        Load a saved pipeline from disk.

        Parameters:
        filepath (str): Path to the saved pipeline

        Returns:
        AMLFeaturePipeline: Loaded pipeline instance
        """
        pipeline = joblib.load(filepath)

        # Create instance and set pipeline
        instance = cls()
        instance.pipeline = pipeline
        instance.is_fitted = True

        print(f"Pipeline loaded from: {filepath}")
        return instance

    def get_feature_names(self):
        """
        Get the names of output features.

        Returns:
        list: Feature names
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted to get feature names")

        # For sklearn pipelines, we need to get feature names from the last step
        last_step = self.pipeline.steps[-1][1]

        if hasattr(last_step, 'get_feature_names_out'):
            # For transformers that support it
            return last_step.get_feature_names_out()
        elif hasattr(self.pipeline, 'get_feature_names_out'):
            return self.pipeline.get_feature_names_out()
        else:
            # Fallback: return column names if DataFrame
            return None

    def get_pipeline_info(self):
        """
        Get information about the pipeline configuration.

        Returns:
        dict: Pipeline information
        """
        enabled_steps = [step['name'] for step in self.config['pipeline']['steps'] if step['enabled']]

        return {
            'enabled_steps': enabled_steps,
            'config_file': str(self.config_path),
            'is_fitted': self.is_fitted,
            'temporal_enabled': self.config['features']['temporal']['enabled'],
            'network_enabled': self.config['features']['network']['enabled'],
            'patterns_enabled': self.config['features']['patterns']['enabled'],
            'selection_enabled': self.config['features']['selection']['enabled']
        }


def create_production_pipeline(config_path=None):
    """
    Factory function to create a production-ready AML feature pipeline.

    Parameters:
    config_path (str): Path to configuration file

    Returns:
    AMLFeaturePipeline: Configured pipeline instance
    """
    return AMLFeaturePipeline(config_path)


# Example usage function
def example_usage():
    """
    Example of how to use the AML feature pipeline.
    """
    # Create pipeline
    pipeline = create_production_pipeline()

    # Load sample data (replace with your data loading logic)
    # df_raw = load_raw_transactions(data_path='../data/raw')

    # Fit and transform
    # df_processed = pipeline.fit_transform(df_raw)

    # Save pipeline
    # pipeline.save_pipeline()

    print("Example usage completed. Uncomment and adapt the code above for actual use.")


if __name__ == "__main__":
    example_usage()