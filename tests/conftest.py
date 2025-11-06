"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """Generate sample data matching HI-Small structure."""
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='h'),
        'amount': np.random.exponential(100, n_samples),
        'payment_format': np.random.randint(0, 5, n_samples),
        'is_laundering': np.random.binomial(1, 0.1, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'transaction_count': np.random.poisson(10, n_samples),
    })
    
    return df


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.binomial(1, 0.1, n_samples)
    
    return X, y


@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def config_dict():
    """Provide test configuration."""
    return {
        'model': {
            'target_column': 'is_laundering',
            'random_state': 42
        },
        'paths': {
            'data': 'data/processed',
            'model_output': 'models'
        },
        'data': {
            'raw_file': 'features_with_patterns.parquet',
            'test_size': 0.2
        }
    }
