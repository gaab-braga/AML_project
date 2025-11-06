"""
Tests for model training module.
"""
import pytest
from sklearn.datasets import make_classification

from src.models.train import get_model, train_model, save_model, load_model


def test_get_model_xgboost():
    """Test XGBoost model instantiation."""
    model = get_model('xgboost')
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_get_model_lightgbm():
    """Test LightGBM model instantiation."""
    model = get_model('lightgbm')
    
    assert model is not None
    assert hasattr(model, 'fit')


def test_get_model_random_forest():
    """Test Random Forest model instantiation."""
    model = get_model('random_forest')
    
    assert model is not None
    assert hasattr(model, 'fit')


def test_get_model_invalid_name():
    """Test invalid model name raises error."""
    with pytest.raises(ValueError):
        get_model('invalid_model')


def test_train_model(sample_features):
    """Test model training."""
    X, y = sample_features
    
    model = train_model(X, y, model_name='random_forest')
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_save_and_load_model(sample_features, temp_model_path):
    """Test model serialization."""
    X, y = sample_features
    
    model = train_model(X, y, model_name='random_forest')
    
    model_path = temp_model_path / "test_model.pkl"
    save_model(model, str(model_path))
    
    loaded_model = load_model(str(model_path))
    
    assert loaded_model is not None
    
    # Verify predictions match
    pred_original = model.predict(X[:10])
    pred_loaded = loaded_model.predict(X[:10])
    
    assert (pred_original == pred_loaded).all()
