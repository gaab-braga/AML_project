"""
Model training utilities.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'xgboost': XGBClassifier,
    'lightgbm': LGBMClassifier
}


def get_model(model_name: str = None):
    """
    Instantiate model with parameters from config.
    
    Args:
        model_name: Model name from registry
        
    Returns:
        Model instance
    """
    if model_name is None:
        model_name = config.get('model.name', 'xgboost')
    
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    params = config.get('model.params', {})
    logger.info(f"Instantiating {model_name} with params: {params}")
    
    return model_class(**params)


def train_model(X_train, y_train, model_name: str = None):
    """
    Train model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_name: Model name
        
    Returns:
        Trained model
    """
    logger.info("Starting model training")
    
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model


def save_model(model, filename: str = None):
    """
    Save trained model.
    
    Args:
        model: Trained model
        filename: Output filename
    """
    if filename is None:
        filename = config.get('paths.model_output', 'models/model.pkl')
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filename: str = None):
    """
    Load trained model.
    
    Args:
        filename: Model filename
        
    Returns:
        Loaded model
    """
    if filename is None:
        filename = config.get('paths.model_output', 'models/model.pkl')
    
    logger.info(f"Loading model from {filename}")
    return joblib.load(filename)
