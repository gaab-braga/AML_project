"""
Model prediction utilities.
"""
import pandas as pd
from src.models.train import load_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def predict(X, model=None, return_proba: bool = True):
    """
    Make predictions.
    
    Args:
        X: Features
        model: Trained model (loads default if None)
        return_proba: Return probabilities if True
        
    Returns:
        Predictions or probabilities
    """
    if model is None:
        model = load_model()
    
    logger.info(f"Predicting for {len(X)} samples")
    
    if return_proba and hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)


def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Batch prediction with structured output.
    
    Args:
        df: Input features dataframe
        model: Trained model
        
    Returns:
        Dataframe with predictions and probabilities
    """
    if model is None:
        model = load_model()
    
    result = df.copy()
    result['prediction'] = predict(df, model, return_proba=False)
    result['probability'] = predict(df, model, return_proba=True)
    
    return result
