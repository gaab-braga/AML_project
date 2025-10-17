"""
Modeling Pipeline Module for AML Detection
Handles pipeline building, training, and model management.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Build sklearn pipeline based on config.

    Args:
        config: Dict with model type, params

    Returns:
        Pipeline: Configured pipeline
    """
    logger.info("Building ML pipeline")
    model_type = config.get('model', 'rf')
    if model_type == 'rf':
        model = RandomForestClassifier(**config.get('params', {}))
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(**config.get('params', {}))
    else:
        raise ValueError("Unsupported model")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    logger.info("Pipeline built")
    return pipeline

def train_pipeline(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, cv: int = 5, temporal: bool = False) -> Dict[str, Any]:
    """
    Train pipeline with cross-validation (temporal or stratified).

    In AML, temporal split prevents data leakage from future transactions,
    which is critical for regulatory compliance and realistic production deployment.
    """
    logger.info("Training pipeline with CV")
    if temporal:
        # Using TimeSeriesSplit because financial data has temporal dependencies;
        # stratified would mix past/future, violating AML best practices.
        from sklearn.model_selection import TimeSeriesSplit
        cv_split = TimeSeriesSplit(n_splits=cv)
    else:
        from sklearn.model_selection import StratifiedKFold
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv_split, scoring='roc_auc')
    pipeline.fit(X, y)
    results = {'cv_scores': scores.tolist(), 'mean_score': scores.mean(), 'temporal': temporal}
    logger.info(f"Training completed. Mean AUC: {scores.mean()}")
    return results

def load_model(path: str) -> Any:
    """
    Load model from disk.

    Args:
        path: Model path

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {path}")
    return joblib.load(path)