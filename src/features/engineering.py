"""
Feature engineering utilities.
"""
import pandas as pd
import numpy as np
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling.
    Converts datetime columns to numeric and selects numeric features.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with numeric features only
    """
    logger.info("Preparing features")
    
    df = df.copy()
    
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df[col] = df[col].astype('int64') // 10**9
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    logger.info(f"Selected {len(numeric_cols)} numeric features")
    return df_numeric


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline.
    For HI-Small dataset, features are already pre-aggregated.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe ready for modeling
    """
    logger.info("Building features")
    
    df_features = prepare_features(df)
    
    logger.info("Feature engineering completed")
    return df_features
