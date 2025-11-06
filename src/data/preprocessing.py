"""
Data preprocessing utilities.
"""
import pandas as pd
import numpy as np
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning: remove duplicates, handle nulls and infinities.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting data cleaning")
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    logger.info("Data cleaning completed")
    return df


def temporal_split(df: pd.DataFrame, target_col: str = None, test_size: float = 0.2) -> tuple:
    """
    Temporal train-test split.
    Train on oldest data, test on most recent.
    
    Args:
        df: Complete dataframe (must be sorted by time)
        target_col: Target column name
        test_size: Fraction for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if target_col is None:
        target_col = config.get('model.target_column', 'is_laundering')
    
    split_idx = int(len(df) * (1 - test_size))
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test
