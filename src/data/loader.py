"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_raw_data(filename: str = None) -> pd.DataFrame:
    """
    Load raw data.
    
    Args:
        filename: Filename in data/raw/ or data/processed/
        
    Returns:
        Raw dataframe
    """
    if filename is None:
        filename = config.get('data.raw_file', 'features_with_patterns.parquet')
    
    filepath = Path(filename)
    if not filepath.exists():
        filepath = config.data_path / 'raw' / filename
    if not filepath.exists():
        filepath = config.data_path / 'processed' / filename
    
    logger.info(f"Loading data from {filepath}")
    
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_pickle(filepath)
    
    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
    return df


def load_processed_data(split: str = 'train') -> pd.DataFrame:
    """
    Load processed data.
    
    Args:
        split: 'train' or 'test'
        
    Returns:
        Processed dataframe
    """
    filepath = config.data_path / 'processed' / f"{split}.pkl"
    logger.info(f"Loading processed data from {filepath}")
    return pd.read_pickle(filepath)


def save_processed_data(df: pd.DataFrame, split: str):
    """
    Save processed data.
    
    Args:
        df: Dataframe to save
        split: 'train' or 'test'
    """
    filepath = config.data_path / 'processed' / f"{split}.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(filepath)
    logger.info(f"Saved {len(df)} records to {filepath}")
