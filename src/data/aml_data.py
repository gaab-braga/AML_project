"""
Data loading and preprocessing utilities for AML project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def load_transaction_data(data_path: str = '../data') -> pd.DataFrame:
    """
    Load raw transaction data with proper column mapping.

    Args:
        data_path: Path to data directory

    Returns:
        DataFrame with standardized column names
    """
    data_dir = Path(data_path)

    # Find transaction file
    trans_file = data_dir / 'HI-Small_Trans.csv'
    if not trans_file.exists():
        raise FileNotFoundError(f"Transaction file not found: {trans_file}")

    # Load data
    df = pd.read_csv(trans_file)
    logger.info(f"Loaded {len(df):,} transactions from {trans_file.name}")

    # Standardize column names
    column_mapping = {
        'Timestamp': 'timestamp',
        'From Bank': 'from_bank',
        'Account': 'source',
        'To Bank': 'to_bank',
        'Account.1': 'target',
        'Amount Paid': 'amount',
        'Payment Format': 'payment_format',
        'Is Laundering': 'is_fraud'
    }

    df = df.rename(columns=column_mapping)

    # Convert data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_fraud'] = df['is_fraud'].astype(int)

    return df

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data: remove duplicates, handle missing values, validate types.

    Args:
        df: Raw transaction DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")

    initial_count = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)

    # Handle missing values
    df = df.dropna(subset=['amount', 'timestamp'])

    # Convert and validate amount
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])  # Remove invalid amounts
    df = df[df['amount'] > 0]  # Remove non-positive amounts

    logger.info(f"Cleaning complete: {duplicates_removed} duplicates removed, {len(df)} transactions remaining")

    return df

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return summary statistics.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_transactions': len(df),
        'fraud_rate': df['is_fraud'].mean(),
        'date_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }

    return quality_report