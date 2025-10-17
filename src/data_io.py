"""
Data I/O Module for AML Detection
Handles loading, saving, and basic data operations with compliance logging.
"""

import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

# Configure logging for compliance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_transactions(path: str) -> pd.DataFrame:
    """
    Load raw transaction data from CSV with anonymization for compliance.

    Args:
        path: Path to CSV file or directory containing AML dataset

    Returns:
        pd.DataFrame: Loaded and anonymized DataFrame
    """
    path_obj = Path(path)
    if path_obj.is_dir():
        # Assume AML dataset structure - use small for demo
        trans_file = path_obj / "HI-Small_Trans.csv"
        if trans_file.exists():
            path = str(trans_file)
        else:
            raise FileNotFoundError(f"Transaction file not found in {path}")

    logger.info(f"Loading raw transactions from {path} for AML analysis")
    df = pd.read_csv(path, nrows=10000)  # Load only first 10k rows for demo

    # Rename columns to standard format
    column_mapping = {
        'Timestamp': 'timestamp',
        'From Bank': 'from_bank',
        'Account': 'source',  # First Account is From
        'To Bank': 'to_bank',
        'Account.1': 'target',  # Second Account is To
        'Amount Paid': 'amount',
        'Payment Currency': 'currency',
        'Payment Format': 'payment_format',
        'Is Laundering': 'is_fraud'
    }
    df = df.rename(columns=column_mapping)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Anonymize sensitive fields (e.g., hash account IDs)
    if 'source' in df.columns:
        df['source'] = df['source'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
        df['target'] = df['target'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
        logger.info("Anonymized account IDs for compliance")

    logger.info(f"Loaded {len(df)} transactions")
    return df

def save_model(model: Any, path: str) -> None:
    """
    Save trained model to disk with metadata logging.

    Args:
        model: Trained model object
        path: Save path
    """
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)
    logger.info("Model saved successfully")

def validate_data_compliance(df: pd.DataFrame) -> bool:
    """
    Basic compliance check: ensure no PII in data.

    Args:
        df: DataFrame to check

    Returns:
        bool: True if compliant
    """
    sensitive_cols = ['ssn', 'name', 'address']  # Example
    for col in sensitive_cols:
        if col in df.columns:
            logger.warning(f"Potential PII detected in column: {col}")
            return False
    logger.info("Data passed basic compliance check")
    return True