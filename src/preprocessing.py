"""
Preprocessing Module for AML Detection
Handles data cleaning, imputation, and encoding with compliance focus.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data: remove duplicates, handle missing values.

    Args:
        df: Raw DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info("Cleaning transactions: removing duplicates and invalid entries")
    df = df.drop_duplicates()
    # Check for critical fields - use timestamp if date not available
    critical_fields = ['amount']
    if 'date' in df.columns:
        critical_fields.append('date')
    elif 'timestamp' in df.columns:
        critical_fields.append('timestamp')
    df = df.dropna(subset=critical_fields)
    logger.info(f"Cleaned to {len(df)} transactions")
    return df

def impute_and_encode(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Impute missing values and encode categorical features.

    Args:
        df: DataFrame to process
        config: Configuration dict (e.g., {'categorical_cols': ['type']})

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    logger.info("Imputing and encoding data")

    # Impute numerical
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Encode categorical
    cat_cols = config.get('categorical_cols', [])
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df = df.drop(cat_cols, axis=1).join(encoded_df)

    logger.info("Imputation and encoding completed")
    return df