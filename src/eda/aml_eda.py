"""
Exploratory Data Analysis utilities for AML project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def analyze_distributions(df: pd.DataFrame, sample_size: int = 100000) -> Dict:
    """
    Analyze distributions of key variables with sampling for performance.

    Args:
        df: Transaction DataFrame
        sample_size: Size of sample for analysis

    Returns:
        Dictionary with distribution statistics
    """
    # Sample data for performance
    if len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=42)
        logger.info(f"Using sample of {sample_size} transactions for analysis")
    else:
        sample_df = df

    distributions = {}

    # Numeric variables
    numeric_cols = ['amount', 'from_bank', 'to_bank']
    for col in numeric_cols:
        if col in sample_df.columns:
            distributions[col] = {
                'mean': sample_df[col].mean(),
                'median': sample_df[col].median(),
                'std': sample_df[col].std(),
                'skewness': sample_df[col].skew(),
                'kurtosis': sample_df[col].kurtosis()
            }

    # Categorical variables
    categorical_cols = ['payment_format']
    for col in categorical_cols:
        if col in sample_df.columns:
            distributions[col] = sample_df[col].value_counts().to_dict()

    # Target distribution
    distributions['is_fraud'] = df['is_fraud'].value_counts(normalize=True).to_dict()

    return distributions

def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric variables.

    Args:
        df: Transaction DataFrame

    Returns:
        Correlation matrix DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    return corr_matrix

def analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns in transaction data.

    Args:
        df: Transaction DataFrame

    Returns:
        Dictionary with temporal insights
    """
    # Extract temporal features
    df_temp = df.copy()
    df_temp['date'] = df_temp['timestamp'].dt.date
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['day_of_week'] = df_temp['timestamp'].dt.day_name()

    temporal_analysis = {
        'daily_volume': df_temp.groupby('date').size().to_dict(),
        'hourly_volume': df_temp.groupby('hour').size().to_dict(),
        'weekly_pattern': df_temp.groupby('day_of_week').size().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).to_dict(),
        'fraud_by_hour': df_temp.groupby('hour')['is_fraud'].mean().to_dict()
    }

    return temporal_analysis

def detect_anomalies(df: pd.DataFrame) -> Dict:
    """
    Detect potential anomalies and suspicious patterns.

    Args:
        df: Transaction DataFrame

    Returns:
        Dictionary with anomaly insights
    """
    anomalies = {}

    # Large transactions (top 0.1%)
    threshold_large = df['amount'].quantile(0.999)
    anomalies['large_transactions'] = {
        'threshold': threshold_large,
        'count': len(df[df['amount'] > threshold_large]),
        'percentage': len(df[df['amount'] > threshold_large]) / len(df) * 100
    }

    # Same bank transactions
    same_bank = df[df['from_bank'] == df['to_bank']]
    anomalies['same_bank_transactions'] = {
        'count': len(same_bank),
        'percentage': len(same_bank) / len(df) * 100
    }

    # Fraud rate by payment format
    fraud_by_format = df.groupby('payment_format')['is_fraud'].agg(['count', 'mean'])
    anomalies['fraud_by_payment_format'] = fraud_by_format.to_dict()

    # Top fraudulent accounts
    top_fraud_accounts = df[df['is_fraud'] == 1]['source'].value_counts().head(10).to_dict()
    anomalies['top_fraud_accounts'] = top_fraud_accounts

    return anomalies

def create_visualization_summary(df: pd.DataFrame) -> Dict:
    """
    Create summary of key visualizations for reporting.

    Args:
        df: Transaction DataFrame

    Returns:
        Dictionary with visualization insights
    """
    summary = {
        'total_transactions': len(df),
        'fraud_rate': df['is_fraud'].mean(),
        'avg_transaction_amount': df['amount'].mean(),
        'unique_payment_formats': df['payment_format'].nunique(),
        'date_range': {
            'start': df['timestamp'].min().strftime('%Y-%m-%d'),
            'end': df['timestamp'].max().strftime('%Y-%m-%d')
        }
    }

    return summary