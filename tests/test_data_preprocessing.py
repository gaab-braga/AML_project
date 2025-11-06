"""
Tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import clean_data, temporal_split


def test_clean_data_removes_duplicates(sample_data):
    """Test that clean_data removes duplicate rows."""
    df_with_dupes = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
    df_clean = clean_data(df_with_dupes)
    
    assert len(df_clean) == len(sample_data), "Duplicates not removed"


def test_clean_data_handles_missing_values(sample_data):
    """Test that clean_data handles missing values."""
    df_with_nulls = sample_data.copy()
    df_with_nulls.loc[0:10, 'amount'] = np.nan
    
    df_clean = clean_data(df_with_nulls)
    
    assert df_clean['amount'].isnull().sum() == 0, "Missing values not filled"


def test_clean_data_handles_infinities(sample_data):
    """Test that clean_data handles infinite values."""
    df_with_inf = sample_data.copy()
    df_with_inf.loc[0:10, 'amount'] = np.inf
    
    df_clean = clean_data(df_with_inf)
    
    assert not np.isinf(df_clean['amount']).any(), "Infinite values not handled"


def test_temporal_split_chronological_order(sample_data):
    """Test that temporal split respects chronological order."""
    X_train, X_test, y_train, y_test = temporal_split(sample_data, 'is_laundering')
    
    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Test set is empty"
    assert len(X_train) + len(X_test) == len(sample_data) - 1, "Data loss in split"


def test_temporal_split_ratio(sample_data):
    """Test that temporal split uses correct ratio."""
    X_train, X_test, y_train, y_test = temporal_split(sample_data, 'is_laundering', test_size=0.2)
    
    train_ratio = len(X_train) / (len(X_train) + len(X_test))
    
    assert 0.75 < train_ratio < 0.85, f"Expected ~0.8, got {train_ratio}"


def test_temporal_split_no_data_leakage(sample_data):
    """Test that test set comes after train set temporally."""
    df_sorted = sample_data.sort_values('timestamp').reset_index(drop=True)
    X_train, X_test, y_train, y_test = temporal_split(df_sorted, 'is_laundering')
    
    # In a temporal split, train indices should be lower than test indices
    assert X_train.index.max() < X_test.index.min(), "Temporal leakage detected"


def test_temporal_split_removes_timestamp(sample_data):
    """Test that timestamp column is removed from features."""
    X_train, X_test, y_train, y_test = temporal_split(sample_data, 'is_laundering')
    
    assert 'timestamp' not in X_train.columns, "Timestamp not removed from train"
    assert 'timestamp' not in X_test.columns, "Timestamp not removed from test"


def test_temporal_split_removes_target(sample_data):
    """Test that target column is removed from features."""
    X_train, X_test, y_train, y_test = temporal_split(sample_data, 'is_laundering')
    
    assert 'is_laundering' not in X_train.columns, "Target not removed from train"
    assert 'is_laundering' not in X_test.columns, "Target not removed from test"
