"""
Integration tests for end-to-end pipeline.
"""
import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data, temporal_split
from src.features.engineering import build_features
from src.models.train import train_model, save_model, load_model
from src.models.predict import predict
from src.models.evaluate import evaluate_model


def test_full_pipeline(sample_data, tmp_path):
    """Test complete pipeline from data to predictions."""
    df_clean = clean_data(sample_data)
    
    df_features = build_features(df_clean)
    
    X_train, X_test, y_train, y_test = temporal_split(df_features, 'is_laundering')
    
    model = train_model(X_train, y_train, model_name='random_forest')
    
    model_path = tmp_path / "test_model.pkl"
    save_model(model, str(model_path))
    
    loaded_model = load_model(str(model_path))
    
    y_pred = loaded_model.predict(X_test)
    y_proba = predict(X_test, loaded_model, return_proba=True)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    assert metrics["accuracy"] >= 0
    assert metrics["roc_auc"] >= 0
    assert metrics["pr_auc"] >= 0


def test_pipeline_consistency(sample_data):
    """Test that pipeline produces consistent results."""
    df_clean = clean_data(sample_data)
    df_features = build_features(df_clean)
    
    X_train, X_test, y_train, y_test = temporal_split(df_features, 'is_laundering')
    
    model = train_model(X_train, y_train, model_name='random_forest')
    
    y_pred1 = model.predict(X_test)
    y_pred2 = model.predict(X_test)
    
    assert (y_pred1 == y_pred2).all(), "Predictions should be deterministic"


def test_pipeline_with_new_data(sample_data, tmp_path):
    """Test pipeline with unseen data."""
    train_data = sample_data.iloc[:800]
    new_data = sample_data.iloc[800:]
    
    df_clean = clean_data(train_data)
    df_features = build_features(df_clean)
    
    X_train, X_test, y_train, y_test = temporal_split(df_features, 'is_laundering')
    
    model = train_model(X_train, y_train, model_name='random_forest')
    
    new_clean = clean_data(new_data)
    new_features = build_features(new_clean)
    
    new_features = new_features.drop(columns=['is_laundering'], errors='ignore')
    new_features = new_features.select_dtypes(include=['int64', 'float64'])
    
    # Align columns
    for col in X_train.columns:
        if col not in new_features.columns:
            new_features[col] = 0
    new_features = new_features[X_train.columns]
    
    y_pred_new = predict(new_features, model, return_proba=False)
    
    assert len(y_pred_new) == len(new_data)
    assert y_pred_new.min() >= 0
    assert y_pred_new.max() <= 1
