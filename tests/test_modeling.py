"""
Tests for modeling module
"""

from src.modeling import build_pipeline
import pandas as pd

def test_build_pipeline():
    config = {'model': 'rf', 'params': {'n_estimators': 10}}
    pipeline = build_pipeline(config)
    assert pipeline is not None
    assert hasattr(pipeline, 'fit')

def test_build_pipeline_xgb():
    config = {'model': 'xgb', 'params': {'n_estimators': 10}}
    pipeline = build_pipeline(config)
    assert pipeline is not None

def test_aml_edge_case():
    # Test with imbalanced data (AML typical)
    X = pd.DataFrame({'feature1': [1, 2, 3, 100], 'feature2': [0, 0, 0, 1]})
    y = pd.Series([0, 0, 0, 1])  # Highly imbalanced
    config = {'model': 'rf', 'params': {}}
    pipeline = build_pipeline(config)
    pipeline.fit(X, y)
    pred = pipeline.predict(X)
    assert len(pred) == len(y)