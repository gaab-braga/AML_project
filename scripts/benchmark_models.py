#!/usr/bin/env python3
"""
Benchmark AML Model Against Open Source Baselines
Compares with LightGBM/XGBoost from Kaggle.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

def compute_metrics(y_true, y_proba):
    """Compute ROC-AUC and PR-AUC"""
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

def benchmark_against_baselines():
    """
    Compare LGBM and XGB baselines on sample AML data.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    X = pd.DataFrame({
        'amount': np.random.exponential(1000, n_samples),
        'source_degree': np.random.randint(1, 10, n_samples),
        'target_degree': np.random.randint(1, 10, n_samples),
        'pair_frequency': np.random.randint(1, 5, n_samples)
    })
    y = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM baseline
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(X_train, y_train)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_metrics = compute_metrics(y_test, lgb_proba)

    # XGBoost baseline
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba)

    results = {
        'LightGBM': lgb_metrics,
        'XGBoost': xgb_metrics
    }

    print("Benchmark Results on Sample Data:")
    for model, metrics in results.items():
        print(f"{model}: ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")

    return results

if __name__ == '__main__':
    benchmark_against_baselines()