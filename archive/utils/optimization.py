"""
Optimization utilities for hyperparameter tuning and model optimization.
Provides Bayesian optimization, cross-validation strategies, and optimization helpers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
from datetime import datetime

# Optimization libraries
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ML libraries
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from sklearn.base import clone

# Local imports
from .data import save_artifact
from .modeling import FraudMetrics


class BayesianOptimizer:
    """Bayesian optimization wrapper for hyperparameter tuning."""

    def __init__(self, n_trials: int = 50, timeout: int = 3600, random_state: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

    def create_study(self, direction: str = 'maximize') -> optuna.Study:
        """Create Optuna study with optimized settings."""
        return optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )

    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                         cv_splits: List[Tuple], study: optuna.Study) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        def objective(trial):
            return self._objective_lgbm(trial, X, y, cv_splits)

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return self._extract_results(study, 'lightgbm')

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series,
                        cv_splits: List[Tuple], study: optuna.Study) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        def objective(trial):
            return self._objective_xgb(trial, X, y, cv_splits)

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return self._extract_results(study, 'xgboost')

    def _objective_lgbm(self, trial: optuna.Trial, X: pd.DataFrame,
                       y: pd.Series, cv_splits: List[Tuple]) -> float:
        """LightGBM objective function."""
        from lightgbm import LGBMClassifier

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': self.random_state,
            'verbosity': -1
        }

        return self._evaluate_model(LGBMClassifier(**params), X, y, cv_splits)

    def _objective_xgb(self, trial: optuna.Trial, X: pd.DataFrame,
                      y: pd.Series, cv_splits: List[Tuple]) -> float:
        """XGBoost objective function."""
        import xgboost as xgb

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': self.random_state
        }

        return self._evaluate_model(xgb.XGBClassifier(**params), X, y, cv_splits)

    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                       cv_splits: List[Tuple]) -> float:
        """Evaluate model using temporal cross-validation."""
        cv_scores = []

        for train_idx, test_idx in cv_splits:
            X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
            y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_test)
            score = recall_score(y_fold_test, y_pred)
            cv_scores.append(score)

        return np.mean(cv_scores)

    def _extract_results(self, study: optuna.Study, model_type: str) -> Dict[str, Any]:
        """Extract optimization results."""
        return {
            'model': self._create_optimized_model(study.best_params, model_type),
            'best_params': study.best_params,
            'best_score': study.best_value,
            'optimization_time': time.time(),  # Would need to track start time
            'study': study,
            'model_type': model_type,
            'n_trials': len(study.trials)
        }

    def _create_optimized_model(self, params: Dict, model_type: str):
        """Create optimized model instance."""
        if model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params, random_state=self.random_state, verbosity=-1)
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(**params, random_state=self.random_state)


def create_temporal_cv_splits(X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5, test_size: int = 30,
                            gap: int = 0) -> List[Tuple]:
    """Create temporal cross-validation splits."""
    n_samples = len(X)
    splits = []

    for i in range(n_splits):
        test_end = n_samples - (n_splits - i) * test_size
        test_start = max(0, test_end - test_size)
        train_end = test_start - gap

        if train_end <= 0:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def run_comprehensive_optimization(X: pd.DataFrame, y: pd.Series,
                                 cv_splits: List[Tuple],
                                 models: List[str] = ['lightgbm', 'xgboost'],
                                 n_trials: int = 30) -> Dict[str, Dict]:
    """Run comprehensive hyperparameter optimization for multiple models."""
    print(f"🎯 Starting comprehensive optimization for {len(models)} models")
    print(f"   Trials per model: {n_trials}, CV folds: {len(cv_splits)}")

    optimizer = BayesianOptimizer(n_trials=n_trials)
    results = {}

    for model_name in models:
        print(f"\nOptimizing {model_name.upper()}...")

        study = optimizer.create_study()

        if model_name.lower() == 'lightgbm':
            result = optimizer.optimize_lightgbm(X, y, cv_splits, study)
        elif model_name.lower() == 'xgboost':
            result = optimizer.optimize_xgboost(X, y, cv_splits, study)
        else:
            print(f"⚠️  Unsupported model: {model_name}")
            continue

        results[model_name.upper()] = result
        print(f"✓ {model_name.upper()} optimization complete - Best score: {result['best_score']:.4f}")

    print(f"\n✅ Optimization complete: {len(results)} models optimized")
    return results