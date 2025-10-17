"""
Ensemble methods for fraud detection models.
Provides stacking, voting, and weighted ensemble implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

# Local imports
from .data import save_artifact
from .modeling import FraudMetrics


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Custom weighted ensemble classifier."""

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models.keys()}
        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to 1 and all models are present."""
        if set(self.weights.keys()) != set(self.models.keys()):
            raise ValueError("Weights keys must match model names")

        if not np.isclose(sum(self.weights.values()), 1.0):
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all base models."""
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted average."""
        probas = []

        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            probas.append(proba * self.weights[name])

        return np.column_stack([1 - np.sum(probas, axis=0), np.sum(probas, axis=0)])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes using weighted voting."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)


def create_stacking_ensemble(base_models: Dict[str, Any],
                           meta_model=None,
                           cv_folds: int = 5) -> StackingClassifier:
    """Create stacking ensemble with cross-validated predictions."""
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)

    estimators = list(base_models.items())

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv_folds,
        stack_method='predict_proba',
        passthrough=False
    )


def create_voting_ensemble(base_models: Dict[str, Any],
                          voting: str = 'soft') -> VotingClassifier:
    """Create voting ensemble (hard or soft voting)."""
    estimators = list(base_models.items())

    return VotingClassifier(
        estimators=estimators,
        voting=voting
    )


def optimize_ensemble_weights(models: Dict[str, Any], X: pd.DataFrame,
                            y: pd.Series, cv_splits: List[Tuple]) -> Dict[str, float]:
    """Optimize ensemble weights using cross-validation."""
    from scipy.optimize import minimize

    def objective(weights):
        """Objective function for weight optimization."""
        ensemble = WeightedEnsemble(models, dict(zip(models.keys(), weights)))
        cv_scores = []

        for train_idx, test_idx in cv_splits:
            X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
            y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

            ensemble.fit(X_fold_train, y_fold_train)
            y_pred = ensemble.predict(X_fold_test)
            score = recall_score(y_fold_test, y_pred)
            cv_scores.append(score)

        return -np.mean(cv_scores)  # Negative because we minimize

    # Initial weights (equal)
    n_models = len(models)
    initial_weights = np.ones(n_models) / n_models

    # Constraints: weights sum to 1, each weight >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
    ]
    bounds = [(0, 1) for _ in range(n_models)]  # Each weight between 0 and 1

    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return dict(zip(models.keys(), result.x))


def evaluate_ensemble_methods(base_models: Dict[str, Any], X: pd.DataFrame,
                            y: pd.Series, cv_splits: List[Tuple]) -> Dict[str, Dict]:
    """Evaluate different ensemble methods."""
    print("Evaluating ensemble methods...")

    results = {}

    # Weighted ensemble with equal weights
    print("Testing weighted ensemble (equal weights)...")
    weighted_equal = WeightedEnsemble(base_models)
    results['weighted_equal'] = evaluate_model_cv(weighted_equal, X, y, cv_splits)

    # Weighted ensemble with optimized weights
    print("Optimizing ensemble weights...")
    try:
        optimal_weights = optimize_ensemble_weights(base_models, X, y, cv_splits)
        weighted_optimized = WeightedEnsemble(base_models, optimal_weights)
        results['weighted_optimized'] = evaluate_model_cv(weighted_optimized, X, y, cv_splits)
        results['optimal_weights'] = optimal_weights
    except Exception as e:
        print(f"Weight optimization failed: {e}")
        results['weighted_optimized'] = None

    # Voting ensemble (soft)
    print("Testing soft voting ensemble...")
    voting_soft = create_voting_ensemble(base_models, voting='soft')
    results['voting_soft'] = evaluate_model_cv(voting_soft, X, y, cv_splits)

    # Voting ensemble (hard)
    print("Testing hard voting ensemble...")
    voting_hard = create_voting_ensemble(base_models, voting='hard')
    results['voting_hard'] = evaluate_model_cv(voting_hard, X, y, cv_splits)

    # Stacking ensemble
    print("Testing stacking ensemble...")
    stacking = create_stacking_ensemble(base_models)
    results['stacking'] = evaluate_model_cv(stacking, X, y, cv_splits)

    return results


def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series,
                     cv_splits: List[Tuple]) -> Dict[str, float]:
    """Evaluate model using cross-validation."""
    cv_scores = {'recall': [], 'auc': [], 'ap': []}

    for train_idx, test_idx in cv_splits:
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred_proba = model.predict_proba(X_fold_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        cv_scores['recall'].append(recall_score(y_fold_test, y_pred))
        cv_scores['auc'].append(roc_auc_score(y_fold_test, y_pred_proba))
        cv_scores['ap'].append(average_precision_score(y_fold_test, y_pred_proba))

    return {metric: np.mean(scores) for metric, scores in cv_scores.items()}


def select_best_ensemble(ensemble_results: Dict[str, Dict],
                        metric: str = 'recall') -> Tuple[str, Dict]:
    """Select the best performing ensemble method."""
    valid_results = {k: v for k, v in ensemble_results.items()
                    if v is not None and isinstance(v, dict)}

    if not valid_results:
        raise ValueError("No valid ensemble results found")

    best_method = max(valid_results.keys(),
                     key=lambda x: valid_results[x][metric])

    return best_method, valid_results[best_method]


def create_final_ensemble(base_models: Dict[str, Any],
                         ensemble_results: Dict[str, Dict],
                         method: str = 'auto') -> Any:
    """Create the final ensemble model based on evaluation results."""
    if method == 'auto':
        method, _ = select_best_ensemble(ensemble_results)

    print(f"Creating final ensemble using method: {method}")

    if method == 'weighted_equal':
        return WeightedEnsemble(base_models)
    elif method == 'weighted_optimized':
        weights = ensemble_results.get('optimal_weights', {})
        return WeightedEnsemble(base_models, weights)
    elif method == 'voting_soft':
        return create_voting_ensemble(base_models, voting='soft')
    elif method == 'voting_hard':
        return create_voting_ensemble(base_models, voting='hard')
    elif method == 'stacking':
        return create_stacking_ensemble(base_models)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")