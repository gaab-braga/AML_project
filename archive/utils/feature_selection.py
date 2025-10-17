"""
Advanced feature selection methods for fraud detection.
Provides RFE, permutation importance, SHAP-based selection, and hybrid approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score
import shap

# Local imports
from .data import save_artifact
from .modeling import FraudMetrics


class AdvancedFeatureSelector:
    """Advanced feature selection combining multiple methods."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = {}
        self.importance_scores = {}

    def select_features_rfe(self, model, X: pd.DataFrame, y: pd.Series,
                           n_features: Optional[int] = None,
                           cv_folds: int = 5) -> List[str]:
        """Recursive Feature Elimination with cross-validation."""
        print(f"Running RFE with CV (target features: {n_features or 'auto'})...")

        if n_features is None:
            # Use RFECV to automatically select optimal number of features
            selector = RFECV(
                estimator=model,
                step=1,
                cv=cv_folds,
                scoring=make_scorer(recall_score),
                min_features_to_select=5
            )
        else:
            selector = RFE(
                estimator=model,
                n_features_to_select=n_features,
                step=1
            )

        selector.fit(X, y)

        selected = X.columns[selector.support_].tolist()
        rankings = dict(zip(X.columns, selector.ranking_))

        self.selected_features['rfe'] = selected
        self.importance_scores['rfe_rankings'] = rankings

        print(f"RFE selected {len(selected)} features")
        return selected

    def select_features_permutation(self, model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 10, cv_folds: int = 5) -> List[str]:
        """Permutation importance-based feature selection."""
        print(f"Running permutation importance (repeats: {n_repeats})...")

        # Get permutation importance scores
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=make_scorer(recall_score)
        )

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        # Select features with positive importance
        selected = importance_df[importance_df['importance_mean'] > 0]['feature'].tolist()

        self.selected_features['permutation'] = selected
        self.importance_scores['permutation'] = importance_df

        print(f"Permutation importance selected {len(selected)} features")
        return selected

    def select_features_shap(self, model, X: pd.DataFrame, y: pd.Series,
                            max_evals: int = 1000, n_features: Optional[int] = None) -> List[str]:
        """SHAP-based feature selection."""
        print(f"Running SHAP feature selection (max_evals: {max_evals})...")

        try:
            # Use TreeExplainer for tree-based models
            if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X, max_evals=max_evals)
            else:
                # Fallback to general explainer
                explainer = shap.Explainer(model)
                shap_values = explainer(X)

            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values.values).mean(axis=0)

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)

            # Select top features or those above threshold
            if n_features:
                selected = importance_df.head(n_features)['feature'].tolist()
            else:
                # Select features with importance > mean
                threshold = importance_df['shap_importance'].mean()
                selected = importance_df[importance_df['shap_importance'] > threshold]['feature'].tolist()

            self.selected_features['shap'] = selected
            self.importance_scores['shap'] = importance_df

            print(f"SHAP selected {len(selected)} features")
            return selected

        except Exception as e:
            print(f"SHAP selection failed: {e}")
            # Fallback to all features
            return X.columns.tolist()

    def select_features_hybrid(self, model, X: pd.DataFrame, y: pd.Series,
                              methods: List[str] = ['rfe', 'permutation', 'shap'],
                              min_votes: int = 2) -> List[str]:
        """Hybrid feature selection using voting across methods."""
        print(f"Running hybrid selection with methods: {methods}")

        method_results = {}

        for method in methods:
            if method == 'rfe':
                method_results[method] = self.select_features_rfe(model, X, y)
            elif method == 'permutation':
                method_results[method] = self.select_features_permutation(model, X, y)
            elif method == 'shap':
                method_results[method] = self.select_features_shap(model, X, y)

        # Count votes for each feature
        all_features = set()
        for features in method_results.values():
            all_features.update(features)

        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for method_features in method_results.values()
                       if feature in method_features)
            feature_votes[feature] = votes

        # Select features with minimum votes
        selected = [f for f, v in feature_votes.items() if v >= min_votes]

        self.selected_features['hybrid'] = selected
        self.importance_scores['hybrid_votes'] = feature_votes

        print(f"Hybrid selection ({min_votes}+ votes) selected {len(selected)} features")
        return selected

    def get_feature_stability(self, X: pd.DataFrame, y: pd.Series,
                            model_factory: Callable, n_bootstraps: int = 10) -> Dict[str, float]:
        """Assess feature selection stability using bootstrapping."""
        print(f"Assessing feature stability with {n_bootstraps} bootstraps...")

        selected_features_list = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            # Create and fit model
            model = model_factory()
            model.fit(X_boot, y_boot)

            # Select features using hybrid method
            selected = self.select_features_hybrid(model, X_boot, y_boot)
            selected_features_list.append(set(selected))

        # Calculate stability scores
        all_features = set()
        for features in selected_features_list:
            all_features.update(features)

        stability_scores = {}
        for feature in all_features:
            selection_count = sum(1 for features in selected_features_list
                                if feature in features)
            stability_scores[feature] = selection_count / n_bootstraps

        self.importance_scores['stability'] = stability_scores
        return stability_scores

    def get_consensus_features(self, stability_threshold: float = 0.7) -> List[str]:
        """Get features selected by consensus across methods."""
        if 'stability' not in self.importance_scores:
            raise ValueError("Run get_feature_stability first")

        stable_features = [f for f, s in self.importance_scores['stability'].items()
                          if s >= stability_threshold]

        print(f"Consensus features (stability ≥ {stability_threshold}): {len(stable_features)}")
        return stable_features

    def save_feature_selection_results(self, output_dir: str = 'artifacts'):
        """Save feature selection results."""
        results = {
            'selected_features': self.selected_features,
            'importance_scores': self.importance_scores
        }

        save_artifact(results, 'feature_selection_results.json', output_dir)
        print("Feature selection results saved")


def optimize_feature_subset(model, X: pd.DataFrame, y: pd.Series,
                           feature_sets: List[List[str]], cv_folds: int = 5) -> Tuple[List[str], float]:
    """Optimize feature subset using cross-validation."""
    print(f"Optimizing feature subset from {len(feature_sets)} candidates...")

    best_score = 0
    best_features = None

    for i, features in enumerate(feature_sets):
        X_subset = X[features]

        scores = cross_val_score(
            model, X_subset, y,
            cv=cv_folds,
            scoring=make_scorer(recall_score)
        )

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_features = features

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(feature_sets)} subsets - Best score: {best_score:.4f}")

    print(f"Optimal subset: {len(best_features)} features, CV score: {best_score:.4f}")
    return best_features, best_score