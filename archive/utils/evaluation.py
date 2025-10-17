"""
Comprehensive model evaluation utilities for fraud detection.
Provides SHAP analysis, business impact assessment, robustness testing, and monitoring setup.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
import shap
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score,
                           average_precision_score, confusion_matrix, classification_report)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.base import clone
from scipy import stats
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from .data import save_artifact

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_results = {}

    def evaluate_model_comprehensive(self, model, X: pd.DataFrame, y: pd.Series,
                                   model_name: str = "Model") -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        print(f"Evaluating {model_name}...")

        # Basic predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

        # Core metrics
        metrics = self._calculate_core_metrics(y, y_pred, y_proba)

        # Confusion matrix analysis
        cm_analysis = self._analyze_confusion_matrix(y, y_pred)

        # Business impact (assuming cost structure)
        business_impact = self._calculate_business_impact(y, y_pred, y_proba)

        results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix_analysis': cm_analysis,
            'business_impact': business_impact,
            'predictions': {
                'y_true': y,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        }

        self.evaluation_results[model_name] = results
        return results

    def _calculate_core_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate core classification metrics."""
        metrics = {
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': np.mean(y_true == y_pred),
            'fraud_rate': np.mean(y_pred),
            'baseline_recall': np.mean(y_true)  # Random prediction baseline
        }

        if y_proba is not None:
            metrics.update({
                'auc': roc_auc_score(y_true, y_proba),
                'average_precision': average_precision_score(y_true, y_proba)
            })

        return metrics

    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detailed confusion matrix analysis."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value
            'total_cases': len(y_true),
            'fraud_cases': np.sum(y_true),
            'predicted_fraud_cases': np.sum(y_pred)
        }

    def _calculate_business_impact(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 fp_cost: float = 1000, fn_cost: float = 50000,
                                 investigation_cost: float = 500) -> Dict[str, Any]:
        """Calculate business impact based on misclassification costs."""
        cm_analysis = self._analyze_confusion_matrix(y_true, y_pred)

        # Direct costs from misclassifications
        direct_costs = (cm_analysis['false_positives'] * fp_cost +
                       cm_analysis['false_negatives'] * fn_cost)

        # Investigation costs
        investigation_costs = cm_analysis['predicted_fraud_cases'] * investigation_cost

        # Fraud prevented (true positives)
        fraud_prevented = cm_analysis['true_positives']
        fraud_prevented_value = fraud_prevented * fn_cost  # Value of prevented fraud

        # Net benefit
        net_benefit = fraud_prevented_value - direct_costs - investigation_costs

        return {
            'direct_costs': direct_costs,
            'investigation_costs': investigation_costs,
            'fraud_prevented_value': fraud_prevented_value,
            'net_benefit': net_benefit,
            'benefit_cost_ratio': net_benefit / (direct_costs + investigation_costs) if (direct_costs + investigation_costs) > 0 else 0,
            'fp_cost_per_case': fp_cost,
            'fn_cost_per_case': fn_cost,
            'investigation_cost_per_case': investigation_cost
        }

    def recall_at_k(self, y_true: pd.Series, y_proba: np.ndarray,
                   k_values: List[int] = [50, 100, 200, 500]) -> Dict[str, float]:
        """Calculate Recall@K for fraud detection."""
        results = {}
        n_positives = y_true.sum()

        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]

        for k in k_values:
            if k > len(y_true):
                k = len(y_true)
            top_k_indices = sorted_indices[:k]
            n_detected = y_true.iloc[top_k_indices].sum()
            recall_k = n_detected / n_positives if n_positives > 0 else 0
            results[f'recall@{k}'] = recall_k

        return results

    def precision_at_k(self, y_true: pd.Series, y_proba: np.ndarray,
                      k_values: List[int] = [50, 100, 200, 500]) -> Dict[str, float]:
        """Calculate Precision@K for fraud detection."""
        results = {}

        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]

        for k in k_values:
            if k > len(y_true):
                k = len(y_true)
            top_k_indices = sorted_indices[:k]
            n_detected = y_true.iloc[top_k_indices].sum()
            precision_k = n_detected / k if k > 0 else 0
            results[f'precision@{k}'] = precision_k

        return results


class SHAPAnalyzer:
    """SHAP-based model interpretability analysis."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.shap_values = None
        self.explainer = None

    def explain_model(self, model, X: pd.DataFrame, max_evals: int = 1000,
                     background_size: int = 100) -> Dict[str, Any]:
        """Generate comprehensive SHAP explanations."""
        print("Generating SHAP explanations...")

        try:
            # Create explainer based on model type
            if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
                # Tree-based model
                self.explainer = shap.TreeExplainer(model)
                background = X.sample(min(background_size, len(X)), random_state=self.random_state)
                self.shap_values = self.explainer(X, max_evals=max_evals)
            else:
                # General explainer
                background = X.sample(min(background_size, len(X)), random_state=self.random_state)
                self.explainer = shap.Explainer(model, background)
                self.shap_values = self.explainer(X)

            # Calculate feature importance
            if hasattr(self.shap_values, 'values'):
                shap_importance = np.abs(self.shap_values.values).mean(axis=0)
            else:
                shap_importance = np.abs(self.shap_values).mean(axis=0)

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)

            # Identify most important features for fraud detection
            fraud_cases = X[self.shap_values.values[:, 1] > np.median(self.shap_values.values[:, 1])]
            top_fraud_features = self._get_top_features_for_subset(fraud_cases, n_top=10)

            results = {
                'feature_importance': feature_importance,
                'shap_values': self.shap_values,
                'explainer': self.explainer,
                'top_fraud_features': top_fraud_features,
                'background_size': background_size,
                'max_evals': max_evals
            }

            return results

        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return {'error': str(e)}

    def _get_top_features_for_subset(self, X_subset: pd.DataFrame, n_top: int = 10) -> List[str]:
        """Get top features for a specific subset of data."""
        if self.shap_values is None:
            return []

        # Get SHAP values for subset
        subset_indices = X_subset.index
        subset_shap = self.shap_values.values[subset_indices]

        # Average absolute SHAP values for subset
        mean_abs_shap = np.abs(subset_shap).mean(axis=0)

        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-n_top:][::-1]
        top_features = X_subset.columns[top_indices].tolist()

        return top_features

    def plot_shap_analysis(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Create SHAP analysis plots."""
        if self.shap_values is None:
            print("No SHAP values available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Summary plot
        plt.sca(axes[0, 0])
        shap.summary_plot(self.shap_values, X, show=False, max_display=15)
        axes[0, 0].set_title('SHAP Summary Plot')

        # Bar plot of mean absolute SHAP values
        plt.sca(axes[0, 1])
        shap.summary_plot(self.shap_values, X, plot_type='bar', show=False, max_display=15)
        axes[0, 1].set_title('SHAP Feature Importance')

        # Waterfall plot for first prediction
        plt.sca(axes[1, 0])
        if hasattr(self.shap_values, 'values') and len(self.shap_values.values) > 0:
            shap.waterfall_plot(self.shap_values[0], show=False)
            axes[1, 0].set_title('SHAP Waterfall Plot (First Sample)')

        # Beeswarm plot
        plt.sca(axes[1, 1])
        shap.plots.beeswarm(self.shap_values, show=False, max_display=15)
        axes[1, 1].set_title('SHAP Beeswarm Plot')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP analysis plots saved to {save_path}")

        plt.show()

    def analyze_explanation_consistency(self, shap_values: pd.DataFrame,
                                      permutation_importance: pd.DataFrame,
                                      top_n: int = 10) -> Dict[str, Any]:
        """Analyze consistency between global (SHAP) and local (permutation) explanations."""
        logger.info("Analyzing explanation consistency...")

        # Get top features from both methods
        shap_top = set(shap_values.head(top_n)['feature'].tolist())
        perm_top = set(permutation_importance.head(top_n)['feature'].tolist())

        # Calculate overlap
        overlap = shap_top.intersection(perm_top)
        overlap_ratio = len(overlap) / top_n

        logger.info(f"Top {top_n} features overlap: {len(overlap)}/{top_n} ({overlap_ratio:.1%})")
        logger.info(f"Common features: {sorted(list(overlap))}")

        # Features unique to each method
        shap_only = shap_top - perm_top
        perm_only = perm_top - shap_top

        if shap_only:
            logger.info(f"SHAP-only features: {sorted(list(shap_only))}")
        if perm_only:
            logger.info(f"Permutation-only features: {sorted(list(perm_only))}")

        return {
            'overlap_ratio': overlap_ratio,
            'common_features': list(overlap),
            'shap_only': list(shap_only),
            'perm_only': list(perm_only)
        }


class RobustnessAnalyzer:
    """Model robustness and stability analysis."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def analyze_learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Analyze learning curves to detect overfitting/underfitting."""
        logger.info("Analyzing learning curves...")

        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_state
        )

        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Detect overfitting/underfitting
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        score_gap = final_train_score - final_val_score

        if score_gap > 0.1:
            overfitting_status = "Overfitting detected"
            overfitting_severity = "High" if score_gap > 0.2 else "Moderate"
        elif score_gap < 0.05:
            overfitting_status = "Underfitting detected"
            overfitting_severity = "High" if score_gap < 0.02 else "Moderate"
        else:
            overfitting_status = "Good fit"
            overfitting_severity = "N/A"

        learning_curve_analysis = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'final_train_score': float(final_train_score),
            'final_val_score': float(final_val_score),
            'score_gap': float(score_gap),
            'overfitting_status': overfitting_status,
            'overfitting_severity': overfitting_severity
        }

        logger.info(f"Final training score: {final_train_score:.4f}")
        logger.info(f"Final validation score: {final_val_score:.4f}")
        logger.info(f"Score gap: {score_gap:.4f}")
        logger.info(f"Status: {overfitting_status} ({overfitting_severity})")

        return learning_curve_analysis

    def analyze_feature_stability(self, model_factory: Callable, X: pd.DataFrame,
                                y: pd.Series, n_bootstraps: int = 10) -> Dict[str, Any]:
        """Analyze feature importance stability using bootstrapping."""
        print(f"Analyzing feature stability with {n_bootstraps} bootstraps...")

        feature_importance_list = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            # Train model
            model = model_factory()
            model.fit(X_boot, y_boot)

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                # Fallback: use permutation importance
                from sklearn.inspection import permutation_importance
                perm_imp = permutation_importance(model, X_boot, y_boot, n_repeats=5, random_state=self.random_state)
                importance = perm_imp.importances_mean

            feature_importance_list.append(importance)

        # Calculate stability metrics
        importance_array = np.array(feature_importance_list)
        importance_mean = np.mean(importance_array, axis=0)
        importance_std = np.std(importance_array, axis=0)
        importance_cv = importance_std / (importance_mean + 1e-10)  # Coefficient of variation

        stability_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': importance_mean,
            'importance_std': importance_std,
            'coefficient_of_variation': importance_cv,
            'stability_score': 1 / (1 + importance_cv)  # Higher is more stable
        }).sort_values('importance_mean', ascending=False)

        return {
            'stability_analysis': stability_df,
            'n_bootstraps': n_bootstraps,
            'mean_stability_score': stability_df['stability_score'].mean()
        }


def create_monitoring_dashboard_config(model, X: pd.DataFrame, y: pd.Series,
                                     output_dir: str = 'artifacts') -> Dict[str, Any]:
    """Create configuration for model monitoring dashboard."""
    print("Creating monitoring dashboard configuration...")

    # Basic model info
    model_info = {
        'model_type': type(model).__name__,
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist(),
        'target_distribution': {
            'fraud_rate': y.mean(),
            'total_samples': len(y),
            'fraud_cases': y.sum()
        }
    }

    # Performance thresholds for alerts
    performance_thresholds = {
        'recall_threshold': 0.85,  # Alert if recall drops below this
        'auc_threshold': 0.80,
        'max_false_positive_rate': 0.10
    }

    # Feature drift thresholds
    feature_drift_config = {
        'drift_method': 'kolmogorov_smirnov',
        'threshold': 0.05,  # p-value threshold
        'features_to_monitor': X.columns.tolist()[:20]  # Top 20 features
    }

    # Prediction drift monitoring
    prediction_drift_config = {
        'monitor_score_distributions': True,
        'alert_on_score_drift': True,
        'drift_threshold': 0.05
    }

    monitoring_config = {
        'model_info': model_info,
        'performance_thresholds': performance_thresholds,
        'feature_drift_config': feature_drift_config,
        'prediction_drift_config': prediction_drift_config,
        'monitoring_frequency': 'daily',
        'alert_channels': ['email', 'slack'],
        'created_at': pd.Timestamp.now().isoformat()
    }

    # Save configuration
    save_artifact(monitoring_config, 'monitoring_config.json', output_dir)

    return monitoring_config


class StatisticalValidator:
    """Statistical validation utilities with confidence intervals."""

    def __init__(self, random_state: int = 42):
        """Initialize validator with random state."""
        self.random_state = random_state

    def bootstrap_confidence_interval(self, y_true: pd.Series, y_proba: np.ndarray,
                                    metric_func: callable, n_bootstraps: int = 1000,
                                    alpha: float = 0.05) -> Dict[str, float]:
        """Calculate bootstrap confidence interval for a metric."""
        bootstrapped_scores = []

        n_samples = len(y_true)
        rng = np.random.RandomState(self.random_state)

        for _ in range(n_bootstraps):
            # Bootstrap sample with replacement
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true.iloc[indices]
            y_proba_boot = y_proba[indices]

            try:
                score = metric_func(y_true_boot, y_proba_boot)
                bootstrapped_scores.append(score)
            except:
                continue

        # Calculate confidence interval
        sorted_scores = np.sort(bootstrapped_scores)
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(sorted_scores, lower_percentile)
        ci_upper = np.percentile(sorted_scores, upper_percentile)
        mean_score = np.mean(sorted_scores)

        return {
            'mean': mean_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'n_bootstraps': len(bootstrapped_scores)
        }

    def mcnemar_test(self, y_true: pd.Series, y_pred1: np.ndarray, y_pred2: np.ndarray,
                    alpha: float = 0.05) -> Dict[str, Any]:
        """McNemar's test for comparing two classification models."""
        # Create contingency table
        both_correct = ((y_pred1 == y_true) & (y_pred2 == y_true)).sum()
        model1_correct_only = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
        model2_correct_only = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()
        both_wrong = ((y_pred1 != y_true) & (y_pred2 != y_true)).sum()

        # McNemar's test statistic
        n12 = model1_correct_only
        n21 = model2_correct_only

        if n12 + n21 == 0:
            return {'p_value': 1.0, 'significant': False, 'statistic': 0}

        # Use continuity correction
        statistic = abs(n12 - n21) - 1
        statistic = max(0, statistic) ** 2 / (n12 + n21)

        # Chi-square distribution with 1 df
        p_value = 1 - stats.chi2.cdf(statistic, 1)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'n12': n12,  # Model1 correct, Model2 wrong
            'n21': n21   # Model1 wrong, Model2 correct
        }

    def calculate_statistical_validation(self, y_test: pd.Series, y_test_proba: np.ndarray,
                                       cv_results: Optional[Dict] = None,
                                       n_bootstraps: int = 1000) -> Dict[str, Any]:
        """Comprehensive statistical validation."""
        logger.info("Performing statistical validation...")

        # Bootstrap confidence intervals for key metrics
        logger.info("Calculating bootstrap confidence intervals...")

        roc_auc_ci = self.bootstrap_confidence_interval(
            y_test, y_test_proba,
            lambda y, proba: roc_auc_score(y, proba),
            n_bootstraps=n_bootstraps
        )

        pr_auc_ci = self.bootstrap_confidence_interval(
            y_test, y_test_proba,
            lambda y, proba: average_precision_score(y, proba),
            n_bootstraps=n_bootstraps
        )

        # Bootstrap for Recall@100
        evaluator = ModelEvaluator()
        recall_100_ci = self.bootstrap_confidence_interval(
            y_test, y_test_proba,
            lambda y, proba: evaluator.recall_at_k(y, proba, [100])['recall@100'],
            n_bootstraps=n_bootstraps
        )

        logger.info(f"Bootstrap Results (95% CI):")
        logger.info(f"  ROC-AUC: {roc_auc_ci['mean']:.4f} [{roc_auc_ci['ci_lower']:.4f}, {roc_auc_ci['ci_upper']:.4f}]")
        logger.info(f"  PR-AUC: {pr_auc_ci['mean']:.4f} [{pr_auc_ci['ci_lower']:.4f}, {pr_auc_ci['ci_upper']:.4f}]")
        logger.info(f"  Recall@100: {recall_100_ci['mean']:.4f} [{recall_100_ci['ci_lower']:.4f}, {recall_100_ci['ci_upper']:.4f}]")

        # Model comparison if we have multiple models
        model_comparison_results = {}
        if cv_results and len(cv_results) > 1:
            logger.info("\nComparing models statistically...")

            # Get predictions from different models if available
            model_names = list(cv_results.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]

                    # Compare mean PR-AUC scores statistically
                    score1 = cv_results[model1]['mean_metrics']['pr_auc_mean']
                    score2 = cv_results[model2]['mean_metrics']['pr_auc_mean']

                    # Simplified statistical comparison
                    mean_diff = score1 - score2
                    model_comparison_results[f'{model1}_vs_{model2}'] = {
                        'mean_diff': mean_diff,
                        'model1_score': score1,
                        'model2_score': score2
                    }

                    logger.info(f"  {model1} vs {model2}: diff = {mean_diff:.4f}")

        return {
            'bootstrap_ci': {
                'roc_auc': roc_auc_ci,
                'pr_auc': pr_auc_ci,
                'recall_100': recall_100_ci
            },
            'model_comparison': model_comparison_results,
            'n_bootstraps': n_bootstraps,
            'confidence_level': 0.95
        }


class ThresholdOptimizer:
    """Threshold optimization and cost-benefit analysis."""

    def __init__(self, cost_benefit_matrix: Optional[Dict[str, float]] = None):
        """Initialize optimizer with cost-benefit matrix."""
        if cost_benefit_matrix is None:
            # Default cost-benefit matrix for fraud detection
            self.cost_benefit_matrix = {
                'tp': 10,  # Correctly identified fraud
                'tn': 1,   # Correctly identified normal
                'fp': -2,  # False alarm (investigation cost)
                'fn': -5   # Missed fraud (severe consequence)
            }
        else:
            self.cost_benefit_matrix = cost_benefit_matrix

    def threshold_optimization_analysis(self, y_true: pd.Series, y_proba: np.ndarray,
                                      thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Comprehensive threshold optimization analysis."""
        if thresholds is None:
            thresholds = np.linspace(0.001, 0.999, 100)

        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Confusion matrix
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            # Standard metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Expected value
            expected_value = (tp * self.cost_benefit_matrix['tp'] +
                            tn * self.cost_benefit_matrix['tn'] +
                            fp * self.cost_benefit_matrix['fp'] +
                            fn * self.cost_benefit_matrix['fn'])

            # Cost-benefit ratio
            total_cost = abs(fp * self.cost_benefit_matrix['fp'] + fn * self.cost_benefit_matrix['fn'])
            total_benefit = tp * self.cost_benefit_matrix['tp'] + tn * self.cost_benefit_matrix['tn']
            cost_benefit_ratio = total_benefit / total_cost if total_cost > 0 else float('inf')

            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'expected_value': expected_value,
                'cost_benefit_ratio': cost_benefit_ratio
            })

        return pd.DataFrame(results)

    def find_optimal_thresholds(self, threshold_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Find optimal thresholds for different criteria."""
        optimals = {}

        # Maximum F1-score
        best_f1_idx = threshold_df['f1'].idxmax()
        optimals['f1_optimal'] = {
            'threshold': threshold_df.loc[best_f1_idx, 'threshold'],
            'f1': threshold_df.loc[best_f1_idx, 'f1'],
            'precision': threshold_df.loc[best_f1_idx, 'precision'],
            'recall': threshold_df.loc[best_f1_idx, 'recall']
        }

        # Maximum Expected Value
        best_ev_idx = threshold_df['expected_value'].idxmax()
        optimals['expected_value_optimal'] = {
            'threshold': threshold_df.loc[best_ev_idx, 'threshold'],
            'expected_value': threshold_df.loc[best_ev_idx, 'expected_value'],
            'precision': threshold_df.loc[best_ev_idx, 'precision'],
            'recall': threshold_df.loc[best_ev_idx, 'recall']
        }

        # Balanced precision-recall (where they are equal)
        threshold_df['precision_recall_diff'] = abs(threshold_df['precision'] - threshold_df['recall'])
        balanced_idx = threshold_df['precision_recall_diff'].idxmin()
        optimals['balanced_precision_recall'] = {
            'threshold': threshold_df.loc[balanced_idx, 'threshold'],
            'precision': threshold_df.loc[balanced_idx, 'precision'],
            'recall': threshold_df.loc[balanced_idx, 'recall'],
            'f1': threshold_df.loc[balanced_idx, 'f1']
        }

        # High precision (90%+)
        high_precision = threshold_df[threshold_df['precision'] >= 0.9]
        if not high_precision.empty:
            best_high_precision_idx = high_precision['recall'].idxmax()
            optimals['high_precision'] = {
                'threshold': high_precision.loc[best_high_precision_idx, 'threshold'],
                'precision': high_precision.loc[best_high_precision_idx, 'precision'],
                'recall': high_precision.loc[best_high_precision_idx, 'recall']
            }

        # High recall (90%+)
        high_recall = threshold_df[threshold_df['recall'] >= 0.9]
        if not high_recall.empty:
            best_high_recall_idx = high_recall['precision'].idxmax()
            optimals['high_recall'] = {
                'threshold': high_recall.loc[best_high_recall_idx, 'threshold'],
                'precision': high_recall.loc[best_high_recall_idx, 'precision'],
                'recall': high_recall.loc[best_high_recall_idx, 'recall']
            }

        return optimals


class CrossValidationAnalyzer:
    """Cross-validation and generalization analysis utilities."""

    def __init__(self, random_state: int = 42):
        """Initialize analyzer with random state."""
        self.random_state = random_state

    def final_cross_validation_test(self, X_test: pd.DataFrame, y_test: pd.Series,
                                  model, n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on the test set to assess generalization."""
        logger.info(f"Performing {n_splits}-fold cross-validation on test set...")

        # Use stratified k-fold to maintain class balance
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        cv_results = {
            'fold_metrics': [],
            'summary': {}
        }

        fold_scores = {
            'roc_auc': [],
            'pr_auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_test, y_test), 1):
            logger.info(f"  Fold {fold}/{n_splits}...")

            # Split data
            X_fold_train, X_fold_val = X_test.iloc[train_idx], X_test.iloc[val_idx]
            y_fold_train, y_fold_val = y_test.iloc[train_idx], y_test.iloc[val_idx]

            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_fold_train, y_fold_train)

            # Predict
            y_fold_proba = model_clone.predict_proba(X_fold_val)[:, 1]
            y_fold_pred = (y_fold_proba >= 0.5).astype(int)

            # Calculate metrics
            precision_val = ((y_fold_pred == 1) & (y_fold_val == 1)).sum() / (y_fold_pred == 1).sum() if (y_fold_pred == 1).sum() > 0 else 0
            recall_val = ((y_fold_pred == 1) & (y_fold_val == 1)).sum() / (y_fold_val == 1).sum() if (y_fold_val == 1).sum() > 0 else 0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

            fold_metrics = {
                'fold': fold,
                'roc_auc': roc_auc_score(y_fold_val, y_fold_proba),
                'pr_auc': average_precision_score(y_fold_val, y_fold_proba),
                'accuracy': (y_fold_pred == y_fold_val).mean(),
                'precision': precision_val,
                'recall': recall_val,
                'f1': f1_val
            }

            cv_results['fold_metrics'].append(fold_metrics)

            # Store scores for summary
            for metric in fold_scores.keys():
                fold_scores[metric].append(fold_metrics[metric])

            logger.info(f"    ROC-AUC: {fold_metrics['roc_auc']:.4f}, PR-AUC: {fold_metrics['pr_auc']:.4f}")

        # Calculate summary statistics
        for metric, scores in fold_scores.items():
            cv_results['summary'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            }

        return cv_results

    def analyze_generalization(self, cv_results: Dict[str, Any],
                             training_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze generalization performance."""
        logger.info("\nGeneralization Analysis:")

        summary = cv_results['summary']

        for metric, stats in summary.items():
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"  CV: {stats['cv']:.4f}")

            # Stability assessment
            if stats['cv'] < 0.1:
                stability = "Very Stable"
            elif stats['cv'] < 0.2:
                stability = "Stable"
            elif stats['cv'] < 0.3:
                stability = "Moderately Variable"
            else:
                stability = "Highly Variable"

            logger.info(f"  Stability: {stability}")

            # Compare with training if available
            if training_metrics and metric in training_metrics:
                train_mean = training_metrics[metric].get('mean', training_metrics[metric])
                diff = stats['mean'] - train_mean
                logger.info(f"  Train-Test Gap: {diff:+.4f}")
            logger.info()

        # Overall assessment
        avg_cv = np.mean([stats['cv'] for stats in summary.values()])
        if avg_cv < 0.15:
            overall_stability = "EXCELLENT - Model generalizes very well"
        elif avg_cv < 0.25:
            overall_stability = "GOOD - Model generalizes well with minor variability"
        elif avg_cv < 0.35:
            overall_stability = "FAIR - Model shows moderate generalization issues"
        else:
            overall_stability = "POOR - Model has significant generalization problems"

        logger.info(f"Overall Generalization Assessment: {overall_stability}")

        return {
            'overall_cv': avg_cv,
            'assessment': overall_stability
        }


class BusinessImpactAnalyzer:
    """Business impact analysis utilities."""

    def __init__(self, costs: Optional[Dict[str, float]] = None):
        """Initialize analyzer with cost configuration."""
        if costs is None:
            # Default costs for money laundering detection
            self.costs = {
                'false_positive_cost': 1000,  # Cost of investigating legitimate transaction
                'false_negative_cost': 50000,  # Cost of missing fraud (average loss)
                'true_positive_benefit': 50000,  # Benefit of catching fraud
                'true_negative_benefit': 0      # No benefit for correct non-fraud
            }
        else:
            self.costs = costs

    def calculate_business_impact(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                thresholds: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate business impact metrics for fraud detection."""
        logger.info("Calculating business impact metrics...")

        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)

        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calculate business impact
            total_cost = (
                fp * self.costs['false_positive_cost'] +
                fn * self.costs['false_negative_cost'] -
                tp * self.costs['true_positive_benefit'] -
                tn * self.costs['true_negative_benefit']
            )

            # Cost per prediction
            cost_per_pred = total_cost / len(y_true)

            # Expected value per prediction
            expected_value = -cost_per_pred  # Negative because costs are losses

            # ROI calculation (benefit/cost ratio)
            total_investigation_cost = fp * self.costs['false_positive_cost']
            total_fraud_prevented = tp * self.costs['true_positive_benefit']
            total_fraud_loss = fn * self.costs['false_negative_cost']

            if total_investigation_cost > 0:
                roi = (total_fraud_prevented - total_fraud_loss) / total_investigation_cost
            else:
                roi = 0

            results.append({
                'threshold': threshold,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'total_cost': float(total_cost),
                'cost_per_prediction': float(cost_per_pred),
                'expected_value': float(expected_value),
                'roi': float(roi),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0
            })

        return pd.DataFrame(results), self.costs

    def find_optimal_business_threshold(self, business_results_df: pd.DataFrame) -> Dict[str, Any]:
        """Find optimal threshold based on business metrics."""
        logger.info("Finding optimal business threshold...")

        # Find threshold with maximum expected value (minimum cost)
        optimal_idx = business_results_df['expected_value'].idxmax()
        optimal_row = business_results_df.loc[optimal_idx]

        logger.info(f"Optimal threshold: {optimal_row['threshold']:.3f}")
        logger.info(f"Expected value: ${optimal_row['expected_value']:,.2f} per prediction")
        logger.info(f"ROI: {optimal_row['roi']:.2%}")
        logger.info(f"Precision: {optimal_row['precision']:.3f}")
        logger.info(f"Recall: {optimal_row['recall']:.3f}")

        return optimal_row.to_dict()


class MonitoringSetup:
    """Model monitoring setup utilities."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize monitoring setup with alert thresholds."""
        if thresholds is None:
            self.thresholds = {
                'auc_drop_threshold': 0.05,  # Alert if AUC drops by more than 5%
                'precision_drop_threshold': 0.10,  # Alert if precision drops by more than 10%
                'recall_drop_threshold': 0.10,  # Alert if recall drops by more than 10%
                'drift_threshold': 0.15  # Alert if feature drift exceeds 15%
            }
        else:
            self.thresholds = thresholds

    def setup_performance_monitoring(self, model, X_reference: pd.DataFrame,
                                   y_reference: pd.Series) -> Dict[str, Any]:
        """Setup performance monitoring with alerts and thresholds."""
        logger.info("Setting up performance monitoring...")

        # Calculate baseline performance
        y_pred_proba_ref = model.predict_proba(X_reference)[:, 1]
        baseline_auc = roc_auc_score(y_reference, y_pred_proba_ref)

        y_pred_ref = (y_pred_proba_ref >= 0.5).astype(int)
        baseline_report = classification_report(y_pred_ref, y_reference, output_dict=True)

        baseline_metrics = {
            'auc': float(baseline_auc),
            'precision': float(baseline_report['weighted avg']['precision']),
            'recall': float(baseline_report['weighted avg']['recall']),
            'f1': float(baseline_report['weighted avg']['f1-score'])
        }

        monitoring_config = {
            'baseline_metrics': baseline_metrics,
            'alert_thresholds': self.thresholds,
            'monitoring_features': X_reference.columns.tolist()[:20],  # Monitor top 20 features
            'check_frequency': 'daily',  # How often to run monitoring
            'retraining_trigger': {
                'auc_drop': 0.10,  # Retrain if AUC drops by 10%
                'performance_window': 30,  # Days to evaluate before retraining
                'min_samples_for_retraining': 1000
            },
            'drift_detection': {
                'method': 'kolmogorov_smirnov',  # Statistical test for drift
                'significance_level': 0.05,
                'features_to_monitor': X_reference.columns.tolist()[:10]  # Top 10 features
            }
        }

        logger.info(f"Baseline AUC: {baseline_auc:.4f}")
        logger.info(f"Alert thresholds configured for {len(self.thresholds)} metrics")
        logger.info(f"Monitoring {len(monitoring_config['monitoring_features'])} features")

        return monitoring_config

    def create_monitoring_dashboard_template(self, monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a template for monitoring dashboard."""
        logger.info("Creating monitoring dashboard template...")

        dashboard_template = {
            'dashboard_title': 'Fraud Detection Model Monitor',
            'panels': [
                {
                    'title': 'Performance Metrics',
                    'metrics': ['auc', 'precision', 'recall', 'f1'],
                    'chart_type': 'time_series',
                    'alert_rules': [
                        {'metric': 'auc', 'condition': 'drop', 'threshold': monitoring_config['alert_thresholds']['auc_drop_threshold']},
                        {'metric': 'precision', 'condition': 'drop', 'threshold': monitoring_config['alert_thresholds']['precision_drop_threshold']}
                    ]
                },
                {
                    'title': 'Feature Drift',
                    'metrics': monitoring_config['drift_detection']['features_to_monitor'],
                    'chart_type': 'drift_heatmap',
                    'alert_rules': [
                        {'metric': 'drift_score', 'condition': 'exceeds', 'threshold': monitoring_config['alert_thresholds']['drift_threshold']}
                    ]
                },
                {
                    'title': 'Data Quality',
                    'metrics': ['missing_rate', 'outlier_rate', 'distribution_shift'],
                    'chart_type': 'quality_indicators',
                    'alert_rules': []
                }
            ],
            'alert_channels': ['email', 'slack', 'dashboard'],
            'report_schedule': 'weekly'
        }

        return dashboard_template