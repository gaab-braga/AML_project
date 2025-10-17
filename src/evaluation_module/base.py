# src/evaluation/base.py
"""
Base classes and interfaces for model evaluation and validation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from ..utils import logger
from ..config import settings


class Evaluator(ABC):
    """Abstract base class for evaluators"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger.getChild(f"evaluation.{name}")

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate predictions

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Evaluation results
        """
        pass

    def validate_input(self, y_true: pd.Series, y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None) -> None:
        """Validate evaluation inputs"""
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")

        if y_proba is not None:
            if len(y_true) != len(y_proba):
                raise ValueError(f"y_true and y_proba must have same length: {len(y_true)} vs {len(y_proba)}")

            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification probabilities
                pass
            elif y_proba.ndim == 1:
                # Binary classification single probability column
                pass
            else:
                raise ValueError(f"Invalid y_proba shape: {y_proba.shape}")


class ClassificationEvaluator(Evaluator):
    """Comprehensive classification evaluator"""

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Evaluation results
        """
        self.validate_input(y_true, y_pred, y_proba)

        results = {
            'basic_metrics': self._calculate_basic_metrics(y_true, y_pred),
            'threshold_metrics': {},
            'curves': {},
            'business_metrics': {}
        }

        if y_proba is not None:
            results['threshold_metrics'] = self._calculate_threshold_metrics(y_true, y_proba)
            results['curves'] = self._calculate_curves(y_true, y_proba)
            results['business_metrics'] = self._calculate_business_metrics(y_true, y_proba)

        self.logger.info("Classification evaluation completed",
                        accuracy=results['basic_metrics'].get('accuracy', 'N/A'))

        return results

    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate basic classification metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                   confusion_matrix, classification_report)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # True/False positives/negatives
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)

        # Detailed classification report
        metrics['classification_report'] = classification_report(y_true, y_pred,
                                                               output_dict=True, zero_division=0)

        return metrics

    def _calculate_threshold_metrics(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate threshold-dependent metrics"""
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Handle probability format
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        metrics = {}

        # AUC scores
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            metrics['average_precision'] = average_precision_score(y_true, y_proba_pos)
        except Exception as e:
            self.logger.warning(f"Could not calculate AUC metrics: {e}")

        # Optimal threshold (Youden's J statistic)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        metrics['optimal_threshold'] = float(thresholds[optimal_idx])
        metrics['optimal_threshold_metrics'] = {
            'tpr': float(tpr[optimal_idx]),
            'fpr': float(fpr[optimal_idx]),
            'j_score': float(j_scores[optimal_idx])
        }

        return metrics

    def _calculate_curves(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC and PR curves"""
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        curves = {}

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba_pos)
        curves['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist(),
            'auc': auc(fpr, tpr)
        }

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba_pos)
        curves['precision_recall'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist(),
            'auc': auc(recall, precision)
        }

        return curves

    def _calculate_business_metrics(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate business-relevant metrics for AML"""
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        metrics = {}

        # Recall at different top-k percentages (common in AML)
        recall_at_k = self._calculate_recall_at_k(y_true, y_proba_pos)
        metrics['recall_at_k'] = recall_at_k

        # Precision at different top-k percentages
        precision_at_k = self._calculate_precision_at_k(y_true, y_proba_pos)
        metrics['precision_at_k'] = precision_at_k

        # Alert efficiency metrics
        alert_metrics = self._calculate_alert_efficiency(y_true, y_proba_pos)
        metrics['alert_efficiency'] = alert_metrics

        return metrics

    def _calculate_recall_at_k(self, y_true: pd.Series, y_proba: np.ndarray, k_values: List[float] = None) -> Dict[str, float]:
        """Calculate recall at top-k% predictions"""
        if k_values is None:
            k_values = [0.01, 0.05, 0.1, 0.2, 0.5]  # 1%, 5%, 10%, 20%, 50%

        results = {}
        n_total_positives = y_true.sum()

        for k in k_values:
            n_predictions = max(1, int(len(y_true) * k))
            top_indices = np.argsort(y_proba)[::-1][:n_predictions]
            n_true_positives_found = y_true.iloc[top_indices].sum()
            recall = n_true_positives_found / n_total_positives if n_total_positives > 0 else 0
            results[f"recall_at_{k:.3f}"] = float(recall)

        return results

    def _calculate_precision_at_k(self, y_true: pd.Series, y_proba: np.ndarray, k_values: List[float] = None) -> Dict[str, float]:
        """Calculate precision at top-k% predictions"""
        if k_values is None:
            k_values = [0.01, 0.05, 0.1, 0.2, 0.5]

        results = {}

        for k in k_values:
            n_predictions = max(1, int(len(y_true) * k))
            top_indices = np.argsort(y_proba)[::-1][:n_predictions]
            n_true_positives = y_true.iloc[top_indices].sum()
            precision = n_true_positives / n_predictions
            results[f"precision_at_{k:.3f}"] = float(precision)

        return results

    def _calculate_alert_efficiency(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate alert efficiency metrics"""
        # Sort by probability descending
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_true = y_true.iloc[sorted_indices]

        # Cumulative true positives
        cum_tp = np.cumsum(sorted_true)
        total_positives = y_true.sum()

        # Find thresholds for different efficiency levels
        efficiency_levels = [0.5, 0.7, 0.8, 0.9, 0.95]  # 50%, 70%, etc.
        results = {}

        for eff in efficiency_levels:
            target_tp = int(total_positives * eff)
            n_alerts_needed = np.where(cum_tp >= target_tp)[0]

            if len(n_alerts_needed) > 0:
                n_alerts = n_alerts_needed[0] + 1
                alert_rate = n_alerts / len(y_true)
                results[f"alerts_for_{eff:.0%}_efficiency"] = {
                    'n_alerts': int(n_alerts),
                    'alert_rate': float(alert_rate),
                    'efficiency_achieved': float(cum_tp[n_alerts-1] / total_positives)
                }

        return results


class CrossValidator(Evaluator):
    """Cross-validation evaluator"""

    def __init__(self, name: str, cv_folds: int = 5, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.cv_folds = cv_folds

    def evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation

        Args:
            model: Model to evaluate (must have fit/predict methods)
            X: Feature matrix
            y: Target vector

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        self.logger.info(f"Performing {self.cv_folds}-fold cross-validation")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        fold_results = []
        all_predictions = []
        all_true_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"Evaluating fold {fold + 1}/{self.cv_folds}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit model
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)

            # Predict
            y_pred = model_copy.predict(X_val)
            y_proba = model_copy.predict_proba(X_val) if hasattr(model_copy, 'predict_proba') else None

            # Calculate metrics
            fold_metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
            }

            if y_proba is not None and y_proba.ndim == 2:
                from sklearn.metrics import roc_auc_score
                try:
                    fold_metrics['roc_auc'] = roc_auc_score(y_val, y_proba[:, 1])
                except:
                    pass

            fold_results.append(fold_metrics)

            # Collect predictions for overall metrics
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_val)

        # Calculate overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(all_true_labels, all_predictions),
            'precision': precision_score(all_true_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_true_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_true_labels, all_predictions, zero_division=0),
        }

        # Calculate fold statistics
        fold_df = pd.DataFrame(fold_results)
        fold_stats = {
            'mean': fold_df.drop('fold', axis=1).mean().to_dict(),
            'std': fold_df.drop('fold', axis=1).std().to_dict(),
            'min': fold_df.drop('fold', axis=1).min().to_dict(),
            'max': fold_df.drop('fold', axis=1).max().to_dict(),
        }

        results = {
            'fold_results': fold_results,
            'fold_statistics': fold_stats,
            'overall_metrics': overall_metrics,
            'cv_folds': self.cv_folds
        }

        self.logger.info("Cross-validation completed",
                        mean_accuracy=fold_stats['mean'].get('accuracy', 'N/A'))

        return results

    def _clone_model(self, model):
        """Clone model for cross-validation"""
        import copy
        return copy.deepcopy(model)


class ValidationPipeline:
    """Pipeline for comprehensive model validation"""

    def __init__(self, name: str, evaluators: List[Evaluator]):
        self.name = name
        self.evaluators = evaluators
        self.logger = logger.getChild(f"validation_pipeline.{name}")

    def validate(self, model, X: pd.DataFrame, y: pd.Series,
                X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation

        Args:
            model: Fitted model
            X: Test features
            y: Test target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Validation results
        """
        self.logger.info(f"Running validation pipeline: {self.name}")

        results = {
            'pipeline_name': self.name,
            'evaluations': {},
            'summary': {},
            'recommendations': []
        }

        # Run each evaluator
        for evaluator in self.evaluators:
            try:
                self.logger.info(f"Running evaluator: {evaluator.name}")

                if isinstance(evaluator, CrossValidator):
                    # Cross-validation on training data
                    eval_results = evaluator.evaluate(model, X, y)
                else:
                    # Standard evaluation
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                    eval_results = evaluator.evaluate(y, y_pred, y_proba)

                results['evaluations'][evaluator.name] = eval_results

            except Exception as e:
                error_msg = f"Evaluator {evaluator.name} failed: {str(e)}"
                self.logger.error(error_msg)
                results['evaluations'][evaluator.name] = {'error': error_msg}

        # Generate summary and recommendations
        results['summary'] = self._generate_summary(results['evaluations'])
        results['recommendations'] = self._generate_recommendations(results['evaluations'])

        self.logger.info("Validation pipeline completed")
        return results

    def _generate_summary(self, evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'total_evaluators': len(evaluations),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'key_metrics': {}
        }

        for eval_name, eval_results in evaluations.items():
            if 'error' in eval_results:
                summary['failed_evaluations'] += 1
            else:
                summary['successful_evaluations'] += 1

                # Extract key metrics
                if 'basic_metrics' in eval_results:
                    metrics = eval_results['basic_metrics']
                    for metric in ['accuracy', 'precision', 'recall', 'f1']:
                        if metric in metrics:
                            summary['key_metrics'][f"{eval_name}_{metric}"] = metrics[metric]

        return summary

    def _generate_recommendations(self, evaluations: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations"""
        recommendations = []

        # Check for evaluation failures
        failed_evals = [name for name, results in evaluations.items() if 'error' in results]
        if failed_evals:
            recommendations.append(f"Fix failed evaluations: {', '.join(failed_evals)}")

        # Check classification metrics
        for eval_name, eval_results in evaluations.items():
            if 'basic_metrics' in eval_results:
                metrics = eval_results['basic_metrics']

                # Low recall warning (critical for AML)
                if metrics.get('recall', 1.0) < 0.7:
                    recommendations.append(f"Improve recall in {eval_name} (currently {metrics['recall']:.3f})")

                # Class imbalance check
                if 'confusion_matrix' in metrics:
                    cm = np.array(metrics['confusion_matrix'])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        if tp + fn > 0 and tp / (tp + fn) < 0.1:  # Less than 10% positive class
                            recommendations.append(f"Address class imbalance in {eval_name}")

        return recommendations


__all__ = [
    "Evaluator",
    "ClassificationEvaluator",
    "CrossValidator",
    "ValidationPipeline"
]