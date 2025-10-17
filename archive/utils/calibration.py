"""
Calibration and threshold optimization for fraud detection models.
Provides probability calibration, threshold tuning, and cost-sensitive optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from .data import save_artifact
from .visualization import plot_calibration_curve, plot_threshold_analysis


class ThresholdOptimizer:
    """Optimize classification thresholds for fraud detection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.optimal_thresholds = {}
        self.threshold_curves = {}

    def optimize_threshold_recall(self, y_true: np.ndarray, y_proba: np.ndarray,
                                target_recall: float = 0.95) -> float:
        """Find threshold that achieves target recall."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Find threshold closest to target recall
        recall_diff = np.abs(recalls[:-1] - target_recall)
        optimal_idx = np.argmin(recall_diff)
        optimal_threshold = thresholds[optimal_idx]

        self.optimal_thresholds['recall_target'] = {
            'threshold': optimal_threshold,
            'achieved_recall': recalls[optimal_idx],
            'precision': precisions[optimal_idx],
            'target_recall': target_recall
        }

        return optimal_threshold

    def optimize_threshold_cost_sensitive(self, y_true: np.ndarray, y_proba: np.ndarray,
                                        fp_cost: float = 1.0, fn_cost: float = 10.0) -> float:
        """Optimize threshold based on misclassification costs."""
        thresholds = np.linspace(0.01, 0.99, 99)
        costs = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            total_cost = fp * fp_cost + fn * fn_cost
            costs.append(total_cost)

        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]

        self.optimal_thresholds['cost_sensitive'] = {
            'threshold': optimal_threshold,
            'total_cost': costs[optimal_idx],
            'fp_cost': fp_cost,
            'fn_cost': fn_cost
        }

        return optimal_threshold

    def optimize_threshold_fraud_rate(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    target_fraud_rate: float = 0.05) -> float:
        """Optimize threshold to achieve target fraud investigation rate."""
        thresholds = np.linspace(0.01, 0.99, 99)
        fraud_rates = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            fraud_rate = np.mean(y_pred)
            fraud_rates.append(fraud_rate)

        # Find threshold closest to target fraud rate
        rate_diff = np.abs(np.array(fraud_rates) - target_fraud_rate)
        optimal_idx = np.argmin(rate_diff)
        optimal_threshold = thresholds[optimal_idx]

        self.optimal_thresholds['fraud_rate_target'] = {
            'threshold': optimal_threshold,
            'achieved_rate': fraud_rates[optimal_idx],
            'target_rate': target_fraud_rate
        }

        return optimal_threshold

    def get_threshold_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive threshold analysis curves."""
        thresholds = np.linspace(0.01, 0.99, 99)

        recalls, precisions, f1s, fraud_rates = [], [], [], []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            recalls.append(recall_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))
            fraud_rates.append(np.mean(y_pred))

        self.threshold_curves = {
            'thresholds': thresholds,
            'recall': recalls,
            'precision': precisions,
            'f1': f1s,
            'fraud_rate': fraud_rates
        }

        return self.threshold_curves

    def find_optimal_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Find optimal thresholds using multiple strategies."""
        print("Optimizing classification thresholds...")

        optimal_thresholds = {}

        # Recall target (95%)
        optimal_thresholds['recall_95'] = self.optimize_threshold_recall(
            y_true, y_proba, target_recall=0.95
        )

        # Cost-sensitive (FP cost = 1, FN cost = 10)
        optimal_thresholds['cost_sensitive'] = self.optimize_threshold_cost_sensitive(
            y_true, y_proba, fp_cost=1.0, fn_cost=10.0
        )

        # Fraud rate target (5%)
        optimal_thresholds['fraud_rate_5'] = self.optimize_threshold_fraud_rate(
            y_true, y_proba, target_fraud_rate=0.05
        )

        # Youden's J statistic (maximizes TPR - FPR)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_thresholds['youden_j'] = roc_thresholds[optimal_idx]

        print(f"Optimal thresholds found: {optimal_thresholds}")
        return optimal_thresholds


class ProbabilityCalibrator:
    """Probability calibration for fraud detection models."""

    def __init__(self, method: str = 'isotonic', cv: int = 5):
        self.method = method
        self.cv = cv
        self.calibrators = {}

    def calibrate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
        """Calibrate model probabilities."""
        print(f"Calibrating model using {self.method} regression...")

        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=self.cv
        )

        calibrated_model.fit(X_train, y_train)

        # Evaluate calibration
        y_proba_uncalibrated = model.predict_proba(X_val)[:, 1]
        y_proba_calibrated = calibrated_model.predict_proba(X_val)[:, 1]

        calibration_metrics = self._evaluate_calibration(y_val, y_proba_uncalibrated, y_proba_calibrated)

        self.calibrators['model'] = calibrated_model
        self.calibrators['metrics'] = calibration_metrics

        return calibrated_model

    def _evaluate_calibration(self, y_true: np.ndarray, y_proba_uncal: np.ndarray,
                            y_proba_cal: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality."""
        from sklearn.metrics import brier_score_loss

        # Brier scores
        brier_uncal = brier_score_loss(y_true, y_proba_uncal)
        brier_cal = brier_score_loss(y_true, y_proba_cal)

        # Expected calibration error (ECE)
        ece_uncal = self._expected_calibration_error(y_true, y_proba_uncal)
        ece_cal = self._expected_calibration_error(y_true, y_proba_cal)

        return {
            'brier_score_uncalibrated': brier_uncal,
            'brier_score_calibrated': brier_cal,
            'ece_uncalibrated': ece_uncal,
            'ece_calibrated': ece_cal,
            'brier_improvement': brier_uncal - brier_cal,
            'ece_improvement': ece_uncal - ece_cal
        }

    def _expected_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray,
                                  n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        return np.mean(np.abs(prob_pred - prob_true))

    def plot_calibration_comparison(self, y_true: np.ndarray, y_proba_uncal: np.ndarray,
                                  y_proba_cal: np.ndarray, save_path: Optional[str] = None):
        """Plot calibration curves comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calibration curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_proba_uncal, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_proba_cal, n_bins=10)

        ax1.plot(prob_pred_uncal, prob_true_uncal, 's-', label='Uncalibrated', alpha=0.8)
        ax1.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrated', alpha=0.8)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean predicted probability')
        ax1.set_ylabel('Fraction of positives')
        ax1.set_title('Calibration Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of probabilities
        ax2.hist(y_proba_uncal[y_true == 0], alpha=0.5, label='Negatives (uncal)', bins=20)
        ax2.hist(y_proba_uncal[y_true == 1], alpha=0.5, label='Positives (uncal)', bins=20)
        ax2.hist(y_proba_cal[y_true == 0], alpha=0.5, label='Negatives (cal)', bins=20, histtype='step')
        ax2.hist(y_proba_cal[y_true == 1], alpha=0.5, label='Positives (cal)', bins=20, histtype='step')
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Probability Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plot saved to {save_path}")

        plt.show()


def optimize_threshold_and_calibrate(model, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Complete pipeline: calibrate model and optimize thresholds."""
    print("Starting threshold optimization and calibration pipeline...")

    # Step 1: Calibrate probabilities
    calibrator = ProbabilityCalibrator()
    calibrated_model = calibrator.calibrate_model(model, X_train, y_train, X_val, y_val)

    # Step 2: Get calibrated probabilities on validation set
    y_proba_calibrated = calibrated_model.predict_proba(X_val)[:, 1]

    # Step 3: Optimize thresholds
    threshold_optimizer = ThresholdOptimizer()
    optimal_thresholds = threshold_optimizer.find_optimal_thresholds(y_val, y_proba_calibrated)

    # Step 4: Evaluate on test set
    y_proba_test = calibrated_model.predict_proba(X_test)[:, 1]

    test_results = {}
    for name, threshold in optimal_thresholds.items():
        y_pred_test = (y_proba_test >= threshold).astype(int)

        test_results[name] = {
            'threshold': threshold,
            'recall': recall_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'fraud_rate': np.mean(y_pred_test)
        }

    # Step 5: Get threshold curves for plotting
    threshold_curves = threshold_optimizer.get_threshold_curves(y_val, y_proba_calibrated)

    results = {
        'calibrated_model': calibrated_model,
        'calibration_metrics': calibrator.calibrators['metrics'],
        'optimal_thresholds': optimal_thresholds,
        'test_results': test_results,
        'threshold_curves': threshold_curves,
        'threshold_optimizer': threshold_optimizer
    }

    print("Threshold optimization and calibration complete")
    return results