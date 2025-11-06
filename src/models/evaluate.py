"""
Model evaluation utilities.
"""
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        y_proba: Probabilities (optional)
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Calculating evaluation metrics")
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    return metrics


def save_evaluation_report(metrics: dict, filepath: str = "artifacts/evaluation.json"):
    """
    Save evaluation report to JSON.
    
    Args:
        metrics: Metrics dictionary
        filepath: Output filepath
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation report saved to {filepath}")


def print_evaluation_summary(metrics: dict):
    """
    Print evaluation metrics summary.
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print("="*50 + "\n")
