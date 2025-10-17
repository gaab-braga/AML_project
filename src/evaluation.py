"""
Evaluation Module for AML Detection
Handles metrics computation and plotting.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred_proba) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC.

    In AML, PR-AUC is prioritized over ROC-AUC because classes are imbalanced;
    we care more about precision/recall trade-off to minimize false negatives (missed fraud).
    """
    logger.info("Computing evaluation metrics")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    metrics = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
    logger.info(f"Metrics: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")
    return metrics

def plot_roc_pr(y_true, y_pred_proba, interactive: bool = False) -> None:
    """
    Plot ROC and PR curves, with AML thresholds.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        interactive: Use Plotly
    """
    logger.info("Plotting ROC and PR curves")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC Curve', 'Precision-Recall Curve'))
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC = {roc_auc:.3f}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'), row=1, col=1)
            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR AUC = {pr_auc:.3f}'), row=1, col=2)
            fig.update_layout(title_text="AML Model Evaluation Curves")
            fig.show()
        except ImportError:
            logger.warning("Plotly not available, using matplotlib")
            interactive = False

    if not interactive:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()

        ax2.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()

        plt.tight_layout()
        plt.show()
    logger.info("Plots generated")

def plot_shap_summary(shap_values, feature_names, max_display: int = 10):
    """
    Plot SHAP summary for explainability in AML.

    Args:
        shap_values: SHAP values array
        feature_names: Feature names
        max_display: Max features to display
    """
    try:
        import shap
        logger.info("Plotting SHAP summary")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
        plt.title("SHAP Feature Importance (AML Explainability)")
        plt.show()
    except ImportError:
        logger.warning("SHAP not installed, skipping explainability plot")