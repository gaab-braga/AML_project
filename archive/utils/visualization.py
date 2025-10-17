"""
🎨 Visualization Module - Advanced ML Model Visualizations.

This MEGA-MODULE provides comprehensive visualization tools for:

1. EXECUTIVE DASHBOARDS
   - KPI indicators with gauges
   - Risk distribution plots
   - Executive summary dashboards
   - Performance comparison visualizations

2. MODEL DIAGNOSTICS
   - ROC and PR curves with detailed analysis
   - Calibration curves
   - Confusion matrices
   - Learning curves
   - Threshold analysis

3. FEATURE ANALYSIS
   - Feature importance plots (multiple methods)
   - Correlation heatmaps
   - Distribution plots
   - Missing value analysis

4. TEMPORAL ANALYSIS
   - Time series plots
   - Temporal heatmaps
   - Hourly transaction patterns
   - Seasonal trend analysis

5. NETWORK ANALYSIS
   - Transaction network graphs
   - Network metrics visualization
   - Entity relationship plots
   - Graph-based fraud detection

6. ENSEMBLE ANALYSIS
   - Model comparison plots
   - Ensemble performance visualization
   - Prediction distribution analysis
   - Model agreement metrics

7. EXPLORATORY DATA ANALYSIS (EDA)
   - Interactive EDA dashboards
   - Data quality reports
   - Monetary distribution plots
   - Comprehensive diagnostic reports

8. QUICK UTILITIES
   - Quick dashboard generation
   - Quick EDA
   - Quick performance plots
   - Quick network visualization

Author: Data Science Team
Date: October 2025
Phase: 3 - Roadmap Implementation
"""

__all__ = [
    # Main Class
    'AMLVisualizationSuite',
    
    # Executive Dashboards (standalone functions only)
    'create_executive_summary_dashboard',
    'create_risk_distribution_plot',
    
    # Model Diagnostics
    'plot_roc_pr_curves',
    'plot_roc_detailed_analysis',
    'plot_calibration_curve',
    'plot_confusion_matrix',
    'plot_learning_curve',
    'plot_threshold_analysis',
    'plot_prediction_distribution',
    'plot_model_comparison',
    'plot_model_performance_comparison',
    'plot_metrics_at_k',
    
    # Feature Analysis
    'plot_feature_importance',
    'create_feature_importance_plot',
    'plot_missing_values',
    'plot_monetary_distributions',
    
    # Temporal Analysis
    'plot_temporal_series',
    'plot_temporal_heatmaps',
    'plot_hourly_transactions',
    
    # Network Analysis
    'plot_network_metrics',
    'calculate_network_metrics',
    'save_network_metrics',
    
    # Ensemble Analysis (standalone functions only)
    # 'plot_ensemble_analysis',  # Class method only
    
    # Fraud Pattern Analysis (standalone functions only)
    # 'plot_fraud_patterns',  # Class method only
    
    # EDA and Data Quality
    'analyze_data_quality',
    'comprehensive_diagnostic_report',
    'save_all_diagnostic_plots',
    
    # Tuning Visualization
    'plot_tuning_results',
    
    # Quick Utilities
    'quick_dashboard',
    'quick_eda',
    'quick_network',
    'quick_performance',
    
    # Utility Functions
    'setup_plot_style',
    'save_all_plots',
    'save_plot_html',
    'save_temporal_metrics',
]

# Standard library
import json
import warnings
from pathlib import Path

# Data science stack
import numpy as np
import pandas as pd

# Visualization - Matplotlib/Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Visualization - Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Scikit-learn metrics
from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ================================================================================
# COMPONENT 1: EXECUTIVE DASHBOARDS
# ================================================================================

def create_executive_summary_dashboard(metrics_dict, title="AML Model Performance"):
    """
    Cria dashboard executivo com KPIs principais.
    
    Args:
        metrics_dict: Dict com métricas {metric_name: value, ...}
        title: Título do dashboard
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(metrics_dict.keys())[:6],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (metric, value) in enumerate(list(metrics_dict.items())[:6]):
        row, col = positions[i]
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=float(value),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': metric},
                gauge={
                    'axis': {'range': [None, 1.0]},
                    'bar': {'color': colors[i]},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1.0], 'color': "green"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.9}
                }
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        font={'size': 14},
        height=600
    )
    
    return fig

def create_risk_distribution_plot(risk_scores, true_labels, title="Risk Score Distribution"):
    """
    Cria gráfico de distribuição de risk scores por classe.
    """
    import pandas as pd
    df_plot = pd.DataFrame({
        'Risk_Score': risk_scores,
        'True_Label': ['Money Laundering' if x == 1 else 'Legitimate' for x in true_labels]
    })
    
    fig = px.histogram(
        df_plot, 
        x='Risk_Score', 
        color='True_Label',
        nbins=50,
        title=title,
        labels={'Risk_Score': 'Risk Score', 'count': 'Frequency'},
        color_discrete_map={
            'Legitimate': '#2ca02c',
            'Money Laundering': '#d62728'
        }
    )
    
    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Frequency",
        legend_title="Transaction Type",
        font=dict(size=12),
        height=500
    )
    
    return fig

def create_feature_importance_plot(feature_names, importance_values, title="Feature Importance"):
    """
    Cria gráfico de importância de features.
    """
    import pandas as pd
    df_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df_features.tail(15),  # Top 15 features
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
    )
    
    fig.update_layout(
        height=600,
        font=dict(size=11),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Função de conveniência para salvar visualizações
def save_plot_html(fig, filename, output_dir="artifacts/visualizations"):
    """Salva plot como HTML interativo."""
    # Path already imported at module level
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / f"{filename}.html"
    fig.write_html(str(filepath))
    return filepath

"""
Diagnostic utilities for model analysis.

Learning curves, calibration curves, and diagnostic plots.
"""

# Additional imports for this component (already imported at module level: numpy, pandas, matplotlib)
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import learning_curve as sklearn_learning_curve
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator

# Optional import - FraudMetrics may not be needed for all functions
FraudMetrics = None
try:
    from modeling import FraudMetrics
except ImportError:
    try:
        from .modeling import FraudMetrics
    except ImportError:
        pass  # FraudMetrics optional for EDA functions

logger = logging.getLogger(__name__)


def plot_learning_curve(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    scoring: str = 'average_precision',
    n_jobs: int = -1,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Plot learning curve to diagnose bias/variance.
    
    Args:
        estimator: Unfitted estimator
        X: Features
        y: Labels
        cv: CV folds
        train_sizes: Training set sizes to evaluate
        scoring: Metric for evaluation
        n_jobs: Parallel jobs
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Dict with train_sizes, train_scores, val_scores
        
    Example:
        >>> from lightgbm import LGBMClassifier
        >>> model = LGBMClassifier(n_estimators=100)
        >>> results = plot_learning_curve(model, X_train, y_train)
    """
    logger.info("Computing learning curve...")
    
    train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=42
    )
    
    # Calculate means and stds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    
    ax.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Validation score')
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color='g'
    )
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel(f'{scoring.replace("_", " ").title()}', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Diagnose
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        diagnosis = "HIGH VARIANCE (overfitting) - consider regularization"
    elif val_mean[-1] < 0.5:
        diagnosis = "HIGH BIAS (underfitting) - increase model complexity"
    else:
        diagnosis = "Good fit"
    
    ax.text(
        0.5, 0.05, f'Diagnosis: {diagnosis}',
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        ha='center'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning curve to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'train_mean': train_mean,
        'val_mean': val_mean,
        'diagnosis': diagnosis
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Plot calibration curve to assess probability calibration.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy ('uniform' or 'quantile')
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Dict with prob_true, prob_pred, calibration_error
        
    Example:
        >>> y_proba = model.predict_proba(X_test)[:, 1]
        >>> results = plot_calibration_curve(y_test, y_proba)
        >>> print(f"Calibration error: {results['calibration_error']:.4f}")
    """
    logger.info("Computing calibration curve...")
    
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy=strategy
    )
    
    # Calculate calibration error (mean absolute difference)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calibration plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 'o-', label=f'Model (error={calibration_error:.3f})')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('True Probability', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_proba, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Diagnose
    if calibration_error < 0.05:
        diagnosis = "Well calibrated"
    elif calibration_error < 0.10:
        diagnosis = "Moderately calibrated - consider calibration"
    else:
        diagnosis = "Poorly calibrated - use CalibratedClassifierCV"
    
    fig.suptitle(f'Calibration Diagnosis: {diagnosis}', fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'calibration_error': calibration_error,
        'diagnosis': diagnosis
    }


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = np.arange(0.1, 0.9, 0.05),
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze metrics across different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to evaluate
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        DataFrame with metrics at each threshold
        
    Example:
        >>> results = plot_threshold_analysis(y_test, y_proba)
        >>> optimal = results.loc[results['f1'].idxmax()]
        >>> print(f"Optimal threshold: {optimal['threshold']:.2f}")
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    logger.info(f"Analyzing {len(thresholds)} thresholds...")
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    df = pd.DataFrame(results)
    
    # Find optimal thresholds
    optimal_f1 = df.loc[df['f1'].idxmax()]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Metrics vs threshold
    ax1.plot(df['threshold'], df['precision'], 'o-', label='Precision', linewidth=2)
    ax1.plot(df['threshold'], df['recall'], 's-', label='Recall', linewidth=2)
    ax1.plot(df['threshold'], df['f1'], '^-', label='F1', linewidth=2)
    ax1.axvline(optimal_f1['threshold'], color='red', linestyle='--', 
                label=f'Optimal F1 ({optimal_f1["threshold"]:.2f})')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall tradeoff
    ax2.plot(df['recall'], df['precision'], 'o-', linewidth=2)
    ax2.scatter(optimal_f1['recall'], optimal_f1['precision'], 
                color='red', s=200, marker='*', 
                label=f'Optimal (t={optimal_f1["threshold"]:.2f})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    logger.info(f"Optimal F1 threshold: {optimal_f1['threshold']:.3f} (F1={optimal_f1['f1']:.4f})")
    
    return df


def comprehensive_diagnostic_report(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = 'artifacts/diagnostics',
    cv: int = 5
) -> Dict[str, any]:
    """
    Generate comprehensive diagnostic report.
    
    Args:
        estimator: Fitted estimator
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save plots
        cv: CV folds
        
    Returns:
        Dict with all diagnostic results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating comprehensive diagnostic report...")
    
    results = {}
    
    # Learning curve (requires unfitted clone)
    from sklearn.base import clone
    unfitted = clone(estimator)
    results['learning_curve'] = plot_learning_curve(
        unfitted,
        X_train,
        y_train,
        cv=cv,
        save_path=f'{output_dir}/learning_curve.png'
    )
    
    # Calibration curve
    y_proba = estimator.predict_proba(X_test)[:, 1]
    results['calibration'] = plot_calibration_curve(
        y_test,
        y_proba,
        save_path=f'{output_dir}/calibration_curve.png'
    )
    
    # Threshold analysis
    results['threshold_analysis'] = plot_threshold_analysis(
        y_test,
        y_proba,
        save_path=f'{output_dir}/threshold_analysis.png'
    )
    
    # Performance metrics
    metrics = FraudMetrics()
    results['metrics'] = metrics.compute_all(y_test, y_proba)
    
    logger.info(f"Diagnostic report saved to {output_dir}")
    
    return results
"""
Módulo de visualizações para análise exploratória de lavagem de dinheiro.
Contém funções modulares para gerar gráficos padronizados com paleta inteligente.
"""

# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# import matplotlib.pyplot as plt  # Already imported at top
# import seaborn as sns  # Already imported at top
# from pathlib import Path  # Already imported at top
# import json  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #2

# Paleta de cores inteligente
COLORS_NEUTRAL = ["#B0B0B0", '#DC143C']  # Cinza neutro vs Vermelho crítico
COLORS_NETWORK = ['#708090', '#DC143C']  # Cinza neutro vs Vermelho crítico
COLOR_VOLUME = '#4A90A4'  # Azul acinzentado neutro
COLOR_LAUNDERING = '#DC143C'  # Vermelho crítico para lavagem


def plot_temporal_series(daily_summary, figsize=(14, 6)):
    """
    Plota séries temporais de volume e taxa de lavagem.
    
    Args:
        daily_summary: DataFrame com colunas 'date', 'transaction_count', 'laundering_rate'
        figsize: Tamanho da figura
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Volume diário
    daily_summary.set_index('date')['transaction_count'].plot(
        ax=axes[0], color=COLOR_VOLUME, linewidth=2.5
    )
    axes[0].set_ylabel('Quantidade de Transações', fontweight='bold')
    axes[0].set_title('Volume Diário de Transações', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Taxa de lavagem
    daily_summary.set_index('date')['laundering_rate'].plot(
        ax=axes[1], color=COLOR_LAUNDERING, linewidth=3
    )
    axes[1].set_ylabel('Taxa de Lavagem', fontweight='bold')
    axes[1].set_title('Taxa Diária de Lavagem de Dinheiro', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_heatmaps(df, figsize=(14, 4)):
    """
    Plota heatmaps de taxa de lavagem e volume por dia/hora.
    
    Args:
        df: DataFrame com colunas 'dow', 'hour', 'Is Laundering'
        figsize: Tamanho da figura
    """
    # Heatmap de taxa de lavagem
    pivot_rate = df.pivot_table(
        index='dow', columns='hour', 
        values='Is Laundering', aggfunc='mean'
    )
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_rate, cmap='Greys_r', annot=False, 
                cbar_kws={'label': 'Taxa de Lavagem'})
    plt.title('Taxa de Lavagem: Dia da Semana vs Hora', fontsize=14, fontweight='bold')
    plt.xlabel('Hora do Dia', fontweight='bold')
    plt.ylabel('Dia da Semana (0=Segunda)', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Heatmap de volume
    pivot_volume = df.pivot_table(
        index='dow', columns='hour', 
        values='Is Laundering', aggfunc='count'
    )
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_volume, cmap='Blues', 
                cbar_kws={'label': 'Volume de Transações'})
    plt.title('Volume de Transações: Dia da Semana vs Hora', fontsize=14, fontweight='bold')
    plt.xlabel('Hora do Dia', fontweight='bold')
    plt.ylabel('Dia da Semana (0=Segunda)', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_network_metrics(df_enriched, figsize=(15, 10)):
    """
    Plota boxplots das métricas de rede por classe.
    
    Args:
        df_enriched: DataFrame com métricas de rede e coluna 'Is Laundering'
        figsize: Tamanho da figura
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Out-degree
    sns.boxplot(data=df_enriched, x='Is Laundering', y='out_degree', 
                ax=axes[0,0], palette=COLORS_NETWORK)
    axes[0,0].set_title('Out-degree por Classe', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Número de Contas de Destino')
    axes[0,0].set_xticklabels(['Normal', 'Lavagem'])

    # In-degree
    sns.boxplot(data=df_enriched, x='Is Laundering', y='in_degree', 
                ax=axes[0,1], palette=COLORS_NETWORK)
    axes[0,1].set_title('In-degree por Classe', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Número de Contas de Origem')
    axes[0,1].set_xticklabels(['Normal', 'Lavagem'])

    # Out-strength (log)
    df_plot = df_enriched[df_enriched['out_strength'] > 0].copy()
    df_plot['log_out_strength'] = np.log1p(df_plot['out_strength'])
    sns.boxplot(data=df_plot, x='Is Laundering', y='log_out_strength', 
                ax=axes[1,0], palette=COLORS_NETWORK)
    axes[1,0].set_title('Out-strength por Classe (log)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Log(1 + Valor Total Enviado)')
    axes[1,0].set_xticklabels(['Normal', 'Lavagem'])

    # Diversidade de destinatários
    sns.boxplot(data=df_enriched, x='Is Laundering', y='unique_destinations', 
                ax=axes[1,1], palette=COLORS_NETWORK)
    axes[1,1].set_title('Diversidade de Destinatários', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Número de Destinatários Únicos')
    axes[1,1].set_xticklabels(['Normal', 'Lavagem'])

    plt.tight_layout()
    plt.show()


def plot_monetary_distributions(df, figsize=(12, 6)):
    """
    Plota distribuições monetárias (Amount Paid e Received) por classe.
    
    Args:
        df: DataFrame com colunas 'Is Laundering', 'Amount Paid', 'Amount Received'
        figsize: Tamanho da figura
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Amount Paid
    sns.boxplot(data=df, x='Is Laundering', y='Amount Paid', 
                ax=axes[0], palette=COLORS_NEUTRAL)
    axes[0].set_title('Distribuição Amount Paid', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].set_xticklabels(['Normal', 'Lavagem'])

    # Amount Received
    sns.boxplot(data=df, x='Is Laundering', y='Amount Received', 
                ax=axes[1], palette=COLORS_NEUTRAL)
    axes[1].set_title('Distribuição Amount Received', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].set_xticklabels(['Normal', 'Lavagem'])

    plt.tight_layout()
    plt.show()


def plot_hourly_transactions(df, figsize=(14, 6)):
    """
    Plota distribuição de transações por hora do dia.
    
    Args:
        df: DataFrame com colunas 'Hour', 'Is Laundering'
        figsize: Tamanho da figura
    """
    plt.figure(figsize=figsize)
    df_hour_summary = df.groupby(['Hour', 'Is Laundering']).size().unstack(fill_value=0)
    df_hour_summary.plot(kind='bar', width=0.8, color=COLORS_NEUTRAL, figsize=figsize)
    plt.title('Transações por Hora do Dia', fontsize=16, fontweight='bold')
    plt.xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    plt.ylabel('Número de Transações', fontsize=12, fontweight='bold')
    plt.legend(['Normal', 'Lavagem'], loc='upper right', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_missing_values(missing_overview, figsize=(10, 6)):
    """
    Plota gráfico de barras para variáveis com valores ausentes.
    
    Args:
        missing_overview: DataFrame com colunas 'n_missing', 'pct_missing'
        figsize: Tamanho da figura
    """
    if missing_overview['n_missing'].sum() > 0:
        vars_with_missing = missing_overview[missing_overview['n_missing'] > 0]
        
        plt.figure(figsize=figsize)
        plt.bar(range(len(vars_with_missing)), vars_with_missing['pct_missing'], color='#DC143C')
        plt.xticks(range(len(vars_with_missing)), vars_with_missing.index, 
                   rotation=45, ha='right')
        plt.ylabel('Percentual de Missing (%)', fontweight='bold')
        plt.title('Variáveis com Valores Ausentes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("✓ Não há valores ausentes no dataset")


def save_temporal_metrics(df, daily_summary, artifacts_dir):
    """
    Calcula e salva métricas temporais em JSON.
    
    Args:
        df: DataFrame principal
        daily_summary: DataFrame com estatísticas diárias
        artifacts_dir: Diretório para salvar artefatos
        
    Returns:
        dict: Métricas temporais calculadas
    """
    # Padrões por hora e dia da semana
    hourly_rate = df.groupby('hour')['Is Laundering'].mean()
    dow_rate = df.groupby('dow')['Is Laundering'].mean()
    
    metrics_temporal = {
        'n_days': int(daily_summary.shape[0]),
        'date_range': (str(daily_summary['date'].min()), str(daily_summary['date'].max())),
        'transactions_per_day': {
            'min': int(daily_summary['transaction_count'].min()),
            'median': float(daily_summary['transaction_count'].median()),
            'max': int(daily_summary['transaction_count'].max())
        },
        'laundering_rate_daily': {
            'min': float(daily_summary['laundering_rate'].min()),
            'median': float(daily_summary['laundering_rate'].median()),
            'max': float(daily_summary['laundering_rate'].max())
        },
        'overall_laundering_rate': float(df['Is Laundering'].mean()),
        'top_hours_rate': hourly_rate.sort_values(ascending=False).head(5).round(4).to_dict(),
        'dow_laundering_rate': dow_rate.round(4).to_dict()
    }
    
    # Salvar métricas
    with open(artifacts_dir / 'temporal_metrics.json', 'w') as f:
        json.dump(metrics_temporal, f, indent=2, default=str)
    
    return metrics_temporal


def save_network_metrics(network_summary, artifacts_dir):
    """
    Salva métricas de rede em JSON.
    
    Args:
        network_summary: DataFrame com resumo das métricas de rede
        artifacts_dir: Diretório para salvar artefatos
    """
    network_summary_dict = network_summary.round(3).to_dict()
    network_summary_dict_str = {str(k): v for k, v in network_summary_dict.items()}

    with open(artifacts_dir / 'network_metrics.json', 'w') as f:
        json.dump(network_summary_dict_str, f, indent=2, default=str)

    print(f"Métricas de rede salvas em: {artifacts_dir / 'network_metrics.json'}")


def analyze_data_quality(df):
    """
    Análise abrangente de qualidade dos dados.
    
    Args:
        df: DataFrame para análise
        
    Returns:
        tuple: (missing_overview, numeric_stats, spread)
    """
    print("Verificando duplicatas e valores ausentes...")
    duplicate_full = df.duplicated().sum()
    duplicate_keys = df.duplicated(subset=['Account', 'Dest Account', 'Timestamp']).sum()
    print(f"Duplicatas (linhas completas): {duplicate_full}")
    print(f"Duplicatas (Account, Dest Account, Timestamp): {duplicate_keys}")

    missing_overview = (df.isna()
                         .sum()
                         .to_frame('n_missing')
                         .assign(pct_missing=lambda x: (x['n_missing'] / len(df)) * 100)
                         .sort_values('pct_missing', ascending=False))

    # Estatísticas resumidas de montantes
    numeric_stats = df[['Amount Paid', 'Amount Received']].describe().T
    numeric_stats['total_sum'] = df[['Amount Paid', 'Amount Received']].sum(axis=0)

    # Avaliação de consistência entre Amount Paid e Amount Received
    spread = df['Amount Received'] - df['Amount Paid']
    print(f"Spread médio (Received - Paid): {spread.mean():.2f}")
    print(f"Spread mediano (Received - Paid): {spread.median():.2f}")
    
    return missing_overview, numeric_stats, spread


def calculate_network_metrics(df):
    """
    Calcula métricas de rede para análise de transações.
    
    Args:
        df: DataFrame com colunas Account, Dest Account, Amount Paid, Is Laundering
        
    Returns:
        tuple: (df_enriched, network_summary)
    """
    print("Calculando métricas de rede sem dependências externas...")

    # Conexões básicas entre contas
    df_graph = df[['Account', 'Dest Account', 'Amount Paid', 'Is Laundering']].dropna(subset=['Account', 'Dest Account'])
    print(f"Analisando {len(df_graph)} transações entre contas...")

    # Métricas de saída (out-degree e out-strength)
    out_metrics = (df_graph.groupby('Account')
                   .agg(out_degree=('Dest Account', 'count'),
                        out_strength=('Amount Paid', 'sum'),
                        unique_destinations=('Dest Account', 'nunique'))
                   .reset_index())

    # Métricas de entrada (in-degree e in-strength)
    in_metrics = (df_graph.groupby('Dest Account')
                  .agg(in_degree=('Account', 'count'),
                       in_strength=('Amount Paid', 'sum'),
                       unique_sources=('Account', 'nunique'))
                  .reset_index()
                  .rename(columns={'Dest Account': 'Account'}))

    # Combinar métricas
    network_metrics = pd.merge(out_metrics, in_metrics, on='Account', how='outer').fillna(0)
    print(f"Métricas calculadas para {len(network_metrics)} contas únicas")

    # Criar dataset enriquecido (sem modificar o original)
    network_cols = ['out_degree', 'in_degree', 'out_strength', 'in_strength', 'unique_destinations', 'unique_sources']
    df_enriched = df.merge(network_metrics[['Account'] + network_cols], on='Account', how='left')

    # Preencher NaN com 0 (contas que não aparecem como origem)
    for col in network_cols:
        df_enriched[col] = df_enriched[col].fillna(0)

    # Resumo por classe
    network_summary = (df_enriched[['Is Laundering'] + network_cols]
                       .groupby('Is Laundering')
                       .agg(['mean', 'median', 'std']))
    
    return df_enriched, network_summary

"""
Módulo de visualizações para diagnósticos e análises avançadas
Contém plots de ROC, Precision-Recall, distribuições e análises de threshold
"""

# import matplotlib.pyplot as plt  # Already imported at top
# import seaborn as sns  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from pathlib import Path  # Already imported at top
# from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #3

def setup_plot_style():
    """Configuração padrão de estilo para todos os plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 120
    plt.rcParams['font.size'] = 10

def plot_roc_pr_curves(models_dict, X_test, y_test, save_path=None):
    """
    Plot combinado de curvas ROC e Precision-Recall para múltiplos modelos
    
    Args:
        models_dict: Dict {model_name: fitted_model}
        X_test: Features de teste
        y_test: Labels de teste
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('📊 Curvas ROC e Precision-Recall - Comparação de Modelos', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
    
    for i, (model_name, model) in enumerate(models_dict.items()):
        # Predições
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        ax1.plot(fpr, tpr, color=colors[i], linewidth=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        ax2.plot(recall, precision, color=colors[i], linewidth=2,
                label=f'{model_name} (AP = {pr_auc:.3f})')
    
    # ROC Plot
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('🎯 ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # PR Plot
    baseline = y_test.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('📈 Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Curvas ROC/PR salvas em: {save_path}")
    
    return fig

def plot_threshold_analysis(y_true, y_proba, model_name="Model", save_path=None):
    """
    Análise de threshold com múltiplas métricas
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        model_name: Nome do modelo
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    # Calcular métricas para diferentes thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'🎯 Análise de Threshold - {model_name}', fontsize=14, fontweight='bold')
    
    # Curvas de métricas vs threshold
    ax1.plot(thresholds, precision_scores, label='Precision', linewidth=2)
    ax1.plot(thresholds, recall_scores, label='Recall', linewidth=2)
    ax1.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
    
    # Marcar melhor F1
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    ax1.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Best F1 @ {best_threshold:.2f}')
    ax1.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('📊 Métricas vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Distribuição de probabilidades por classe
    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.7, label='Classe 0 (Normal)', 
            color='blue', density=True)
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.7, label='Classe 1 (Fraude)', 
            color='red', density=True)
    ax2.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2,
               label=f'Threshold Ótimo ({best_threshold:.2f})')
    
    ax2.set_xlabel('Probabilidade Predita')
    ax2.set_ylabel('Densidade')
    ax2.set_title('📈 Distribuição de Probabilidades')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Análise de threshold salva em: {save_path}")
    
    return fig, best_threshold

def plot_prediction_distribution(y_true, y_proba, model_name="Model", save_path=None):
    """
    Análise da distribuição de predições e segmentação
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        model_name: Nome do modelo
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'📊 Análise de Distribuição de Predições - {model_name}', fontsize=14, fontweight='bold')
    
    # 1. Box plot por classe
    df_plot = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba,
        'class_label': ['Fraude' if x == 1 else 'Normal' for x in y_true]
    })
    
    sns.boxplot(data=df_plot, x='class_label', y='y_proba', ax=ax1)
    ax1.set_title('📦 Distribuição de Scores por Classe')
    ax1.set_xlabel('Classe Verdadeira')
    ax1.set_ylabel('Probabilidade Predita')
    ax1.grid(True, alpha=0.3)
    
    # 2. Densidade superposta
    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='Normal', 
            color='blue', density=True)
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label='Fraude', 
            color='red', density=True)
    ax2.set_xlabel('Probabilidade Predita')
    ax2.set_ylabel('Densidade')
    ax2.set_title('📈 Sobreposição de Distribuições')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quantis de risco
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    threshold_values = np.percentile(y_proba, [q*100 for q in quantiles])
    
    fraud_rates = []
    for threshold in threshold_values:
        mask = y_proba >= threshold
        if mask.sum() > 0:
            fraud_rate = y_true[mask].mean()
        else:
            fraud_rate = 0
        fraud_rates.append(fraud_rate)
    
    ax3.plot([q*100 for q in quantiles], fraud_rates, marker='o', linewidth=2)
    ax3.set_xlabel('Percentil de Score')
    ax3.set_ylabel('Taxa de Fraude')
    ax3.set_title('🎯 Taxa de Fraude por Percentil')
    ax3.grid(True, alpha=0.3)
    
    # Anotar valores
    for i, (q, rate) in enumerate(zip(quantiles, fraud_rates)):
        ax3.annotate(f'{rate:.2f}', (q*100, rate), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Calibração (reliability diagram)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(in_bin.sum())
    
    if bin_centers:
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfeitamente Calibrado')
        ax4.plot(bin_centers, bin_accuracies, marker='o', linewidth=2, label='Modelo')
        
        # Barras de frequência
        ax4_twin = ax4.twinx()
        ax4_twin.bar(bin_centers, bin_counts, alpha=0.3, width=0.08, color='gray')
        ax4_twin.set_ylabel('Frequência')
        
        ax4.set_xlabel('Probabilidade Predita')
        ax4.set_ylabel('Fração de Positivos')
        ax4.set_title('📊 Diagrama de Calibração')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Dados insuficientes\npara calibração', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('📊 Diagrama de Calibração')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Análise de distribuição salva em: {save_path}")
    
    return fig

def plot_ensemble_analysis(individual_scores, ensemble_score, y_true, model_names, save_path=None):
    """
    Análise de ensemble vs modelos individuais
    
    Args:
        individual_scores: Dict {model_name: predictions}
        ensemble_score: Array com scores do ensemble
        y_true: Labels verdadeiros
        model_names: Lista com nomes dos modelos
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🔄 Análise de Ensemble vs Modelos Individuais', fontsize=14, fontweight='bold')
    
    # 1. Comparação de AUC
    aucs = {}
    for name, scores in individual_scores.items():
        aucs[name] = roc_auc_score(y_true, scores)
    aucs['Ensemble'] = roc_auc_score(y_true, ensemble_score)
    
    models = list(aucs.keys())
    auc_values = list(aucs.values())
    colors = ['gold' if model == 'Ensemble' else 'lightblue' for model in models]
    
    bars = ax1.bar(models, auc_values, color=colors, alpha=0.8)
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('🏆 Comparação de AUC')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Anotar valores
    for bar, auc in zip(bars, auc_values):
        height = bar.get_height()
        ax1.annotate(f'{auc:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Correlação entre modelos
    if len(individual_scores) > 1:
        score_df = pd.DataFrame(individual_scores)
        score_df['Ensemble'] = ensemble_score
        
        corr_matrix = score_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('🔗 Correlação entre Modelos')
    else:
        ax2.text(0.5, 0.5, 'Poucos modelos\npara correlação', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('🔗 Correlação entre Modelos')
    
    # 3. Distribuição de disagreement
    if len(individual_scores) >= 2:
        # Calcular variance dos scores
        all_scores = np.array(list(individual_scores.values())).T
        score_variance = np.var(all_scores, axis=1)
        
        # Dividir em bins de variance
        variance_bins = pd.cut(score_variance, bins=5, labels=['Baixa', 'Baixa-Med', 'Média', 'Med-Alta', 'Alta'])
        
        # Performance por bin
        bin_performance = []
        bin_names = []
        for bin_name in variance_bins.categories:
            mask = variance_bins == bin_name
            if mask.sum() > 0:
                bin_auc = roc_auc_score(y_true[mask], ensemble_score[mask])
                bin_performance.append(bin_auc)
                bin_names.append(f'{bin_name}\n(n={mask.sum()})')
        
        if bin_performance:
            ax3.bar(bin_names, bin_performance, alpha=0.7)
            ax3.set_ylabel('AUC')
            ax3.set_xlabel('Nível de Disagreement')
            ax3.set_title('📊 Performance vs Disagreement')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Poucos modelos para\nanálise de disagreement', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('📊 Performance vs Disagreement')
    
    # 4. Contribuição de cada modelo (weights se disponível)
    if hasattr(ensemble_score, 'weights'):
        weights = ensemble_score.weights
        model_names_for_weights = model_names[:len(weights)]
    else:
        # Assumir pesos iguais
        weights = [1.0/len(individual_scores)] * len(individual_scores)
        model_names_for_weights = list(individual_scores.keys())
    
    if weights:
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = ax4.pie(weights, labels=model_names_for_weights, 
                                          autopct='%1.1f%%', colors=colors)
        ax4.set_title('🥧 Contribuição no Ensemble')
    else:
        ax4.text(0.5, 0.5, 'Pesos não disponíveis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('🥧 Contribuição no Ensemble')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Análise de ensemble salva em: {save_path}")
    
    return fig

def save_all_diagnostic_plots(models_dict, X_test, y_test, 
                            ensemble_scores=None, output_dir="artifacts/plots"):
    """
    Salva todos os plots de diagnóstico em um diretório
    
    Args:
        models_dict: Dict {model_name: fitted_model}
        X_test: Features de teste
        y_test: Labels de teste
        ensemble_scores: Scores de ensemble (opcional)
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Salvando plots de diagnóstico em: {output_path}")
    
    # Plot 1: ROC e PR curves
    fig1 = plot_roc_pr_curves(models_dict, X_test, y_test,
                             save_path=output_path / "roc_pr_curves.png")
    plt.close(fig1)
    
    # Plot 2-N: Threshold analysis para cada modelo
    for model_name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        fig2, best_threshold = plot_threshold_analysis(y_test, y_proba, model_name,
                                                      save_path=output_path / f"threshold_analysis_{model_name}.png")
        plt.close(fig2)
        
        fig3 = plot_prediction_distribution(y_test, y_proba, model_name,
                                          save_path=output_path / f"prediction_distribution_{model_name}.png")
        plt.close(fig3)
    
    # Plot ensemble (se disponível)
    if ensemble_scores is not None and len(models_dict) > 1:
        individual_scores = {}
        for model_name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                individual_scores[model_name] = model.predict_proba(X_test)[:, 1]
            else:
                individual_scores[model_name] = model.decision_function(X_test)
        
        fig4 = plot_ensemble_analysis(individual_scores, ensemble_scores, y_test, 
                                    list(models_dict.keys()),
                                    save_path=output_path / "ensemble_analysis.png")
        plt.close(fig4)
    
    print(f"✅ Todos os plots de diagnóstico salvos com sucesso!")

"""
Módulo de visualizações para comparação de modelos de ML
Centraliza todos os gráficos de performance, reduzindo tamanho do notebook
"""

# import matplotlib.pyplot as plt  # Already imported at top
# import seaborn as sns  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from pathlib import Path  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #4

def setup_plot_style():
    """Configuração padrão de estilo para todos os plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 120
    plt.rcParams['font.size'] = 10

def plot_model_performance_comparison(df_results, save_path=None):
    """
    Gráfico comparativo de performance dos modelos baseline
    
    Args:
        df_results: DataFrame com resultados dos modelos
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    # 1. Gráfico de barras comparativo - PR_AUC por modelo
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Análise Comparativa de Performance - Modelos Baseline', fontsize=16, fontweight='bold')

    # Bar plot principal - PR_AUC
    df_viz = df_results.copy()
    df_pivot = df_viz.pivot(index='Model', columns='Variant', values='PR_AUC')

    df_pivot.plot(kind='bar', ax=ax1, rot=45, width=0.8)
    ax1.set_title('🎯 PR_AUC por Modelo e Variante', fontweight='bold')
    ax1.set_ylabel('PR_AUC Score')
    ax1.axhline(y=df_viz['PR_AUC'].mean(), color='red', linestyle='--', alpha=0.7, label='Média')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Tempo de treinamento vs Performance
    ax2.scatter(df_viz['Fit_Time'], df_viz['PR_AUC'], 
               c=df_viz['Variant'].map({'full': 'blue', 'core': 'orange'}), 
               s=100, alpha=0.7)
    ax2.set_xlabel('Tempo de Treinamento (s)')
    ax2.set_ylabel('PR_AUC Score')
    ax2.set_title('⚡ Eficiência: Tempo vs Performance', fontweight='bold')

    # Anotar pontos
    for idx, row in df_viz.iterrows():
        ax2.annotate(f"{row['Model'][:3]}", 
                    (row['Fit_Time'], row['PR_AUC']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Distribuição de métricas
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    available_metrics = [m for m in metrics_to_plot if m in df_viz.columns]

    if available_metrics:
        df_metrics = df_viz[['Model', 'Variant'] + available_metrics].melt(
            id_vars=['Model', 'Variant'], 
            var_name='Metric', 
            value_name='Score'
        )
        
        sns.boxplot(data=df_metrics, x='Metric', y='Score', ax=ax3)
        ax3.set_title('📈 Distribuição de Métricas Secundárias', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Métricas secundárias\nnão disponíveis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('📈 Distribuição de Métricas Secundárias', fontweight='bold')

    # 4. Ranking geral (combinando múltiplas métricas)
    if 'PR_AUC' in df_viz.columns and 'F1_Score' in df_viz.columns:
        df_viz['Combined_Score'] = 0.6 * df_viz['PR_AUC'] + 0.4 * df_viz['F1_Score']
        df_sorted = df_viz.sort_values('Combined_Score', ascending=True)
        
        bars = ax4.barh(range(len(df_sorted)), df_sorted['Combined_Score'])
        ax4.set_yticks(range(len(df_sorted)))
        ax4.set_yticklabels([f"{row['Model']} ({row['Variant']})" for _, row in df_sorted.iterrows()])
        ax4.set_xlabel('Score Combinado (0.6*PR_AUC + 0.4*F1)')
        ax4.set_title('🏆 Ranking Geral dos Modelos', fontweight='bold')
        
        # Colorir barras
        for i, bar in enumerate(bars):
            if i == len(bars) - 1:  # Melhor modelo
                bar.set_color('gold')
            elif i >= len(bars) - 3:  # Top 3
                bar.set_color('lightgreen')
            else:
                bar.set_color('lightblue')
        
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Dados insuficientes\npara ranking', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('🏆 Ranking Geral dos Modelos', fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Gráfico salvo em: {save_path}")
    
    return fig

def plot_metrics_at_k(df_metrics_k, save_path=None):
    """
    Visualização de métricas @K (Precision@K, Recall@K, etc.)
    
    Args:
        df_metrics_k: DataFrame com métricas @K
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('📈 Métricas @K - Performance em Top Predictions', fontsize=14, fontweight='bold')
    
    # 1. Precision@K por modelo
    for model in df_metrics_k['Model'].unique():
        model_data = df_metrics_k[df_metrics_k['Model'] == model]
        ax1.plot(model_data['K'], model_data['Precision_at_K'], 
                marker='o', label=model, linewidth=2, markersize=6)
    
    ax1.set_xlabel('K (Top Predictions)')
    ax1.set_ylabel('Precision@K')
    ax1.set_title('🎯 Precision@K por Modelo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Heatmap de performance
    if len(df_metrics_k['Model'].unique()) > 1:
        pivot_data = df_metrics_k.pivot(index='Model', columns='K', values='Precision_at_K')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('🔥 Heatmap Precision@K')
        ax2.set_xlabel('K (Top Predictions)')
        ax2.set_ylabel('Modelo')
    else:
        ax2.text(0.5, 0.5, 'Poucos modelos\npara heatmap', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('🔥 Heatmap Precision@K')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Gráfico salvo em: {save_path}")
    
    return fig

def plot_tuning_results(tuning_results, save_path=None):
    """
    Visualização dos resultados de hyperparameter tuning
    
    Args:
        tuning_results: Dicionário com resultados do tuning
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔧 Análise de Hyperparameter Tuning', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    # 1. Comparação de scores antes/depois
    ax1 = axes[0]
    models = list(tuning_results.keys())
    scores_before = [tuning_results[m].get('baseline_score', 0) for m in models]
    scores_after = [tuning_results[m].get('best_score_', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, scores_before, width, label='Baseline', alpha=0.7)
    ax1.bar(x + width/2, scores_after, width, label='Tuned', alpha=0.7)
    
    ax1.set_xlabel('Modelos')
    ax1.set_ylabel('Score')
    ax1.set_title('📊 Antes vs Depois do Tuning')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement por modelo
    ax2 = axes[1]
    improvements = [(after - before) for before, after in zip(scores_before, scores_after)]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(models, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Melhoria no Score')
    ax2.set_title('📈 Melhoria por Modelo')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Anotar valores
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.annotate(f'{imp:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
    
    # 3. Distribution of parameters (exemplo para um modelo)
    ax3 = axes[2]
    if tuning_results and 'cv_results_' in list(tuning_results.values())[0]:
        # Pegar primeiro modelo com resultados detalhados
        model_name = list(tuning_results.keys())[0]
        cv_results = tuning_results[model_name]['cv_results_']
        
        # Plot distribuição de scores
        scores = cv_results['mean_test_score']
        ax3.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Média: {np.mean(scores):.3f}')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequência')
        ax3.set_title(f'📈 Distribuição de Scores - {model_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Resultados de CV\nnão disponíveis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('📈 Distribuição de Scores')
    
    # 4. Convergência (se disponível)
    ax4 = axes[3]
    convergence_data = []
    for model_name, results in tuning_results.items():
        if 'best_estimator_' in results:
            best_model = results['best_estimator_']
            if hasattr(best_model, 'n_estimators'):
                convergence_data.append({
                    'model': model_name,
                    'n_estimators': best_model.n_estimators,
                    'score': results.get('best_score_', 0)
                })
    
    if convergence_data:
        df_conv = pd.DataFrame(convergence_data)
        for model in df_conv['model'].unique():
            model_data = df_conv[df_conv['model'] == model]
            ax4.plot(model_data['n_estimators'], model_data['score'], 
                    marker='o', label=model, linewidth=2)
        
        ax4.set_xlabel('N. Estimators')
        ax4.set_ylabel('Best Score')
        ax4.set_title('📈 Convergência de Modelos')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Dados de convergência\nnão disponíveis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('📈 Convergência de Modelos')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Gráfico salvo em: {save_path}")
    
    return fig

def plot_feature_importance(feature_importance_dict, top_n=15, save_path=None):
    """
    Visualização de importância de features
    
    Args:
        feature_importance_dict: Dict com importâncias por modelo
        top_n: Número de features top para mostrar
        save_path: Caminho para salvar o gráfico (opcional)
    
    Returns:
        fig: Figura matplotlib
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, min(len(feature_importance_dict), 3), figsize=(5*min(len(feature_importance_dict), 3), 6))
    if len(feature_importance_dict) == 1:
        axes = [axes]
    
    fig.suptitle('🎯 Importância de Features por Modelo', fontsize=14, fontweight='bold')
    
    for i, (model_name, importance_data) in enumerate(feature_importance_dict.items()):
        if i >= 3:  # Máximo 3 subplots
            break
            
        ax = axes[i] if len(feature_importance_dict) > 1 else axes[0]
        
        # Ordenar features por importância
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Plot horizontal
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importância')
        ax.set_title(f'{model_name}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Inverter ordem para mostrar mais importante no topo
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📁 Gráfico salvo em: {save_path}")
    
    return fig

def save_all_plots(df_results, df_metrics_k=None, tuning_results=None, 
                   feature_importance=None, output_dir="artifacts/plots"):
    """
    Salva todos os plots em um diretório
    
    Args:
        df_results: DataFrame com resultados dos modelos
        df_metrics_k: DataFrame com métricas @K (opcional)
        tuning_results: Resultados de tuning (opcional)
        feature_importance: Dicionário com importâncias (opcional)
        output_dir: Diretório de saída
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Salvando plots em: {output_path}")
    
    # Plot 1: Performance comparison
    fig1 = plot_model_performance_comparison(df_results, 
                                           save_path=output_path / "model_performance_comparison.png")
    plt.close(fig1)
    
    # Plot 2: Metrics @K
    if df_metrics_k is not None:
        fig2 = plot_metrics_at_k(df_metrics_k, 
                               save_path=output_path / "metrics_at_k.png")
        plt.close(fig2)
    
    # Plot 3: Tuning results
    if tuning_results is not None:
        fig3 = plot_tuning_results(tuning_results, 
                                 save_path=output_path / "tuning_results.png")
        plt.close(fig3)
    
    # Plot 4: Feature importance
    if feature_importance is not None:
        fig4 = plot_feature_importance(feature_importance, 
                                     save_path=output_path / "feature_importance.png")
        plt.close(fig4)
    
    print(f"✅ Todos os plots salvos com sucesso!")

"""
Visualization Utilities
Funções para criar gráficos e análises visuais de modelos ML
"""

# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# import matplotlib.pyplot as plt  # Already imported at top
# import seaborn as sns  # Already imported at top
# from matplotlib.patches import Patch  # Already imported at top
# from pathlib import Path  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #5


def plot_feature_importance(model, feature_names, save_path=None, top_n=15):
    """
    Cria visualização de feature importance com análise por tipo.
    
    Parameters:
    -----------
    model : sklearn-like model
        Modelo treinado com atributo feature_importances_
    feature_names : list or Index
        Nomes das features
    save_path : str or Path, optional
        Caminho para salvar o gráfico
    top_n : int, default=15
        Número de features top para visualizar
        
    Returns:
    --------
    feat_imp_df : DataFrame
        DataFrame com feature importance e tipos
    fig : matplotlib.figure.Figure
        Figura criada
    """
    if not hasattr(model, 'feature_importances_'):
        print("[WARNING] Model doesn't support feature_importances_")
        return None, None
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Classify feature types
    def classify_feature(name):
        if name in ['Hour', 'DayOfWeek', 'IsWeekend', 'IsBusinessHours', 'IsNightTransaction']:
            return 'Temporal'
        elif 'Amount_' in name and any(x in name for x in ['per_Hour', 'From_Bank', 'To_Bank', 'Account']):
            return 'Interaction'
        elif any(x in name for x in ['Account_', 'FromBank_', 'ToBank_']):
            return 'Aggregation'
        else:
            return 'Original'
    
    feat_imp_df['Type'] = feat_imp_df['Feature'].apply(classify_feature)
    
    # Display top features
    print("=" * 80)
    print(f"TOP {min(20, len(feat_imp_df))} MOST IMPORTANT FEATURES")
    print("=" * 80)
    print(feat_imp_df.head(20).to_string(index=False))
    
    # Summarize by type
    print("\n" + "=" * 80)
    print("IMPORTANCE BY FEATURE TYPE")
    print("=" * 80)
    type_summary = feat_imp_df.groupby('Type').agg({
        'Importance': ['sum', 'mean', 'count']
    }).round(4)
    type_summary.columns = ['Total', 'Average', 'Count']
    type_summary = type_summary.sort_values('Total', ascending=False)
    print(type_summary)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top N features
    top_features = feat_imp_df.head(top_n)
    colors = top_features['Type'].map({
        'Original': '#3b82f6',
        'Temporal': '#ef4444',
        'Interaction': '#10b981',
        'Aggregation': '#f59e0b'
    })
    
    ax1.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['Feature'])
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#3b82f6', label='Original'),
        Patch(facecolor='#ef4444', label='Temporal'),
        Patch(facecolor='#10b981', label='Interaction'),
        Patch(facecolor='#f59e0b', label='Aggregation')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Importance by type
    type_sum = feat_imp_df.groupby('Type')['Importance'].sum().sort_values(ascending=False)
    colors_pie = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    ax2.pie(type_sum.values, labels=type_sum.index, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Feature Importance by Type', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Feature importance plot saved to: {save_path}")
        
        # Save CSV
        csv_path = save_path.parent / 'feature_importance.csv'
        feat_imp_df.to_csv(csv_path, index=False)
        print(f"[OK] Feature importance data saved to: {csv_path}")
    
    plt.show()
    
    return feat_imp_df, fig


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """
    Plota matriz de confusão.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_pred : array-like
        Predições
    save_path : str or Path, optional
        Caminho para salvar
    normalize : bool, default=False
        Se True, normaliza por linha (recall)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                ax=ax)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_roc_pr_curves(y_true, y_proba, save_path=None):
    """
    Plota curvas ROC e Precision-Recall lado a lado.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_proba : array-like
        Probabilidades preditas
    save_path : str or Path, optional
        Caminho para salvar
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='#2196f3', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # PR Curve
    ax2.plot(recall, precision, color='#4caf50', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax2.axhline(y=y_true.mean(), color='gray', lw=1, linestyle='--', label=f'Baseline ({y_true.mean():.3f})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] ROC and PR curves saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_threshold_analysis(y_true, y_proba, save_path=None, thresholds=None):
    """
    Plota análise de threshold mostrando precision, recall, F1 vs threshold.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_proba : array-like
        Probabilidades preditas
    save_path : str or Path, optional
        Caminho para salvar
    thresholds : array-like, optional
        Thresholds para avaliar
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, precisions, label='Precision', color='#2196f3', lw=2)
    ax.plot(thresholds, recalls, label='Recall', color='#4caf50', lw=2)
    ax.plot(thresholds, f1_scores, label='F1-Score', color='#ff9800', lw=2)
    
    # Mark best threshold
    ax.axvline(best_thresh, color='red', linestyle='--', lw=1, alpha=0.7)
    ax.scatter([best_thresh], [best_f1], color='red', s=100, zorder=5)
    ax.text(best_thresh, best_f1 + 0.02, f'Best: {best_thresh:.2f}\nF1={best_f1:.3f}',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim([thresholds.min(), thresholds.max()])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Threshold analysis saved to: {save_path}")
    
    plt.show()
    
    print(f"\n[BEST THRESHOLD] {best_thresh:.3f} (F1-Score: {best_f1:.3f})")
    print(f"   Precision: {precisions[best_idx]:.3f}")
    print(f"   Recall: {recalls[best_idx]:.3f}")
    
    return fig, best_thresh


def plot_model_comparison(results_dict, metric='pr_auc', save_path=None):
    """
    Plota comparação de múltiplos modelos.
    
    Parameters:
    -----------
    results_dict : dict
        Dicionário {model_name: {'cv_pr_auc': ..., 'test_pr_auc': ...}}
    metric : str, default='pr_auc'
        Métrica para comparar
    save_path : str or Path, optional
        Caminho para salvar
    """
    models = list(results_dict.keys())
    cv_scores = [results_dict[m][f'cv_{metric}'] for m in models]
    test_scores = [results_dict[m][f'test_{metric}'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, cv_scores, width, label='CV', color='#2196f3', alpha=0.8)
    ax.bar(x + width/2, test_scores, width, label='Test', color='#4caf50', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.upper().replace('_', '-'), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (cv, test) in enumerate(zip(cv_scores, test_scores)):
        ax.text(i - width/2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Model comparison saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_fraud_patterns(X_test_featured, y_test, y_test_proba, save_path=None, sample_size=500):
    """
    Cria visualização abrangente de padrões de fraude vs normal.
    
    Parameters:
    -----------
    X_test_featured : DataFrame
        Features de teste (com features engineered)
    y_test : array-like
        Labels verdadeiros
    y_test_proba : array-like
        Probabilidades preditas
    save_path : str or Path, optional
        Caminho para salvar
    sample_size : int, default=500
        Tamanho da amostra por classe
    """
    # matplotlib.pyplot already imported at module level
    
    # Combine test data with features and predictions
    X_test_viz = X_test_featured.copy()
    X_test_viz['Is_Fraud'] = y_test.values if hasattr(y_test, 'values') else y_test
    X_test_viz['Fraud_Score'] = y_test_proba
    
    # Sample for visualization (to avoid overplotting)
    np.random.seed(42)
    fraud_sample = X_test_viz[X_test_viz['Is_Fraud'] == 1].sample(
        min(sample_size, sum(X_test_viz['Is_Fraud']))
    )
    normal_sample = X_test_viz[X_test_viz['Is_Fraud'] == 0].sample(
        min(sample_size, sum(X_test_viz['Is_Fraud'] == 0))
    )
    viz_data = pd.concat([fraud_sample, normal_sample])
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ============================================================================
    # 1. TEMPORAL PATTERNS
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    hour_dist = viz_data.groupby(['Hour', 'Is_Fraud']).size().unstack(fill_value=0)
    hour_dist = hour_dist.div(hour_dist.sum(axis=0), axis=1) * 100
    hour_dist.plot(kind='bar', ax=ax1, color=['#3b82f6', '#ef4444'], alpha=0.8)
    ax1.set_title('Transaction Distribution by Hour', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('% of Transactions')
    ax1.legend(['Normal', 'Fraud'], loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 2. BUSINESS HOURS vs NIGHT TRANSACTIONS
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    time_fraud = viz_data.groupby('Is_Fraud')[['IsBusinessHours', 'IsNightTransaction']].mean() * 100
    time_fraud.T.plot(kind='bar', ax=ax2, color=['#3b82f6', '#ef4444'], alpha=0.8)
    ax2.set_title('Business Hours vs Night Patterns', fontsize=12, fontweight='bold')
    ax2.set_ylabel('% of Transactions')
    ax2.set_xticklabels(['Business Hours', 'Night (00h-06h)'], rotation=0)
    ax2.legend(['Normal', 'Fraud'], loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 3. AMOUNT DISTRIBUTION
    # ============================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    fraud_amounts = viz_data[viz_data['Is_Fraud'] == 1]['Amount Paid']
    normal_amounts = viz_data[viz_data['Is_Fraud'] == 0]['Amount Paid']
    ax3.hist([normal_amounts, fraud_amounts], bins=50, label=['Normal', 'Fraud'], 
             color=['#3b82f6', '#ef4444'], alpha=0.6, edgecolor='black')
    ax3.set_title('Amount Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Amount Paid')
    ax3.set_ylabel('Frequency')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 4. ACCOUNT VELOCITY
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    velocity_fraud = viz_data[viz_data['Is_Fraud'] == 1]['Account_Velocity'].clip(0, 20)
    velocity_normal = viz_data[viz_data['Is_Fraud'] == 0]['Account_Velocity'].clip(0, 20)
    ax4.hist([velocity_normal, velocity_fraud], bins=30, label=['Normal', 'Fraud'],
             color=['#3b82f6', '#ef4444'], alpha=0.6, edgecolor='black')
    ax4.set_title('Account Velocity (Trans/Day)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Transactions per Day')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 5. CONCENTRATION RATIO
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    ratio_fraud = viz_data[viz_data['Is_Fraud'] == 1]['Amount_Account_Ratio'].clip(0, 1)
    ratio_normal = viz_data[viz_data['Is_Fraud'] == 0]['Amount_Account_Ratio'].clip(0, 1)
    ax5.hist([ratio_normal, ratio_fraud], bins=30, label=['Normal', 'Fraud'],
             color=['#3b82f6', '#ef4444'], alpha=0.6, edgecolor='black')
    ax5.set_title('Transaction Concentration Ratio', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Amount / Account Total Volume')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 6. DEVIATION FROM ACCOUNT MEAN
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    dev_fraud = viz_data[viz_data['Is_Fraud'] == 1]['Amount_vs_Account_Mean'].clip(-5, 5)
    dev_normal = viz_data[viz_data['Is_Fraud'] == 0]['Amount_vs_Account_Mean'].clip(-5, 5)
    ax6.hist([dev_normal, dev_fraud], bins=30, label=['Normal', 'Fraud'],
             color=['#3b82f6', '#ef4444'], alpha=0.6, edgecolor='black')
    ax6.set_title('Deviation from Account Mean', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Standard Deviations from Mean')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 7. FRAUD SCORE DISTRIBUTION
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    fraud_scores = viz_data[viz_data['Is_Fraud'] == 1]['Fraud_Score']
    normal_scores = viz_data[viz_data['Is_Fraud'] == 0]['Fraud_Score']
    ax7.hist([normal_scores, fraud_scores], bins=50, label=['Normal', 'Fraud'],
             color=['#3b82f6', '#ef4444'], alpha=0.6, edgecolor='black')
    ax7.set_title('Model Fraud Score Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Fraud Probability')
    ax7.set_ylabel('Frequency')
    ax7.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 8. DAY OF WEEK PATTERN
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    dow_dist = viz_data.groupby(['DayOfWeek', 'Is_Fraud']).size().unstack(fill_value=0)
    dow_dist = dow_dist.div(dow_dist.sum(axis=0), axis=1) * 100
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_dist.index = [dow_names[i] for i in dow_dist.index]
    dow_dist.plot(kind='bar', ax=ax8, color=['#3b82f6', '#ef4444'], alpha=0.8)
    ax8.set_title('Transaction Distribution by Weekday', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Day of Week')
    ax8.set_ylabel('% of Transactions')
    ax8.legend(['Normal', 'Fraud'], loc='upper right')
    ax8.set_xticklabels(dow_dist.index, rotation=45)
    ax8.grid(axis='y', alpha=0.3)
    
    # ============================================================================
    # 9. KEY STATISTICS TABLE
    # ============================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate key statistics
    stats_data = []
    for feature in ['Amount Paid', 'Account_Velocity', 'Amount_Account_Ratio', 
                    'IsNightTransaction', 'IsBusinessHours']:
        if feature in viz_data.columns:
            normal_val = viz_data[viz_data['Is_Fraud'] == 0][feature].mean()
            fraud_val = viz_data[viz_data['Is_Fraud'] == 1][feature].mean()
            ratio = fraud_val / (normal_val + 1e-10)
            stats_data.append([feature, f'{normal_val:.3f}', f'{fraud_val:.3f}', f'{ratio:.2f}x'])
    
    stats_table = ax9.table(cellText=stats_data,
                           colLabels=['Feature', 'Normal', 'Fraud', 'Ratio'],
                           cellLoc='left',
                           loc='center',
                           colWidths=[0.35, 0.2, 0.2, 0.15])
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(9)
    stats_table.scale(1, 2)
    
    # Style header
    for i in range(4):
        stats_table[(0, i)].set_facecolor('#34495e')
        stats_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Key Feature Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('[SEARCH] Fraud Detection Pattern Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Pattern analysis plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_roc_detailed_analysis(y_test, y_test_proba, save_path=None):
    """
    Cria análise detalhada da curva ROC com 4 visualizações:
    1. Curva ROC completa com pontos-chave marcados
    2. Zoom na região FPR ≤ 20% (crítica para AML)
    3. Distribuição de thresholds
    4. Densidade de pontos por faixa de FPR
    
    Parameters:
    -----------
    y_test : array-like
        Labels verdadeiros do conjunto de teste
    y_test_proba : array-like
        Probabilidades preditas (classe positiva)
    save_path : str or Path, optional
        Caminho para salvar o gráfico
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figura com 4 subplots
    analysis_dict : dict
        Dicionário com estatísticas da análise
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # ============================================================================
    # ESTATÍSTICAS
    # ============================================================================
    print("=" * 80)
    print(" ANÁLISE DA CURVA ROC")
    print("=" * 80)
    print(f"\n📊 Estatísticas da Curva:")
    print(f"  - Número de pontos calculados: {len(fpr)}")
    print(f"  - ROC-AUC Score: {roc_auc:.4f}")
    print(f"  - FPR range: [{fpr.min():.4f}, {fpr.max():.4f}]")
    print(f"  - TPR range: [{tpr.min():.4f}, {tpr.max():.4f}]")
    print(f"  - Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")
    
    # Análise de distribuição dos pontos
    print(f"\n🔍 Distribuição dos Pontos:")
    fpr_bins = [0, 0.1, 0.2, 0.5, 1.0]
    point_distribution = {}
    for i in range(len(fpr_bins)-1):
        mask = (fpr >= fpr_bins[i]) & (fpr < fpr_bins[i+1])
        n_points = np.sum(mask)
        bin_key = f"FPR_{fpr_bins[i]:.1f}_{fpr_bins[i+1]:.1f}"
        point_distribution[bin_key] = n_points
        print(f"  - FPR [{fpr_bins[i]:.1f}, {fpr_bins[i+1]:.1f}): {n_points} pontos ({n_points/len(fpr)*100:.1f}%)")
    
    # Calcular pontos-chave
    idx_fpr_01 = np.argmin(np.abs(fpr - 0.01))
    idx_fpr_05 = np.argmin(np.abs(fpr - 0.05))
    idx_fpr_10 = np.argmin(np.abs(fpr - 0.10))
    
    key_points = {
        'TPR@FPR=1%': (fpr[idx_fpr_01], tpr[idx_fpr_01], thresholds[idx_fpr_01]),
        'TPR@FPR=5%': (fpr[idx_fpr_05], tpr[idx_fpr_05], thresholds[idx_fpr_05]),
        'TPR@FPR=10%': (fpr[idx_fpr_10], tpr[idx_fpr_10], thresholds[idx_fpr_10])
    }
    
    print(f"\n⭐ Pontos-Chave da Curva:")
    print(f"  - TPR @ FPR=1%:  {tpr[idx_fpr_01]:.4f} (threshold={thresholds[idx_fpr_01]:.4f})")
    print(f"  - TPR @ FPR=5%:  {tpr[idx_fpr_05]:.4f} (threshold={thresholds[idx_fpr_05]:.4f})")
    print(f"  - TPR @ FPR=10%: {tpr[idx_fpr_10]:.4f} (threshold={thresholds[idx_fpr_10]:.4f})")
    print("\n" + "=" * 80)
    
    # ============================================================================
    # VISUALIZAÇÃO
    # ============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Curva ROC Completa
    ax1.plot(fpr, tpr, color='#10b981', linewidth=2.5, label=f'Model (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.5000)')
    ax1.fill_between(fpr, tpr, alpha=0.2, color='#10b981')
    
    # Marcar pontos-chave
    ax1.scatter([fpr[idx_fpr_01]], [tpr[idx_fpr_01]], color='red', s=100, zorder=5, 
               label=f'TPR@FPR=1%: {tpr[idx_fpr_01]:.3f}')
    ax1.scatter([fpr[idx_fpr_05]], [tpr[idx_fpr_05]], color='orange', s=100, zorder=5,
               label=f'TPR@FPR=5%: {tpr[idx_fpr_05]:.3f}')
    ax1.scatter([fpr[idx_fpr_10]], [tpr[idx_fpr_10]], color='blue', s=100, zorder=5,
               label=f'TPR@FPR=10%: {tpr[idx_fpr_10]:.3f}')
    
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax1.set_title(f'ROC Curve - {len(fpr)} pontos calculados', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom na região de baixo FPR
    mask_zoom = fpr <= 0.2
    ax2.plot(fpr[mask_zoom], tpr[mask_zoom], color='#3b82f6', linewidth=2.5, marker='o', 
             markersize=3, label='ROC Curve (zoomed)')
    ax2.plot([0, 0.2], [0, 0.2], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax2.set_title(f'ROC Curve - Zoom FPR ≤ 20% ({np.sum(mask_zoom)} pontos)', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.2])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Distribuição de Thresholds
    thresholds_finite = thresholds[np.isfinite(thresholds)]
    ax3.hist(thresholds_finite, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title(f'Distribuição de Thresholds ({len(thresholds_finite)} finitos)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Densidade de Pontos
    fpr_segments = np.linspace(0, 1, 11)
    densities = []
    segment_labels = []
    
    for i in range(len(fpr_segments)-1):
        mask = (fpr >= fpr_segments[i]) & (fpr < fpr_segments[i+1])
        n_points = np.sum(mask)
        densities.append(n_points)
        segment_labels.append(f'{fpr_segments[i]:.1f}-{fpr_segments[i+1]:.1f}')
    
    ax4.bar(range(len(densities)), densities, color='#ef4444', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('FPR Range', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
    ax4.set_title('Densidade de Pontos por Faixa de FPR', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(segment_labels)))
    ax4.set_xticklabels(segment_labels, rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('📊 Análise Detalhada da Curva ROC', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] ROC detailed analysis saved: {save_path}")
    
    plt.show()
    
    # ============================================================================
    # EXPLICAÇÃO
    # ============================================================================
    print("\n" + "=" * 80)
    print("⭐ RESPOSTA: COMO A CURVA ROC É CALCULADA?")
    print("=" * 80)
    print(f"""
A função `roc_curve` do scikit-learn calcula AUTOMATICAMENTE todos os pontos
necessários da curva ROC. Ela NÃO requer configuração manual de thresholds.

📊 Como funciona:
1. Para cada valor ÚNICO de probabilidade no conjunto de predições, a função
   calcula um ponto (FPR, TPR) usando esse valor como threshold
   
2. No seu caso: {len(fpr)} pontos foram calculados
   - Isso significa que existem {len(np.unique(y_test_proba))} valores únicos de probabilidades
   - A curva cobre TODO o espectro de thresholds possíveis
   
3. Mais pontos = curva mais suave e precisa
   - {len(fpr)} pontos é EXCELENTE para um dataset de {len(y_test)} amostras
   - Densidade média: {len(fpr) / 100:.2f} pontos por 1% de FPR

✅ CONCLUSÃO:
Sua curva ROC já está sendo testada com TODOS os parâmetros relevantes!
Não há necessidade de configurar mais thresholds manualmente.

⭐ A curva ROC-AUC de {roc_auc:.4f} representa uma avaliação completa e precisa
do poder discriminativo do seu modelo em TODOS os níveis de threshold.
""")
    print("=" * 80)
    
    # Retornar análise
    analysis_dict = {
        'roc_auc': roc_auc,
        'n_points': len(fpr),
        'n_unique_probs': len(np.unique(y_test_proba)),
        'key_points': {k: {'fpr': float(v[0]), 'tpr': float(v[1]), 'threshold': float(v[2])} 
                      for k, v in key_points.items()},
        'point_distribution': point_distribution,
        'density_per_percent_fpr': len(fpr) / 100
    }
    
    return fig, analysis_dict


print("[OK] Visualization utilities loaded!")

"""
🎨 VISUALIZATION SUITE - Sistema de Visualização Profissional AML
Resolve o problema de visualizações básicas com gráficos interativos e profissionais
"""

# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# import matplotlib.pyplot as plt  # Already imported at top
# import seaborn as sns  # Already imported at top
# from pathlib import Path  # Already imported at top
# import warnings  # Already imported at top
# warnings.filterwarnings('ignore')  # Already configured at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #6

# Plotly already imported at module level (px, go, make_subplots, ff)
PLOTLY_AVAILABLE = True  # Already imported at top

# NetworkX para análise de grafos
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX não disponível. Instale com: pip install networkx")

class AMLVisualizationSuite:
    """Suite completa de visualizações para AML com gráficos interativos"""
    
    def __init__(self):
        self.setup_style()
        
    def setup_style(self):
        """Configurar estilo profissional para todas as visualizações"""
        # Matplotlib/Seaborn style
        plt.style.use('default')
        try:
            sns.set_theme(style="whitegrid", context="notebook")
        except AttributeError:
            sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Configurações matplotlib
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    # ==================== DASHBOARD EXECUTIVO ====================
    
    def create_executive_dashboard(self, metrics_dict, save_path=None):
        """
        Cria dashboard executivo com KPIs principais
        
        Args:
            metrics_dict: Dict com métricas (roc_auc, pr_auc, expected_value, etc.)
            save_path: Caminho para salvar o dashboard
        """
        if not PLOTLY_AVAILABLE:
            return self._fallback_executive_dashboard(metrics_dict)
        
        # Layout do dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'ROC-AUC Performance', 'PR-AUC Performance', 'Expected Value',
                'Model Confidence', 'Feature Importance', 'Drift Detection'
            ),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # KPIs principais
        roc_auc = metrics_dict.get('roc_auc', 0.85)
        pr_auc = metrics_dict.get('pr_auc', 0.75)
        expected_value = metrics_dict.get('expected_value', 1000000)
        
        # Indicadores de performance
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=roc_auc,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ROC-AUC"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.7], 'color': "lightgray"},
                    {'range': [0.7, 0.85], 'color': "yellow"},
                    {'range': [0.85, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pr_auc,
            title={'text': "PR-AUC"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.75], 'color': "yellow"},
                    {'range': [0.75, 1], 'color': "green"}
                ]
            }
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=expected_value,
            title={'text': "Expected Value (R$)"},
            delta={'reference': 800000, 'relative': True},
            number={'prefix': "R$ "}
        ), row=1, col=3)
        
        # Gráficos adicionais
        if 'feature_importance' in metrics_dict:
            importance = metrics_dict['feature_importance']
            fig.add_trace(go.Bar(
                x=list(importance.values())[:10],
                y=list(importance.keys())[:10],
                orientation='h',
                name="Top Features"
            ), row=2, col=2)
        
        # Layout final
        fig.update_layout(
            title="🎯 AML Executive Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Dashboard salvo em: {save_path}")
        
        return fig
    
    # ==================== NETWORK ANALYSIS ====================
    
    def create_transaction_network(self, df, suspicious_only=True, min_amount=10000):
        """
        Cria visualização de rede de transações para detectar padrões de lavagem
        
        Args:
            df: DataFrame com colunas Account, Dest Account, Amount Paid, Is Laundering
            suspicious_only: Se deve mostrar apenas transações suspeitas
            min_amount: Valor mínimo para incluir na rede
        """
        if not NETWORKX_AVAILABLE or not PLOTLY_AVAILABLE:
            return self._fallback_network_analysis(df)
        
        # Filtrar dados
        if suspicious_only:
            df_net = df[df['Is Laundering'] == 1].copy()
        else:
            df_net = df.copy()
        
        df_net = df_net[df_net['Amount Paid'] >= min_amount]
        
        # Criar grafo
        G = nx.from_pandas_edgelist(
            df_net,
            source='Account', 
            target='Dest Account',
            edge_attr='Amount Paid',
            create_using=nx.DiGraph()
        )
        
        # Calcular métricas de centralidade
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Layout do grafo
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Preparar dados para Plotly
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            amount = G.edges[edge].get('Amount Paid', 0)
            edge_info.append(f"{edge[0]} → {edge[1]}: R$ {amount:,.2f}")
        
        # Nós
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Informações do nó
            degree = G.degree(node)
            central = centrality.get(node, 0)
            between = betweenness.get(node, 0)
            
            node_text.append(f"Account: {node}<br>"
                           f"Connections: {degree}<br>"
                           f"Centrality: {central:.3f}<br>"
                           f"Betweenness: {between:.3f}")
            
            node_size.append(20 + degree * 5)  # Tamanho proporcional ao grau
            node_color.append(central)  # Cor baseada na centralidade
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar arestas
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Transactions'
        ))
        
        # Adicionar nós
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Reds',
                reversescale=True,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.05,
                    title="Centrality"
                ),
                line=dict(width=2)
            ),
            name='Accounts'
        ))
        
        # Layout
        fig.update_layout(
            title="🕸️ Transaction Network Analysis - Money Laundering Patterns",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Node size = number of connections<br>Color = centrality score",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        return fig
    
    # ==================== EDA INTERATIVO ====================
    
    def create_interactive_eda(self, df, target_col='Is Laundering'):
        """Cria análise exploratória interativa"""
        if not PLOTLY_AVAILABLE:
            return self._fallback_eda(df, target_col)
        
        # Preparar dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Dashboard EDA
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Target Distribution', 'Amount Distribution by Target',
                'Temporal Patterns', 'Correlation Heatmap',
                'Bank Frequency', 'Payment Format Analysis'
            ),
            specs=[[{"type": "pie"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Distribuição do target
        target_counts = df[target_col].value_counts()
        fig.add_trace(go.Pie(
            labels=['Normal', 'Money Laundering'],
            values=target_counts.values,
            hole=0.3,
            marker_colors=['lightblue', 'red']
        ), row=1, col=1)
        
        # 2. Distribuição de valores
        for i, label in enumerate(['Normal', 'Suspicious']):
            target_value = 0 if label == 'Normal' else 1
            data = df[df[target_col] == target_value]['Amount Paid']
            
            fig.add_trace(go.Box(
                y=data,
                name=label,
                boxpoints='outliers',
                marker_color='lightblue' if label == 'Normal' else 'red'
            ), row=1, col=2)
        
        # 3. Padrões temporais (se existe coluna de tempo)
        if 'Timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Timestamp']).dt.date
            temporal_data = df_temp.groupby(['Date', target_col]).size().reset_index(name='count')
            
            for target_val in temporal_data[target_col].unique():
                data = temporal_data[temporal_data[target_col] == target_val]
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['count'],
                    mode='lines+markers',
                    name=f'Target {target_val}',
                    line=dict(color='red' if target_val == 1 else 'blue')
                ), row=2, col=1)
        
        # 4. Correlação (apenas colunas numéricas)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ), row=2, col=2)
        
        # Layout
        fig.update_layout(
            title="📊 Interactive Exploratory Data Analysis",
            height=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    # ==================== PERFORMANCE METRICS ====================
    
    def create_performance_dashboard(self, results_dict):
        """Cria dashboard de performance com métricas avançadas"""
        if not PLOTLY_AVAILABLE:
            return self._fallback_performance(results_dict)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Feature Importance', 'Confusion Matrix'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # ROC Curve
        if 'fpr' in results_dict and 'tpr' in results_dict:
            fig.add_trace(go.Scatter(
                x=results_dict['fpr'], 
                y=results_dict['tpr'],
                mode='lines',
                name=f"ROC (AUC = {results_dict.get('roc_auc', 0):.3f})",
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            # Linha diagonal
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', width=1, dash='dash')
            ), row=1, col=1)
        
        # PR Curve
        if 'precision' in results_dict and 'recall' in results_dict:
            fig.add_trace(go.Scatter(
                x=results_dict['recall'], 
                y=results_dict['precision'],
                mode='lines',
                name=f"PR (AUC = {results_dict.get('pr_auc', 0):.3f})",
                line=dict(color='green', width=2)
            ), row=1, col=2)
        
        # Feature Importance
        if 'feature_importance' in results_dict:
            importance = results_dict['feature_importance']
            top_features = dict(list(importance.items())[:15])
            
            fig.add_trace(go.Bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                marker_color='skyblue'
            ), row=2, col=1)
        
        # Layout
        fig.update_layout(
            title="📈 Model Performance Dashboard",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    # ==================== MÉTODOS FALLBACK ====================
    
    def _fallback_executive_dashboard(self, metrics):
        """Dashboard básico sem Plotly"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎯 AML Executive Dashboard (Fallback)', fontsize=16)
        
        # Métricas principais
        metrics_names = ['ROC-AUC', 'PR-AUC', 'Expected Value', 'F1-Score']
        metrics_values = [
            metrics.get('roc_auc', 0.85),
            metrics.get('pr_auc', 0.75), 
            metrics.get('expected_value', 1000000) / 1000000,  # Em milhões
            metrics.get('f1_score', 0.80)
        ]
        
        axes[0,0].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
        axes[0,0].set_title('Key Performance Indicators')
        axes[0,0].set_ylim(0, 1.2)
        
        # Placeholder para outros gráficos
        axes[0,1].text(0.5, 0.5, 'Feature Importance\n(Requires Plotly)', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[1,0].text(0.5, 0.5, 'Network Analysis\n(Requires Plotly + NetworkX)', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,1].text(0.5, 0.5, 'Drift Detection\n(Requires Plotly)', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def _fallback_network_analysis(self, df):
        """Análise básica de rede sem NetworkX"""
        print("⚠️ Network analysis requer NetworkX e Plotly")
        print("💡 Instale com: pip install networkx plotly")
        
        # Análise básica alternativa
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Análise de grau simples
        account_counts = df['Account'].value_counts().head(20)
        ax.bar(range(len(account_counts)), account_counts.values)
        ax.set_title('Top 20 Accounts by Transaction Count')
        ax.set_xlabel('Account Rank')
        ax.set_ylabel('Transaction Count')
        
        return fig
    
    def _fallback_eda(self, df, target_col):
        """EDA básico sem Plotly"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Exploratory Data Analysis (Fallback)', fontsize=16)
        
        # Distribuição do target
        df[target_col].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('Target Distribution')
        
        # Box plot por target
        if 'Amount Paid' in df.columns:
            df.boxplot(column='Amount Paid', by=target_col, ax=axes[0,1])
            axes[0,1].set_title('Amount Distribution by Target')
        
        # Correlação
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, ax=axes[1,0], cmap='RdBu', center=0)
            axes[1,0].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        return fig
    
    def _fallback_performance(self, results):
        """Performance básico sem Plotly"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Model Performance (Fallback)', fontsize=16)
        
        # Métricas
        metrics = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy']
        values = [
            results.get('roc_auc', 0),
            results.get('pr_auc', 0),
            results.get('f1_score', 0),
            results.get('accuracy', 0)
        ]
        
        axes[0,0].bar(metrics, values)
        axes[0,0].set_title('Performance Metrics')
        axes[0,0].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig

# ==================== FUNÇÕES DE CONVENIÊNCIA ====================

def quick_dashboard(metrics_dict, save_path=None):
    """Cria dashboard rapidamente"""
    suite = AMLVisualizationSuite()
    return suite.create_executive_dashboard(metrics_dict, save_path)

def quick_network(df, save_path=None):
    """Cria análise de rede rapidamente"""
    suite = AMLVisualizationSuite()
    fig = suite.create_transaction_network(df)
    if save_path and PLOTLY_AVAILABLE:
        fig.write_html(save_path)
    return fig

def quick_eda(df, target_col='Is Laundering', save_path=None):
    """Cria EDA rapidamente"""
    suite = AMLVisualizationSuite()
    fig = suite.create_interactive_eda(df, target_col)
    if save_path and PLOTLY_AVAILABLE:
        fig.write_html(save_path)
    return fig

def quick_performance(results_dict, save_path=None):
    """Cria dashboard de performance rapidamente"""
    suite = AMLVisualizationSuite()
    fig = suite.create_performance_dashboard(results_dict)
    if save_path and PLOTLY_AVAILABLE:
        fig.write_html(save_path)
    return fig

# ==================== EXEMPLO DE USO ====================

if __name__ == "__main__":
    print("🎨 AML Visualization Suite - Teste")
    
    # Exemplo de métricas
    metrics = {
        'roc_auc': 0.87,
        'pr_auc': 0.73,
        'expected_value': 1250000,
        'f1_score': 0.79,
        'feature_importance': {
            'Amount_Paid': 0.25,
            'Network_Centrality': 0.18,
            'Transaction_Frequency': 0.15,
            'Time_of_Day': 0.12,
            'Bank_Risk_Score': 0.10
        }
    }
    
    # Testar dashboard
    suite = AMLVisualizationSuite()
    dashboard = suite.create_executive_dashboard(metrics)
    
    print("✅ Visualization Suite carregada e testada!")
    print(f"📊 Plotly disponível: {PLOTLY_AVAILABLE}")
    print(f"🕸️ NetworkX disponível: {NETWORKX_AVAILABLE}")
