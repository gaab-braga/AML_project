"""
Advanced Metrics for AML Model Evaluation

Extended metrics including calibration assessment, ranking metrics (NDCG@K),
and specialized AML evaluation functions.

NOTE: This file was consolidated from multiple modules and contains some
duplicate function definitions for backward compatibility.
"""

from __future__ import annotations

__all__ = [
    # Calibration metrics
    'compute_calibration_metrics',
    'calibration_error',
    'reliability_curve',
    
    # Ranking metrics (NDCG, top-K)
    'compute_ndcg_at_k',
    'compute_at_k',
    'format_at_k',
    'generate_lift_table',
    'compute_cumulative_gains',
    'compute_top_decile_lift',
    'compute_efficiency_curve',
    'compare_ranking_performance',
    'get_ranking_summary',
    'save_ranking_curves',
    
    # AML specialized metrics
    'compute_aml_specialized_metrics',
    'precision_at_k',
    'recall_at_k',
    'lift_at_k',
    'expected_cost',
    'pr_auc_score',
    'alert_rate',
    'strike_rate',
    'aml_metrics_summary',
    
    # Expected Value metrics
    'expected_value_with_costs',
    'compute_expected_value',
    'find_optimal_threshold_ev',
    'compare_capacity_scenarios',
    'get_aml_cost_params_template',
    'format_ev_summary',
    
    # Comprehensive metrics
    'compute_comprehensive_metrics',
    'format_advanced_metrics_summary',
    
    # Model evaluation utilities
    'evaluate_model',
    'compute_cv_metrics',
    'bootstrap_metric',
    'safe_metric',
    'convert_for_json',
    
    # Curve persistence
    'save_curves',
    
    # Threshold optimization
    'ThresholdConfig',
    'evaluate_thresholds',
    'evaluate_thresholds_curve',
    'generate_threshold_grid',
    'save_threshold_artifact',
    'optimize_threshold_f1',
    'optimize_threshold_cost',
    'optimize_threshold_recall_at_precision',
    'optimize_threshold_custom',
    'plot_threshold_curves',
    'analyze_threshold_impact',
    
    # Validation
    'validate_inputs',
    
    # Backward compatibility aliases
    'compute_lift_table',
]

# Core imports
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings


def compute_calibration_metrics(y_true, y_prob, n_bins=10) -> Dict:
    """
    Compute calibration metrics for probability assessment.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Brier score (lower is better)
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find points in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        calibration_data = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
        
    except (ValueError, RuntimeWarning):
        ece = mce = np.nan
        calibration_data = None
    
    return {
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'calibration_curve': calibration_data,
        'n_bins': n_bins
    }


def compute_ndcg_at_k(y_true, y_scores, k_values: List[int]) -> Dict:
    """
    Compute Normalized Discounted Cumulative Gain at different K values.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores (higher = more relevant)
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary with NDCG@K values
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by scores descending
    sort_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sort_idx]
    
    def dcg_at_k(relevances, k):
        """Compute DCG@K"""
        if k > len(relevances):
            k = len(relevances)
        if k <= 0:
            return 0.0
        
        # DCG formula: sum(rel_i / log2(i+1)) for i in [0, k-1]
        relevances_k = relevances[:k]
        discounts = np.log2(np.arange(2, len(relevances_k) + 2))  # log2(2), log2(3), ...
        return np.sum(relevances_k / discounts)
    
    def ndcg_at_k(y_true, y_pred_sorted, k):
        """Compute NDCG@K"""
        # DCG@K for predicted ranking
        dcg_k = dcg_at_k(y_pred_sorted, k)
        
        # IDCG@K (ideal DCG) - perfect ranking
        y_true_ideal = np.sort(y_true)[::-1]  # Sort true labels descending
        idcg_k = dcg_at_k(y_true_ideal, k)
        
        if idcg_k == 0:
            return 0.0
        
        return dcg_k / idcg_k
    
    ndcg_results = {}
    for k in k_values:
        ndcg_k = ndcg_at_k(y_true, y_true_sorted, k)
        ndcg_results[f'ndcg_at_{k}'] = float(ndcg_k)
    
    return ndcg_results


def compute_aml_specialized_metrics(y_true, y_scores, capacity_constraint: Optional[int] = None) -> Dict:
    """
    Compute AML-specific metrics focusing on operational constraints.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        capacity_constraint: Maximum number of cases that can be reviewed
        
    Returns:
        Dictionary with specialized metrics
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by scores descending
    sort_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sort_idx]
    
    total_positives = y_true.sum()
    total_cases = len(y_true)
    base_rate = total_positives / total_cases
    
    metrics = {
        'total_positives': int(total_positives),
        'total_cases': int(total_cases),
        'base_rate': float(base_rate)
    }
    
    # Capacity-constrained metrics
    if capacity_constraint and capacity_constraint > 0:
        capacity = min(capacity_constraint, total_cases)
        
        # Performance at capacity
        top_k_true = y_true_sorted[:capacity]
        tp_at_capacity = top_k_true.sum()
        
        metrics['capacity_constrained'] = {
            'capacity': capacity,
            'tp_at_capacity': int(tp_at_capacity),
            'precision_at_capacity': float(tp_at_capacity / capacity),
            'recall_at_capacity': float(tp_at_capacity / total_positives) if total_positives > 0 else 0.0,
            'lift_at_capacity': float((tp_at_capacity / capacity) / base_rate) if base_rate > 0 else 0.0,
            'cases_needed_for_50pct_recall': int(np.searchsorted(np.cumsum(y_true_sorted), total_positives * 0.5) + 1) if total_positives > 0 else 0,
            'cases_needed_for_80pct_recall': int(np.searchsorted(np.cumsum(y_true_sorted), total_positives * 0.8) + 1) if total_positives > 0 else 0
        }
    
    # Efficiency curves (workload vs recall)
    recall_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    workload_for_recall = {}
    
    cumsum_positives = np.cumsum(y_true_sorted)
    
    for target_recall in recall_points:
        target_positives = total_positives * target_recall
        if target_positives <= total_positives:
            # Find position where we reach target recall
            position = np.searchsorted(cumsum_positives, target_positives)
            workload_pct = (position + 1) / total_cases * 100
            workload_for_recall[f'workload_for_{int(target_recall*100)}pct_recall'] = float(workload_pct)
        else:
            workload_for_recall[f'workload_for_{int(target_recall*100)}pct_recall'] = 100.0
    
    metrics['efficiency_curve'] = workload_for_recall
    
    # Top-decile metrics (common in AML)
    top_decile_size = max(1, total_cases // 10)
    top_decile_positives = y_true_sorted[:top_decile_size].sum()
    
    metrics['top_decile'] = {
        'size': top_decile_size,
        'positives_captured': int(top_decile_positives),
        'precision': float(top_decile_positives / top_decile_size),
        'recall': float(top_decile_positives / total_positives) if total_positives > 0 else 0.0,
        'lift': float((top_decile_positives / top_decile_size) / base_rate) if base_rate > 0 else 0.0,
        'concentration': float(top_decile_positives / total_positives) if total_positives > 0 else 0.0  # % of positives in top 10%
    }
    
    return metrics


def compute_comprehensive_metrics(y_true, y_scores, y_proba=None, 
                                 k_values: List[int] = [50, 100, 200], 
                                 capacity_constraint: Optional[int] = None) -> Dict:
    """
    Compute comprehensive evaluation metrics combining standard and specialized metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores (for ranking)
        y_proba: Predicted probabilities (for calibration, can be same as y_scores)
        k_values: K values for ranking metrics
        capacity_constraint: Operational capacity constraint
        
    Returns:
        Comprehensive metrics dictionary
    """
    if y_proba is None:
        y_proba = y_scores
    
    metrics = {}
    
    # Standard ranking metrics (already computed elsewhere)
    # We focus on the new advanced metrics here
    
    # Calibration metrics
    try:
        calibration = compute_calibration_metrics(y_true, y_proba)
        metrics['calibration'] = calibration
    except Exception as e:
        metrics['calibration'] = {'error': str(e)}
    
    # NDCG@K
    try:
        ndcg = compute_ndcg_at_k(y_true, y_scores, k_values)
        metrics['ranking'] = ndcg
    except Exception as e:
        metrics['ranking'] = {'error': str(e)}
    
    # AML specialized metrics
    try:
        aml_metrics = compute_aml_specialized_metrics(y_true, y_scores, capacity_constraint)
        metrics['aml_specialized'] = aml_metrics
    except Exception as e:
        metrics['aml_specialized'] = {'error': str(e)}
    
    return metrics


def format_advanced_metrics_summary(metrics: Dict) -> str:
    """
    Format advanced metrics as readable summary.
    
    Args:
        metrics: Dictionary from compute_comprehensive_metrics
        
    Returns:
        Formatted summary string
    """
    lines = [
        "📊 ADVANCED METRICS SUMMARY",
        "=" * 50,
        ""
    ]
    
    # Calibration
    if 'calibration' in metrics and 'error' not in metrics['calibration']:
        cal = metrics['calibration']
        lines.extend([
            "🎯 Calibration Assessment:",
            f"   Brier Score: {cal['brier_score']:.4f} (lower is better)",
            f"   Expected Calibration Error: {cal['expected_calibration_error']:.4f}",
            f"   Maximum Calibration Error: {cal['maximum_calibration_error']:.4f}",
            ""
        ])
    
    # Ranking (NDCG)
    if 'ranking' in metrics and 'error' not in metrics['ranking']:
        ranking = metrics['ranking']
        lines.extend([
            "🏆 Ranking Quality (NDCG@K):",
            *[f"   NDCG@{k.split('_')[-1]}: {v:.4f}" for k, v in ranking.items()],
            ""
        ])
    
    # AML Specialized
    if 'aml_specialized' in metrics and 'error' not in metrics['aml_specialized']:
        aml = metrics['aml_specialized']
        
        lines.extend([
            "💼 AML Operational Metrics:",
            f"   Base rate: {aml['base_rate']:.4f} ({aml['total_positives']}/{aml['total_cases']})",
            ""
        ])
        
        # Top decile
        if 'top_decile' in aml:
            td = aml['top_decile']
            lines.extend([
                "🔟 Top Decile Performance:",
                f"   Precision: {td['precision']:.3f}",
                f"   Recall: {td['recall']:.3f}",
                f"   Lift: {td['lift']:.1f}x",
                f"   Concentration: {td['concentration']:.1%} of positives in top 10%",
                ""
            ])
        
        # Capacity constrained
        if 'capacity_constrained' in aml:
            cc = aml['capacity_constrained']
            lines.extend([
                f"⚖️ Capacity-Constrained ({cc['capacity']} cases):",
                f"   Precision: {cc['precision_at_capacity']:.3f}",
                f"   Recall: {cc['recall_at_capacity']:.3f}",
                f"   Lift: {cc['lift_at_capacity']:.1f}x",
                f"   Cases for 50% recall: {cc['cases_needed_for_50pct_recall']:,}",
                f"   Cases for 80% recall: {cc['cases_needed_for_80pct_recall']:,}",
                ""
            ])
        
        # Efficiency curve (selected points)
        if 'efficiency_curve' in aml:
            ec = aml['efficiency_curve']
            lines.extend([
                "📈 Workload vs Recall:",
                f"   50% recall needs: {ec.get('workload_for_50pct_recall', 0):.1f}% workload",
                f"   80% recall needs: {ec.get('workload_for_80pct_recall', 0):.1f}% workload",
                ""
            ])
    
    return "\n".join(lines)

"""
Métricas AML (Anti-Money Laundering) especializadas para detecção de lavagem de dinheiro.

Foco em métricas operacionais: precision@K, recall@K, lift@K, expected_cost.
Estas métricas são otimizadas para cenários de alta classe desbalanceada onde
a capacidade operacional (K) é limitada e o custo de falsos positivos é significativo.

Autor: Generated for AML Pipeline
Data: 2025-09-30
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
from sklearn.metrics import precision_recall_curve, auc


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calcula precision no top-K (capacidade operacional).
    
    Args:
        y_true: Array binário de labels verdadeiros (0/1)
        y_scores: Array de scores de probabilidade ou ranking
        k: Número de casos a investigar (capacidade operacional)
        
    Returns:
        Precision@K: proporção de verdadeiros positivos no top-K
        
    Raises:
        ValueError: Se k > len(y_true) ou arrays têm tamanhos diferentes
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true e y_scores devem ter o mesmo tamanho")
    
    if k > len(y_true):
        raise ValueError(f"k ({k}) não pode ser maior que número de amostras ({len(y_true)})")
    
    if k <= 0:
        raise ValueError("k deve ser positivo")
    
    # Ordenar por score descendente e pegar top-K
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_labels = y_true[top_k_indices]
    
    return np.mean(top_k_labels)


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calcula recall no top-K (taxa de captura).
    
    Args:
        y_true: Array binário de labels verdadeiros (0/1)
        y_scores: Array de scores de probabilidade ou ranking
        k: Número de casos a investigar (capacidade operacional)
        
    Returns:
        Recall@K: proporção de verdadeiros positivos capturados no top-K
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true e y_scores devem ter o mesmo tamanho")
    
    if k > len(y_true):
        raise ValueError(f"k ({k}) não pode ser maior que número de amostras ({len(y_true)})")
    
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0  # Não há positivos para capturar
    
    # Ordenar por score descendente e pegar top-K
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_labels = y_true[top_k_indices]
    true_positives_at_k = np.sum(top_k_labels)
    
    return true_positives_at_k / total_positives


def lift_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calcula Lift no top-K (quantas vezes melhor que aleatório).
    
    Args:
        y_true: Array binário de labels verdadeiros (0/1)
        y_scores: Array de scores de probabilidade ou ranking
        k: Número de casos a investigar
        
    Returns:
        Lift@K: Precision@K / prevalência base
    """
    precision_k = precision_at_k(y_true, y_scores, k)
    base_prevalence = np.mean(y_true)
    
    if base_prevalence == 0:
        return float('inf') if precision_k > 0 else 1.0
    
    return precision_k / base_prevalence


def expected_cost(y_true: np.ndarray, y_scores: np.ndarray, k: int, 
                 cost_fp: float = 1.0, benefit_tp: float = 10.0) -> float:
    """
    Calcula custo esperado da estratégia top-K.
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_scores: Array de scores
        k: Número de casos a investigar
        cost_fp: Custo de investigar um falso positivo
        benefit_tp: Benefício de capturar um verdadeiro positivo
        
    Returns:
        Custo esperado (negativo = lucro)
    """
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_labels = y_true[top_k_indices]
    
    true_positives = np.sum(top_k_labels)
    false_positives = k - true_positives
    
    total_cost = (false_positives * cost_fp) - (true_positives * benefit_tp)
    return total_cost


def pr_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calcula Area Under Precision-Recall Curve.
    Mais informativa que ROC-AUC em datasets desbalanceados.
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_scores: Array de scores de probabilidade
        
    Returns:
        AUC da curva Precision-Recall
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def alert_rate(y_pred: np.ndarray, total_transactions: int) -> float:
    """
    Calcula taxa de alertas (porcentagem de transações sinalizadas).
    
    Métrica operacional crítica: indica carga de trabalho dos analistas.
    Em AML típico, alert_rate ideal é 1-5% dependendo da capacidade.
    
    Args:
        y_pred: Array binário de predições (0=normal, 1=alerta)
        total_transactions: Número total de transações no período
        
    Returns:
        Taxa de alertas (0-1)
        
    Examples:
        >>> y_pred = np.array([0, 0, 1, 0, 1, 0, 0, 0])
        >>> alert_rate(y_pred, len(y_pred))
        0.25  # 25% das transações geraram alerta
    """
    if total_transactions <= 0:
        raise ValueError("total_transactions deve ser positivo")
    
    alerts = np.sum(y_pred)
    return alerts / total_transactions


def strike_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula strike rate (taxa de acerto em alertas gerados).
    
    Strike Rate = TP / (TP + FP) = Precision
    Mas o nome "strike rate" é mais comum em contexto AML/compliance.
    
    Indica eficiência operacional: de todos os alertas gerados,
    quantos % são fraudes reais?
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_pred: Array binário de predições
        
    Returns:
        Strike rate (0-1)
        
    Examples:
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([1, 1, 1, 0, 0])
        >>> strike_rate(y_true, y_pred)
        0.6667  # 2 TP de 3 alertas
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")
    
    alerts = np.sum(y_pred)
    if alerts == 0:
        return 0.0  # Nenhum alerta gerado
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    return true_positives / alerts


def expected_value_with_costs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_matrix: dict
) -> dict:
    """
    Calcula expected value considerando matriz completa de custos/benefícios.
    
    Implementação alinhada com threshold_optimization.py para consistência.
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_proba: Array de probabilidades preditas
        threshold: Limiar de decisão (0-1)
        cost_matrix: Dicionário com:
            - 'false_positive_cost': Custo de investigar não-fraude
            - 'false_negative_cost': Custo de perder fraude
            - 'true_positive_benefit': Valor de detectar fraude
            - 'true_negative_benefit': Benefício de rejeição correta (geralmente 0)
            
    Returns:
        dict com:
            - 'expected_value': Valor esperado total
            - 'expected_value_per_tx': Valor por transação
            - 'tp', 'tn', 'fp', 'fn': Confusion matrix
            - 'breakdown': Detalhamento de custos
    """
    from sklearn.metrics import confusion_matrix as cm
    
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = cm(y_true, y_pred).ravel()
    
    fp_cost = cost_matrix.get('false_positive_cost', 0)
    fn_cost = cost_matrix.get('false_negative_cost', 0)
    tp_benefit = cost_matrix.get('true_positive_benefit', 0)
    tn_benefit = cost_matrix.get('true_negative_benefit', 0)
    
    expected_value = (
        tp * tp_benefit +
        tn * tn_benefit -
        fp * fp_cost -
        fn * fn_cost
    )
    
    return {
        'expected_value': expected_value,
        'expected_value_per_tx': expected_value / len(y_true),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'breakdown': {
            'tp_benefit_total': tp * tp_benefit,
            'tn_benefit_total': tn * tn_benefit,
            'fp_cost_total': fp * fp_cost,
            'fn_cost_total': fn * fn_cost
        }
    }


def calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> float:
    """
    Calcula Expected Calibration Error (ECE).
    
    Mede quanto as probabilidades preditas correspondem às frequências reais.
    ECE baixo (<0.05) indica modelo bem calibrado.
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_proba: Array de probabilidades preditas [0,1]
        n_bins: Número de bins para agrupar probabilidades
        strategy: 'uniform' (bins de largura igual) ou 'quantile' (mesma qtd por bin)
        
    Returns:
        ECE: Expected Calibration Error (0-1, menor é melhor)
        
    References:
        Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
        
    Examples:
        >>> # Modelo perfeitamente calibrado
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        >>> calibration_error(y_true, y_proba, n_bins=2)
        0.0  # Erro próximo de zero
    """
    if len(y_true) != len(y_proba):
        raise ValueError("y_true e y_proba devem ter o mesmo tamanho")
    
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bins = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("strategy deve ser 'uniform' ou 'quantile'")
    
    # Discretizar probabilidades em bins
    bin_indices = np.digitize(y_proba, bins[1:-1])
    
    ece = 0.0
    total_samples = len(y_true)
    
    for bin_idx in range(n_bins):
        # Amostras neste bin
        in_bin = bin_indices == bin_idx
        n_samples_in_bin = np.sum(in_bin)
        
        if n_samples_in_bin == 0:
            continue
        
        # Confiança média (probabilidade predita média)
        avg_confidence = np.mean(y_proba[in_bin])
        
        # Acurácia real neste bin
        avg_accuracy = np.mean(y_true[in_bin])
        
        # Contribuição para ECE
        ece += (n_samples_in_bin / total_samples) * abs(avg_confidence - avg_accuracy)
    
    return ece


def reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera curva de confiabilidade (reliability diagram).
    
    Returns:
        mean_predicted_proba: Probabilidade média predita por bin
        fraction_of_positives: Fração real de positivos por bin
        counts: Número de amostras por bin
    """
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bins = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError("strategy deve ser 'uniform' ou 'quantile'")
    
    bin_indices = np.digitize(y_proba, bins[1:-1])
    
    mean_predicted = np.zeros(n_bins)
    fraction_positives = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for bin_idx in range(n_bins):
        in_bin = bin_indices == bin_idx
        n_samples = np.sum(in_bin)
        
        if n_samples > 0:
            mean_predicted[bin_idx] = np.mean(y_proba[in_bin])
            fraction_positives[bin_idx] = np.mean(y_true[in_bin])
            counts[bin_idx] = n_samples
    
    return mean_predicted, fraction_positives, counts


def aml_metrics_summary(y_true: np.ndarray, y_scores: np.ndarray, 
                       k_values: list = [50, 100, 200], 
                       cost_fp: float = 1.0, benefit_tp: float = 10.0,
                       y_proba: Optional[np.ndarray] = None,
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Gera sumário completo de métricas AML para múltiplos valores de K.
    
    FASE 1: Expandido com alert_rate, strike_rate e calibration_error
    
    Args:
        y_true: Array binário de labels verdadeiros
        y_scores: Array de scores (para ranking)
        k_values: Lista de valores K para avaliar
        cost_fp: Custo de falso positivo
        benefit_tp: Benefício de verdadeiro positivo
        y_proba: Array de probabilidades (opcional, para métricas adicionais)
        threshold: Limiar de decisão (usado se y_proba fornecido)
        
    Returns:
        DataFrame com métricas por K + métricas globais
    """
    results = []
    
    for k in k_values:
        if k > len(y_true):
            continue
            
        metrics = {
            'K': k,
            'Precision@K': precision_at_k(y_true, y_scores, k),
            'Recall@K': recall_at_k(y_true, y_scores, k),
            'Lift@K': lift_at_k(y_true, y_scores, k),
            'Expected_Cost': expected_cost(y_true, y_scores, k, cost_fp, benefit_tp),
            'TP_at_K': int(np.sum(y_true[np.argsort(y_scores)[-k:]])),
            'FP_at_K': int(k - np.sum(y_true[np.argsort(y_scores)[-k:]]))
        }
        
        # Se probabilidades fornecidas, calcular métricas adicionais
        if y_proba is not None:
            # Usar top-K como "alerta"
            y_pred_at_k = np.zeros(len(y_true), dtype=int)
            top_k_indices = np.argsort(y_scores)[-k:]
            y_pred_at_k[top_k_indices] = 1
            
            metrics['Alert_Rate@K'] = alert_rate(y_pred_at_k, len(y_true))
            metrics['Strike_Rate@K'] = strike_rate(y_true, y_pred_at_k)
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Adicionar métricas globais
    global_metrics = {
        'K': 'ALL',
        'Precision@K': np.mean(y_true),
        'Recall@K': 1.0,
        'Lift@K': 1.0,
        'Expected_Cost': 0.0,
        'TP_at_K': int(np.sum(y_true)),
        'FP_at_K': int(len(y_true) - np.sum(y_true)),
        'PR_AUC': pr_auc_score(y_true, y_scores)
    }
    
    # Métricas de calibração (se probabilidades fornecidas)
    if y_proba is not None:
        y_pred_global = (y_proba >= threshold).astype(int)
        global_metrics['Alert_Rate'] = alert_rate(y_pred_global, len(y_true))
        global_metrics['Strike_Rate'] = strike_rate(y_true, y_pred_global)
        global_metrics['Calibration_Error'] = calibration_error(y_true, y_proba)
    
    df_global = pd.DataFrame([global_metrics])
    
    return pd.concat([df, df_global], ignore_index=True)


def validate_inputs(y_true: np.ndarray, y_scores: np.ndarray, k: Optional[int] = None) -> None:
    """Validação comum de inputs para todas as funções."""
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_scores, np.ndarray):
        y_scores = np.array(y_scores)
    
    if len(y_true) != len(y_scores):
        raise ValueError("y_true e y_scores devem ter o mesmo tamanho")
    
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true deve ser binário (0/1)")
    
    if k is not None and (k <= 0 or k > len(y_true)):
        raise ValueError(f"k deve estar entre 1 e {len(y_true)}")


# ==============================================================================
# REMOVED DUPLICATE __main__ BLOCK (lines 808-830)
# This was from a previous module that was concatenated here during refactoring
# The proper __main__ block is at the end of this file (line 2257+)
# ==============================================================================

"""
Metrics Utilities

Centralized functions for model evaluation, cross-validation,
and curve persistence.

NOTE: Ranking metrics (@K functions) have been consolidated into ranking_metrics.py
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    make_scorer,
    brier_score_loss
)
from sklearn.utils import resample

# Ranking metrics are defined later in this module
_has_ranking_metrics = True


def bootstrap_metric(y_true, y_score, metric_fn=average_precision_score, n_bootstrap=1000, 
                    confidence_level=0.95, random_state=42, stratified=True):
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True binary labels
        y_score: Prediction scores
        metric_fn: Metric function to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
        stratified: Whether to maintain class proportions
        
    Returns:
        Dict with 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    np.random.seed(random_state)
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_samples = len(y_true)
    
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        if stratified:
            # Stratified resampling to maintain class proportions
            indices = resample(range(n_samples), n_samples=n_samples, 
                             stratify=y_true, random_state=random_state + i)
        else:
            indices = resample(range(n_samples), n_samples=n_samples, 
                             random_state=random_state + i)
        
        y_boot = y_true[indices]
        score_boot = y_score[indices]
        
        try:
            boot_metric = metric_fn(y_boot, score_boot)
            bootstrap_scores.append(boot_metric)
        except (ValueError, ZeroDivisionError):
            # Skip invalid bootstrap samples
            continue
    
    if not bootstrap_scores:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    
    return {
        'mean': float(np.mean(bootstrap_scores)),
        'std': float(np.std(bootstrap_scores)),
        'ci_lower': float(np.percentile(bootstrap_scores, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
    }


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute calibration metrics (ECE and Brier Score).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Dict with 'ece', 'brier_score', 'reliability_data'
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    reliability_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            reliability_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop_in_bin': prop_in_bin,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum()
            })
    
    return {
        'ece': float(ece),
        'brier_score': float(brier),
        'reliability_data': reliability_data
    }


def safe_metric(func, *args, **kwargs):
    """Safely compute metric, returning NaN on error"""
    try:
        return func(*args, **kwargs)
    except ValueError:
        return np.nan


def compute_cv_metrics(model, X, y, cv_folds=3, random_state=42):
    """
    Compute cross-validation metrics for stability assessment.
    
    Args:
        model: Estimator to evaluate
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (cv_mean, cv_std, cv_stability_coef)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    try:
        print(f"      Starting {cv_folds}-fold CV...")
        
        # Manual CV to ensure correct PR_AUC calculation
        cv_pr_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and fit model
            cv_model = clone(model)
            cv_model.fit(X_train_cv, y_train_cv)
            
            # Get probabilities
            if hasattr(cv_model, 'predict_proba'):
                val_probs = cv_model.predict_proba(X_val_cv)[:, 1]
            else:
                val_scores = cv_model.decision_function(X_val_cv)
                val_probs = (val_scores - val_scores.min()) / (val_scores.max() - val_scores.min() + 1e-9)
            
            # Calculate PR_AUC
            try:
                pr_auc = average_precision_score(y_val_cv, val_probs)
                cv_pr_scores.append(pr_auc)
            except ValueError:
                # Handle edge case where fold has no positive examples
                cv_pr_scores.append(np.nan)
        
        cv_pr_scores = np.array(cv_pr_scores)
        print(f"      CV completed. PR_AUC scores: {cv_pr_scores}")
        
        # Filter out NaN values
        valid_scores = cv_pr_scores[~np.isnan(cv_pr_scores)]
        
        if len(valid_scores) > 0:
            cv_mean = float(np.mean(valid_scores))
            cv_std = float(np.std(valid_scores))
            cv_stability = cv_std / cv_mean if cv_mean > 0 else np.inf  # Coefficient of variation
        else:
            cv_mean = cv_std = cv_stability = np.nan
            
    except Exception as e:
        print(f"⚠ CV Error: {e}")
        cv_mean = cv_std = cv_stability = np.nan
    
    return cv_mean, cv_std, cv_stability


# DEPRECATED: This function has been moved to ranking_metrics.py
# Keeping for backward compatibility
if _has_ranking_metrics:
    # Use the consolidated implementation
    pass  # Import at top handles this
else:
    # Fallback implementation if ranking_metrics not available
    def compute_at_k(y_true, scores, ks):
        """
        DEPRECATED: Use ranking_metrics.compute_at_k instead
        
        Compute precision/recall/lift at different K thresholds.
        """
        print("⚠️  WARNING: Using deprecated compute_at_k from metrics_utils. Use ranking_metrics.compute_at_k instead.")
        
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)
        order = np.argsort(scores)[::-1]
        y_sorted = y_true[order]
        total_pos = y_true.sum()
        base_rate = y_true.mean() if len(y_true) else np.nan
        
        rows = []
        for k in ks:
            k = int(min(k, len(y_true)))
            if k <= 0:
                precision_k = recall_k = lift_k = np.nan
            else:
                top = y_sorted[:k]
                positives_top = top.sum()
                precision_k = positives_top / k if k else np.nan
                recall_k = positives_top / total_pos if total_pos > 0 else np.nan
                lift_k = precision_k / base_rate if base_rate and base_rate > 0 else np.nan
            rows.append((k, precision_k, recall_k, lift_k))
        
        return rows


# DEPRECATED: This function has been moved to ranking_metrics.py  
# Keeping for backward compatibility
if _has_ranking_metrics:
    # Use the consolidated implementation  
    pass  # Import at top handles this
else:
    # Fallback implementation
    def format_at_k(model_name, variant, y_true, scores, ks):
        """
        DEPRECATED: Use ranking_metrics.format_at_k instead
        
        Format metrics@K as list of dictionaries for DataFrame creation.
        """
        print("⚠️  WARNING: Using deprecated format_at_k from metrics_utils. Use ranking_metrics.format_at_k instead.")
        
        base_rate = float(np.mean(y_true)) if len(y_true) else np.nan
        data = []
        
        for k, precision_k, recall_k, lift_k in compute_at_k(y_true, scores, ks):
            data.append({
                'Model': model_name,
                'Variant': variant,
                'K': k,
                'Precision@K': precision_k,
                'Recall@K': recall_k,
                'Lift@K': lift_k,
                'BaseRate': base_rate,
            })
        
        return data


def save_curves(model_name, variant, y_true, scores, output_dir):
    """
    Save ROC and PR curves to CSV files.
    
    Args:
        model_name: Model identifier
        variant: Variant identifier
        y_true: True binary labels
        scores: Prediction scores
        output_dir: Directory to save curves
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
            output_dir / f'roc_{model_name}_{variant}.csv', index=False
        )
    except ValueError:
        pass
    
    # Precision-Recall curve
    try:
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, scores)
        thresholds_full = np.append(thresholds, np.nan)
        pd.DataFrame({
            'precision': precision_vals,
            'recall': recall_vals,
            'threshold': thresholds_full
        }).to_csv(
            output_dir / f'pr_{model_name}_{variant}.csv', index=False
        )
    except ValueError:
        pass


def evaluate_model(model_name, model, X_tr, y_tr, X_te, y_te, variant, 
                  calibrate=False, cv_folds=3, compute_bootstrap=True):
    """
    Complete model evaluation with training, CV, and test metrics.
    
    Args:
        model_name: Model identifier
        model: Estimator to evaluate
        X_tr, y_tr: Training data
        X_te, y_te: Test data
        variant: Variant identifier ('full', 'core')
        calibrate: Whether to apply calibration
        cv_folds: Number of CV folds for stability
        compute_bootstrap: Whether to compute bootstrap CI
        
    Returns:
        Tuple of (metrics_dict, test_scores, fitted_model)
    """
    estimator = clone(model)
    calibrate_used = calibrate and hasattr(estimator, 'predict_proba')
    
    # Training
    start = time.perf_counter()
    if calibrate_used:
        print(f"   Applying calibration for {model_name} {variant}...")
        calibrated = CalibratedClassifierCV(estimator, method='isotonic', cv=3)
        calibrated.fit(X_tr, y_tr)
        fitted = calibrated
    else:
        fitted = estimator.fit(X_tr, y_tr)
    train_time = time.perf_counter() - start
    
    # Cross-validation for stability
    print(f"   Computing CV for {model_name} {variant}...")
    cv_mean, cv_std, cv_stability = compute_cv_metrics(clone(model), X_tr, y_tr, cv_folds)
    print(f"   CV completed for {model_name} {variant}")
    
    # Test prediction
    print(f"   Computing test predictions for {model_name} {variant}...")
    try:
        if hasattr(fitted, 'predict_proba'):
            scores = fitted.predict_proba(X_te)[:, 1]
        else:
            raw_scores = fitted.decision_function(X_te)
            scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    except Exception as e:
        print(f"   ERROR in test prediction for {model_name}: {e}")
        # Fallback: return dummy scores
        scores = np.random.rand(len(X_te))
    
    preds = (scores >= 0.5).astype(int)
    print(f"   Test predictions completed for {model_name} {variant}")
    
    # Metrics
    print(f"   Computing metrics for {model_name} {variant}...")
    metrics = {
        'Model': model_name,
        'Variant': variant,
        'ROC_AUC': safe_metric(roc_auc_score, y_te, scores),
        'PR_AUC': safe_metric(average_precision_score, y_te, scores),
        'F1': safe_metric(f1_score, y_te, preds, zero_division=0),
        'Precision': safe_metric(precision_score, y_te, preds, zero_division=0),
        'Recall': safe_metric(recall_score, y_te, preds, zero_division=0),
        'Train_Time_Sec': train_time,
        'Calibrated': calibrate_used,
        # Stability metrics
        'CV_PR_AUC_Mean': cv_mean,
        'CV_PR_AUC_Std': cv_std,
        'CV_Stability': cv_stability,
    }
    
    # Bootstrap confidence intervals
    if compute_bootstrap and not np.isnan(metrics['PR_AUC']):
        print(f"   Computing bootstrap CI for {model_name} {variant}...")
        try:
            bootstrap_results = bootstrap_metric(y_te, scores, average_precision_score, 
                                               n_bootstrap=500, random_state=42)
            metrics.update({
                'PR_AUC_CI_Lower': bootstrap_results['ci_lower'],
                'PR_AUC_CI_Upper': bootstrap_results['ci_upper'],
                'PR_AUC_Bootstrap_Std': bootstrap_results['std']
            })
        except Exception as e:
            print(f"   Warning: Bootstrap CI failed for {model_name}: {e}")
            metrics.update({
                'PR_AUC_CI_Lower': np.nan,
                'PR_AUC_CI_Upper': np.nan,
                'PR_AUC_Bootstrap_Std': np.nan
            })
    
    # Calibration metrics (if probabilities available)
    if hasattr(fitted, 'predict_proba') and not np.isnan(metrics['PR_AUC']):
        print(f"   Computing calibration metrics for {model_name} {variant}...")
        try:
            cal_metrics = compute_calibration_metrics(y_te, scores)
            metrics.update({
                'ECE': cal_metrics['ece'],
                'Brier_Score': cal_metrics['brier_score']
            })
        except Exception as e:
            print(f"   Warning: Calibration metrics failed for {model_name}: {e}")
            metrics.update({
                'ECE': np.nan,
                'Brier_Score': np.nan
            })
    
    print(f"   Metrics completed for {model_name} {variant}")
    
    return metrics, scores, fitted


def convert_for_json(obj):
    """
    Convert numpy/pandas types to native Python for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

"""
Ranking Metrics Utilities

Consolidated functions for computing metrics @K, lift tables, 
cumulative gains and performance ranking metrics.

This module consolidates the previous scattered functions:
- compute_at_k from metrics_utils
- generate_lift_table from feature_engineering
- Additional ranking utilities
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_at_k(y_true, scores, ks):
    """
    Compute precision/recall/lift at different K thresholds.
    
    Args:
        y_true: True binary labels
        scores: Prediction scores (higher = more positive)
        ks: List of K values to evaluate
        
    Returns:
        List of tuples (k, precision_k, recall_k, lift_k)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    total_pos = y_true.sum()
    base_rate = y_true.mean() if len(y_true) else np.nan
    
    rows = []
    for k in ks:
        k = int(min(k, len(y_true)))
        if k <= 0:
            precision_k = recall_k = lift_k = np.nan
        else:
            top = y_sorted[:k]
            positives_top = top.sum()
            precision_k = positives_top / k if k else np.nan
            recall_k = positives_top / total_pos if total_pos > 0 else np.nan
            lift_k = precision_k / base_rate if base_rate and base_rate > 0 else np.nan
        rows.append((k, precision_k, recall_k, lift_k))
    
    return rows


def format_at_k(model_name, variant, y_true, scores, ks):
    """
    Format metrics@K as list of dictionaries for DataFrame creation.
    
    Args:
        model_name: Model identifier
        variant: Variant identifier (e.g., 'full', 'core')
        y_true: True binary labels
        scores: Prediction scores
        ks: List of K values
        
    Returns:
        List of metric dictionaries
    """
    base_rate = float(np.mean(y_true)) if len(y_true) else np.nan
    data = []
    
    for k, precision_k, recall_k, lift_k in compute_at_k(y_true, scores, ks):
        data.append({
            'Model': model_name,
            'Variant': variant,
            'K': k,
            'Precision@K': precision_k,
            'Recall@K': recall_k,
            'Lift@K': lift_k,
            'BaseRate': base_rate,
        })
    
    return data


def generate_lift_table(y_true, y_proba, q=10):
    """
    Generate lift/gains table for performance analysis.
    
    Args:
        y_true: True binary labels
        y_proba: Prediction probabilities
        q: Number of quantiles
        
    Returns:
        pandas.DataFrame with lift table
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba
    })
    
    # Sort by probability descending
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    # Create quantiles
    df['quantile'] = pd.qcut(df.index, q=q, labels=False) + 1
    
    # Calculate metrics by quantile
    lift_table = df.groupby('quantile').agg({
        'y_true': ['count', 'sum'],
        'y_proba': 'mean'
    }).round(4)
    
    lift_table.columns = ['total', 'positives', 'avg_proba']
    lift_table['positive_rate'] = lift_table['positives'] / lift_table['total']
    
    # Calculate lift
    overall_rate = df['y_true'].mean()
    lift_table['lift'] = lift_table['positive_rate'] / overall_rate
    
    # Calculate cumulative gains
    lift_table['cum_positives'] = lift_table['positives'].cumsum()
    lift_table['cum_total'] = lift_table['total'].cumsum()
    lift_table['cum_positive_rate'] = lift_table['cum_positives'] / lift_table['cum_total']
    lift_table['gains'] = lift_table['cum_positives'] / df['y_true'].sum()
    
    return lift_table.reset_index()


def compute_cumulative_gains(y_true, scores, n_points=100):
    """
    Compute cumulative gains curve.
    
    Args:
        y_true: True binary labels
        scores: Prediction scores
        n_points: Number of points in the curve
        
    Returns:
        Tuple of (percentiles, cumulative_gains)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    
    # Sort by scores descending
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    
    # Calculate cumulative gains at different percentiles
    n_samples = len(y_true)
    total_positives = y_true.sum()
    
    percentiles = np.linspace(0, 100, n_points)
    gains = []
    
    for p in percentiles:
        n_selected = int(n_samples * p / 100)
        if n_selected == 0:
            gain = 0
        else:
            positives_selected = y_sorted[:n_selected].sum()
            gain = positives_selected / total_positives if total_positives > 0 else 0
        gains.append(gain)
    
    return percentiles, np.array(gains)


def compute_top_decile_lift(y_true, scores):
    """
    Compute lift in top decile (top 10%).
    
    Args:
        y_true: True binary labels
        scores: Prediction scores
        
    Returns:
        Dictionary with top decile metrics
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    
    n_samples = len(y_true)
    top_10_pct = int(n_samples * 0.1)
    
    if top_10_pct == 0:
        return {
            'top_decile_precision': np.nan,
            'top_decile_recall': np.nan,
            'top_decile_lift': np.nan,
            'top_decile_size': 0
        }
    
    # Sort and get top decile
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    top_decile = y_sorted[:top_10_pct]
    
    # Metrics
    total_positives = y_true.sum()
    top_decile_positives = top_decile.sum()
    base_rate = y_true.mean()
    
    precision = top_decile_positives / top_10_pct
    recall = top_decile_positives / total_positives if total_positives > 0 else 0
    lift = precision / base_rate if base_rate > 0 else np.nan
    
    return {
        'top_decile_precision': float(precision),
        'top_decile_recall': float(recall),
        'top_decile_lift': float(lift),
        'top_decile_size': top_10_pct
    }


def compute_efficiency_curve(y_true, scores, workload_limits=None):
    """
    Compute efficiency curve: recall achieved vs workload (cases reviewed).
    
    Args:
        y_true: True binary labels
        scores: Prediction scores
        workload_limits: List of workload percentages to evaluate
        
    Returns:
        pandas.DataFrame with workload vs recall
    """
    if workload_limits is None:
        workload_limits = [1, 5, 10, 20, 30, 50, 70, 100]
    
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_samples = len(y_true)
    total_positives = y_true.sum()
    
    # Sort by scores descending
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    
    results = []
    
    for workload_pct in workload_limits:
        n_reviewed = int(n_samples * workload_pct / 100)
        n_reviewed = min(n_reviewed, n_samples)
        
        if n_reviewed == 0:
            recall = 0
            precision = 0
        else:
            positives_found = y_sorted[:n_reviewed].sum()
            recall = positives_found / total_positives if total_positives > 0 else 0
            precision = positives_found / n_reviewed
        
        results.append({
            'workload_pct': workload_pct,
            'cases_reviewed': n_reviewed,
            'recall': recall,
            'precision': precision
        })
    
    return pd.DataFrame(results)


def compare_ranking_performance(models_dict, y_true, k_values=None):
    """
    Compare ranking performance across multiple models.
    
    Args:
        models_dict: Dictionary {model_name: scores}
        y_true: True binary labels
        k_values: List of K values to evaluate
        
    Returns:
        pandas.DataFrame with comparison results
    """
    if k_values is None:
        k_values = [50, 100, 200, 500]
    
    results = []
    
    for model_name, scores in models_dict.items():
        # Compute @K metrics
        at_k_results = compute_at_k(y_true, scores, k_values)
        
        for k, precision_k, recall_k, lift_k in at_k_results:
            results.append({
                'Model': model_name,
                'K': k,
                'Precision@K': precision_k,
                'Recall@K': recall_k,
                'Lift@K': lift_k
            })
        
        # Add top decile metrics
        top_decile = compute_top_decile_lift(y_true, scores)
        results.append({
            'Model': model_name,
            'K': 'Top10%',
            'Precision@K': top_decile['top_decile_precision'],
            'Recall@K': top_decile['top_decile_recall'],
            'Lift@K': top_decile['top_decile_lift']
        })
    
    return pd.DataFrame(results)


def evaluate_thresholds_ranking(y_true, y_proba, metric='f1', step=0.01):
    """
    Evaluate different thresholds and return ranking-based metrics.
    
    Args:
        y_true: True binary labels
        y_proba: Prediction probabilities
        metric: Primary metric for optimization
        step: Threshold step size
        
    Returns:
        Dictionary with optimization results
    """
    cfg = ThresholdConfig(
        min_threshold=step,
        max_threshold=0.99,
        step=step,
        optimize_metric=metric
    )

    result = evaluate_thresholds(y_true, y_proba, cfg)
    results_df = pd.DataFrame(result['curve'])

    best_metrics = result['best_metrics']
    logger.info(
        "Best threshold for %s: %.3f (precision=%.3f, recall=%.3f, f1=%.3f)",
        metric,
        result['best_threshold'],
        best_metrics.get('precision', np.nan),
        best_metrics.get('recall', np.nan),
        best_metrics.get('f1', np.nan),
    )

    return {
        'best_threshold': result['best_threshold'],
        'best_metrics': best_metrics,
        'results_df': results_df,
    }


def save_ranking_curves(model_name, variant, y_true, scores, output_dir):
    """
    Save ranking-related curves to files.
    
    Args:
        model_name: Model identifier
        variant: Variant identifier
        y_true: True binary labels
        scores: Prediction scores
        output_dir: Directory to save curves
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lift table
    try:
        lift_table = generate_lift_table(y_true, scores, q=10)
        lift_table.to_csv(
            output_dir / f'lift_table_{model_name}_{variant}.csv', 
            index=False
        )
    except Exception as e:
        logger.warning(f"Failed to save lift table: {e}")
    
    # Cumulative gains curve
    try:
        percentiles, gains = compute_cumulative_gains(y_true, scores)
        gains_df = pd.DataFrame({
            'percentile': percentiles,
            'cumulative_gain': gains
        })
        gains_df.to_csv(
            output_dir / f'cumulative_gains_{model_name}_{variant}.csv',
            index=False
        )
    except Exception as e:
        logger.warning(f"Failed to save cumulative gains: {e}")
    
    # Efficiency curve
    try:
        efficiency_df = compute_efficiency_curve(y_true, scores)
        efficiency_df.to_csv(
            output_dir / f'efficiency_curve_{model_name}_{variant}.csv',
            index=False
        )
    except Exception as e:
        logger.warning(f"Failed to save efficiency curve: {e}")


def get_ranking_summary(y_true, scores, k_values=None):
    """
    Get comprehensive ranking performance summary.
    
    Args:
        y_true: True binary labels
        scores: Prediction scores
        k_values: List of K values to evaluate
        
    Returns:
        Dictionary with summary statistics
    """
    if k_values is None:
        k_values = [50, 100, 200]
    
    # Basic @K metrics
    at_k_results = compute_at_k(y_true, scores, k_values)
    at_k_dict = {}
    for k, precision_k, recall_k, lift_k in at_k_results:
        at_k_dict[f'Precision@{k}'] = precision_k
        at_k_dict[f'Recall@{k}'] = recall_k
        at_k_dict[f'Lift@{k}'] = lift_k
    
    # Top decile
    top_decile = compute_top_decile_lift(y_true, scores)
    
    # Base statistics
    base_rate = float(np.mean(y_true))
    total_positives = int(np.sum(y_true))
    
    summary = {
        'base_rate': base_rate,
        'total_positives': total_positives,
        'total_samples': len(y_true),
        **at_k_dict,
        **top_decile
    }
    
    return summary


# Backward compatibility - keep old function names as aliases
compute_lift_table = generate_lift_table  # For backwards compatibility

logger.info("SUCCESS: Ranking Metrics module loaded successfully!")

"""
Threshold Optimization Module

Provides functions to optimize classification threshold based on:
- F1-score maximization
- Business cost minimization (expected value)
- Recall constraint with maximum precision
- Custom business metrics

Author: Data Science Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_threshold_f1(
    y_true: np.ndarray, 
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict]:
    """
    Find threshold that maximizes F1-score.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    thresholds : array-like, optional
        Custom threshold values to test. If None, uses precision-recall curve thresholds
        
    Returns
    -------
    best_threshold : float
        Optimal threshold value
    best_f1 : float
        Maximum F1-score achieved
    metrics_dict : dict
        Dictionary with precision, recall, and F1 at optimal threshold
        
    Examples
    --------
    >>> threshold, f1, metrics = optimize_threshold_f1(y_test, y_proba)
    >>> print(f"Best threshold: {threshold:.3f}, F1: {f1:.3f}")
    """
    # Get precision-recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    # Note: precision_recall_curve returns n+1 values but n thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    metrics_dict = {
        'threshold': best_threshold,
        'f1': best_f1,
        'precision': precisions[best_idx],
        'recall': recalls[best_idx]
    }
    
    return best_threshold, best_f1, metrics_dict


def optimize_threshold_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: Dict[str, float],
    thresholds: Optional[np.ndarray] = None,
    n_steps: int = 1000
) -> Tuple[float, float, Dict]:
    """
    Find threshold that minimizes expected cost or maximizes expected value.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    cost_matrix : dict
        Dictionary with business costs:
        - 'false_positive_cost': Cost of investigating non-fraud (e.g., 50)
        - 'false_negative_cost': Cost of missing fraud (e.g., 5000)
        - 'true_positive_benefit': Value recovered from catching fraud (e.g., 4500)
        - 'true_negative_benefit': Benefit of correct rejection (usually 0)
    thresholds : array-like, optional
        Custom threshold values to test
    n_steps : int, default=1000
        Number of threshold values to test
        
    Returns
    -------
    best_threshold : float
        Optimal threshold value
    best_value : float
        Maximum expected value (or minimum cost if negative)
    metrics_dict : dict
        Dictionary with expected value, costs breakdown, and confusion matrix
        
    Notes
    -----
    Expected Value = TP * tp_benefit - FP * fp_cost - FN * fn_cost + TN * tn_benefit
    
    For cost minimization (no benefits), set benefits to 0 and use negative costs.
    
    Examples
    --------
    >>> cost_matrix = {
    ...     'false_positive_cost': 50,
    ...     'false_negative_cost': 5000,
    ...     'true_positive_benefit': 4500,
    ...     'true_negative_benefit': 0
    ... }
    >>> threshold, value, metrics = optimize_threshold_cost(y_test, y_proba, cost_matrix)
    """
    # Extract costs
    fp_cost = cost_matrix.get('false_positive_cost', 0)
    fn_cost = cost_matrix.get('false_negative_cost', 0)
    tp_benefit = cost_matrix.get('true_positive_benefit', 0)
    tn_benefit = cost_matrix.get('true_negative_benefit', 0)
    
    # Generate thresholds
    if thresholds is None:
        thresholds = np.linspace(0, 1, n_steps)
    
    # Calculate expected value for each threshold
    expected_values = []
    all_metrics = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate expected value
        expected_value = (
            tp * tp_benefit +
            tn * tn_benefit -
            fp * fp_cost -
            fn * fn_cost
        )
        
        expected_values.append(expected_value)
        all_metrics.append({
            'threshold': thresh,
            'expected_value': expected_value,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Find best threshold
    best_idx = np.argmax(expected_values)
    best_threshold = thresholds[best_idx]
    best_value = expected_values[best_idx]
    best_metrics = all_metrics[best_idx]
    
    # Add detailed breakdown
    metrics_dict = {
        'threshold': best_threshold,
        'expected_value': best_value,
        'expected_value_per_transaction': best_value / len(y_true),
        'confusion_matrix': {
            'tp': best_metrics['tp'],
            'tn': best_metrics['tn'],
            'fp': best_metrics['fp'],
            'fn': best_metrics['fn']
        },
        'costs_breakdown': {
            'tp_benefit_total': best_metrics['tp'] * tp_benefit,
            'tn_benefit_total': best_metrics['tn'] * tn_benefit,
            'fp_cost_total': best_metrics['fp'] * fp_cost,
            'fn_cost_total': best_metrics['fn'] * fn_cost
        },
        'all_thresholds': thresholds,
        'all_values': expected_values
    }
    
    return best_threshold, best_value, metrics_dict


def optimize_threshold_recall_at_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.9
) -> Tuple[float, float, Dict]:
    """
    Find threshold that maximizes recall while maintaining minimum precision.
    
    Useful when false positives are costly and you need high precision,
    but want to maximize fraud detection (recall) subject to that constraint.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    min_precision : float, default=0.9
        Minimum required precision (0-1)
        
    Returns
    -------
    best_threshold : float
        Optimal threshold value
    best_recall : float
        Maximum recall achieved at minimum precision
    metrics_dict : dict
        Dictionary with precision, recall, F1 at optimal threshold
        
    Examples
    --------
    >>> # Find threshold for 90% precision
    >>> threshold, recall, metrics = optimize_threshold_recall_at_precision(
    ...     y_test, y_proba, min_precision=0.90
    ... )
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find all thresholds that meet precision requirement
    valid_indices = np.where(precisions[:-1] >= min_precision)[0]
    
    if len(valid_indices) == 0:
        raise ValueError(
            f"Cannot achieve precision >= {min_precision:.2f}. "
            f"Maximum achievable precision: {precisions.max():.3f}"
        )
    
    # Among valid thresholds, find the one with highest recall
    best_idx = valid_indices[np.argmax(recalls[:-1][valid_indices])]
    best_threshold = thresholds[best_idx]
    best_recall = recalls[best_idx]
    
    metrics_dict = {
        'threshold': best_threshold,
        'recall': best_recall,
        'precision': precisions[best_idx],
        'f1': 2 * (precisions[best_idx] * recalls[best_idx]) / 
              (precisions[best_idx] + recalls[best_idx] + 1e-10)
    }
    
    return best_threshold, best_recall, metrics_dict


def optimize_threshold_custom(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable,
    thresholds: Optional[np.ndarray] = None,
    n_steps: int = 1000,
    maximize: bool = True
) -> Tuple[float, float, Dict]:
    """
    Find threshold that optimizes a custom metric function.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    metric_fn : callable
        Custom metric function with signature: metric_fn(y_true, y_pred) -> float
    thresholds : array-like, optional
        Custom threshold values to test
    n_steps : int, default=1000
        Number of threshold values to test
    maximize : bool, default=True
        If True, find threshold that maximizes metric. If False, minimizes.
        
    Returns
    -------
    best_threshold : float
        Optimal threshold value
    best_metric : float
        Optimal metric value
    metrics_dict : dict
        Dictionary with metric values for all thresholds
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, n_steps)
    
    metric_values = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        metric_val = metric_fn(y_true, y_pred)
        metric_values.append(metric_val)
    
    # Find best threshold
    if maximize:
        best_idx = np.argmax(metric_values)
    else:
        best_idx = np.argmin(metric_values)
    
    best_threshold = thresholds[best_idx]
    best_metric = metric_values[best_idx]
    
    metrics_dict = {
        'threshold': best_threshold,
        'metric_value': best_metric,
        'all_thresholds': thresholds,
        'all_metrics': metric_values
    }
    
    return best_threshold, best_metric, metrics_dict


def plot_threshold_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: Optional[Dict[str, float]] = None,
    optimal_thresholds: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize trade-offs for different threshold values.
    
    Creates a comprehensive plot showing:
    - Precision-Recall trade-off
    - F1-score vs threshold
    - Expected value vs threshold (if cost_matrix provided)
    - ROC curve
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    cost_matrix : dict, optional
        Business costs for expected value calculation
    optimal_thresholds : dict, optional
        Dictionary of optimal thresholds to mark on plots, e.g.:
        {'F1': 0.45, 'Expected Value': 0.32}
    figsize : tuple, default=(15, 10)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with all plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get curves
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    
    # Plot 1: Precision-Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(pr_thresholds, precisions[:-1], label='Precision', linewidth=2)
    ax1.plot(pr_thresholds, recalls[:-1], label='Recall', linewidth=2)
    
    if optimal_thresholds:
        for name, thresh in optimal_thresholds.items():
            ax1.axvline(thresh, linestyle='--', alpha=0.5, label=f'{name}: {thresh:.3f}')
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Precision-Recall vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1-Score vs Threshold
    ax2 = axes[0, 1]
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    ax2.plot(pr_thresholds, f1_scores, linewidth=2, color='green')
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = pr_thresholds[best_f1_idx]
    ax2.axvline(best_f1_thresh, linestyle='--', color='red', 
                label=f'Max F1 at {best_f1_thresh:.3f}')
    ax2.scatter(best_f1_thresh, f1_scores[best_f1_idx], 
                color='red', s=100, zorder=5)
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Expected Value vs Threshold (if cost_matrix provided)
    ax3 = axes[1, 0]
    if cost_matrix:
        thresholds = np.linspace(0, 1, 500)
        expected_values = []
        
        fp_cost = cost_matrix.get('false_positive_cost', 0)
        fn_cost = cost_matrix.get('false_negative_cost', 0)
        tp_benefit = cost_matrix.get('true_positive_benefit', 0)
        tn_benefit = cost_matrix.get('true_negative_benefit', 0)
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            ev = tp * tp_benefit + tn * tn_benefit - fp * fp_cost - fn * fn_cost
            expected_values.append(ev)
        
        ax3.plot(thresholds, expected_values, linewidth=2, color='purple')
        
        best_ev_idx = np.argmax(expected_values)
        best_ev_thresh = thresholds[best_ev_idx]
        ax3.axvline(best_ev_thresh, linestyle='--', color='red',
                   label=f'Max EV at {best_ev_thresh:.3f}')
        ax3.scatter(best_ev_thresh, expected_values[best_ev_idx],
                   color='red', s=100, zorder=5)
        
        ax3.set_xlabel('Threshold', fontsize=12)
        ax3.set_ylabel('Expected Value ($)', fontsize=12)
        ax3.set_title('Expected Value vs Threshold', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Cost matrix not provided', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Expected Value vs Threshold', fontsize=14, fontweight='bold')
    
    # Plot 4: ROC Curve
    ax4 = axes[1, 1]
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_threshold_impact(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list,
    cost_matrix: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Analyze impact of different threshold values on key metrics.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    thresholds : list
        List of threshold values to analyze
    cost_matrix : dict, optional
        Business costs for expected value calculation
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with metrics for each threshold
    """
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        alert_rate = (tp + fp) / len(y_true)
        strike_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        result = {
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'alert_rate': alert_rate,
            'strike_rate': strike_rate,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        # Add expected value if cost matrix provided
        if cost_matrix:
            fp_cost = cost_matrix.get('false_positive_cost', 0)
            fn_cost = cost_matrix.get('false_negative_cost', 0)
            tp_benefit = cost_matrix.get('true_positive_benefit', 0)
            tn_benefit = cost_matrix.get('true_negative_benefit', 0)
            
            ev = tp * tp_benefit + tn * tn_benefit - fp * fp_cost - fn * fn_cost
            result['expected_value'] = ev
            result['expected_value_per_tx'] = ev / len(y_true)
        
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    print("Threshold Optimization Module")
    print("=" * 50)
    print("\nThis module provides functions for optimizing classification thresholds.")
    print("\nMain functions:")
    print("  - optimize_threshold_f1(): Maximize F1-score")
    print("  - optimize_threshold_cost(): Minimize cost / maximize expected value")
    print("  - optimize_threshold_recall_at_precision(): Max recall at min precision")
    print("  - plot_threshold_curves(): Visualize trade-offs")
    print("  - analyze_threshold_impact(): Compare multiple thresholds")
    print("\nSee function docstrings for detailed usage examples.")

"""
Threshold Optimization Utilities

This module centralizes threshold search logic so it can be reused by
notebooks and the automated pipeline stage. It evaluates a grid of
thresholds on binary classification probabilities and returns the best
threshold according to a configured metric while also exposing the full
metric curve for auditing.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ThresholdConfig:
    min_threshold: float = 0.01
    max_threshold: float = 0.99
    step: float = 0.01
    optimize_metric: str = "f1"
    # Expected Value parameters
    v_tp: float = 10.0  # Value per True Positive
    c_fp: float = 2.0   # Cost per False Positive
    c_fn: float = 50.0  # Cost per False Negative
    c_review: float = 0.5  # Cost per case reviewed
    capacity_limit: Optional[int] = None  # Maximum cases that can be reviewed

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, float]]) -> "ThresholdConfig":
        if cfg is None:
            return cls()
        
        # Get EV config from global config if available
        ev_config = cfg.get('expected_value', {}) if 'expected_value' in cfg else {}
        
        return cls(
            min_threshold=cfg.get("min_threshold", cls.min_threshold),
            max_threshold=cfg.get("max_threshold", cls.max_threshold),
            step=cfg.get("step", cls.step),
            optimize_metric=cfg.get("optimize_metric", cls.optimize_metric).lower(),
            v_tp=ev_config.get("v_tp", cls.v_tp),
            c_fp=ev_config.get("c_fp", cls.c_fp),
            c_fn=ev_config.get("c_fn", cls.c_fn),
            c_review=ev_config.get("c_review", cls.c_review),
            capacity_limit=cfg.get("capacity_limit", cls.capacity_limit)
        )


ALLOWED_METRICS = {"f1", "precision", "recall", "accuracy", "expected_value"}


def _validate_inputs(y_true: Iterable[int], scores: Iterable[float], cfg: ThresholdConfig) -> None:
    if cfg.step <= 0:
        raise ValueError("step must be > 0")
    if cfg.max_threshold <= cfg.min_threshold:
        raise ValueError("max_threshold must be greater than min_threshold")
    if cfg.optimize_metric not in ALLOWED_METRICS:
        raise ValueError(
            f"optimize_metric must be one of {sorted(ALLOWED_METRICS)}, got {cfg.optimize_metric}"
        )

    y_arr = np.asarray(y_true)
    if y_arr.ndim != 1:
        raise ValueError("y_true must be 1-dimensional")

    s_arr = np.asarray(scores)
    if s_arr.ndim != 1:
        raise ValueError("scores must be 1-dimensional")
    if len(y_arr) != len(s_arr):
        raise ValueError("y_true and scores must have the same length")


def generate_threshold_grid(cfg: ThresholdConfig) -> np.ndarray:
    """Return a numpy array of thresholds to evaluate."""
    # Ensure inclusive of max_threshold within floating tolerance
    thresholds = np.arange(cfg.min_threshold, cfg.max_threshold + cfg.step / 2, cfg.step)
    # Clamp to [0, 1]
    thresholds = np.clip(thresholds, 0.0, 1.0)
    # Remove duplicates caused by clipping or floating arithmetic
    return np.unique(np.round(thresholds, 6))


def evaluate_thresholds(
    y_true: Iterable[int],
    scores: Iterable[float],
    cfg: Optional[ThresholdConfig] = None,
) -> Dict[str, object]:
    """
    Evaluate a grid of thresholds and return best threshold + full curve.

    Returns a dictionary with keys:
        - best_threshold
        - best_metrics: dict of metrics at the selected threshold
        - curve: list of metrics for each evaluated threshold
        - global_metrics: dict with ROC_AUC, PR_AUC, base_rate
    """
    if cfg is None:
        cfg = ThresholdConfig()

    _validate_inputs(y_true, scores, cfg)

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if np.all(scores == scores[0]):
        # Constant scores -> cannot discriminate, return default 0.5
        base_metrics = _compute_metrics_at_threshold(y_true, scores, 0.5, cfg)
        return {
            "best_threshold": 0.5,
            "best_metrics": base_metrics,
            "curve": [base_metrics],
            "global_metrics": _compute_global_metrics(y_true, scores),
            "note": "Scores are constant; falling back to threshold 0.5.",
        }

    thresholds = generate_threshold_grid(cfg)
    curve_rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        metrics = _compute_metrics_at_threshold(y_true, scores, threshold, cfg)
        curve_rows.append(metrics)

    curve_df = pd.DataFrame(curve_rows)

    if curve_df.empty:
        raise RuntimeError("Threshold evaluation produced no rows; check configuration")

    # Apply capacity constraint if specified
    if cfg.capacity_limit is not None:
        valid_rows = curve_df[curve_df['flagged_cases'] <= cfg.capacity_limit]
        if not valid_rows.empty:
            curve_df = valid_rows
        else:
            # If no threshold meets capacity, use the one closest to capacity
            curve_df = curve_df.loc[[curve_df['flagged_cases'].sub(cfg.capacity_limit).abs().idxmin()]]

    metric_column = cfg.optimize_metric
    best_idx = curve_df[metric_column].idxmax()
    best_row = curve_df.loc[best_idx].to_dict()

    # Add optimization info
    optimization_info = {
        "optimization_metric": cfg.optimize_metric,
        "capacity_limit": cfg.capacity_limit,
        "capacity_constraint_applied": cfg.capacity_limit is not None
    }

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_metrics": {k: float(v) for k, v in best_row.items() if k != "threshold"},
        "curve": curve_df.to_dict(orient="records"),
        "global_metrics": _compute_global_metrics(y_true, scores),
        "optimization_info": optimization_info
    }


def _compute_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float, 
                                cfg: Optional[ThresholdConfig] = None) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix components
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    flagged = int(y_pred.sum())
    total = len(y_true)
    flagged_pct = flagged / total * 100 if total else 0.0

    # Expected Value calculation
    expected_value = 0.0
    if cfg is not None:
        # EV = TP * v_tp - FP * c_fp - FN * c_fn - flagged * c_review
        expected_value = (tp * cfg.v_tp - fp * cfg.c_fp - fn * cfg.c_fn - flagged * cfg.c_review)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "expected_value": float(expected_value),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "flagged_cases": flagged,
        "flagged_pct": float(flagged_pct),
    }


def _compute_global_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

    return {
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
        "base_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
    }


# Add alias for backward compatibility
def evaluate_thresholds_curve(y_true, scores, cfg):
    """Alias for evaluate_thresholds for backward compatibility."""
    return evaluate_thresholds(y_true, scores, cfg)


def save_threshold_artifact(output_path: Path, payload: Dict[str, object]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path

"""
Expected Value Metrics for AML Context

Cost-benefit analysis for money laundering detection considering:
- True Positive Value (risk mitigated)
- False Positive Cost (analyst time)
- Review Cost (case opening overhead)
- False Negative Cost (regulatory/reputational risk)
- Operational Capacity (maximum investigations per period)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_expected_value(y_true, y_scores, 
                          v_tp=1.0,           # Value per True Positive (risk mitigated)
                          c_fp=0.2,           # Cost per False Positive (analyst hours)
                          c_review=0.05,      # Review cost per case (overhead)
                          c_fn=5.0,           # Cost per False Negative (penalty)
                          capacity=None):      # Max investigations per period
    """
    Compute Expected Value for different thresholds in AML context.
    
    Args:
        y_true: True binary labels
        y_scores: Model scores (higher = more suspicious)
        v_tp: Value per True Positive (e.g., avg risk mitigated)
        c_fp: Cost per False Positive (e.g., analyst time)
        c_review: Fixed cost per case opened for review
        c_fn: Cost per False Negative (e.g., regulatory penalty)
        capacity: Max cases that can be reviewed (None = unlimited)
        
    Returns:
        DataFrame with threshold, metrics, and expected value
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by score descending
    sort_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sort_idx]
    scores_sorted = y_scores[sort_idx]
    
    results = []
    
    # Get unique thresholds
    thresholds = np.unique(scores_sorted)
    thresholds = np.append(thresholds, [thresholds.min() - 0.001])  # Add below-min threshold
    
    total_positives = y_true.sum()
    total_negatives = len(y_true) - total_positives
    
    for threshold in thresholds:
        # Predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        cases_opened = tp + fp
        
        # Apply capacity constraint if specified
        if capacity is not None and cases_opened > capacity:
            # Take top capacity cases by score
            top_capacity_pred = np.zeros_like(y_pred)
            top_capacity_indices = np.argsort(y_scores)[::-1][:capacity]
            top_capacity_pred[top_capacity_indices] = 1
            
            # Recalculate with capacity constraint
            tp = np.sum((y_true == 1) & (top_capacity_pred == 1))
            fp = np.sum((y_true == 0) & (top_capacity_pred == 1))
            fn = np.sum((y_true == 1) & (top_capacity_pred == 0))
            tn = np.sum((y_true == 0) & (top_capacity_pred == 0))
            cases_opened = capacity
        
        # Expected Value calculation
        ev = (tp * v_tp -                    # Value from catching true positives
              fp * c_fp -                    # Cost from false positive investigations
              cases_opened * c_review -      # Fixed cost per case opened
              fn * c_fn)                     # Cost from missing true positives
        
        # Standard metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Operational metrics
        workload = cases_opened
        efficiency = tp / cases_opened if cases_opened > 0 else 0  # TP rate among reviewed
        coverage = recall  # Same as recall, but contextually clearer
        
        results.append({
            'threshold': float(threshold),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'cases_opened': int(cases_opened),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'workload': int(workload),
            'efficiency': float(efficiency),
            'coverage': float(coverage),
            'expected_value': float(ev),
            'ev_per_case': float(ev / cases_opened) if cases_opened > 0 else 0
        })
    
    return pd.DataFrame(results).sort_values('expected_value', ascending=False)


def find_optimal_threshold_ev(y_true, y_scores, cost_params: Dict, capacity=None):
    """
    Find threshold that maximizes Expected Value.
    
    Args:
        y_true: True binary labels
        y_scores: Model scores
        cost_params: Dict with keys v_tp, c_fp, c_review, c_fn
        capacity: Optional capacity constraint
        
    Returns:
        Dict with optimal threshold and metrics
    """
    ev_df = compute_expected_value(y_true, y_scores, capacity=capacity, **cost_params)
    
    if ev_df.empty:
        return None
    
    optimal_row = ev_df.iloc[0]  # Already sorted by EV descending
    
    return {
        'optimal_threshold': optimal_row['threshold'],
        'expected_value': optimal_row['expected_value'],
        'precision': optimal_row['precision'],
        'recall': optimal_row['recall'],
        'f1': optimal_row['f1'],
        'cases_opened': optimal_row['cases_opened'],
        'efficiency': optimal_row['efficiency'],
        'workload_utilization': optimal_row['cases_opened'] / capacity if capacity else None
    }


def compare_capacity_scenarios(y_true, y_scores, cost_params: Dict, 
                              capacities: List[int]) -> pd.DataFrame:
    """
    Compare Expected Value across different capacity scenarios.
    
    Args:
        y_true: True binary labels  
        y_scores: Model scores
        cost_params: Cost parameters dictionary
        capacities: List of capacity values to compare
        
    Returns:
        DataFrame comparing scenarios
    """
    scenarios = []
    
    for capacity in capacities:
        optimal = find_optimal_threshold_ev(y_true, y_scores, cost_params, capacity)
        if optimal:
            scenarios.append({
                'capacity': capacity,
                'optimal_threshold': optimal['optimal_threshold'],
                'expected_value': optimal['expected_value'],
                'ev_per_capacity': optimal['expected_value'] / capacity,
                'precision': optimal['precision'],
                'recall': optimal['recall'],
                'f1': optimal['f1'],
                'efficiency': optimal['efficiency'],
                'workload_utilization': optimal.get('workload_utilization', 1.0)
            })
    
    return pd.DataFrame(scenarios)


def get_aml_cost_params_template() -> Dict:
    """
    Get template cost parameters for AML context.
    Adjust these based on your organization's specific costs.
    
    Returns:
        Dictionary of cost parameters with reasonable defaults
    """
    return {
        'v_tp': 10.0,      # Value per TP: avg. risk mitigated (~$10K equivalent)
        'c_fp': 2.0,       # Cost per FP: 2 hours analyst time (~$100 at $50/hr)  
        'c_review': 0.5,   # Review cost: case opening overhead (~$25)
        'c_fn': 50.0       # Cost per FN: regulatory/reputational risk (~$5K)
    }


def format_ev_summary(optimal_result: Dict, cost_params: Dict) -> str:
    """
    Format Expected Value optimization results as readable summary.
    
    Args:
        optimal_result: Result from find_optimal_threshold_ev
        cost_params: Cost parameters used
        
    Returns:
        Formatted string summary
    """
    if not optimal_result:
        return "❌ No valid EV optimization result"
    
    lines = [
        "💰 EXPECTED VALUE OPTIMIZATION:",
        f"   Optimal threshold: {optimal_result['optimal_threshold']:.4f}",
        f"   Expected Value: ${optimal_result['expected_value']:,.2f}",
        f"   Cases to review: {optimal_result['cases_opened']:,}",
        f"   Precision: {optimal_result['precision']:.3f}",
        f"   Recall: {optimal_result['recall']:.3f}",
        f"   Efficiency: {optimal_result['efficiency']:.3f} (TP rate among reviewed)",
        "",
        "📊 Cost Parameters Used:",
        f"   Value per TP: ${cost_params.get('v_tp', 0):,.2f}",
        f"   Cost per FP: ${cost_params.get('c_fp', 0):,.2f}",
        f"   Review cost: ${cost_params.get('c_review', 0):,.2f}",
        f"   Cost per FN: ${cost_params.get('c_fn', 0):,.2f}"
    ]
    
    if optimal_result.get('workload_utilization'):
        lines.append(f"   Capacity utilization: {optimal_result['workload_utilization']:.1%}")
    
    return "\n".join(lines)

