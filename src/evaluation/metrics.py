"""
Métricas customizadas para avaliação de modelos AML.
Foco em métricas @k para detecção de fraudes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    recall_score,
    precision_score
)
from typing import Dict, List, Union


def calculate_metrics_at_k(
    y_true: Union[pd.Series, np.ndarray],
    y_scores: np.ndarray,
    k_list: List[int]
) -> Dict[str, float]:
    """
    Calcula Precision@k e Recall@k para múltiplos valores de k.
    
    Essencial para AML: mede performance nas transações de maior risco.
    
    Args:
        y_true: Labels verdadeiros (0/1)
        y_scores: Scores de probabilidade do modelo
        k_list: Lista de valores k para avaliar (ex: [100, 500, 1000])
        
    Returns:
        Dict com métricas 'precision@k' e 'recall@k' para cada k
        
    Examples:
        >>> metrics = calculate_metrics_at_k(y_test, y_pred_proba, [100, 500])
        >>> print(f"Precision@100: {metrics['precision@100']:.3f}")
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.reset_index(drop=True)
    
    metrics_at_k = {}
    total_positives = np.sum(y_true)
    
    for k in k_list:
        # Validar k
        if k > len(y_scores):
            k = len(y_scores)
        
        # Pegar top k predições
        top_k_idx = np.argsort(y_scores)[-k:]
        
        # Calcular precision@k
        if isinstance(y_true, pd.Series):
            true_positives_at_k = y_true.iloc[top_k_idx].sum()
        else:
            true_positives_at_k = y_true[top_k_idx].sum()
        
        precision = true_positives_at_k / k if k > 0 else 0
        
        # Calcular recall@k
        recall = true_positives_at_k / total_positives if total_positives > 0 else 0
        
        metrics_at_k[f'precision@{k}'] = precision
        metrics_at_k[f'recall@{k}'] = recall
    
    return metrics_at_k


def calculate_standard_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas padrão de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições binárias (0/1)
        y_pred_proba: Probabilidades preditas
        
    Returns:
        Dict com ROC-AUC, PR-AUC, F1, Recall, Precision
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'f1_score': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred)
    }
    return metrics


def calculate_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_proba: np.ndarray,
    k_list: List[int] = [100, 500, 1000],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula todas as métricas (padrão + @k) de uma vez.
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        k_list: Lista de valores k para métricas @k
        threshold: Threshold para converter probabilidades em classes
        
    Returns:
        Dict unificado com todas as métricas
    """
    # Converter probabilidades em predições binárias
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Métricas padrão
    metrics = calculate_standard_metrics(y_true, y_pred, y_pred_proba)
    
    # Métricas @k
    metrics_at_k = calculate_metrics_at_k(y_true, y_pred_proba, k_list)
    metrics.update(metrics_at_k)
    
    return metrics


def print_metrics_report(metrics: Dict[str, float], title: str = "Métricas de Avaliação"):
    """
    Imprime relatório formatado de métricas.
    
    Args:
        metrics: Dict com métricas
        title: Título do relatório
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Separar métricas padrão e @k
    standard_metrics = {k: v for k, v in metrics.items() if '@' not in k}
    k_metrics = {k: v for k, v in metrics.items() if '@' in k}
    
    print("\nMétricas Padrão:")
    print("-" * 60)
    for metric, value in standard_metrics.items():
        print(f"  {metric:.<25} {value:.4f}")
    
    if k_metrics:
        print("\nMétricas @k (Foco em Alto Risco):")
        print("-" * 60)
        for metric, value in k_metrics.items():
            print(f"  {metric:.<25} {value:.4f}")
    
    print("=" * 60)


def calculate_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred_proba: np.ndarray,
    k_list: List[int] = [100, 500, 1000],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula todas as métricas (padrão + @k) de uma vez.
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        k_list: Lista de valores k para métricas @k
        threshold: Threshold para converter probabilidades em classes
        
    Returns:
        Dict unificado com todas as métricas
    """
    # Converter probabilidades em predições binárias
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Métricas padrão
    metrics = calculate_standard_metrics(y_true, y_pred, y_pred_proba)
    
    # Métricas @k
    metrics_at_k = calculate_metrics_at_k(y_true, y_pred_proba, k_list)
    metrics.update(metrics_at_k)
    
    return metrics


# Compatibilidade com código legado
def calculate_precision_recall_at_k(y_true, y_scores, k_list):
    """
    Wrapper de compatibilidade. Use calculate_metrics_at_k() para novo código.
    """
    return calculate_metrics_at_k(y_true, y_scores, k_list)
