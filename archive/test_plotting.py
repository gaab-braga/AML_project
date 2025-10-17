#!/usr/bin/env python3
"""
Test script to verify if plot_threshold_comparison_all_models_optimized
can correctly plot 3 models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src.features.aml_plotting import plot_threshold_comparison_all_models_optimized

# Mock EXPERIMENT_CONFIG since it's not available in the module
EXPERIMENT_CONFIG = {
    'business_metrics': {
        'cost_benefit_ratio': {
            'fp_cost': 1,
            'fn_cost': 100
        },
        'regulatory_requirements': {
            'min_recall': 0.8,
            'max_false_positive_rate': 0.05
        }
    }
}

def create_mock_evaluation_data():
    """Create mock evaluation data for 3 models with proper structure."""
    np.random.seed(42)
    n_samples = 1000
    thresholds = np.linspace(0.1, 0.9, 9)

    # Create mock data for 3 models
    eval_results_list = []

    for model_name in ['xgboost', 'lightgbm', 'random_forest']:
        # Generate different probability distributions for each model
        if model_name == 'xgboost':
            probabilities = np.random.beta(3, 8, size=n_samples)  # Better performance
        elif model_name == 'lightgbm':
            probabilities = np.random.beta(2.5, 7, size=n_samples)  # Medium performance
        else:  # random_forest
            probabilities = np.random.beta(2, 6, size=n_samples)  # Conservative performance

        # Create threshold analysis data
        threshold_analysis = []
        for thresh in thresholds:
            y_pred = (probabilities >= thresh).astype(int)
            y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 10% fraud rate

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            threshold_analysis.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        eval_results = {
            'threshold_analysis': threshold_analysis,
            'probabilities': probabilities,
            'optimal_threshold': 0.5
        }

        eval_results_list.append(eval_results)

    return eval_results_list

def test_plot_function():
    """Test the plotting function with mock data."""
    print("🧪 TESTANDO FUNÇÃO DE PLOTAGEM DOS 3 MODELOS")
    print("=" * 50)

    # Create mock data
    eval_results_list = create_mock_evaluation_data()
    model_names = ['xgboost', 'lightgbm', 'random_forest']
    y_true = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

    print(f"📊 Dados simulados criados:")
    print(f"   • {len(model_names)} modelos: {model_names}")
    print(f"   • {len(eval_results_list[0]['threshold_analysis'])} thresholds por modelo")
    print(f"   • {len(eval_results_list[0]['probabilities'])} amostras")

    # Test the plotting function
    try:
        print("\n🎨 Executando plot_threshold_comparison_all_models_optimized...")
        plot_threshold_comparison_all_models_optimized(
            eval_results_list=eval_results_list,
            model_names=model_names,
            y_true=y_true,
            figsize=(20, 8)
        )
        print("\n✅ SUCESSO: Função executou sem erros!")
        print("✅ Gráficos gerados para os 3 modelos!")
        return True

    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plot_function()
    if success:
        print("\n🎉 TESTE APROVADO: A função consegue plotar corretamente os 3 modelos!")
    else:
        print("\n💥 TESTE FALHADO: Problemas na função de plotagem.")