import sys
sys.path.insert(0, '.')

# Simular o que acontece no notebook
import pandas as pd
import numpy as np
from pathlib import Path

# Carregar dados como no notebook
artifacts_dir = Path('artifacts')
features_pkl = artifacts_dir / 'features_processed.pkl'

try:
    import joblib
    df = joblib.load(features_pkl)

    # Separar features e target
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)

    # Usar apenas features numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f'Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features')

    # Simular training_results como seria criado no notebook
    training_results = {
        'results': {},
        'successful_model_names': ['xgboost', 'lightgbm', 'random_forest']
    }

    # Simular eval_results_list com pipelines (usando dados pequenos para simular o problema)
    eval_results_list = []
    for name in training_results['successful_model_names']:
        # Simular evaluation_results (seria carregado do cache ou calculado)
        eval_results = {
            'threshold_analysis': [
                {'threshold': 0.1, 'recall': 0.95, 'precision': 0.05, 'f1': 0.09, 'custo_total': 50000},
                {'threshold': 0.3, 'recall': 0.90, 'precision': 0.08, 'f1': 0.15, 'custo_total': 30000},
                {'threshold': 0.5, 'recall': 0.85, 'precision': 0.12, 'f1': 0.21, 'custo_total': 20000},
                {'threshold': 0.7, 'recall': 0.75, 'precision': 0.18, 'f1': 0.29, 'custo_total': 15000},
                {'threshold': 0.9, 'recall': 0.60, 'precision': 0.25, 'f1': 0.35, 'custo_total': 10000}
            ],
            'probabilities': np.random.random(1000),  # Pequeno para simular erro
            'roc_auc': 0.85,
            'pr_auc': 0.75
        }
        eval_results_list.append(eval_results)

    model_names_list = training_results['successful_model_names']

    print(f'Simulação criada: {len(eval_results_list)} modelos')
    print(f'Tamanho y: {len(y)}, probabilidades: {[len(er["probabilities"]) for er in eval_results_list]}')

    # Agora testar a função corrigida
    from src.features.aml_plotting import plot_threshold_comparison_all_models_optimized

    print('Testando função corrigida...')
    plot_threshold_comparison_all_models_optimized(eval_results_list, model_names_list, y, X)
    print('Sucesso!')

except Exception as e:
    print(f'Erro: {e}')
    import traceback
    traceback.print_exc()