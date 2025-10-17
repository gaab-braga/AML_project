import sys
sys.path.insert(0, 'c:/Users/gafeb/OneDrive/Desktop/lavagem_dev')
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.aml_plotting import plot_threshold_comparison_all_models_optimized

# Simular dados similares ao que o notebook teria
np.random.seed(42)
n_samples = 1000
y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.95, 0.05]))
X = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'feature_{i}' for i in range(10)])

# Simular eval_results_list como seria gerado pelo notebook
eval_results_list = []
model_names = ['xgboost', 'lightgbm', 'random_forest']

for model_name in model_names:
    # Simular threshold_analysis
    thresholds = np.linspace(0.1, 0.9, 9)
    threshold_analysis = []

    for thresh in thresholds:
        # Simular métricas
        recall = np.random.uniform(0.7, 0.95)
        precision = np.random.uniform(0.1, 0.5)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        threshold_analysis.append({
            'threshold': thresh,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'custo_total': np.random.randint(1000, 10000)
        })

    # Simular probabilities (com tamanho diferente para testar correção)
    probabilities = np.random.random(500)  # Menor que n_samples

    eval_results = {
        'threshold_analysis': threshold_analysis,
        'probabilities': probabilities,
        'roc_auc': np.random.uniform(0.8, 0.95),
        'pr_auc': np.random.uniform(0.7, 0.9)
    }

    eval_results_list.append(eval_results)

print('Testando plot_threshold_comparison_all_models_optimized com correção automática...')
print(f'Modelos: {model_names}')
print(f'Número de eval_results: {len(eval_results_list)}')
print(f'Tamanho y: {len(y)}, X: {X.shape}')
print(f'Tamanho probabilidades originais: {[len(er["probabilities"]) for er in eval_results_list]}')

try:
    # Chamar a função como no notebook
    plot_threshold_comparison_all_models_optimized(eval_results_list, model_names, y, X)
    print(' Função executada com sucesso!')
except Exception as e:
    print(f' Erro na execução: {e}')
    import traceback
    traceback.print_exc()