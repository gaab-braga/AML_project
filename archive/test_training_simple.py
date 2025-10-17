#!/usr/bin/env python3
"""
Script simples para testar treinamento otimizado sem duplicatas
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time

# Adicionar diretório raiz ao sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuração SILENCIOSA de logging
logging.getLogger().setLevel(logging.CRITICAL)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

print("=== TESTE DE TREINAMENTO OTIMIZADO ===")

# Carregar dados
artifacts_dir = Path('artifacts')
features_pkl = artifacts_dir / 'features_processed.pkl'

try:
    import joblib
    df = joblib.load(features_pkl)
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")

    # Configuração simples
    config = {
        'models': {
            'xgboost': {
                'model_type': 'xgb',
                'params': {
                    'n_estimators': 10,  # Muito pequeno para teste rápido
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbosity': 0,
                    'n_jobs': 1
                }
            }
        }
    }

    # Importar função de treinamento
    from src.modeling.train_individual_models import train_xgboost_model

    # Treinar
    start_time = time.time()
    results = train_xgboost_model(
        X=X[:1000],  # Usar apenas 1000 amostras para teste rápido
        y=y[:1000],
        config=config,
        artifacts_dir=artifacts_dir,
        force_retrain=True,  # Forçar retreinamento para teste
        enable_gpu=False
    )
    training_time = time.time() - start_time

    # Resultado otimizado
    if 'model_name' in results and results['model_name'] == 'xgboost':
        eval_results = results['evaluation_results']
        print("\n✅ XGBOOST - {:.1f}s".format(training_time))
        print("   ROC-AUC: {:.4f} | PR-AUC: {:.4f} | Threshold: {:.3f}".format(
            eval_results.get('roc_auc', 0.0),
            eval_results.get('pr_auc', 0.0),
            eval_results.get('optimal_threshold', 0.5)
        ))
        print("   Recall: {:.4f} | Precision: {:.4f} | F1: {:.4f}".format(
            eval_results.get('recall', 0.0),
            eval_results.get('precision', 0.0),
            eval_results.get('f1', 0.0)
        ))
    else:
        print("Falha no treinamento")

    print("\n=== TESTE CONCLUÍDO ===")

except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()