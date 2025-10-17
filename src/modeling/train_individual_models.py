"""
Funções de treinamento individuais para modelos AML.
Permite controle fino e refinamento específico por modelo.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging
from datetime import datetime

# Configuração centralizada de logging
def setup_logging():
    """Configura logging centralizado."""
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    artifacts_dir: Path,
    force_retrain: bool = False,
    enable_gpu: bool = False
) -> Dict[str, Any]:
    """
    Treina modelo XGBoost com configurações otimizadas para AML.

    XGBoost é excelente para dados estruturados e desbalanceados,
    com suporte nativo a early stopping e GPU acceleration.

    Args:
        X: Features
        y: Target
        config: Configuração experimental
        artifacts_dir: Diretório de artefatos
        force_retrain: Forçar retreinamento
        enable_gpu: Habilitar aceleração GPU

    Returns:
        Resultados do treinamento
    """
    print("🚀 Treinando XGBoost - Otimizado para CPU/GPU")

    # Configuração específica XGBoost
    xgb_config = config.copy()
    xgb_config['models']['xgboost']['params'].update({
        'tree_method': 'gpu_hist' if enable_gpu else 'hist',
        'predictor': 'gpu_predictor' if enable_gpu else 'cpu_predictor',
        'n_jobs': -1,  # Paralelização CPU
        'verbosity': 0
    })

    # Treinar usando função base
    from .train_single_model import train_single_model
    try:
        result = train_single_model(X, y, 'xgboost', xgb_config, artifacts_dir, force_retrain)
        print("✅ XGBoost treinado com sucesso")
        return {
            'results': {'xgboost': result},
            'successful_model_names': ['xgboost'],
            'model_names': ['xgboost']
        }
    except Exception as e:
        print(f"❌ Erro no XGBoost: {e}")
        return {
            'results': {'xgboost': {'error': str(e)}},
            'successful_model_names': [],
            'model_names': ['xgboost']
        }


def train_lightgbm_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    artifacts_dir: Path,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Treina modelo LightGBM com configurações otimizadas para AML.

    LightGBM é eficiente em memória e rápido, mas requer cuidados
    específicos com paralelização e early stopping.

    Args:
        X: Features
        y: Target
        config: Configuração experimental
        artifacts_dir: Diretório de artefatos
        force_retrain: Forçar retreinamento

    Returns:
        Resultados do treinamento
    """
    print("🚀 Treinando LightGBM - Otimizado para velocidade e memória")

    # Configuração específica LightGBM
    lgb_config = config.copy()
    lgb_config['models']['lightgbm']['params'].update({
        'n_jobs': 1,  # LightGBM tem problemas com paralelização
        'verbosity': -1,  # Silenciar completamente
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'is_unbalance': True,
        'metric': 'auc'
    })

    # Garantir métricas corretas
    lgb_config['metrics'] = ['roc_auc', 'average_precision', 'recall', 'precision', 'f1']

    # Treinar usando função base
    from .train_single_model import train_single_model
    try:
        result = train_single_model(X, y, 'lightgbm', lgb_config, artifacts_dir, force_retrain)
        print("✅ LightGBM treinado com sucesso")
        return {
            'results': {'lightgbm': result},
            'successful_model_names': ['lightgbm'],
            'model_names': ['lightgbm']
        }
    except Exception as e:
        print(f"❌ Erro no LightGBM: {e}")
        return {
            'results': {'lightgbm': {'error': str(e)}},
            'successful_model_names': [],
            'model_names': ['lightgbm']
        }


def train_random_forest_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    artifacts_dir: Path,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Treina modelo Random Forest com configurações otimizadas para AML.

    Random Forest é robusto e interpretável, mas pode ser lento
    em datasets grandes. Foca em controle de overfitting.

    Args:
        X: Features
        y: Target
        config: Configuração experimental
        artifacts_dir: Diretório de artefatos
        force_retrain: Forçar retreinamento

    Returns:
        Resultados do treinamento
    """
    print("🚀 Treinando Random Forest - Focado em robustez e interpretabilidade")

    # Configuração específica Random Forest
    rf_config = config.copy()
    rf_config['models']['random_forest']['params'].update({
        'n_jobs': -1,  # Paralelização máxima
        'class_weight': 'balanced',
        'random_state': 42,
        'bootstrap': True,
        'oob_score': True  # Out-of-bag scoring
    })

    # Treinar usando função base
    from .train_single_model import train_single_model
    try:
        result = train_single_model(X, y, 'random_forest', rf_config, artifacts_dir, force_retrain)
        print("✅ Random Forest treinado com sucesso")
        return {
            'results': {'random_forest': result},
            'successful_model_names': ['random_forest'],
            'model_names': ['random_forest']
        }
    except Exception as e:
        print(f"❌ Erro no Random Forest: {e}")
        return {
            'results': {'random_forest': {'error': str(e)}},
            'successful_model_names': [],
            'model_names': ['random_forest']
        }