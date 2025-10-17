#!/usr/bin/env python3
"""
Configuração Ultra-Rápida para Desenvolvimento AML
Versão avançada com múltiplos modos de otimização.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Configuração Ultra-Rápida (Desenvolvimento)
FAST_DEVELOPMENT_CONFIG = {
    'random_seed': 42,
    'temporal_splits': 2,  # Corrigido: mínimo 2 splits para cross-validation
    'early_stopping': {
        'enabled': True,
        'rounds': 10,  # Early stopping mais agressivo
        'metric': 'auc',
        'min_delta': 0.005,  # Menos rigoroso para velocidade
        'max_rounds': 500  # Máximo reduzido
    },
    'models': {
        'xgboost': {
            'model_type': 'xgb',
            'params': {
                'n_estimators': 500,  # Metade do tempo normal
                'max_depth': 4,  # Menos profundo
                'learning_rate': 0.15,  # Maior para convergir mais rápido
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'random_state': 42,
                'scale_pos_weight': 10,
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'verbosity': 0,
                'n_jobs': -1,
                # GPU (será ativado se disponível)
                'tree_method': 'hist',  # Default, será sobrescrito se GPU disponível
            }
        },
        'lightgbm': {
            'model_type': 'lgb',
            'params': {
                'n_estimators': 500,
                'max_depth': 4,
                'learning_rate': 0.15,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'random_state': 42,
                'scale_pos_weight': 10,
                'verbosity': -1,
                'metric': 'auc',
                'n_jobs': -1,
                # GPU (será ativado se disponível)
                'device': 'cpu',  # Default, será sobrescrito se GPU disponível
            }
        },
        'random_forest': {
            'model_type': 'rf',
            'params': {
                'n_estimators': 50,  # Muito reduzido para velocidade
                'max_depth': 8,  # Menos profundo
                'min_samples_split': 20,  # Mais restritivo
                'min_samples_leaf': 10,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
        }
    },
    'metrics': ['roc_auc', 'average_precision', 'recall', 'precision', 'f1'],
    'aml_thresholds': [0.3, 0.5, 0.7],  # Menos thresholds para velocidade
    'business_metrics': {
        'cost_benefit_ratio': {'fp_cost': 1, 'fn_cost': 100},
        'regulatory_requirements': {
            'min_recall': 0.8,
            'max_false_positive_rate': 0.05
        }
    }
}

def enable_gpu_acceleration(config: dict) -> dict:
    """
    Habilita aceleração GPU se disponível.

    Args:
        config: Configuração base

    Returns:
        Configuração com GPU habilitada
    """
    try:
        import torch
        gpu_available = torch.cuda.is_available()

        if gpu_available:
            print("[GPU] GPU detectada! Habilitando aceleração GPU...")

            # XGBoost GPU
            config['models']['xgboost']['params']['tree_method'] = 'gpu_hist'
            config['models']['xgboost']['params']['gpu_id'] = 0

            # LightGBM GPU
            config['models']['lightgbm']['params']['device'] = 'gpu'
            config['models']['lightgbm']['params']['gpu_platform_id'] = 0
            config['models']['lightgbm']['params']['gpu_device_id'] = 0

            print("[SUCCESS] Configuração GPU aplicada!")
        else:
            print("[WARNING] GPU não detectada. Usando CPU otimizada.")

    except ImportError:
        print("[WARNING] PyTorch não instalado. GPU não disponível.")

    return config

def get_sample_data(X: pd.DataFrame, y: pd.Series, sample_frac: float = 0.25,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Retorna uma amostra estratificada dos dados para desenvolvimento rápido.

    Args:
        X: Features
        y: Target
        sample_frac: Fração dos dados (default 25%)
        stratify: Manter proporção de classes

    Returns:
        X_sample, y_sample: Dados amostrados
    """
    if stratify:
        # Amostragem estratificada para manter proporção de classes
        from sklearn.model_selection import train_test_split

        # Usar train_test_split para estratificação
        _, X_sample, _, y_sample = train_test_split(
            X, y,
            test_size=sample_frac,
            random_state=42,
            stratify=y
        )
    else:
        # Amostragem simples
        X_sample = X.sample(frac=sample_frac, random_state=42)
        y_sample = y.loc[X_sample.index]

    print(f"[DATA] Dados amostrados: {len(X_sample):,} transações ({sample_frac:.0%})")
    print(f"   [TARGET] Fraude na amostra: {y_sample.mean():.3%} (original: {y.mean():.3%})")
    print(f"   [SPEED] Redução esperada: ~{1/sample_frac:.0f}x mais rápido")

    return X_sample, y_sample

def get_feature_selection(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """
    Seleciona as top N features mais importantes usando RandomForest rápido.

    Args:
        X: Features
        y: Target
        top_n: Número de features para selecionar

    Returns:
        X_selected: Features selecionadas
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel

    print(f"[FEATURES] Selecionando top {top_n} features...")

    # Treinar RF rápido para seleção
    rf_selector = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    rf_selector.fit(X, y)

    # Selecionar features
    selector = SelectFromModel(rf_selector, prefit=True, max_features=top_n)
    X_selected = selector.transform(X)

    # Obter nomes das features selecionadas
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"[SUCCESS] {len(selected_features)} features selecionadas:")
    for i, feature in enumerate(selected_features[:10], 1):  # Mostrar primeiras 10
        print(f"   {i:2d}. {feature}")

    if len(selected_features) > 10:
        print(f"   ... e mais {len(selected_features) - 10} features")

    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

def create_ultra_fast_config(sample_fraction: float = 0.25,
                           use_gpu: bool = True,
                           feature_selection: Optional[int] = None) -> dict:
    """
    Cria configuração ultra-otimizada com múltiplas estratégias.

    Args:
        sample_fraction: Fração dos dados para usar
        use_gpu: Habilitar GPU se disponível
        feature_selection: Número de features para selecionar (None = usar todas)

    Returns:
        Configuração otimizada
    """
    config = FAST_DEVELOPMENT_CONFIG.copy()

    print("[CONFIG] CRIANDO CONFIGURAÇÃO ULTRA-OTIMIZADA")
    print("=" * 50)
    print(f"[DATA] Amostragem: {sample_fraction:.0%} dos dados")
    print(f"[GPU] GPU: {'Habilitada' if use_gpu else 'Desabilitada'}")
    print(f"[FEATURES] Features: {'Seleção automática' if feature_selection else 'Todas'}")

    if use_gpu:
        config = enable_gpu_acceleration(config)

    # Ajustar parâmetros baseado na amostragem
    if sample_fraction < 0.5:
        # Com poucos dados, reduzir ainda mais a complexidade
        for model_name in ['xgboost', 'lightgbm']:
            config['models'][model_name]['params']['max_depth'] = 3
            config['models'][model_name]['params']['n_estimators'] = 300

        config['models']['random_forest']['params']['n_estimators'] = 30
        config['models']['random_forest']['params']['max_depth'] = 6

        print("[PARAMS] Parâmetros ajustados para dados reduzidos")

    print(f"[TIMING] Tempo estimado: ~{2-5 if use_gpu else 5-10} minutos")
    print(f"[CONFIG] Configuração pronta!")

    return config

if __name__ == "__main__":
    print("🚀 CONFIGURAÇÃO ULTRA-RÁPIDA PARA DESENVOLVIMENTO AML")
    print("=" * 60)
    print("Estratégias implementadas:")
    print("✅ Amostragem estratificada de dados")
    print("✅ Configuração GPU automática")
    print("✅ Seleção automática de features")
    print("✅ Parâmetros otimizados por tamanho de dados")
    print()
    print("Para usar:")
    print("1. Importe as funções no seu notebook")
    print("2. Use create_ultra_fast_config() para configuração otimizada")
    print("3. Use get_sample_data(X, y) para amostrar dados")
    print("4. Use get_feature_selection(X, y) para reduzir features")
    print()
    print("[WARNING] NÃO use em produção - apenas para desenvolvimento!")