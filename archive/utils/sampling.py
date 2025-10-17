"""
Baseline Models Training and Class Imbalance Handling Utilities.

This module consolidates two major components:
1. Baseline Models Training: LightGBM and XGBoost with default parameters
2. Sampling Strategies: SMOTE, SMOTE-ENN, ADASYN, undersampling for imbalanced data

FASE 1: Added undersampling and class_weight_only strategies to maintain
realistic fraud distribution and avoid over-sampling artifacts.
"""

__all__ = [
    # Baseline training
    'train_baseline_models',
    
    # Sampling utilities
    'create_balanced_dataset',
    'create_sampling_strategies',
    'get_sampling_summary',
    'select_best_strategy',
]

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple

# ML libraries
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# Imbalanced-learn (imported dynamically in functions to handle missing dependency)
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.combine import SMOTEENN, SMOTETomek
# from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


def train_baseline_models(X_train, y_train, X_test, y_test, config):
    """
    Treina modelos baseline (LightGBM e XGBoost) com parâmetros padrão.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de treino
    y_train : Series
        Target de treino
    X_test : DataFrame
        Features de teste
    y_test : Series
        Target de teste
    config : dict
        Configuração com random_state e cv_folds
        
    Returns:
    --------
    baseline_results : dict
        Dicionário com métricas por modelo
    models : dict
        Modelos treinados
    """
    print("\n" + "=" * 70)
    print("[BASELINE] TRAINING MODELS WITH DEFAULT PARAMETERS")
    print("=" * 70)
    
    baseline_results = {}
    models = {}
    
    # ============================================================================
    # 1. LIGHTGBM BASELINE
    # ============================================================================
    print("\n[1/2] Training LightGBM (default params)...")
    lgbm_baseline = LGBMClassifier(
        random_state=config['random_state'],
        n_estimators=100,
        verbose=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(
        lgbm_baseline, 
        X_train, 
        y_train,
        cv=config['cv_folds'],
        scoring='average_precision',
        n_jobs=-1
    )
    
    # Train on full train set
    lgbm_baseline.fit(X_train, y_train)
    
    # Evaluate on test
    y_pred_proba = lgbm_baseline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    baseline_results['LightGBM'] = {
        'cv_pr_auc': cv_scores.mean(),
        'cv_pr_auc_std': cv_scores.std(),
        'test_pr_auc': average_precision_score(y_test, y_pred_proba),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_f1': f1_score(y_test, y_pred)
    }
    
    models['LightGBM'] = lgbm_baseline
    
    print(f"   CV PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Test PR-AUC: {baseline_results['LightGBM']['test_pr_auc']:.4f}")
    print(f"   Test ROC-AUC: {baseline_results['LightGBM']['test_roc_auc']:.4f}")
    
    # ============================================================================
    # 2. XGBOOST BASELINE
    # ============================================================================
    print("\n[2/2] Training XGBoost (default params)...")
    xgb_baseline = XGBClassifier(
        random_state=config['random_state'],
        n_estimators=100,
        eval_metric='logloss',
        verbosity=0
    )
    
    # Cross-validation
    cv_scores = cross_val_score(
        xgb_baseline, 
        X_train, 
        y_train,
        cv=config['cv_folds'],
        scoring='average_precision',
        n_jobs=-1
    )
    
    # Train on full train set
    xgb_baseline.fit(X_train, y_train)
    
    # Evaluate on test
    y_pred_proba = xgb_baseline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    baseline_results['XGBoost'] = {
        'cv_pr_auc': cv_scores.mean(),
        'cv_pr_auc_std': cv_scores.std(),
        'test_pr_auc': average_precision_score(y_test, y_pred_proba),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_f1': f1_score(y_test, y_pred)
    }
    
    models['XGBoost'] = xgb_baseline
    
    print(f"   CV PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Test PR-AUC: {baseline_results['XGBoost']['test_pr_auc']:.4f}")
    print(f"   Test ROC-AUC: {baseline_results['XGBoost']['test_roc_auc']:.4f}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY:")
    print("=" * 70)
    print(f"{'Model':<15} {'CV PR-AUC':<12} {'Test PR-AUC':<12} {'Test ROC-AUC':<12}")
    print("-" * 70)
    for model, metrics in baseline_results.items():
        print(f"{model:<15} {metrics['cv_pr_auc']:.4f}      {metrics['test_pr_auc']:.4f}       {metrics['test_roc_auc']:.4f}")
    print("=" * 70)
    
    print("\n[OK] Baseline models trained successfully!")
    print("     Next step: Hyperparameter tuning to improve these scores")
    print("=" * 70)
    
    return baseline_results, models


# ================================================================================
# SAMPLING STRATEGIES
# ================================================================================
# Professional sampling strategies: SMOTE, SMOTE-ENN, ADASYN, with 
# memory-efficient implementation for large datasets.
# ================================================================================


def create_balanced_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'smote',
    random_state: int = 42,
    sampling_strategy: str = 'auto',
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create balanced dataset using specified sampling method.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: 'smote', 'smote_enn', 'adasyn', or 'none'
        random_state: Random seed
        sampling_strategy: Sampling strategy ('auto', float, or dict)
        **kwargs: Additional parameters for samplers
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if method == 'none':
        logger.info("No sampling applied")
        return X.copy(), y.copy()
    
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=kwargs.get('k_neighbors', 5)
            )
        
        elif method == 'smote_enn':
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                n_neighbors=kwargs.get('n_neighbors', 5)
            )
        
        elif method == 'smote_tomek':
            from imblearn.combine import SMOTETomek
            sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        
        else:
            raise ValueError(f"Method '{method}' not supported")
        
        logger.info(f"Applying {method} sampling...")
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        fraud_rate = y_resampled.sum() / len(y_resampled)
        logger.info(
            f"{method.upper()}: {len(y_resampled):,} samples, "
            f"fraud rate: {fraud_rate:.2%}"
        )
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"Sampling with {method} failed: {e}")
        logger.info("Returning original dataset")
        return X.copy(), y.copy()


def create_sampling_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    methods: list = ['original', 'smote', 'smote_enn', 'undersampling', 'class_weight_only'],
    random_state: int = 42,
    enable_compact: bool = True,
    compact_size: int = 500000
) -> Dict[str, Dict[str, Any]]:
    """
    Create multiple balanced versions of dataset for comparison.
    
    FASE 1: Adicionadas alternativas 'undersampling' e 'class_weight_only'
    para evitar super-amostragem excessiva e manter distribuição realista.
    
    Args:
        X: Feature matrix
        y: Target labels
        methods: List of sampling methods to apply
            - 'original': No balancing
            - 'smote': SMOTE oversampling
            - 'smote_enn': SMOTE + ENN cleaning
            - 'undersampling': Random undersampling of majority class
            - 'class_weight_only': Original data (for use with class_weight in models)
        random_state: Random seed
        enable_compact: Create compact version for large datasets
        compact_size: Max samples for compact version
        
    Returns:
        Dictionary with balanced datasets and metadata
    """
    logger.info(f"Creating {len(methods)} sampling strategies...")
    
    balanced_datasets = {}
    original_size = len(y)
    original_fraud_rate = y.sum() / len(y)
    
    # Original dataset (no balancing)
    if 'original' in methods:
        balanced_datasets['original'] = {
            'X': X.copy(),
            'y': y.copy(),
            'method': 'none',
            'description': 'Original dataset without balancing',
            'use_class_weight': False
        }
    
    # Class weight only (no resampling)
    if 'class_weight_only' in methods:
        balanced_datasets['class_weight_only'] = {
            'X': X.copy(),
            'y': y.copy(),
            'method': 'class_weight',
            'description': 'Original data (use class_weight in model)',
            'use_class_weight': True
        }
    
    # Undersampling (FASE 1 addition)
    if 'undersampling' in methods:
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            # Target ratio: 3:1 (majority:minority) instead of 50:50
            # More conservative than SMOTE, maintains realistic distribution
            fraud_count = y.sum()
            target_normal_count = fraud_count * 3
            
            rus = RandomUnderSampler(
                sampling_strategy={0: target_normal_count, 1: fraud_count},
                random_state=random_state
            )
            
            X_under, y_under = rus.fit_resample(X, y)
            
            balanced_datasets['undersampling'] = {
                'X': X_under,
                'y': y_under,
                'method': 'undersampling',
                'description': f'Random undersampling (3:1 ratio)',
                'use_class_weight': False
            }
            
            logger.info(
                f"UNDERSAMPLING: {len(y_under):,} samples, "
                f"fraud rate: {y_under.sum()/len(y_under):.2%}"
            )
        
        except Exception as e:
            logger.warning(f"Undersampling failed: {e}")
    
    # Apply each method
    for method in methods:
        if method in ['original', 'class_weight_only', 'undersampling']:
            continue
        
        try:
            X_balanced, y_balanced = create_balanced_dataset(
                X, y, method=method, random_state=random_state
            )
            
            balanced_datasets[method] = {
                'X': X_balanced,
                'y': y_balanced,
                'method': method,
                'description': f'{method.upper()} balanced dataset',
                'use_class_weight': False
            }
        
        except Exception as e:
            logger.warning(f"Failed to create {method} dataset: {e}")
    
    # Compact version for large datasets
    if enable_compact and original_size > 1000000:
        logger.info(f"Large dataset detected ({original_size:,}) - creating compact version...")
        
        try:
            X_compact, _, y_compact, _ = train_test_split(
                X, y,
                train_size=min(compact_size, original_size),
                stratify=y,
                random_state=random_state
            )
            
            X_compact_smote, y_compact_smote = create_balanced_dataset(
                X_compact, y_compact, 
                method='smote', 
                random_state=random_state
            )
            
            balanced_datasets['compact_smote'] = {
                'X': X_compact_smote,
                'y': y_compact_smote,
                'method': 'compact_smote',
                'description': f'SMOTE on stratified sample ({compact_size:,})'
            }
        
        except Exception as e:
            logger.warning(f"Compact version creation failed: {e}")
    
    # Log summary
    logger.info(f"✓ Created {len(balanced_datasets)} balanced datasets")
    
    return balanced_datasets


def get_sampling_summary(balanced_datasets: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate summary table of sampling strategies.
    
    Args:
        balanced_datasets: Dictionary from create_sampling_strategies
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for name, data in balanced_datasets.items():
        y = data['y']
        fraud_count = y.sum()
        total_count = len(y)
        fraud_rate = fraud_count / total_count
        normal_count = total_count - fraud_count
        
        summary_data.append({
            'Strategy': name.upper(),
            'Method': data['method'],
            'Total_Samples': f"{total_count:,}",
            'Normal': f"{normal_count:,}",
            'Fraud': f"{fraud_count:,}",
            'Fraud_Rate': f"{fraud_rate:.2%}",
            'Imbalance_Ratio': f"{normal_count/fraud_count:.1f}:1" if fraud_count > 0 else 'N/A'
        })
    
    return pd.DataFrame(summary_data)


def select_best_strategy(
    balanced_datasets: Dict[str, Dict],
    dataset_size: int,
    prefer_compact: bool = True,
    size_threshold: int = 2000000
) -> str:
    """
    Select best sampling strategy based on dataset size.
    
    Args:
        balanced_datasets: Available sampling strategies
        dataset_size: Original dataset size
        prefer_compact: Use compact version for large datasets
        size_threshold: Threshold for considering dataset "large"
        
    Returns:
        Name of recommended strategy
    """
    if dataset_size > size_threshold and 'compact_smote' in balanced_datasets and prefer_compact:
        logger.info(f"Large dataset ({dataset_size:,}) - recommending 'compact_smote'")
        return 'compact_smote'
    
    elif 'smote' in balanced_datasets:
        logger.info("Standard size dataset - recommending 'smote'")
        return 'smote'
    
    elif 'smote_enn' in balanced_datasets:
        logger.info("SMOTE not available - using 'smote_enn'")
        return 'smote_enn'
    
    else:
        logger.warning("No balanced dataset available - using 'original'")
        return 'original'
