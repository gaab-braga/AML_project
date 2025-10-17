"""
Advanced model training utilities for Phase 3.

Simplified functions to reduce notebook code clutter.

This module was consolidated from multiple sub-modules including:
- Advanced model training utilities
- CatBoost integration
- Model selection utilities
- Monte Carlo cross-validation
- Fraud detection metrics
"""

__all__ = [
    # Advanced training
    'train_advanced_models',
    'create_tuning_comparison',
    'ensure_catboost',
    
    # CatBoost integration
    'CatBoostAMLModel',
    'create_catboost_param_space',
    'compare_catboost_vs_lightgbm',
    'add_catboost_to_pipeline',
    'get_catboost_categorical_encoding_comparison',
    
    # Model selection
    'normalize_metrics',
    'compute_retention',
    'rank_models',
    'apply_core_full_policy',
    'decide_best_model',
    'create_model_metadata',
    'save_best_model_meta',
    'load_best_model_meta',
    'select_best_model',
    'get_model_file_path',
    'format_selection_summary',
    
    # Monte Carlo CV
    'monte_carlo_cross_validation',
    'compare_models_mccv',
    
    # Modeling utilities
    'FraudMetrics',
    'get_cv_strategy',
    'train_with_early_stopping',
    'cross_validate_with_metrics',
    'calculate_class_weights',
]

from typing import Dict, Any, Optional
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def train_advanced_models(
    tuning_results: Dict[str, Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Treina modelos finais com parâmetros otimizados e avalia no test set.
    
    Args:
        tuning_results: Resultados do hyperparameter tuning
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        random_state: Seed para reprodutibilidade
        
    Returns:
        Dict com modelos treinados e métricas
    """
    from .modeling import FraudMetrics
    
    models = {}
    metrics = {}
    
    for model_name, tuning_result in tuning_results.items():
        best_params = tuning_result['best_params']
        
        # Criar modelo com parâmetros otimizados
        if model_name == 'LightGBM':
            model = LGBMClassifier(
                **best_params,
                n_estimators=1000,
                random_state=random_state,
                verbosity=-1
            )
        elif model_name == 'XGBoost':
            model = XGBClassifier(
                **best_params,
                n_estimators=1000,
                random_state=random_state,
                verbosity=0
            )
        elif model_name == 'CatBoost':
            if not CATBOOST_AVAILABLE:
                print(f"[WARNING] CatBoost not available, skipping {model_name}")
                continue
            model = CatBoostClassifier(
                **best_params,
                iterations=1000,
                random_state=random_state,
                verbose=False
            )
        else:
            print(f"[WARNING] Unknown model: {model_name}")
            continue
        
        # Treinar
        model.fit(X_train, y_train)
        models[model_name] = model
        
        # Avaliar
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[model_name] = {
            'pr_auc': FraudMetrics.pr_auc_score(y_test, y_proba),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"  {model_name}: PR-AUC={metrics[model_name]['pr_auc']:.4f} | "
              f"ROC-AUC={metrics[model_name]['roc_auc']:.4f}")
    
    return {'models': models, 'metrics': metrics}


def create_tuning_comparison(
    tuning_results: Dict[str, Dict],
    baseline_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Cria tabela comparativa entre tuning básico e avançado.
    
    Args:
        tuning_results: Resultados do tuning avançado
        baseline_scores: Scores do tuning básico
        
    Returns:
        DataFrame com comparação
    """
    comparison_data = []
    
    for model_name in baseline_scores.keys():
        if model_name not in tuning_results:
            continue
            
        baseline = baseline_scores[model_name]
        advanced = tuning_results[model_name]['best_score']
        improvement = advanced - baseline
        improvement_pct = (improvement / baseline) * 100
        
        comparison_data.append({
            'Modelo': model_name,
            'Baseline': f"{baseline:.4f}",
            'Advanced': f"{advanced:.4f}",
            'Ganho': f"+{improvement:.4f} ({improvement_pct:+.1f}%)"
        })
    
    # Adicionar novos modelos
    for model_name in tuning_results.keys():
        if model_name not in baseline_scores:
            comparison_data.append({
                'Modelo': model_name,
                'Baseline': 'N/A',
                'Advanced': f"{tuning_results[model_name]['best_score']:.4f}",
                'Ganho': 'NOVO'
            })
    
    return pd.DataFrame(comparison_data)


def ensure_catboost():
    """Instala CatBoost se necessário."""
    try:
        import catboost
        return True
    except ImportError:
        print("[INFO] Installing CatBoost...")
        import subprocess
        try:
            subprocess.check_call(['pip', 'install', 'catboost'], 
                                stdout=subprocess.DEVNULL)
            print("[OK] CatBoost installed")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to install CatBoost: {e}")
            return False

"""
CatBoost Integration Module

Integrates CatBoost into the AML pipeline with:
- Automatic categorical feature detection
- Optimized hyperparameter spaces
- GPU support (if available)
- Comparison with LightGBM/XGBoost

CatBoost Advantages for AML:
1. Native handling of categorical features (no encoding needed)
2. Robust to overfitting
3. Fast training with GPU
4. Excellent performance on tabular data
5. Built-in overfitting detector

Author: Data Science Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from catboost import CatBoostClassifier, Pool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatBoostAMLModel:
    """
    CatBoost wrapper optimized for AML detection.
    
    Parameters
    ----------
    categorical_features : list of str, optional
        List of categorical feature names. If None, auto-detected.
    use_gpu : bool, default=False
        Whether to use GPU for training
    random_state : int, default=42
        Random seed
    **kwargs : dict
        Additional CatBoost parameters
        
    Attributes
    ----------
    model_ : CatBoostClassifier
        Fitted model
    categorical_features_ : list
        List of categorical feature names
    feature_importance_ : pd.DataFrame
        Feature importance scores
        
    Examples
    --------
    >>> model = CatBoostAMLModel(categorical_features=['From Bank', 'To Bank'])
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        use_gpu: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        self.categorical_features = categorical_features
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None
        self.categorical_features_ = None
        self.feature_importance_ = None
    
    def _detect_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        Auto-detect categorical features.
        
        Heuristic: object dtype or int with < 50 unique values
        """
        cat_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                cat_features.append(col)
            elif X[col].dtype in ['int64', 'int32']:
                if X[col].nunique() < 50:
                    cat_features.append(col)
        
        logger.info(f"Auto-detected {len(cat_features)} categorical features")
        return cat_features
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ):
        """
        Fit CatBoost model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping
        early_stopping_rounds : int, default=50
            Early stopping patience
        verbose : bool, default=False
            Print training progress
        """
        # Detect categorical features if not provided
        if self.categorical_features is None:
            self.categorical_features_ = self._detect_categorical_features(X)
        else:
            self.categorical_features_ = self.categorical_features
        
        # Get categorical feature indices
        cat_indices = [X.columns.get_loc(col) for col in self.categorical_features_ 
                      if col in X.columns]
        
        # Default parameters optimized for fraud detection
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': self.random_state,
            'verbose': verbose,
            'task_type': 'GPU' if self.use_gpu else 'CPU',
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'auto_class_weights': 'Balanced',  # For imbalanced data
            'od_type': 'Iter',  # Overfitting detector
            'od_wait': early_stopping_rounds
        }
        
        # Merge with user parameters
        params = {**default_params, **self.kwargs}
        
        # Create model
        self.model_ = CatBoostClassifier(**params)
        
        # Create Pool objects for faster training
        train_pool = Pool(
            X,
            y,
            cat_features=cat_indices
        )
        
        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_pool = Pool(
                X_val,
                y_val,
                cat_features=cat_indices
            )
        
        # Fit model
        logger.info(f"Training CatBoost with {len(cat_indices)} categorical features...")
        
        self.model_.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=verbose
        )
        
        # Calculate feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"✅ Training complete. Best iteration: {self.model_.best_iteration_}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.predict(X)
    
    def get_feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Get top K most important features.
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.feature_importance_.head(top_k)


def create_catboost_param_space(trial) -> Dict:
    """
    Create Optuna hyperparameter search space for CatBoost.
    
    Optimized for fraud detection with imbalanced data.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
        
    Returns
    -------
    params : dict
        Dictionary of hyperparameters
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'max_bin': trial.suggest_int('max_bin', 200, 400)
    }


def compare_catboost_vs_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: Optional[List[str]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare CatBoost vs LightGBM performance.
    
    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    X_test, y_test : pd.DataFrame, pd.Series
        Test data
    categorical_features : list, optional
        Categorical feature names
    random_state : int
        Random seed
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with metrics
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score
    )
    import time
    
    results = []
    
    # CatBoost
    logger.info("Training CatBoost...")
    start_time = time.time()
    
    catboost_model = CatBoostAMLModel(
        categorical_features=categorical_features,
        random_state=random_state,
        iterations=500,
        verbose=False
    )
    catboost_model.fit(X_train, y_train)
    
    cb_train_time = time.time() - start_time
    
    # Inference time
    start_time = time.time()
    cb_pred_proba = catboost_model.predict_proba(X_test)[:, 1]
    cb_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    cb_pred = (cb_pred_proba >= 0.5).astype(int)
    
    results.append({
        'Model': 'CatBoost',
        'ROC_AUC': roc_auc_score(y_test, cb_pred_proba),
        'PR_AUC': average_precision_score(y_test, cb_pred_proba),
        'F1': f1_score(y_test, cb_pred),
        'Precision': precision_score(y_test, cb_pred),
        'Recall': recall_score(y_test, cb_pred),
        'Train_Time_s': cb_train_time,
        'Inference_Time_ms': cb_inference_time
    })
    
    # LightGBM
    logger.info("Training LightGBM...")
    start_time = time.time()
    
    lgbm_model = LGBMClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight='balanced',
        verbosity=-1
    )
    lgbm_model.fit(X_train, y_train)
    
    lgbm_train_time = time.time() - start_time
    
    # Inference time
    start_time = time.time()
    lgbm_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_inference_time = (time.time() - start_time) / len(X_test) * 1000
    
    lgbm_pred = (lgbm_pred_proba >= 0.5).astype(int)
    
    results.append({
        'Model': 'LightGBM',
        'ROC_AUC': roc_auc_score(y_test, lgbm_pred_proba),
        'PR_AUC': average_precision_score(y_test, lgbm_pred_proba),
        'F1': f1_score(y_test, lgbm_pred),
        'Precision': precision_score(y_test, lgbm_pred),
        'Recall': recall_score(y_test, lgbm_pred),
        'Train_Time_s': lgbm_train_time,
        'Inference_Time_ms': lgbm_inference_time
    })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('PR_AUC', ascending=False)
    
    # Add difference
    if len(comparison_df) == 2:
        pr_auc_diff = comparison_df.iloc[0]['PR_AUC'] - comparison_df.iloc[1]['PR_AUC']
        logger.info(f"\n{'='*60}")
        logger.info(f"Winner: {comparison_df.iloc[0]['Model']}")
        logger.info(f"PR-AUC Difference: {pr_auc_diff:+.4f}")
        logger.info(f"{'='*60}")
    
    return comparison_df


def add_catboost_to_pipeline(
    tuning_config: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: Optional[List[str]] = None
) -> Dict:
    """
    Add CatBoost to existing tuning pipeline.
    
    Parameters
    ----------
    tuning_config : dict
        Existing tuning configuration
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    categorical_features : list, optional
        Categorical feature names
        
    Returns
    -------
    updated_config : dict
        Configuration with CatBoost added
    """
    from .tuning import run_staged_tuning
    
    logger.info("Adding CatBoost to tuning pipeline...")
    
    # Run staged tuning for CatBoost
    catboost_results = run_staged_tuning(
        model_name='CatBoost',
        X_train=X_train,
        y_train=y_train,
        config=tuning_config,
        metric='pr_auc',
        categorical_features=categorical_features
    )
    
    return catboost_results


def get_catboost_categorical_encoding_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare CatBoost (native) vs manual encoding strategies.
    
    Shows advantage of CatBoost's native categorical handling.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features with categorical columns
    y : pd.Series
        Target
    categorical_cols : list
        Names of categorical columns
    random_state : int
        Random seed
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison of encoding strategies
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier
    import time
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    results = []
    
    # 1. CatBoost (native categorical handling)
    logger.info("Testing CatBoost (native)...")
    start_time = time.time()
    
    cb_model = CatBoostAMLModel(
        categorical_features=categorical_cols,
        random_state=random_state,
        iterations=300,
        verbose=False
    )
    cb_model.fit(X_train, y_train)
    
    cb_time = time.time() - start_time
    cb_pred = cb_model.predict_proba(X_test)[:, 1]
    cb_auc = roc_auc_score(y_test, cb_pred)
    
    results.append({
        'Strategy': 'CatBoost (Native)',
        'ROC_AUC': cb_auc,
        'Train_Time_s': cb_time,
        'Features': X_train.shape[1]
    })
    
    # 2. Label Encoding + LightGBM
    logger.info("Testing Label Encoding...")
    X_train_le = X_train.copy()
    X_test_le = X_test.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_le[col] = le.fit_transform(X_train_le[col].astype(str))
        X_test_le[col] = le.transform(X_test_le[col].astype(str))
    
    start_time = time.time()
    lgbm_le = LGBMClassifier(n_estimators=300, random_state=random_state, verbosity=-1)
    lgbm_le.fit(X_train_le, y_train)
    le_time = time.time() - start_time
    
    le_pred = lgbm_le.predict_proba(X_test_le)[:, 1]
    le_auc = roc_auc_score(y_test, le_pred)
    
    results.append({
        'Strategy': 'Label Encoding',
        'ROC_AUC': le_auc,
        'Train_Time_s': le_time,
        'Features': X_train_le.shape[1]
    })
    
    # 3. One-Hot Encoding + LightGBM (if feasible)
    unique_counts = X_train[categorical_cols].nunique()
    if unique_counts.max() < 50:  # Only if not too many categories
        logger.info("Testing One-Hot Encoding...")
        
        X_train_ohe = pd.get_dummies(X_train, columns=categorical_cols)
        X_test_ohe = pd.get_dummies(X_test, columns=categorical_cols)
        
        # Align columns
        missing_cols = set(X_train_ohe.columns) - set(X_test_ohe.columns)
        for col in missing_cols:
            X_test_ohe[col] = 0
        X_test_ohe = X_test_ohe[X_train_ohe.columns]
        
        start_time = time.time()
        lgbm_ohe = LGBMClassifier(n_estimators=300, random_state=random_state, verbosity=-1)
        lgbm_ohe.fit(X_train_ohe, y_train)
        ohe_time = time.time() - start_time
        
        ohe_pred = lgbm_ohe.predict_proba(X_test_ohe)[:, 1]
        ohe_auc = roc_auc_score(y_test, ohe_pred)
        
        results.append({
            'Strategy': 'One-Hot Encoding',
            'ROC_AUC': ohe_auc,
            'Train_Time_s': ohe_time,
            'Features': X_train_ohe.shape[1]
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('ROC_AUC', ascending=False)
    
    return comparison_df


if __name__ == "__main__":
    print("CatBoost Integration Module")
    print("=" * 60)
    print("\nKey Features:")
    print("  - Native categorical feature handling")
    print("  - Auto-detection of categorical columns")
    print("  - GPU acceleration support")
    print("  - Optimized hyperparameter spaces")
    print("  - Direct comparison with LightGBM")
    print("\nExample Usage:")
    print("""
from catboost_integration import CatBoostAMLModel

# Define categorical features
cat_features = ['From Bank', 'To Bank', 'Account.1', 'Receiving Currency']

# Train model
model = CatBoostAMLModel(categorical_features=cat_features)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Get feature importance
importance = model.get_feature_importance(top_k=20)
    """)

"""
Model Selection Utilities

Unified logic for selecting the best model across baseline (n06), 
tuning (n07), and threshold calculation (n08) phases.

Key principles:
- Single primary metric (PR_AUC by default)
- Configurable tie-breakers 
- Core vs Full variant retention policy
- Minimum improvement thresholds for complexity increases
- Persistent best model metadata artifact
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def normalize_metrics(df_models: pd.DataFrame, 
                     df_at_k: Optional[pd.DataFrame] = None,
                     config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Normalize and enrich model metrics DataFrame with required columns.
    
    Args:
        df_models: DataFrame with columns like Model, Variant, PR_AUC, ROC_AUC, F1, etc.
        df_at_k: Optional DataFrame with metrics@K (Model, Variant, K, Precision_at_K, Recall_at_K)
        config: Configuration dict with model_selection settings
        
    Returns:
        Enhanced DataFrame with Recall_at_k column added if available
    """
    df = df_models.copy()
    
    # Ensure required columns exist
    required_cols = ['Model', 'Variant', 'PR_AUC', 'ROC_AUC', 'F1', 'Precision', 'Recall']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # Add Recall@K if available
    if df_at_k is not None and config is not None:
        k_ref = config.get('modeling', {}).get('model_selection', {}).get('k_reference', 100)
        
        # Filter for the reference K value
        k_metrics = df_at_k[df_at_k['K'] == k_ref].copy()
        if not k_metrics.empty:
            # Merge Recall@K
            merge_cols = ['Model', 'Variant']
            if 'Recall_at_K' in k_metrics.columns:
                recall_k = k_metrics[merge_cols + ['Recall_at_K']].rename(columns={'Recall_at_K': 'Recall_at_k'})
            elif 'Recall@K' in k_metrics.columns:
                recall_k = k_metrics[merge_cols + ['Recall@K']].rename(columns={'Recall@K': 'Recall_at_k'})
            else:
                recall_k = None
                
            if recall_k is not None:
                df = df.merge(recall_k, on=merge_cols, how='left')
    
    # Fallback: use regular Recall if Recall_at_k not available
    if 'Recall_at_k' not in df.columns:
        df['Recall_at_k'] = df['Recall']
        
    return df


def compute_retention(df_models: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core/full retention ratios for each model.
    
    Args:
        df_models: DataFrame with Model, Variant, PR_AUC columns
        
    Returns:
        DataFrame with retention_core_full column added
    """
    df = df_models.copy()
    
    # Initialize retention column
    df['retention_core_full'] = np.nan
    
    # Group by Model and compute retention
    for model_name in df['Model'].unique():
        model_rows = df[df['Model'] == model_name]
        
        # Find full and core variants
        full_rows = model_rows[model_rows['Variant'] == 'full']
        core_rows = model_rows[model_rows['Variant'] == 'core']
        
        if not full_rows.empty and not core_rows.empty:
            full_pr_auc = full_rows['PR_AUC'].iloc[0]
            core_pr_auc = core_rows['PR_AUC'].iloc[0]
            
            if pd.notna(full_pr_auc) and pd.notna(core_pr_auc) and full_pr_auc > 0:
                retention = core_pr_auc / full_pr_auc
                
                # Update both core and full rows
                df.loc[df['Model'] == model_name, 'retention_core_full'] = retention
                
    return df


def rank_models(df_models: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Rank models according to configured criteria.
    
    Args:
        df_models: Normalized DataFrame with all required metrics
        config: Configuration dict with model_selection settings
        
    Returns:
        DataFrame sorted by ranking criteria with rank column added
    """
    df = df_models.copy()
    
    # Get selection config
    selection_config = config.get('modeling', {}).get('model_selection', {})
    primary_metric = selection_config.get('primary_metric', 'PR_AUC')
    tie_breakers = selection_config.get('tie_breakers', ['ROC_AUC', 'Recall_at_k', 'F1'])
    complexity_order = selection_config.get('complexity_order', [])
    
    # Filter out rows with missing primary metric
    df = df.dropna(subset=[primary_metric]).copy()
    
    if df.empty:
        return df
    
    # Create complexity rank (lower is better)
    if complexity_order:
        df['complexity_rank'] = df['Model'].map(
            {model: idx for idx, model in enumerate(complexity_order)}
        ).fillna(999)  # Unknown models get worst complexity rank
    else:
        df['complexity_rank'] = 0
    
    # Create stability penalty (higher CV_Stability = worse)
    if 'CV_Stability' in df.columns:
        # Penalize high coefficient of variation
        df['stability_rank'] = df['CV_Stability'].fillna(np.inf).rank(ascending=True)
    else:
        df['stability_rank'] = 0
    
    # Build sort criteria
    sort_cols = [primary_metric]
    sort_ascending = [False]  # Primary metric: higher is better
    
    for tie_breaker in tie_breakers:
        if tie_breaker in df.columns:
            sort_cols.append(tie_breaker)
            sort_ascending.append(False)  # All metrics: higher is better
    
    # Add stability as tie-breaker (lower is better)
    if 'CV_Stability' in df.columns:
        sort_cols.append('CV_Stability')
        sort_ascending.append(True)  # Lower coefficient of variation is better
    
    # Add complexity as final tie-breaker (lower is better)
    sort_cols.append('complexity_rank')
    sort_ascending.append(True)
    
    # Sort and add rank
    df = df.sort_values(sort_cols, ascending=sort_ascending).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def apply_core_full_policy(df_ranked: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply core vs full variant selection policy.
    
    Args:
        df_ranked: Ranked DataFrame from rank_models()
        config: Configuration dict with model_selection settings
        
    Returns:
        DataFrame with core/full policy applied
    """
    selection_config = config.get('modeling', {}).get('model_selection', {})
    retention_threshold = selection_config.get('prefer_core_if_retention_gte', 0.95)
    
    df = df_ranked.copy()
    
    # Group by Model and apply policy
    model_groups = []
    
    for model_name in df['Model'].unique():
        model_rows = df[df['Model'] == model_name].copy()
        
        # Find best full and core variants
        full_rows = model_rows[model_rows['Variant'] == 'full']
        core_rows = model_rows[model_rows['Variant'] == 'core']
        
        if not full_rows.empty and not core_rows.empty:
            retention = core_rows['retention_core_full'].iloc[0]
            
            # If core retention is good enough, prefer core
            if pd.notna(retention) and retention >= retention_threshold:
                selected_rows = core_rows
            else:
                selected_rows = full_rows
        else:
            # Keep all variants if no policy applies
            selected_rows = model_rows
            
        model_groups.append(selected_rows)
    
    # Combine and re-rank
    if model_groups:
        df_filtered = pd.concat(model_groups, ignore_index=True)
        # Re-rank after filtering
        df_filtered = rank_models(df_filtered, config)
    else:
        df_filtered = df
        
    return df_filtered


def decide_best_model(df_candidates: pd.DataFrame,
                     existing_meta: Optional[Dict[str, Any]] = None,
                     config: Dict[str, Any] = None) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Apply improvement thresholds and decide final best model.
    
    Args:
        df_candidates: Ranked candidate models
        existing_meta: Previously selected model metadata (from JSON)
        config: Configuration dict with model_selection settings
        
    Returns:
        Tuple of (best_model_row, decision_metadata)
    """
    if df_candidates.empty:
        raise ValueError("No candidate models provided")
        
    selection_config = config.get('modeling', {}).get('model_selection', {}) if config else {}
    primary_metric = selection_config.get('primary_metric', 'PR_AUC')
    min_improvement = selection_config.get('min_improvement_primary', 0.005)
    
    # Get top candidate
    best_candidate = df_candidates.iloc[0]
    
    decision_path = []
    
    if existing_meta is None:
        # First selection (baseline phase)
        decision = {
            'action': 'initial_selection',
            'reason': 'First model selection from baseline evaluation',
            'improvement': None
        }
        decision_path.append(f"Initial: {best_candidate['Model']} {best_candidate['Variant']} "
                           f"{primary_metric}={best_candidate[primary_metric]:.4f}")
        
    else:
        # Comparing against existing model
        existing_metric = existing_meta.get('primary_value', 0)
        current_metric = best_candidate[primary_metric]
        improvement = current_metric - existing_metric
        
        decision_path.append(f"Existing: {existing_meta.get('model_name', 'unknown')} "
                           f"{primary_metric}={existing_metric:.4f}")
        decision_path.append(f"Candidate: {best_candidate['Model']} {best_candidate['Variant']} "
                           f"{primary_metric}={current_metric:.4f}")
        
        if improvement >= min_improvement:
            decision = {
                'action': 'upgrade',
                'reason': f'Improvement {improvement:.4f} >= threshold {min_improvement}',
                'improvement': float(improvement)
            }
            decision_path.append(f"Decision: UPGRADE (+{improvement:.4f} >= {min_improvement})")
        else:
            decision = {
                'action': 'retain',
                'reason': f'Improvement {improvement:.4f} < threshold {min_improvement}',
                'improvement': float(improvement)
            }
            decision_path.append(f"Decision: RETAIN (+{improvement:.4f} < {min_improvement})")
            
            # Return existing model info instead
            if 'model_name' in existing_meta:
                # Create a synthetic row representing the existing model
                existing_row = pd.Series({
                    'Model': existing_meta['model_name'],
                    'Variant': existing_meta.get('variant', 'unknown'),
                    primary_metric: existing_meta.get('primary_value', 0)
                })
                return existing_row, {**decision, 'decision_path': decision_path}
    
    return best_candidate, {**decision, 'decision_path': decision_path}


def create_model_metadata(model_row: pd.Series,
                         decision_meta: Dict[str, Any],
                         config: Dict[str, Any],
                         source: str = "baseline",
                         candidates_data: Optional[Dict] = None,
                         rejected_models: Optional[List] = None) -> Dict[str, Any]:
    """
    Create comprehensive model metadata for persistence.
    
    Args:
        model_row: Selected model row from DataFrame
        decision_meta: Decision metadata from decide_best_model
        config: Full configuration dict
        source: Source phase ("baseline" or "tuning")
        candidates_data: Gating candidates data (if available)
        rejected_models: List of rejected models with reasons
        
    Returns:
        Complete metadata dictionary
    """
    selection_config = config.get('modeling', {}).get('model_selection', {})
    primary_metric = selection_config.get('primary_metric', 'PR_AUC')
    
    # Determine selection stage
    if source == "baseline":
        selection_stage = "baseline_evaluation"
    elif model_row.get('Stage') == 'coarse':
        selection_stage = "coarse_tuning"
    elif model_row.get('Stage') == 'fine':
        selection_stage = "fine_tuning"
    else:
        selection_stage = "tuning"
    
    # Build metadata
    metadata = {
        'model_name': model_row.get('Model', 'unknown'),
        'variant': model_row.get('Variant', 'unknown'),
        'source': source,
        'selection_stage': selection_stage,
        'primary_metric': primary_metric,
        'primary_value': float(model_row.get(primary_metric, 0)),
        
        # Tie-breaker metrics
        'tie_breakers': {
            'ROC_AUC': float(model_row.get('ROC_AUC', np.nan)),
            'F1': float(model_row.get('F1', np.nan)),
            'Recall_at_k': float(model_row.get('Recall_at_k', np.nan)),
            'Precision': float(model_row.get('Precision', np.nan)),
            'Recall': float(model_row.get('Recall', np.nan))
        },
        
        # Stability metrics (new)
        'stability_metrics': {
            'cv_pr_auc_mean': float(model_row.get('CV_PR_AUC_Mean', np.nan)),
            'cv_pr_auc_std': float(model_row.get('CV_PR_AUC_Std', np.nan)),
            'cv_stability_coef': float(model_row.get('CV_Stability', np.nan))  # CV = std/mean
        },
        
        # Additional info
        'retention_core_full': float(model_row.get('retention_core_full', np.nan)) 
                               if pd.notna(model_row.get('retention_core_full')) else None,
        'n_features_used': int(model_row.get('n_features_used', 0)) 
                          if pd.notna(model_row.get('n_features_used')) else None,
        'train_time_sec': float(model_row.get('Train_Time_Sec', np.nan))
                         if pd.notna(model_row.get('Train_Time_Sec')) else None,
        
        # Tuning-specific info (if applicable)
        'tuning_info': {
            'stage': model_row.get('Stage'),
            'trials_executed': int(model_row.get('Tuning_Trials', 0)) 
                              if pd.notna(model_row.get('Tuning_Trials')) else None,
            'best_params': model_row.get('Best_Params'),
            'baseline_score': float(model_row.get('Baseline_Score', np.nan))
                             if pd.notna(model_row.get('Baseline_Score')) else None,
            'improvement_vs_baseline': float(model_row.get('Improvement_vs_Baseline', np.nan))
                                      if pd.notna(model_row.get('Improvement_vs_Baseline')) else None
        } if source == "tuning" else None,
        
        # Decision tracking
        'improvement_over_baseline': decision_meta.get('improvement'),
        'decision_action': decision_meta.get('action', 'unknown'),
        'decision_reason': decision_meta.get('reason', ''),
        'decision_path': decision_meta.get('decision_path', []),
        
        # Gating info (if available)
        'gating_info': {
            'candidates_evaluated': candidates_data.get('candidates_count', 0) if candidates_data else 0,
            'rejected_count': candidates_data.get('rejected_count', 0) if candidates_data else 0,
            'gating_criteria': candidates_data.get('criteria_applied') if candidates_data else None,
            'rejected_models': rejected_models or []
        } if candidates_data or rejected_models else None,
        
        # Timestamps and versioning
        'selected_at': datetime.utcnow().isoformat(),
        'config_snapshot': {
            'primary_metric': selection_config.get('primary_metric'),
            'tie_breakers': selection_config.get('tie_breakers'),
            'min_improvement_primary': selection_config.get('min_improvement_primary'),
            'prefer_core_if_retention_gte': selection_config.get('prefer_core_if_retention_gte'),
            'k_reference': selection_config.get('k_reference'),
            'gating_config': config.get('modeling', {}).get('gating', {}) if config.get('modeling', {}).get('gating') else None
        }
    }
    
    # Clean NaN values for JSON serialization
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj
            
    return clean_nans(metadata)


def save_best_model_meta(metadata: Dict[str, Any], artifacts_dir: Path) -> Path:
    """
    Save best model metadata to JSON file.
    
    Args:
        metadata: Model metadata dictionary
        artifacts_dir: Artifacts directory path
        
    Returns:
        Path to saved JSON file
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)
    
    json_path = artifacts_dir / "best_model_meta.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    return json_path


def load_best_model_meta(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load existing best model metadata from JSON file.
    
    Args:
        artifacts_dir: Artifacts directory path
        
    Returns:
        Metadata dictionary or None if file doesn't exist
    """
    json_path = Path(artifacts_dir) / "best_model_meta.json"
    
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_path}: {e}")
            return None
    
    return None


def select_best_model(df_models: pd.DataFrame,
                     df_at_k: Optional[pd.DataFrame] = None,
                     config: Dict[str, Any] = None,
                     artifacts_dir: Path = None,
                     source: str = "baseline") -> Dict[str, Any]:
    """
    Complete model selection pipeline.
    
    Args:
        df_models: Model metrics DataFrame
        df_at_k: Optional metrics@K DataFrame  
        config: Configuration dictionary
        artifacts_dir: Directory to save/load metadata
        source: Selection source ("baseline" or "tuning")
        
    Returns:
        Dictionary with selected model metadata and file paths
    """
    if df_models.empty:
        raise ValueError("No models provided for selection")
        
    if config is None:
        config = {}
        
    if artifacts_dir is None:
        artifacts_dir = Path("../artifacts")
    
    # Step 1: Normalize and enrich metrics
    df_normalized = normalize_metrics(df_models, df_at_k, config)
    
    # Step 2: Compute retention ratios
    df_with_retention = compute_retention(df_normalized)
    
    # Step 3: Rank models
    df_ranked = rank_models(df_with_retention, config)
    
    # Step 4: Apply core/full policy
    df_candidates = apply_core_full_policy(df_ranked, config)
    
    # Step 5: Load existing metadata and decide
    existing_meta = load_best_model_meta(artifacts_dir)
    best_model, decision_meta = decide_best_model(df_candidates, existing_meta, config)
    
    # Step 6: Create and save metadata
    metadata = create_model_metadata(best_model, decision_meta, config, source)
    json_path = save_best_model_meta(metadata, artifacts_dir)
    
    return {
        'metadata': metadata,
        'json_path': str(json_path),
        'selected_row': best_model,
        'candidates_df': df_candidates,
        'decision': decision_meta
    }


def get_model_file_path(metadata: Dict[str, Any], models_dir: Path) -> Path:
    """
    Get the file path for the selected model based on metadata.
    
    Args:
        metadata: Model metadata from best_model_meta.json
        models_dir: Models directory path
        
    Returns:
        Path to the model pickle file
    """
    models_dir = Path(models_dir)
    
    model_name = metadata.get('model_name', '')
    variant = metadata.get('variant', '')
    source = metadata.get('source', '')
    
    # Determine file name based on source and variant
    if source == "tuning":
        return models_dir / "best_model_tuned.pkl"
    elif variant == "core":
        return models_dir / "best_baseline_core.pkl"
    else:  # baseline full or unknown
        return models_dir / "best_baseline.pkl"


# Helper function for backward compatibility
def format_selection_summary(result: Dict[str, Any]) -> str:
    """
    Format selection result as readable summary.
    
    Args:
        result: Result from select_best_model()
        
    Returns:
        Formatted summary string
    """
    metadata = result['metadata']
    decision = result['decision']
    
    lines = [
        f"🎯 Best Model Selected:",
        f"   Model: {metadata['model_name']} ({metadata['variant']})",
        f"   {metadata['primary_metric']}: {metadata['primary_value']:.4f}",
        f"   Source: {metadata['source']}",
        f"   Action: {decision['action'].upper()}",
        f"   Reason: {decision['reason']}",
    ]
    
    if metadata.get('retention_core_full'):
        lines.append(f"   Core Retention: {metadata['retention_core_full']:.1%}")
        
    if decision['improvement'] is not None:
        lines.append(f"   Improvement: +{decision['improvement']:.4f}")
        
    lines.append(f"   Metadata: {result['json_path']}")
    
    return "\n".join(lines)


# ============================================================================
# FASE 2: Monte Carlo Cross-Validation
# ============================================================================

def monte_carlo_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 30,
    test_size: float = 0.3,
    metric_fn: callable = None,
    random_state: int = 42,
    stratify: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Monte Carlo Cross-Validation (MCCV) for robust performance estimation.
    
    Repeatedly splits data randomly and evaluates model to get distribution
    of performance metrics. More robust than k-fold CV for small datasets
    or when you want confidence intervals.
    
    Parameters
    ----------
    model : estimator
        Model to evaluate (will be cloned for each iteration)
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_iterations : int, default=30
        Number of random train/test splits
    test_size : float, default=0.3
        Proportion of data for testing
    metric_fn : callable, optional
        Custom metric function with signature: metric_fn(y_true, y_pred) -> dict
        If None, uses default fraud detection metrics
    random_state : int, default=42
        Base random seed (each iteration adds iteration number)
    stratify : bool, default=True
        Whether to stratify splits by target variable
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'scores': List of scores from each iteration
        - 'mean': Mean score
        - 'std': Standard deviation
        - 'ci_95': 95% confidence interval
        - 'median': Median score
        - 'min': Minimum score
        - 'max': Maximum score
        - 'cv_coefficient': Coefficient of variation (std/mean)
        - 'all_metrics': DataFrame with all metrics per iteration
        
    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> model = LGBMClassifier(random_state=42)
    >>> results = monte_carlo_cross_validation(
    ...     model, X, y, n_iterations=30, test_size=0.3
    ... )
    >>> print(f"Score: {results['mean']:.4f} ± {results['std']:.4f}")
    >>> print(f"95% CI: [{results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f}]")
    
    References
    ----------
    - Xu & Goodacre (2018) "On Splitting Training and Validation Set: A Comparative Study"
    - Arlot & Celisse (2010) "A survey of cross-validation procedures for model selection"
    """
    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from scipy import stats
    
    if metric_fn is None:
        # Use default AML metrics
        from .metrics import pr_auc_score
        from .metrics import roc_auc_score, f1_score, precision_score, recall_score
        
        def default_metric_fn(y_true, y_pred_proba):
            y_pred = (y_pred_proba >= 0.5).astype(int)
            return {
                'pr_auc': pr_auc_score(y_true, y_pred_proba),
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0)
            }
        
        metric_fn = default_metric_fn
    
    if verbose:
        print(f"Running Monte Carlo CV: {n_iterations} iterations...")
        print(f"Test size: {test_size:.1%}, Stratify: {stratify}")
    
    all_metrics = []
    
    for i in range(n_iterations):
        # Random split
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state + i,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state + i
            )
        
        # Clone and train model
        model_iter = clone(model)
        model_iter.fit(X_train, y_train)
        
        # Predict
        if hasattr(model_iter, 'predict_proba'):
            y_pred_proba = model_iter.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model_iter.decision_function(X_test)
        
        # Calculate metrics
        metrics = metric_fn(y_test.values, y_pred_proba)
        metrics['iteration'] = i + 1
        all_metrics.append(metrics)
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations...")
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    
    # Calculate statistics for primary metric (assume first metric is primary)
    metric_names = [col for col in df_metrics.columns if col != 'iteration']
    primary_metric = metric_names[0]
    scores = df_metrics[primary_metric].values
    
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    median_score = np.median(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # 95% confidence interval using t-distribution
    ci_95 = stats.t.interval(
        0.95,
        len(scores) - 1,
        loc=mean_score,
        scale=stats.sem(scores)
    )
    
    # Coefficient of variation
    cv_coef = std_score / mean_score if mean_score != 0 else np.inf
    
    results = {
        'scores': scores.tolist(),
        'mean': mean_score,
        'std': std_score,
        'ci_95': ci_95,
        'median': median_score,
        'min': min_score,
        'max': max_score,
        'cv_coefficient': cv_coef,
        'all_metrics': df_metrics,
        'primary_metric': primary_metric,
        'n_iterations': n_iterations,
        'test_size': test_size
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Monte Carlo CV Results")
        print(f"{'='*60}")
        print(f"Primary Metric: {primary_metric}")
        print(f"Mean: {mean_score:.4f} ± {std_score:.4f}")
        print(f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"Median: {median_score:.4f}")
        print(f"Range: [{min_score:.4f}, {max_score:.4f}]")
        print(f"CV Coefficient: {cv_coef:.4f}")
        print(f"{'='*60}\n")
    
    return results


def compare_models_mccv(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 30,
    test_size: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple models using Monte Carlo Cross-Validation.
    
    Parameters
    ----------
    models : dict
        Dictionary of {model_name: model_instance}
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_iterations : int
        Number of MCCV iterations
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with mean, std, CI, and statistical tests
        
    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> from xgboost import XGBClassifier
    >>> 
    >>> models = {
    ...     'LightGBM': LGBMClassifier(random_state=42),
    ...     'XGBoost': XGBClassifier(random_state=42)
    ... }
    >>> 
    >>> comparison = compare_models_mccv(models, X, y, n_iterations=30)
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}")
        print("="*60)
        
        result = monte_carlo_cross_validation(
            model, X, y,
            n_iterations=n_iterations,
            test_size=test_size,
            random_state=random_state,
            verbose=True
        )
        
        results[model_name] = result
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Mean_Score': result['mean'],
            'Std_Score': result['std'],
            'CI_Lower': result['ci_95'][0],
            'CI_Upper': result['ci_95'][1],
            'Median': result['median'],
            'Min': result['min'],
            'Max': result['max'],
            'CV_Coefficient': result['cv_coefficient']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_Score', ascending=False)
    
    # Add statistical significance tests
    if len(models) >= 2:
        comparison_df = _add_pairwise_tests_mccv(comparison_df, results)
    
    return comparison_df


def _add_pairwise_tests_mccv(comparison_df: pd.DataFrame, results: Dict) -> pd.DataFrame:
    """
    Add pairwise Wilcoxon signed-rank tests for MCCV results.
    """
    from scipy.stats import wilcoxon, mannwhitneyu
    
    best_model = comparison_df.iloc[0]['Model']
    best_scores = results[best_model]['scores']
    
    significance = []
    p_values = []
    
    for model_name in comparison_df['Model']:
        if model_name == best_model:
            significance.append('Best')
            p_values.append(np.nan)
        else:
            current_scores = results[model_name]['scores']
            
            # Use Wilcoxon signed-rank test (paired)
            # Or Mann-Whitney U if sample sizes differ
            if len(best_scores) == len(current_scores):
                try:
                    _, pvalue = wilcoxon(best_scores, current_scores)
                except:
                    _, pvalue = mannwhitneyu(best_scores, current_scores)
            else:
                _, pvalue = mannwhitneyu(best_scores, current_scores)
            
            p_values.append(pvalue)
            
            if pvalue < 0.01:
                significance.append('*** (p<0.01)')
            elif pvalue < 0.05:
                significance.append('** (p<0.05)')
            elif pvalue < 0.10:
                significance.append('* (p<0.10)')
            else:
                significance.append('n.s.')
    
    comparison_df['P_Value_vs_Best'] = p_values
    comparison_df['Significance'] = significance
    
    return comparison_df

"""
Modeling utilities for fraud detection.

Professional, tested functions for metrics, cross-validation, 
and model training with early stopping.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, 
    average_precision_score, recall_score, precision_score, 
    f1_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class FraudMetrics:
    """Specialized metrics for fraud detection scenarios."""
    
    @staticmethod
    def pr_auc_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calculate PR-AUC (primary metric for imbalanced classes).
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            PR-AUC score (Area Under Precision-Recall Curve)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return auc(recall, precision)
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
        """
        Calculate recall in top-k highest scores.
        
        Critical for fraud detection where investigation capacity is limited.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Recall score for top-k predictions
        """
        if len(y_true) < k:
            k = len(y_true)
        
        indices = np.argsort(y_proba)[::-1][:k]
        y_pred_at_k = np.zeros_like(y_true)
        y_pred_at_k[indices] = 1
        
        return recall_score(y_true, y_pred_at_k)
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
        """Calculate precision in top-k highest scores."""
        if len(y_true) < k:
            k = len(y_true)
        
        indices = np.argsort(y_proba)[::-1][:k]
        y_pred_at_k = np.zeros_like(y_true)
        y_pred_at_k[indices] = 1
        
        return precision_score(y_true, y_pred_at_k, zero_division=0)
    
    @staticmethod
    def compute_all(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        threshold: float = 0.5,
        k_values: list = [50, 100, 200, 500]
    ) -> Dict[str, float]:
        """
        Compute all fraud detection metrics in a single call.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            threshold: Classification threshold
            k_values: List of k values for recall@k
            
        Returns:
            Dictionary with all computed metrics
        """
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'pr_auc': FraudMetrics.pr_auc_score(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'avg_precision': average_precision_score(y_true, y_proba),
            'recall_fraud': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'precision_fraud': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_fraud': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Metrics @k
        metrics['metrics_at_k'] = {}
        for k in k_values:
            if k <= len(y_true):
                metrics['metrics_at_k'][f'recall_at_{k}'] = FraudMetrics.recall_at_k(y_true, y_proba, k)
                metrics['metrics_at_k'][f'precision_at_{k}'] = FraudMetrics.precision_at_k(y_true, y_proba, k)
        
        return metrics


def get_cv_strategy(
    strategy: str = 'stratified', 
    n_splits: int = 5, 
    test_size: Optional[float] = None,
    random_state: int = 42
):
    """
    Return appropriate cross-validation strategy.
    
    Args:
        strategy: 'stratified' or 'timeseries'
        n_splits: Number of CV folds
        test_size: For TimeSeriesSplit (optional)
        random_state: Random seed
        
    Returns:
        CV splitter object
    """
    if strategy == 'stratified':
        return StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
    elif strategy == 'timeseries':
        if test_size is None:
            test_size = 1000
        return TimeSeriesSplit(n_splits=n_splits, test_size=int(test_size))
    else:
        raise ValueError(f"Strategy '{strategy}' not supported. Use 'stratified' or 'timeseries'.")


def train_with_early_stopping(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = 'lightgbm',
    early_stopping_rounds: int = 50,
    eval_metric: str = 'auc',
    verbose: bool = False
):
    """
    Wrapper for training with early stopping (GBDT models).
    
    Args:
        model: Model instance (LightGBM, XGBoost, etc.)
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        model_type: 'lightgbm', 'xgboost', or 'other'
        early_stopping_rounds: Rounds without improvement before stopping
        eval_metric: Metric to monitor
        verbose: Print training logs
        
    Returns:
        Trained model
    """
    model_type = model_type.lower()
    
    if model_type in ['lightgbm', 'xgboost'] and X_val is not None and y_val is not None:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=eval_metric,
                callbacks=[
                    model.early_stopping(early_stopping_rounds, verbose=verbose)
                ] if model_type == 'lightgbm' else None,
                early_stopping_rounds=early_stopping_rounds if model_type == 'xgboost' else None,
                verbose=verbose
            )
        except Exception as e:
            logger.warning(f"Early stopping failed: {e}. Training without validation.")
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    return model


def cross_validate_with_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_strategy,
    metric_fn: Callable = None,
    return_models: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Cross-validation with fraud-specific metrics.
    
    Args:
        model: Model instance to evaluate
        X: Features
        y: Labels
        cv_strategy: CV splitter
        metric_fn: Custom metric function (default: FraudMetrics.compute_all)
        return_models: Whether to return trained models per fold
        verbose: Print progress
        
    Returns:
        Dictionary with CV results
    """
    if metric_fn is None:
        metric_fn = FraudMetrics.compute_all
    
    results = {
        'fold_metrics': [],
        'fold_models': [] if return_models else None
    }
    
    for fold_num, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)
        
        # Train
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
        
        # Metrics
        fold_metrics = metric_fn(y_val_fold.values, y_proba)
        fold_metrics['fold'] = fold_num
        results['fold_metrics'].append(fold_metrics)
        
        if return_models:
            results['fold_models'].append(fold_model)
        
        if verbose:
            pr_auc = fold_metrics.get('pr_auc', 0)
            logger.info(f"Fold {fold_num}: PR-AUC = {pr_auc:.4f}")
    
    # Aggregate metrics
    results['mean_metrics'] = _aggregate_fold_metrics(results['fold_metrics'])
    
    return results


def _aggregate_fold_metrics(fold_metrics: list) -> Dict[str, float]:
    """Aggregate metrics across CV folds."""
    aggregated = {}
    
    # Simple metrics
    for key in ['pr_auc', 'roc_auc', 'avg_precision', 'recall_fraud', 'precision_fraud', 'f1_fraud']:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
    
    # Metrics @k
    if 'metrics_at_k' in fold_metrics[0]:
        aggregated['metrics_at_k'] = {}
        for k_metric in fold_metrics[0]['metrics_at_k']:
            values = [m['metrics_at_k'][k_metric] for m in fold_metrics]
            aggregated['metrics_at_k'][f'{k_metric}_mean'] = np.mean(values)
            aggregated['metrics_at_k'][f'{k_metric}_std'] = np.std(values)
    
    return aggregated


def calculate_class_weights(y: pd.Series, method: str = 'balanced') -> Dict[int, float]:
    """
    Calculate class weights for balancing.
    
    Args:
        y: Target labels
        method: 'balanced' or custom method
        
    Returns:
        Dictionary mapping class labels to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    if method == 'balanced':
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    return None
