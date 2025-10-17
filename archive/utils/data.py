"""
Data Module - Central Pipeline & I/O Functions

This module provides two main sets of functionality:

1. PIPELINE FUNCTIONS (Lines ~50-300):
   - setup_standard_environment: Standardized notebook setup
   - load_datasets_standard: Load train/test datasets
   - evaluate_model_standard: Quick model evaluation
   - run_feature_engineering_pipeline: Integrated feature engineering
   - run_feature_selection_pipeline: Integrated feature selection

2. I/O & UTILITIES (Lines ~300-900):
   - optimize_dtypes: Memory-efficient DataFrame optimization
   - load_data/save_artifact: Flexible data loading/saving
   - Temporal split functions: Time-aware train/test splits
   
USAGE:
    from utils.data import setup_standard_environment, load_datasets_standard
    from utils import load_data, save_artifact  # Via __init__.py
    
⚠️ NOTE: All imports from other utils modules use relative imports (.metrics, .config)
         to ensure proper package structure.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

# ============================================================================
# IMPORTAÇÕES CENTRALIZADAS (reutiliza código existente)
# ============================================================================

# Métricas (usa os módulos existentes)
_has_metrics = False
try:
    # Try relative imports first (when imported as package)
    from .metrics import bootstrap_metric, compute_cv_metrics
    from .metrics import compute_at_k, format_at_k
    from .metrics import compute_calibration_metrics
    _has_metrics = True
except (ImportError, ValueError):
    # Fall back to absolute imports (when imported directly)
    try:
        from utils.metrics import bootstrap_metric, compute_cv_metrics
        from utils.metrics import compute_at_k, format_at_k
        from utils.metrics import compute_calibration_metrics
        _has_metrics = True
    except ImportError:
        pass

# Data processing (usa os módulos existentes)
_has_data_processing = False
try:
    # Try relative imports first
    from .preprocessing import (
        apply_frequency_encoding, 
        create_temporal_features,
        apply_feature_scaling
    )
    from .preprocessing import permutation_ranking, incremental_subset_evaluation
    from .preprocessing import create_temporal_split, validate_split_quality
    _has_data_processing = True
except (ImportError, ValueError):
    # Fall back to absolute imports
    try:
        from utils.preprocessing import (
            apply_frequency_encoding, 
            create_temporal_features,
            apply_feature_scaling
        )
        from utils.preprocessing import permutation_ranking, incremental_subset_evaluation
        from utils.preprocessing import create_temporal_split, validate_split_quality
        _has_data_processing = True
    except ImportError:
        pass

# Configuration
_has_config = False
try:
    # Try relative imports first
    from .config import load_config
    _has_config = True
except (ImportError, ValueError):
    # Fall back to absolute imports
    try:
        from utils.config import load_config
        _has_config = True
    except ImportError:
        pass

# ============================================================================
# FUNÇÕES CENTRALIZADAS (elimina redundância entre notebooks)
# ============================================================================

def setup_standard_environment(stage_name: str, project_root: Optional[Path] = None):
    """
    Setup padrão para todos os notebooks - elimina código repetitivo.
    Reutiliza módulos existentes sem deletar nada.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # Auto-detect project root
    if project_root is None:
        current_dir = Path.cwd()
        if current_dir.name == 'notebooks':
            project_root = current_dir.parent
        else:
            project_root = current_dir
    
    # Standard paths
    data_dir = project_root / 'data'
    artifacts_dir = project_root / 'artifacts'
    models_dir = project_root / 'models'
    
    artifacts_dir.mkdir(exist_ok=True)
    
    # Load config using existing module
    config = {}
    if _has_config:
        try:
            config = load_config(project_root / 'config.yaml')
            print(f"✅ Config loaded via existing config_loader module")
        except Exception as e:
            print(f"⚠️ Config loading failed: {e}")
    
    print(f"🚀 {stage_name} - Setup via central pipeline functions")
    
    return {
        'data_dir': data_dir,
        'artifacts_dir': artifacts_dir,
        'models_dir': models_dir,
        'config': config,
        'project_root': project_root
    }

def load_datasets_standard(data_dir: Path, variant: str = 'engineered'):
    """
    Carregamento padrão de datasets - elimina código repetitivo.
    Versão única para todos os notebooks.
    """
    try:
        X_train = pd.read_csv(data_dir / f'X_train_{variant}.csv')
        X_test = pd.read_csv(data_dir / f'X_test_{variant}.csv')
        y_train = pd.read_csv(data_dir / f'y_train_{variant}.csv').iloc[:, 0]
        y_test = pd.read_csv(data_dir / f'y_test_{variant}.csv').iloc[:, 0]
        
        print(f"✅ Datasets ({variant}) - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"🎯 Target rates - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Dataset loading failed: {e}")
        available_files = list(data_dir.glob('*.csv'))
        print(f"💡 Available files: {[f.name for f in available_files[:5]]}")
        return None, None, None, None

def evaluate_model_standard(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Avaliação padrão de modelo - reutiliza funções existentes de métricas.
    Elimina código duplicado entre notebooks.
    """
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
    
    # Basic metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    
    # Advanced metrics usando módulos existentes
    results = {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'base_rate': y_test.mean(),
        'predictions': y_pred,
        'scores': y_scores,
        'fitted_model': model
    }
    
    # Add bootstrap CI if available
    if _has_metrics:
        try:
            bootstrap_ci = bootstrap_metric(y_test, y_scores)
            results['pr_auc_ci'] = bootstrap_ci
            print(f"📊 {model_name}: PR-AUC = {pr_auc:.4f} [{bootstrap_ci['ci_lower']:.4f}, {bootstrap_ci['ci_upper']:.4f}]")
        except Exception as e:
            print(f"📊 {model_name}: PR-AUC = {pr_auc:.4f} (CI failed: {e})")
    else:
        print(f"📊 {model_name}: PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}")
    
    return results

def get_core_features_standard(artifacts_dir: Path):
    """
    Carregamento padrão de core features - elimina repetição.
    """
    core_path = artifacts_dir / 'core_features.txt'
    if core_path.exists():
        with open(core_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        print(f"✅ Core features loaded: {len(features)}")
        return features
    else:
        print("⚠️ Core features not found")
        return []

def save_results_standard(results: Dict[str, Any], artifacts_dir: Path, filename: str):
    """
    Salvamento padrão de resultados - elimina código repetitivo.
    """
    import json
    from datetime import datetime
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    # Save
    filepath = artifacts_dir / filename
    if filename.endswith('.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            # Handle numpy types
            def json_serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                else:
                    return str(obj)
            
            json.dump(results, f, indent=2, default=json_serialize, ensure_ascii=False)
    elif filename.endswith('.csv'):
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        else:
            pd.DataFrame(results).to_csv(filepath, index=False)
    
    print(f"✅ Results saved: {filepath}")
    return filepath

# ============================================================================
# FUNÇÕES DE INTEGRAÇÃO (usa módulos existentes)
# ============================================================================

def run_feature_engineering_pipeline(X_train, X_test, config):
    """
    Pipeline integrado de feature engineering - usa módulos existentes.
    """
    if not _has_data_processing:
        print("⚠️ Data processing modules not available")
        return X_train, X_test
    
    print("🔧 Running integrated feature engineering...")
    
    # Temporal features
    try:
        X_train_temporal = create_temporal_features(X_train)
        X_test_temporal = create_temporal_features(X_test)
        print("✅ Temporal features created")
    except Exception as e:
        print(f"⚠️ Temporal features failed: {e}")
        X_train_temporal, X_test_temporal = X_train, X_test
    
    # Frequency encoding
    try:
        fe_config = config.get('data_processing', {}).get('feature_engineering', {})
        high_card_cols = fe_config.get('high_cardinality_cols', [])
        if high_card_cols:
            available_cols = [col for col in high_card_cols if col in X_train_temporal.columns]
            if available_cols:
                X_train_encoded, X_test_encoded, _ = apply_frequency_encoding(
                    X_train_temporal, X_test_temporal, available_cols
                )
                print(f"✅ Frequency encoding applied to {len(available_cols)} columns")
            else:
                X_train_encoded, X_test_encoded = X_train_temporal, X_test_temporal
        else:
            X_train_encoded, X_test_encoded = X_train_temporal, X_test_temporal
    except Exception as e:
        print(f"⚠️ Frequency encoding failed: {e}")
        X_train_encoded, X_test_encoded = X_train_temporal, X_test_temporal
    
    return X_train_encoded, X_test_encoded

def run_feature_selection_pipeline(X_train, y_train, X_test, config):
    """
    Pipeline integrado de feature selection - usa módulos existentes.
    """
    if not _has_data_processing:
        print("⚠️ Feature selection modules not available")
        return X_train, X_test, X_train.columns.tolist()
    
    print("🎯 Running integrated feature selection...")
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Quick model for feature ranking
        model = GradientBoostingClassifier(
            n_estimators=50, 
            random_state=config.get('random_state', 42)
        )
        
        # Get feature ranking
        ranking_df = permutation_ranking(
            model, X_train, y_train, 
            metric='pr_auc', 
            n_repeats=3, 
            random_state=config.get('random_state', 42)
        )
        
        # Select top features (e.g., top 80% by importance)
        n_features = max(5, int(len(ranking_df) * 0.8))
        selected_features = ranking_df['feature'].head(n_features).tolist()
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        print(f"✅ Feature selection: {len(X_train.columns)} → {len(selected_features)}")
        
        return X_train_selected, X_test_selected, selected_features
        
    except Exception as e:
        print(f"⚠️ Feature selection failed: {e}")
        return X_train, X_test, X_train.columns.tolist()

# ============================================================================
# EXPORTS - Complete public interface
# ============================================================================

__all__ = [
    # Pipeline Functions (Notebook Setup & Evaluation)
    'setup_standard_environment',
    'load_datasets_standard', 
    'evaluate_model_standard',
    'get_core_features_standard',
    'save_results_standard',
    'run_feature_engineering_pipeline',
    'run_feature_selection_pipeline',
    
    # I/O Functions (Data Loading & Artifacts)
    'optimize_dtypes',
    'load_data',
    'save_artifact',
    'load_artifact',
    'check_artifact_exists',
    'create_run_metadata',
    'setup_caching',
    
    # Temporal Split Functions (Time-aware Splits)
    'create_temporal_split',
    'validate_split_quality',
    'prepare_temporal_datasets',
    'format_temporal_split_report'
]

"""
I/O utilities for model artifacts, data optimization, and caching.

Memory-efficient data loading, dtype optimization, and artifact management.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def optimize_dtypes(
    df: pd.DataFrame,
    int_downcast: str = 'unsigned',
    float_downcast: str = 'float',
    category_threshold: int = 50,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Reduce memory footprint by optimizing dtypes.
    
    Typically achieves 60-70% memory reduction.
    
    Args:
        df: Input DataFrame
        int_downcast: Downcast strategy for integers
        float_downcast: Downcast strategy for floats
        category_threshold: Max unique values to convert to category
        verbose: Print memory savings
        
    Returns:
        DataFrame with optimized dtypes
    """
    if verbose:
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory before optimization: {start_mem:.2f} MB")
    
    # Downcast integers
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast=int_downcast)
    
    # Downcast floats
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast=float_downcast)
    
    # Convert low-cardinality strings to category
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < category_threshold:
            df[col] = df[col].astype('category')
    
    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(
            f"Memory after optimization: {end_mem:.2f} MB "
            f"(↓ {reduction:.1f}%)"
        )
    
    return df


def load_data(
    path: Union[str, Path],
    optimize_memory: bool = True,
    columns: Optional[list] = None,
    dtypes: Optional[Dict[str, str]] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data with automatic format detection and optimization.
    
    Args:
        path: Path to data file (CSV, Parquet, Feather)
        optimize_memory: Apply dtype optimization
        columns: Subset of columns to load
        dtypes: Explicit dtype specification
        nrows: Number of rows to load (for sampling)
        
    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading data from {path.name}...")
    
    # Load based on file extension
    if path.suffix == '.parquet':
        df = pd.read_parquet(path, columns=columns)
    
    elif path.suffix == '.feather':
        df = pd.read_feather(path, columns=columns)
    
    elif path.suffix in ['.csv', '.txt']:
        df = pd.read_csv(
            path, 
            usecols=columns, 
            dtype=dtypes,
            nrows=nrows
        )
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Optimize memory
    if optimize_memory:
        df = optimize_dtypes(df, verbose=True)
    
    return df


def save_artifact(
    obj: Any,
    path: Union[str, Path],
    artifact_type: str = 'auto',
    compress: bool = False
) -> Path:
    """
    Save model artifact with appropriate format.
    
    Args:
        obj: Object to save (model, dict, DataFrame, etc.)
        path: Save path
        artifact_type: 'model', 'json', 'pickle', 'parquet', 'auto'
        compress: Use compression
        
    Returns:
        Path to saved artifact
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect type
    if artifact_type == 'auto':
        if isinstance(obj, (dict, list)):
            artifact_type = 'json'
        elif isinstance(obj, pd.DataFrame):
            artifact_type = 'parquet'
        else:
            artifact_type = 'pickle'
    
    # Save
    if artifact_type == 'json':
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    
    elif artifact_type == 'parquet':
        obj.to_parquet(path, compression='gzip' if compress else None)
    
    elif artifact_type in ['pickle', 'model']:
        if compress:
            joblib.dump(obj, path, compress=3)
        else:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
    
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    logger.info(f"✓ Saved {artifact_type} to {path}")
    return path


def load_artifact(
    path: Union[str, Path],
    artifact_type: str = 'auto'
) -> Any:
    """
    Load artifact with automatic format detection.
    
    Args:
        path: Path to artifact
        artifact_type: 'model', 'json', 'pickle', 'parquet', 'auto'
        
    Returns:
        Loaded object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    # Auto-detect type
    if artifact_type == 'auto':
        suffix = path.suffix.lower()
        if suffix == '.json':
            artifact_type = 'json'
        elif suffix == '.parquet':
            artifact_type = 'parquet'
        elif suffix in ['.pkl', '.pickle', '.joblib']:
            artifact_type = 'pickle'
        else:
            artifact_type = 'pickle'
    
    # Load
    if artifact_type == 'json':
        with open(path, 'r') as f:
            obj = json.load(f)
    
    elif artifact_type == 'parquet':
        obj = pd.read_parquet(path)
    
    elif artifact_type in ['pickle', 'model']:
        try:
            obj = joblib.load(path)
        except:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
    
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    logger.info(f"✓ Loaded {artifact_type} from {path}")
    return obj


def check_artifact_exists(
    path: Union[str, Path],
    max_age_hours: Optional[float] = None
) -> bool:
    """
    Check if artifact exists and is fresh.
    
    Args:
        path: Path to artifact
        max_age_hours: Maximum age in hours (optional)
        
    Returns:
        True if exists and fresh
    """
    path = Path(path)
    
    if not path.exists():
        return False
    
    if max_age_hours is not None:
        from datetime import datetime, timedelta
        
        file_time = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - file_time
        
        if age > timedelta(hours=max_age_hours):
            logger.info(f"Artifact {path.name} is stale (age: {age})")
            return False
    
    return True


def create_run_metadata(
    config: Dict[str, Any],
    model_name: str,
    metrics: Dict[str, float],
    additional_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create standardized metadata for model runs.
    
    Args:
        config: Configuration dictionary
        model_name: Model identifier
        metrics: Performance metrics
        additional_info: Extra metadata
        
    Returns:
        Structured metadata dictionary
    """
    from datetime import datetime
    import platform
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'metrics': metrics,
        'config': {
            'random_state': config.get('random_state'),
            'cv_folds': config.get('cv_folds'),
            'primary_metric': config.get('primary_metric')
        },
        'environment': {
            'python_version': platform.python_version(),
            'platform': platform.platform()
        }
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    # Try to add git info
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        metadata['git_commit'] = git_hash
    except:
        pass
    
    return metadata


def setup_caching(cache_dir: Union[str, Path], verbose: int = 0):
    """
    Setup joblib memory caching.
    
    Args:
        cache_dir: Directory for cache
        verbose: Verbosity level
        
    Returns:
        Memory object for caching
    """
    from joblib import Memory
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    memory = Memory(location=cache_dir, verbose=verbose)
    logger.info(f"✓ Caching enabled at {cache_dir}")
    
    return memory

"""
Temporal Split Utilities

Functions for creating temporal train/test splits that prevent data leakage
in time series scenarios like AML detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split


def create_temporal_split(df: pd.DataFrame, 
                         timestamp_col: str,
                         entity_cols: Optional[List[str]] = None,
                         test_size: float = 0.3,
                         gap_days: int = 0,
                         min_train_days: int = 30,
                         stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create temporal train/test split ensuring no data leakage.
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        entity_cols: List of entity identifier columns (e.g., ['From Bank', 'Account'])
        test_size: Fraction for test set
        gap_days: Gap between train and test periods to prevent leakage
        min_train_days: Minimum days for training period
        stratify_col: Column to stratify on (e.g., target variable)
        
    Returns:
        Tuple of (train_df, test_df, split_info)
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Calculate split point
    min_date = df[timestamp_col].min()
    max_date = df[timestamp_col].max()
    total_days = (max_date - min_date).days
    
    if total_days < min_train_days + gap_days + 1:
        raise ValueError(f"Dataset spans only {total_days} days, need at least {min_train_days + gap_days + 1}")
    
    # Calculate cutoff dates
    test_days = int(total_days * test_size)
    train_days = total_days - test_days - gap_days
    
    train_end_date = min_date + pd.Timedelta(days=train_days)
    test_start_date = train_end_date + pd.Timedelta(days=gap_days)
    
    # Split data
    train_mask = df[timestamp_col] <= train_end_date
    test_mask = df[timestamp_col] >= test_start_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Validate entity leakage if specified
    entity_leakage = {}
    if entity_cols:
        for col in entity_cols:
            if col in df.columns:
                train_entities = set(train_df[col])
                test_entities = set(test_df[col])
                overlap = train_entities.intersection(test_entities)
                entity_leakage[col] = {
                    'overlapping_count': len(overlap),
                    'train_unique': len(train_entities),
                    'test_unique': len(test_entities),
                    'overlap_percentage': len(overlap) / len(train_entities) * 100 if train_entities else 0
                }
    
    # Split info
    split_info = {
        'split_type': 'temporal',
        'timestamp_column': timestamp_col,
        'total_days': total_days,
        'train_days': train_days,
        'test_days': test_days,
        'gap_days': gap_days,
        'train_period': (train_df[timestamp_col].min().isoformat(), 
                        train_df[timestamp_col].max().isoformat()),
        'test_period': (test_df[timestamp_col].min().isoformat(), 
                       test_df[timestamp_col].max().isoformat()),
        'train_count': len(train_df),
        'test_count': len(test_df),
        'entity_leakage': entity_leakage,
        'stratification': None
    }
    
    # Add stratification info if used
    if stratify_col and stratify_col in df.columns:
        train_dist = train_df[stratify_col].value_counts(normalize=True)
        test_dist = test_df[stratify_col].value_counts(normalize=True)
        split_info['stratification'] = {
            'column': stratify_col,
            'train_distribution': train_dist.to_dict(),
            'test_distribution': test_dist.to_dict()
        }
    
    return train_df, test_df, split_info


def validate_split_quality(train_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          split_info: Dict,
                          target_col: str = 'Is Laundering') -> Dict:
    """
    Validate the quality of the temporal split.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        split_info: Split information from create_temporal_split
        target_col: Target variable column name
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'temporal_leakage': False,
        'entity_leakage_summary': {},
        'target_distribution_shift': {},
        'size_balance': {},
        'warnings': [],
        'recommendations': []
    }
    
    # Check temporal leakage
    if split_info.get('gap_days', 0) < 1:
        validation['temporal_leakage'] = True
        validation['warnings'].append("No temporal gap between train and test periods")
    
    # Entity leakage summary
    entity_leakage = split_info.get('entity_leakage', {})
    high_leakage_entities = []
    
    for entity, leakage_info in entity_leakage.items():
        overlap_pct = leakage_info['overlap_percentage']
        validation['entity_leakage_summary'][entity] = {
            'overlap_percentage': overlap_pct,
            'severity': 'high' if overlap_pct > 50 else 'medium' if overlap_pct > 20 else 'low'
        }
        
        if overlap_pct > 50:
            high_leakage_entities.append(f"{entity}: {overlap_pct:.1f}%")
    
    if high_leakage_entities:
        validation['warnings'].append(f"High entity leakage: {', '.join(high_leakage_entities)}")
    
    # Target distribution shift
    if target_col in train_df.columns and target_col in test_df.columns:
        train_pos_rate = train_df[target_col].mean()
        test_pos_rate = test_df[target_col].mean()
        shift_ratio = abs(test_pos_rate - train_pos_rate) / train_pos_rate if train_pos_rate > 0 else 0
        
        validation['target_distribution_shift'] = {
            'train_positive_rate': train_pos_rate,
            'test_positive_rate': test_pos_rate,
            'shift_ratio': shift_ratio,
            'severity': 'high' if shift_ratio > 0.3 else 'medium' if shift_ratio > 0.1 else 'low'
        }
        
        if shift_ratio > 0.3:
            validation['warnings'].append(f"Large target distribution shift: {shift_ratio:.2%}")
    
    # Size balance
    train_size = len(train_df)
    test_size = len(test_df)
    total_size = train_size + test_size
    
    validation['size_balance'] = {
        'train_size': train_size,
        'test_size': test_size,
        'train_percentage': train_size / total_size * 100,
        'test_percentage': test_size / total_size * 100
    }
    
    # Recommendations
    if validation['temporal_leakage']:
        validation['recommendations'].append("Add temporal gap (gap_days > 0) between train and test")
    
    if high_leakage_entities:
        validation['recommendations'].append("Consider entity-based splitting or accept leakage if temporal patterns are more important")
    
    if validation['target_distribution_shift'].get('severity') == 'high':
        validation['recommendations'].append("Large target shift detected - verify this reflects real concept drift")
    
    return validation


def prepare_temporal_datasets(raw_data_path: str,
                             timestamp_col: str = 'Timestamp',
                             target_col: str = 'Is Laundering',
                             entity_cols: List[str] = ['From Bank', 'Account'],
                             test_size: float = 0.3,
                             gap_days: int = 1,
                             output_dir: str = '../data') -> Dict:
    """
    End-to-end temporal split preparation with validation.
    
    Args:
        raw_data_path: Path to raw dataset
        timestamp_col: Timestamp column name
        target_col: Target variable column
        entity_cols: Entity identifier columns
        test_size: Test set fraction
        gap_days: Gap between train and test
        output_dir: Output directory
        
    Returns:
        Dictionary with results and paths
    """
    # Load data
    df = pd.read_csv(raw_data_path)
    print(f"📊 Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    
    # Create temporal split
    train_df, test_df, split_info = create_temporal_split(
        df, timestamp_col, entity_cols, test_size, gap_days, stratify_col=target_col
    )
    
    # Validate split
    validation = validate_split_quality(train_df, test_df, split_info, target_col)
    
    # Save datasets
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / 'train_temporal.csv'
    test_path = output_dir / 'test_temporal.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save split metadata
    split_metadata = {
        'split_info': split_info,
        'validation': validation,
        'files_created': {
            'train': str(train_path),
            'test': str(test_path)
        }
    }
    
    metadata_path = output_dir / 'temporal_split_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2, default=str)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'split_info': split_info,
        'validation': validation,
        'paths': {
            'train': train_path,
            'test': test_path,
            'metadata': metadata_path
        }
    }


def format_temporal_split_report(split_info: Dict, validation: Dict) -> str:
    """
    Format temporal split results as readable report.
    
    Args:
        split_info: Split information dictionary
        validation: Validation results dictionary
        
    Returns:
        Formatted report string
    """
    lines = [
        "⏰ TEMPORAL SPLIT REPORT",
        "=" * 50,
        "",
        "📅 Split Configuration:",
        f"   Split type: {split_info['split_type']}",
        f"   Total days: {split_info['total_days']}",
        f"   Train period: {split_info['train_period'][0]} to {split_info['train_period'][1]}",
        f"   Test period: {split_info['test_period'][0]} to {split_info['test_period'][1]}",
        f"   Gap days: {split_info['gap_days']}",
        f"   Train size: {split_info['train_count']:,} ({validation['size_balance']['train_percentage']:.1f}%)",
        f"   Test size: {split_info['test_count']:,} ({validation['size_balance']['test_percentage']:.1f}%)",
        ""
    ]
    
    # Entity leakage summary
    if validation['entity_leakage_summary']:
        lines.extend([
            "👥 Entity Leakage Analysis:",
            *[f"   {entity}: {info['overlap_percentage']:.1f}% overlap ({info['severity']} risk)"
              for entity, info in validation['entity_leakage_summary'].items()],
            ""
        ])
    
    # Target distribution
    if validation['target_distribution_shift']:
        shift = validation['target_distribution_shift']
        lines.extend([
            "🎯 Target Distribution:",
            f"   Train positive rate: {shift['train_positive_rate']:.4f}",
            f"   Test positive rate: {shift['test_positive_rate']:.4f}",
            f"   Distribution shift: {shift['shift_ratio']:.2%} ({shift['severity']} severity)",
            ""
        ])
    
    # Warnings and recommendations
    if validation['warnings']:
        lines.extend([
            "⚠ Warnings:",
            *[f"   - {warning}" for warning in validation['warnings']],
            ""
        ])
    
    if validation['recommendations']:
        lines.extend([
            "💡 Recommendations:",
            *[f"   - {rec}" for rec in validation['recommendations']],
            ""
        ])
    
    return "\n".join(lines)

def save_trained_model_for_production(model, model_name: str, experiment_id: str = None,
                                   artifacts_dir: str = None, metadata: dict = None):
    """
    Save a trained model from notebook directly to production-ready format.

    This function allows notebooks to save models that can be directly promoted
    to production without retraining, solving the duplication issue.

    Args:
        model: Trained model object
        model_name: Name for the model file
        experiment_id: Optional experiment ID for tracking
        artifacts_dir: Directory to save model (defaults to artifacts/models)
        metadata: Additional metadata to save

    Returns:
        Dict with save information
    """
    from pathlib import Path
    import pickle
    import json
    from datetime import datetime

    # Set default artifacts directory
    if artifacts_dir is None:
        # Try to find project root
        current_dir = Path.cwd()
        # Look for artifacts directory
        if (current_dir / 'artifacts').exists():
            artifacts_dir = current_dir / 'artifacts'
        elif (current_dir.parent / 'artifacts').exists():
            artifacts_dir = current_dir.parent / 'artifacts'
        else:
            # Create artifacts directory
            artifacts_dir = current_dir / 'artifacts'
            artifacts_dir.mkdir(exist_ok=True)

    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / 'notebook_models'  # Changed from 'models' to 'notebook_models'
    models_dir.mkdir(exist_ok=True, parents=True)

    # Generate model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_id:
        filename = f"{model_name}_{experiment_id}_{timestamp}.pkl"
    else:
        filename = f"{model_name}_{timestamp}.pkl"

    model_path = models_dir / filename

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Create metadata
    model_metadata = {
        'model_name': model_name,
        'experiment_id': experiment_id,
        'saved_at': datetime.now().isoformat(),
        'model_path': str(model_path),
        'model_type': type(model).__name__,
        'notebook_saved': True,
        'ready_for_production': True
    }

    # Add custom metadata
    if metadata:
        model_metadata.update(metadata)

    # Try to add model info
    try:
        if hasattr(model, 'feature_importances_'):
            model_metadata['feature_importances'] = list(model.feature_importances_)
        if hasattr(model, 'n_features_'):
            model_metadata['n_features'] = model.n_features_
        if hasattr(model, 'classes_'):
            model_metadata['classes'] = list(model.classes_) if hasattr(model.classes_, '__iter__') else str(model.classes_)
    except:
        pass  # Skip if not available

    # Save metadata
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)

    print("✅ Model saved for production!")
    print(f"📁 Model: {model_path}")
    print(f"📄 Metadata: {metadata_path}")
    print(f"🧪 Experiment ID: {experiment_id or 'N/A'}")
    print()
    print("To promote to production, run:")
    print(f"python src/modeling/train.py --promote_notebook_model {model_name}")

    return {
        'model_path': str(model_path),
        'metadata_path': str(metadata_path),
        'experiment_id': experiment_id,
        'model_name': model_name,
        'promotion_command': f"python src/modeling/train.py --promote_notebook_model {model_name}"
    }