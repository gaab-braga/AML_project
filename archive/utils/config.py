"""
Configuration loader for Money Laundering Detection Pipeline
Centralizes all parameters from config.yaml

⚠️ NOTE: This module has duplicate code in utils/integrations.py
    The recommended way to use these functions is:
    
    from utils import load_config, get_paths  # ✅ Preferred
    from utils.config import load_config      # ✅ Also works
    
    This duplication exists for backward compatibility and will be 
    consolidated in a future refactor.
"""
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    
from pathlib import Path
from typing import Dict, Any, Optional

__all__ = [
    'get_default_config',
    'load_config',
    'get_paths',
    'get_model_params',
    'get_k_values',
    'get_psi_interpretation',
    'setup_paths',
    'get_ensemble_config'
]

def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration as fallback when YAML is not available.
    """
    return {
        'random_state': 42,
        'paths': {
            'data': 'data',
            'artifacts': 'artifacts',
            'models': 'models',
            'utils': 'utils'
        },
        'scoring': {
            'k_values': [50, 100, 200, 500],
            'risk_levels': {
                'HIGH': 0.95,
                'MEDIUM': 0.85,
                'LOW': 0.70
            }
        },
        'ensemble': {
            'weights': {
                'supervised': 0.7,
                'anomaly': 0.3,
                'graph': 0.0
            },
            'blend_method': 'rank_mean'
        },
        'monitoring': {
            'psi_thresholds': {
                'stable': 0.1,
                'small_change': 0.2,
                'significant_change': 0.3
            }
        },
        'modeling': {
            'primary_metric': 'pr_auc'
        }
    }

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in project root.
        
    Returns:
        Dictionary with all configuration parameters
    """
    if not HAS_YAML:
        print("⚠️  PyYAML não disponível. Usando configuração padrão.")
        return get_default_config()
        
    if config_path is None:
        # Try to find config.yaml from current location
        current = Path.cwd()
        
        # Check current directory and parent directories
        for path in [current] + list(current.parents):
            config_file = path / "config.yaml"
            if config_file.exists():
                config_path = str(config_file)
                break
        
        if config_path is None:
            print("⚠️  config.yaml não encontrado. Usando configuração padrão.")
            return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Erro ao carregar config.yaml: {e}. Usando configuração padrão.")
        return get_default_config()

def get_paths(config: Dict[str, Any], base_path: Optional[Path] = None) -> Dict[str, Path]:
    """
    Convert path strings from config to Path objects.
    
    Args:
        config: Configuration dictionary
        base_path: Base path to resolve relative paths. If None, uses current directory.
        
    Returns:
        Dictionary of Path objects
    """
    if base_path is None:
        base_path = Path.cwd()
        
    paths = {}
    for key, path_str in config['paths'].items():
        paths[key] = base_path / path_str
        
    return paths

def get_model_params(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get model parameters for specific algorithm.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model (e.g., 'GradientBoosting')
        
    Returns:
        Dictionary of model parameters
    """
    baseline_models = config['modeling']['baseline_models']
    
    if model_name in baseline_models:
        params = baseline_models[model_name].copy()
        
        # Add random_state if not present
        if 'random_state' not in params and model_name in ['GradientBoosting', 'RandomForest', 'LightGBM']:
            params['random_state'] = config['random_state']
            
        return params
    
    return {}

def get_k_values(config: Dict[str, Any], n_samples: int) -> list:
    """
    Get adaptive K values for @K metrics based on dataset size.
    
    Args:
        config: Configuration dictionary
        n_samples: Number of samples in test set
        
    Returns:
        List of K values adapted to dataset size
    """
    base_k_values = config['scoring']['k_values']
    
    # Adapt K values to dataset size
    adapted_k = []
    for k in base_k_values:
        adapted = min(k, n_samples // 2)  # Don't exceed half of dataset
        if adapted > 0:
            adapted_k.append(adapted)
    
    return sorted(set(adapted_k))  # Remove duplicates and sort

def get_psi_interpretation(psi_value: float, config: Dict[str, Any]) -> str:
    """
    Get PSI drift interpretation based on configured thresholds.
    
    Args:
        psi_value: PSI value to interpret
        config: Configuration dictionary
        
    Returns:
        String interpretation of drift level
    """
    thresholds = config['monitoring']['psi_thresholds']
    
    if psi_value < thresholds['stable']:
        return 'Estável'
    elif psi_value < thresholds['small_change']:
        return 'Mudança pequena'
    elif psi_value < thresholds['significant_change']:
        return 'Mudança moderada'
    else:
        return 'Mudança significativa'

# Example usage functions
def setup_paths(config: Dict[str, Any], create_dirs: bool = True) -> Dict[str, Path]:
    """
    Setup project directory structure.
    
    Args:
        config: Configuration dictionary
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        Dictionary of Path objects
    """
    paths = get_paths(config)
    
    if create_dirs:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    return paths

def get_ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ensemble configuration with computed parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with ensemble parameters
    """
    ensemble_config = config['ensemble'].copy()
    
    # Add computed parameters
    ensemble_config['primary_metric'] = config['modeling']['primary_metric']
    ensemble_config['random_state'] = config['random_state']
    
    return ensemble_config