"""
Setup utilities for AML notebooks
==================================

Centralized setup functions to eliminate code duplication across notebooks.

Functions:
----------
- load_config_and_paths(): Load config.yaml and setup paths
- get_common_imports(): Return dict of commonly used imports
- setup_notebook_environment(): Complete notebook setup
"""

import sys
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


def load_config_and_paths(notebook_dir: Path = None, config_path: str = '../../config.yaml') -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """
    Load configuration and setup paths for notebooks.

    Parameters
    ----------
    notebook_dir : Path, optional
        Notebook directory. If None, uses current working directory.
    config_path : str
        Path to config.yaml relative to notebook_dir

    Returns
    -------
    config : dict
        Loaded configuration
    paths : dict
        Dictionary with common paths (data_dir, artifacts_dir, etc.)
    """
    if notebook_dir is None:
        notebook_dir = Path.cwd()  # Use current working directory instead of script location

    # Load config
    config_file = notebook_dir / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths - project root is the directory containing config.yaml
    project_root = config_file.parent
    paths = {
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'artifacts_dir': project_root / 'artifacts',
        'models_dir': project_root / 'artifacts' / 'models',
        'notebooks_dir': project_root / 'notebooks',
        'utils_dir': project_root / 'utils'
    }

    # Create directories if they don't exist
    paths['artifacts_dir'].mkdir(exist_ok=True, parents=True)
    paths['models_dir'].mkdir(exist_ok=True, parents=True)

    return config, paths


def get_common_imports() -> Dict[str, Any]:
    """
    Get dictionary of commonly used imports across notebooks.

    Returns
    -------
    imports : dict
        Dictionary with import names as keys and imported modules as values
    """
    imports = {}

    # Standard libraries
    try:
        import sys
        imports['sys'] = sys
    except ImportError:
        pass

    try:
        import os
        imports['os'] = os
    except ImportError:
        pass

    try:
        import pickle
        imports['pickle'] = pickle
    except ImportError:
        pass

    try:
        import json
        imports['json'] = json
    except ImportError:
        pass

    try:
        import warnings
        imports['warnings'] = warnings
    except ImportError:
        pass

    # Data Science
    try:
        import pandas as pd
        imports['pd'] = pd
    except ImportError:
        pass

    try:
        import numpy as np
        imports['np'] = np
    except ImportError:
        pass

    # Visualization
    try:
        import matplotlib.pyplot as plt
        imports['plt'] = plt
    except ImportError:
        pass

    try:
        import seaborn as sns
        imports['sns'] = sns
    except ImportError:
        pass

    # Progress tracking
    try:
        from tqdm import tqdm
        imports['tqdm'] = tqdm
    except ImportError:
        pass

    return imports


def setup_notebook_environment(notebook_name: str = "Notebook",
                              config_path: str = '../../config.yaml',
                              verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """
    Complete notebook environment setup.

    Parameters
    ----------
    notebook_name : str
        Name of the notebook for logging
    config_path : str
        Path to config.yaml
    verbose : bool
        Whether to print setup information

    Returns
    -------
    config : dict
        Loaded configuration
    paths : dict
        Dictionary with common paths
    """
    if verbose:
        print(f"🚀 Setting up {notebook_name}")
        print("=" * 50)

    # Load config and paths
    config, paths = load_config_and_paths(config_path=config_path)

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    if verbose:
        print("✅ Configuration loaded")
        print(f"  Primary metric: {config['modeling']['primary_metric'].upper()}")
        print(f"  CV folds: {config['modeling']['cv_folds']}")
        print(f"  Random state: {config['random_state']}")
        print(f"  Data directory: {paths['data_dir']}")
        print(f"  Artifacts directory: {paths['artifacts_dir']}")
        print(f"  Models directory: {paths['models_dir']}")
        print()

    return config, paths


def validate_setup(config: Dict[str, Any], paths: Dict[str, Path], verbose: bool = True) -> bool:
    """
    Validate that setup is complete and paths exist.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    paths : dict
        Paths dictionary
    verbose : bool
        Whether to print validation results

    Returns
    -------
    is_valid : bool
        True if setup is valid
    """
    checks = [
        ("Config loaded", bool(config), "Configuration dictionary not empty"),
        ("Data directory", paths['data_dir'].exists(), f"Data dir exists: {paths['data_dir']}"),
        ("Artifacts directory", paths['artifacts_dir'].exists(), f"Artifacts dir exists: {paths['artifacts_dir']}"),
        ("Models directory", paths['models_dir'].exists(), f"Models dir exists: {paths['models_dir']}"),
    ]

    all_passed = True
    for check_name, passed, details in checks:
        status = "✅" if passed else "❌"
        if verbose:
            print(f"{status} {check_name}: {details}")
        if not passed:
            all_passed = False

    if verbose:
        print(f"\n{'🎉 SETUP VALID!' if all_passed else '⚠️  SETUP ISSUES!'}")

    return all_passed