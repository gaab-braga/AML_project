# AML Platform - Enterprise ML Pipeline
# Version: 1.0.0
# Description: Modular AML detection platform with DAG orchestration

__version__ = "1.0.0"
__author__ = "AML Platform Team"
__description__ = "Enterprise AML detection platform with parallel execution and intelligent caching"

# Import main components for easy access
from .config import settings
from .utils import logger
from .data_io import load_raw_transactions, save_model
from .preprocessing import clean_transactions, impute_and_encode
from .features import aggregate_by_entity, compute_network_features
from .modeling import build_pipeline, train_pipeline, load_model
from .evaluation import compute_metrics, plot_roc_pr
from .viz import plot_time_series, plot_network_subgraph

# Make key classes available at package level
__all__ = [
    "settings",
    "logger",
    "load_raw_transactions",
    "save_model",
    "clean_transactions",
    "impute_and_encode",
    "aggregate_by_entity",
    "compute_network_features",
    "build_pipeline",
    "train_pipeline",
    "load_model",
    "compute_metrics",
    "plot_roc_pr",
    "plot_time_series",
    "plot_network_subgraph"
]