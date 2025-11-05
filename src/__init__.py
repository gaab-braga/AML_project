# AML Platform - Enterprise ML Pipeline
# Version: 1.0.0
# Description: Modular AML detection platform with production-ready components

__version__ = "1.0.0"
__author__ = "AML Platform Team"
__description__ = "Enterprise AML detection platform with executive reporting and MLOps capabilities"

# Configuration
from .config import settings, AMLSettings, load_settings

# Utilities
from .utils import logger, setup_logging, Timer

# Data processing
from .data import DataLoader, CSVLoader, DataPipeline

# Feature engineering
from .features import (
    aggregate_by_entity,
    compute_network_features,
    create_temporal_features,
    create_network_features,
    AMLFeaturePipeline
)

# Modeling
from .modeling import (
    build_pipeline,
    train_pipeline,
    load_model,
    AMLModelTrainer,
    AMLModel
)

# Evaluation
from .evaluation import (
    calculate_all_metrics,
    calculate_metrics_at_k,
    print_metrics_report
)

# Visualization
from .visualization import (
    set_aml_style,
    create_model_comparison_plot,
    create_feature_importance_plot
)

# Reporting (Executive Summary)
from .reporting import (
    calculate_baseline_costs,
    calculate_ml_impact,
    calculate_roi_metrics,
    generate_executive_dashboard,
    print_executive_summary
)

# Make key components available at package level
__all__ = [
    # Configuration
    "settings",
    "AMLSettings",
    "load_settings",
    
    # Utilities
    "logger",
    "setup_logging",
    "Timer",
    
    # Data
    "DataLoader",
    "CSVLoader",
    "DataPipeline",
    
    # Features
    "aggregate_by_entity",
    "compute_network_features",
    "create_temporal_features",
    "create_network_features",
    "AMLFeaturePipeline",
    
    # Modeling
    "build_pipeline",
    "train_pipeline",
    "load_model",
    "AMLModelTrainer",
    "AMLModel",
    
    # Evaluation
    "calculate_all_metrics",
    "calculate_metrics_at_k",
    "print_metrics_report",
    
    # Visualization
    "set_aml_style",
    "create_model_comparison_plot",
    "create_feature_importance_plot",
    
    # Reporting
    "calculate_baseline_costs",
    "calculate_ml_impact",
    "calculate_roi_metrics",
    "generate_executive_dashboard",
    "print_executive_summary"
]