# src/visualization/__init__.py
"""
Visualization utilities for AML project.
"""

from .plot_config import (
    set_aml_style,
    AML_COLORS,
    MODEL_COLORS,
    FRAUD_COLORS,
    FEATURE_IMPORTANCE_COLORS,
    get_model_palette,
    get_fraud_palette,
    get_feature_importance_palette,
    apply_consistent_formatting,
    create_model_comparison_plot,
    create_feature_importance_plot,
    plot_aml_optimization_history,
    plot_aml_param_importances
)

__all__ = [
    'set_aml_style',
    'AML_COLORS',
    'MODEL_COLORS',
    'FRAUD_COLORS',
    'FEATURE_IMPORTANCE_COLORS',
    'get_model_palette',
    'get_fraud_palette',
    'get_feature_importance_palette',
    'apply_consistent_formatting',
    'create_model_comparison_plot',
    'create_feature_importance_plot',
    'plot_aml_optimization_history',
    'plot_aml_param_importances'
]