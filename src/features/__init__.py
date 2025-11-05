# Feature engineering module
# Handles feature creation, selection, and anti-leakage validation

from .temporal import TemporalFeatures, aggregate_by_entity, compute_network_features

# AML-specific features
from .aml_features import (
    create_temporal_features,
    create_network_features,
    encode_categorical_features,
    AMLFeaturePipeline
)

# AML plotting functions
from .aml_plotting import (
    plot_threshold_comparison_all_models_optimized,
    plot_executive_summary_aml_new,
    plot_feature_importance,
    plot_shap_summary,
    generate_executive_summary,
    process_training_results
)

__all__ = [
    "TemporalFeatures",
    "aggregate_by_entity",
    "compute_network_features",
    # AML additions
    "create_temporal_features",
    "create_network_features",
    "encode_categorical_features",
    "AMLFeaturePipeline",
    # AML plotting functions
    "plot_threshold_comparison_all_models_optimized",
    "plot_executive_summary_aml_new",
    "plot_feature_importance",
    "plot_shap_summary",
    "generate_executive_summary",
    "process_training_results"
]