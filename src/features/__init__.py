# Feature engineering module
# Handles feature creation, selection, and anti-leakage validation

from .base import (
    FeatureEngineer,
    LeakageDetector,
    FeatureSelector,
    FeaturePipeline
)

from .temporal import TemporalFeatures, aggregate_by_entity, compute_network_features
from .statistical import StatisticalFeatures
from .leakage_detector import remove_leaky_features

# AML-specific features
from .aml_features import (
    create_temporal_features,
    create_network_features,
    encode_categorical_features,
    create_aml_feature_pipeline,
    AMLFeaturePipeline,
    # Dashboard utilities
    ensure_server_running,
    wait_for_client_connection,
    SSELiveCallback,
    setup_live_dashboard,
    start_live_dashboard
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
    "FeatureEngineer",
    "LeakageDetector",
    "FeatureSelector",
    "FeaturePipeline",
    "TemporalFeatures",
    "aggregate_by_entity",
    "compute_network_features",
    "StatisticalFeatures",
    "remove_leaky_features",
    # AML additions
    "create_temporal_features",
    "create_network_features",
    "encode_categorical_features",
    "create_aml_feature_pipeline",
    "AMLFeaturePipeline",
    # Dashboard utilities
    "ensure_server_running",
    "wait_for_client_connection",
    "SSELiveCallback",
    "setup_live_dashboard",
    "start_live_dashboard",
    # AML plotting functions
    "plot_threshold_comparison_all_models_optimized",
    "plot_executive_summary_aml_new",
    "plot_feature_importance",
    "plot_shap_summary",
    "generate_executive_summary",
    "process_training_results"
]