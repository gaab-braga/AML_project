"""
Utils package for AML Pipeline

This package contains utility modules for the Anti-Money Laundering (AML) pipeline,
including metrics calculation, model optimization, validation, and production tools.
"""

# Import commonly used functions for convenience
try:
    from .metrics import bootstrap_metric, compute_cv_metrics
    from .metrics import compute_calibration_metrics
    from .metrics import ThresholdConfig, evaluate_thresholds
except ImportError as e:
    print(f"⚠️ Warning importing metrics: {e}")
    bootstrap_metric = compute_cv_metrics = compute_calibration_metrics = None
    ThresholdConfig = evaluate_thresholds = None

try:
    from .governance import create_model_card
except ImportError as e:
    print(f"⚠️ Warning importing governance: {e}")
    create_model_card = None

# Configuration
try:
    from .config import (
        load_config,
        get_default_config,
        get_paths,
        get_model_params,
        get_k_values,
        get_psi_interpretation,
        setup_paths,
        get_ensemble_config
    )
except ImportError as e:
    print(f"⚠️ Warning importing config: {e}")
    load_config = get_default_config = get_paths = None
    get_model_params = get_k_values = get_psi_interpretation = None
    setup_paths = get_ensemble_config = None

# New professional modules
from .modeling import (
    FraudMetrics,
    get_cv_strategy,
    train_with_early_stopping,
    cross_validate_with_metrics,
    calculate_class_weights
)
from .sampling import (
    create_balanced_dataset,
    create_sampling_strategies,
    get_sampling_summary,
    select_best_strategy
)
from .tuning import (
    run_staged_tuning,
    apply_gating
)
try:
    from .data import (
        # I/O Functions
        optimize_dtypes,
        load_data,
        save_artifact,
        load_artifact,
        check_artifact_exists,
        setup_caching,
        # Pipeline Functions
        setup_standard_environment,
        load_datasets_standard,
        evaluate_model_standard,
        get_core_features_standard,
        save_results_standard,
        # Temporal Split Functions
        create_temporal_split,
        validate_split_quality,
        prepare_temporal_datasets,
        format_temporal_split_report
    )
except ImportError as e:
    print(f"⚠️ Warning importing data functions: {e}")
    optimize_dtypes = load_data = save_artifact = None
    load_artifact = check_artifact_exists = setup_caching = None
    setup_standard_environment = load_datasets_standard = evaluate_model_standard = None
    get_core_features_standard = save_results_standard = None
    create_temporal_split = validate_split_quality = None
    prepare_temporal_datasets = format_temporal_split_report = None
from .explainability import (
    compute_shap_values as analyze_shap,
    compute_permutation_importance
)
try:
    from .ensemble import (
        # Custom Stacking
        StackingEnsemble,
        WeightedVotingEnsemble,
        create_default_stacking_ensemble,
        compare_ensemble_strategies,
        # Enhanced Ensemble
        EnhancedEnsembleRiskScorer,
        # Score Fusion
        blend_scores,
        build_ensemble_outputs,
        # sklearn-based
        create_stacking_ensemble,
        create_voting_ensemble,
        evaluate_ensemble,
        compare_ensemble_vs_individual,
        quick_lgbm_xgb_ensemble
    )
except ImportError as e:
    print(f"⚠️ Warning importing ensemble: {e}")
    StackingEnsemble = WeightedVotingEnsemble = None
    create_default_stacking_ensemble = compare_ensemble_strategies = None
    EnhancedEnsembleRiskScorer = None
    blend_scores = build_ensemble_outputs = None
    create_stacking_ensemble = create_voting_ensemble = None
    evaluate_ensemble = compare_ensemble_vs_individual = None
    quick_lgbm_xgb_ensemble = None
# Funções de diagnóstico estão em visualization.py
from .visualization import (
    plot_learning_curve,
    plot_calibration_curve,
    plot_threshold_analysis,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_model_comparison,
    plot_fraud_patterns,
    plot_roc_detailed_analysis
)

# Verificar se existe comprehensive_diagnostic_report
try:
    from .visualization import comprehensive_diagnostic_report
except ImportError:
    comprehensive_diagnostic_report = None

__version__ = "2.1.0"
__author__ = "AML Pipeline Team"

__all__ = [
    # Configuration
    'load_config',
    'get_default_config',
    'get_paths',
    'get_model_params',
    'get_k_values',
    'get_psi_interpretation',
    'setup_paths',
    'get_ensemble_config',
    
    # Metrics
    'bootstrap_metric',
    'compute_cv_metrics', 
    'compute_calibration_metrics',
    'ThresholdConfig',
    'evaluate_thresholds',
    # 'compute_expected_value',  # TODO: Implementar se necessário
    # 'find_optimal_threshold_ev',  # TODO: Implementar se necessário
    'create_model_card',
    
    # Modeling
    'FraudMetrics',
    'get_cv_strategy',
    'train_with_early_stopping',
    'cross_validate_with_metrics',
    'calculate_class_weights',
    
    # Sampling
    'create_balanced_dataset',
    'create_sampling_strategies',
    'get_sampling_summary',
    'select_best_strategy',
    
    # Tuning
    'run_staged_tuning',
    'apply_gating',
    
    # I/O & Data
    'optimize_dtypes',
    'load_data',
    'save_artifact',
    'load_artifact',
    'check_artifact_exists',
    'setup_caching',
    'setup_standard_environment',
    'load_datasets_standard',
    'evaluate_model_standard',
    'get_core_features_standard',
    'save_results_standard',
    'create_temporal_split',
    'validate_split_quality',
    'prepare_temporal_datasets',
    'format_temporal_split_report',
    
    # Explainability
    'analyze_shap',
    'compute_permutation_importance',
    # 'prune_features',  # TODO: Implementar se necessário
    
    # Ensemble
    'StackingEnsemble',
    'WeightedVotingEnsemble',
    'create_default_stacking_ensemble',
    'compare_ensemble_strategies',
    'EnhancedEnsembleRiskScorer',
    'blend_scores',
    'build_ensemble_outputs',
    'create_stacking_ensemble',
    'create_voting_ensemble',
    'evaluate_ensemble',
    'compare_ensemble_vs_individual',
    'quick_lgbm_xgb_ensemble',
    
    # Diagnostics
    'plot_learning_curve',
    'plot_calibration_curve',
    'plot_threshold_analysis',
    'comprehensive_diagnostic_report',
    
    # Visualization
    'plot_feature_importance',
    'plot_confusion_matrix',
    'plot_roc_pr_curves',
    'plot_model_comparison',
    'plot_fraud_patterns',
    'plot_roc_detailed_analysis',
    'save_trained_model_for_production',
]