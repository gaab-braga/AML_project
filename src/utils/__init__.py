# Utils module
# Handles logging, monitoring, exceptions, and helpers

from .logging import (
    setup_logging,
    get_task_logger,
    get_pipeline_logger,
    get_model_logger,
    log_performance_metrics,
    log_error_with_context,
    logger
)
from .exceptions import (
    AMLPlatformError,
    ConfigurationError,
    DataError,
    ValidationError,
    FeatureError,
    ModelingError,
    EvaluationError,
    OrchestrationError,
    CacheError,
    TaskError,
    PipelineError,
    ResourceError,
    TimeoutError,
    DependencyError,
    handle_exception,
    create_error_context
)
from .helpers import (
    generate_unique_id,
    calculate_hash,
    format_duration,
    safe_divide,
    deep_merge_dicts,
    ensure_directory,
    get_file_size_mb,
    truncate_text,
    validate_email,
    create_batches,
    get_memory_usage_mb,
    retry_with_backoff,
    Timer
)

__all__ = [
    # Logging
    "setup_logging",
    "get_task_logger",
    "get_pipeline_logger",
    "get_model_logger",
    "log_performance_metrics",
    "log_error_with_context",
    "logger",

    # Exceptions
    "AMLPlatformError",
    "ConfigurationError",
    "DataError",
    "ValidationError",
    "FeatureError",
    "ModelingError",
    "EvaluationError",
    "OrchestrationError",
    "CacheError",
    "TaskError",
    "PipelineError",
    "ResourceError",
    "TimeoutError",
    "DependencyError",
    "handle_exception",
    "create_error_context",

    # Helpers
    "generate_unique_id",
    "calculate_hash",
    "format_duration",
    "safe_divide",
    "deep_merge_dicts",
    "ensure_directory",
    "get_file_size_mb",
    "truncate_text",
    "validate_email",
    "create_batches",
    "get_memory_usage_mb",
    "retry_with_backoff",
    "Timer"
]