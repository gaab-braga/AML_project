# src/utils/exceptions.py
"""
Custom exception hierarchy for AML Platform
"""

from typing import Dict, Any, Optional


class AMLPlatformError(Exception):
    """Base exception for AML Platform"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(AMLPlatformError):
    """Configuration-related errors"""
    pass


class DataError(AMLPlatformError):
    """Data processing errors"""
    pass


class ValidationError(AMLPlatformError):
    """Data validation errors"""
    pass


class FeatureError(AMLPlatformError):
    """Feature engineering errors"""
    pass


class ModelingError(AMLPlatformError):
    """ML modeling errors"""
    pass


class EvaluationError(AMLPlatformError):
    """Model evaluation errors"""
    pass


class OrchestrationError(AMLPlatformError):
    """DAG orchestration errors"""
    pass


class CacheError(AMLPlatformError):
    """Cache system errors"""
    pass


class TaskError(OrchestrationError):
    """Task execution errors"""

    def __init__(self, message: str, task_id: str, retry_count: int = 0, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.task_id = task_id
        self.retry_count = retry_count


class PipelineError(OrchestrationError):
    """Pipeline execution errors"""

    def __init__(self, message: str, pipeline_name: str, execution_id: str, failed_tasks: Optional[list] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.pipeline_name = pipeline_name
        self.execution_id = execution_id
        self.failed_tasks = failed_tasks or []


class ResourceError(AMLPlatformError):
    """Resource-related errors (memory, disk, etc.)"""
    pass


class TimeoutError(AMLPlatformError):
    """Timeout errors"""
    pass


class DependencyError(AMLPlatformError):
    """Dependency resolution errors"""
    pass


def handle_exception(exc: Exception, logger=None, re_raise: bool = True) -> None:
    """
    Centralized exception handling

    Args:
        exc: Exception to handle
        logger: Optional logger for error reporting
        re_raise: Whether to re-raise the exception
    """
    if logger:
        from .logging import log_error_with_context
        log_error_with_context(logger, exc)

    if re_raise:
        raise exc


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Create error context dictionary

    Args:
        operation: Operation that failed
        **kwargs: Additional context

    Returns:
        Error context dictionary
    """
    context = {
        "operation": operation,
        "timestamp": "auto",  # Will be filled by logging system
    }
    context.update(kwargs)
    return context


__all__ = [
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
    "create_error_context"
]