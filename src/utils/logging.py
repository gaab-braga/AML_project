# src/utils/logging.py
"""
Structured logging system with JSON format and monitoring integration
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

from ..config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.utcnow().isoformat()

        log_entry = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add structured fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add context from structlog
        if hasattr(record, 'structured_context'):
            log_entry.update(record.structured_context)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def setup_structlog():
    """Setup structlog for structured logging"""
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add JSON renderer for production
    if settings.environment.value in ["staging", "production"]:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_logging(
    level: str = None,
    log_file: Optional[Path] = None,
    enable_structlog: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging system

    Args:
        level: Logging level (defaults to settings)
        log_file: Log file path (defaults to settings)
        enable_structlog: Whether to enable structlog

    Returns:
        Configured logger
    """
    # Use settings if not provided
    if level is None:
        level = settings.monitoring.log_level.value
    if log_file is None:
        log_file = settings.monitoring.log_file

    # Setup structlog first
    if enable_structlog:
        setup_structlog()

    # Create logger
    logger = logging.getLogger("aml_platform")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create JSON formatter
    formatter = JSONFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)

    # Set logging level for other loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context"""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)

    def process(self, msg, kwargs):
        # Add extra fields to log record
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_task_logger(task_id: str, execution_id: Optional[str] = None) -> LoggerAdapter:
    """Get logger with task context"""
    logger = logging.getLogger("aml_platform.task")
    extra = {"task_id": task_id}
    if execution_id:
        extra["execution_id"] = execution_id
    return LoggerAdapter(logger, extra)


def get_pipeline_logger(pipeline_name: str, execution_id: str) -> LoggerAdapter:
    """Get logger with pipeline context"""
    logger = logging.getLogger("aml_platform.pipeline")
    extra = {
        "pipeline_name": pipeline_name,
        "execution_id": execution_id
    }
    return LoggerAdapter(logger, extra)


def get_model_logger(model_name: str, version: Optional[str] = None) -> LoggerAdapter:
    """Get logger with model context"""
    logger = logging.getLogger("aml_platform.model")
    extra = {"model_name": model_name}
    if version:
        extra["model_version"] = version
    return LoggerAdapter(logger, extra)


def log_performance_metrics(logger: logging.Logger,
                          operation: str,
                          duration: float,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log performance metrics"""
    log_data = {
        "operation": operation,
        "duration_seconds": duration,
        "performance_metric": True
    }

    if metadata:
        log_data.update(metadata)

    logger.info(f"Performance: {operation}", extra={"extra_fields": log_data})


def log_error_with_context(logger: logging.Logger,
                          error: Exception,
                          context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with additional context"""
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_logged": True
    }

    if context:
        log_data.update(context)

    logger.error(f"Error: {type(error).__name__}",
                exc_info=error,
                extra={"extra_fields": log_data})


# Global logger instance
logger = setup_logging()

__all__ = [
    "setup_logging",
    "get_task_logger",
    "get_pipeline_logger",
    "get_model_logger",
    "log_performance_metrics",
    "log_error_with_context",
    "logger"
]