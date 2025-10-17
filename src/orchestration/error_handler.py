# src/orchestration/error_handler.py
"""
Error Handler for AML Pipeline Orchestration
"""

import time
import traceback
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

from .task_definition import Task, TaskResult
from ..utils import logger
from ..features.metrics import feature_monitor


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    RESOURCE = "resource"
    DATA = "data"
    MODEL = "model"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error"""
    task_id: str
    task_name: str
    error_message: str
    error_type: str
    traceback: str
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'traceback': self.traceback,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    function: Callable
    description: str
    max_attempts: int = 1
    backoff_seconds: float = 1.0
    requires_approval: bool = False

    def execute(self, error_context: ErrorContext) -> bool:
        """
        Execute recovery action

        Args:
            error_context: Error context

        Returns:
            True if recovery successful, False otherwise
        """
        try:
            logger.info(f"Executing recovery action: {self.name}")
            result = self.function(error_context)
            return result if isinstance(result, bool) else True
        except Exception as e:
            logger.error(f"Recovery action {self.name} failed: {e}")
            return False


class ErrorHandler:
    """
    Comprehensive error handler with recovery strategies

    Handles different types of errors and applies appropriate recovery actions
    """

    def __init__(self,
                 max_retries: int = 3,
                 enable_alerts: bool = True,
                 alert_channels: Optional[List[str]] = None):
        """
        Initialize error handler

        Args:
            max_retries: Default maximum retry attempts
            enable_alerts: Whether to enable alerting
            alert_channels: List of alert channels (email, slack, etc.)
        """
        self.max_retries = max_retries
        self.enable_alerts = enable_alerts
        self.alert_channels = alert_channels or []

        self.logger = logger.getChild("error_handler")

        # Error tracking
        self._error_history: List[ErrorContext] = []
        self._error_lock = threading.Lock()

        # Recovery strategies by error category
        self._recovery_strategies = self._initialize_recovery_strategies()

        # Error counters
        self._error_counts = {
            'total': 0,
            'by_category': {},
            'by_severity': {},
            'by_task': {}
        }

    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryAction]]:
        """Initialize recovery strategies for different error categories"""
        return {
            ErrorCategory.RESOURCE: [
                RecoveryAction(
                    name="wait_for_resources",
                    function=self._wait_for_resources,
                    description="Wait for system resources to become available",
                    max_attempts=5,
                    backoff_seconds=10.0
                ),
                RecoveryAction(
                    name="scale_resources",
                    function=self._scale_resources,
                    description="Attempt to scale system resources",
                    requires_approval=True
                )
            ],

            ErrorCategory.DATA: [
                RecoveryAction(
                    name="validate_data",
                    function=self._validate_data_integrity,
                    description="Validate and repair data integrity",
                    max_attempts=2
                ),
                RecoveryAction(
                    name="fallback_data_source",
                    function=self._use_fallback_data_source,
                    description="Switch to fallback data source",
                    requires_approval=True
                )
            ],

            ErrorCategory.MODEL: [
                RecoveryAction(
                    name="reload_model",
                    function=self._reload_model,
                    description="Reload model from cache/storage",
                    max_attempts=3,
                    backoff_seconds=2.0
                ),
                RecoveryAction(
                    name="use_fallback_model",
                    function=self._use_fallback_model,
                    description="Switch to backup model",
                    requires_approval=True
                )
            ],

            ErrorCategory.TIMEOUT: [
                RecoveryAction(
                    name="increase_timeout",
                    function=self._increase_timeout,
                    description="Increase timeout and retry",
                    max_attempts=2
                ),
                RecoveryAction(
                    name="optimize_task",
                    function=self._optimize_task_execution,
                    description="Optimize task for better performance",
                    requires_approval=True
                )
            ],

            ErrorCategory.INFRASTRUCTURE: [
                RecoveryAction(
                    name="restart_service",
                    function=self._restart_service,
                    description="Restart affected service",
                    max_attempts=2,
                    backoff_seconds=5.0
                ),
                RecoveryAction(
                    name="failover",
                    function=self._failover_to_backup,
                    description="Failover to backup system",
                    requires_approval=True
                )
            ],

            ErrorCategory.CONFIGURATION: [
                RecoveryAction(
                    name="reload_config",
                    function=self._reload_configuration,
                    description="Reload configuration from source",
                    max_attempts=2
                ),
                RecoveryAction(
                    name="use_default_config",
                    function=self._use_default_configuration,
                    description="Fallback to default configuration",
                    requires_approval=True
                )
            ]
        }

    def handle_error(self,
                    task: Task,
                    error: Exception,
                    retry_count: int = 0) -> Dict[str, Any]:
        """
        Handle an error that occurred during task execution

        Args:
            task: Task that failed
            error: Exception that occurred
            retry_count: Current retry count

        Returns:
            Recovery decision and actions
        """
        # Create error context
        error_context = self._create_error_context(task, error, retry_count)

        # Categorize error
        self._categorize_error(error_context)

        # Log error
        self._log_error(error_context)

        # Update error statistics
        self._update_error_stats(error_context)

        # Determine recovery strategy
        recovery_decision = self._determine_recovery_strategy(error_context)

        # Execute recovery if appropriate
        if recovery_decision['should_retry'] and retry_count < error_context.max_retries:
            recovery_success = self._execute_recovery(error_context)
            recovery_decision['recovery_attempted'] = True
            recovery_decision['recovery_success'] = recovery_success

        # Send alerts if needed
        if self.enable_alerts and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert(error_context, recovery_decision)

        return recovery_decision

    def _create_error_context(self,
                             task: Task,
                             error: Exception,
                             retry_count: int) -> ErrorContext:
        """Create error context from task and exception"""
        return ErrorContext(
            task_id=task.id,
            task_name=task.name,
            error_message=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
            retry_count=retry_count,
            max_retries=task.retry_count or self.max_retries,
            metadata={
                'task_priority': task.priority,
                'resource_requirements': task.resource_requirements,
                'function_name': getattr(task.function, '__name__', 'unknown')
            }
        )

    def _categorize_error(self, error_context: ErrorContext):
        """Categorize error by type and severity"""
        error_msg = error_context.error_message.lower()
        error_type = error_context.error_type

        # Categorize by error type
        if any(keyword in error_msg for keyword in ['memory', 'cpu', 'gpu', 'disk', 'resource']):
            error_context.category = ErrorCategory.RESOURCE
        elif any(keyword in error_msg for keyword in ['data', 'dataframe', 'csv', 'parquet']):
            error_context.category = ErrorCategory.DATA
        elif any(keyword in error_msg for keyword in ['model', 'predict', 'fit', 'sklearn']):
            error_context.category = ErrorCategory.MODEL
        elif 'timeout' in error_msg or error_type == 'TimeoutError':
            error_context.category = ErrorCategory.TIMEOUT
        elif any(keyword in error_msg for keyword in ['config', 'yaml', 'json', 'settings']):
            error_context.category = ErrorCategory.CONFIGURATION
        elif any(keyword in error_msg for keyword in ['connection', 'network', 'service']):
            error_context.category = ErrorCategory.INFRASTRUCTURE
        else:
            error_context.category = ErrorCategory.UNKNOWN

        # Determine severity
        if error_context.category == ErrorCategory.INFRASTRUCTURE:
            error_context.severity = ErrorSeverity.CRITICAL
        elif error_context.category in [ErrorCategory.RESOURCE, ErrorCategory.DATA]:
            error_context.severity = ErrorSeverity.HIGH
        elif error_context.retry_count >= error_context.max_retries:
            error_context.severity = ErrorSeverity.HIGH
        else:
            error_context.severity = ErrorSeverity.MEDIUM

    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level"""
        log_data = {
            "error_category": error_context.category.value,
            "error_severity": error_context.severity.value,
            "task_id": error_context.task_id,
            "retry_count": error_context.retry_count,
            "max_retries": error_context.max_retries
        }

        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Critical error in task {error_context.task_id}: {error_context.error_message}",
                extra={"extra_fields": log_data}
            )
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"High severity error in task {error_context.task_id}: {error_context.error_message}",
                extra={"extra_fields": log_data}
            )
        else:
            self.logger.warning(
                f"Error in task {error_context.task_id}: {error_context.error_message}",
                extra={"extra_fields": log_data}
            )

    def _update_error_stats(self, error_context: ErrorContext):
        """Update error statistics"""
        with self._error_lock:
            self._error_history.append(error_context)

            # Keep only last 1000 errors
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-1000:]

            # Update counters
            self._error_counts['total'] += 1
            self._error_counts['by_category'][error_context.category.value] = \
                self._error_counts['by_category'].get(error_context.category.value, 0) + 1
            self._error_counts['by_severity'][error_context.severity.value] = \
                self._error_counts['by_severity'].get(error_context.severity.value, 0) + 1
            self._error_counts['by_task'][error_context.task_id] = \
                self._error_counts['by_task'].get(error_context.task_id, 0) + 1

    def _determine_recovery_strategy(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Determine recovery strategy for error"""
        decision = {
            'should_retry': False,
            'retry_delay': 0.0,
            'max_retries': error_context.max_retries,
            'recovery_actions': [],
            'requires_approval': False,
            'estimated_recovery_time': 0.0
        }

        # Don't retry if already at max retries
        if error_context.retry_count >= error_context.max_retries:
            decision['should_retry'] = False
            return decision

        # Get recovery actions for error category
        recovery_actions = self._recovery_strategies.get(error_context.category, [])

        if not recovery_actions:
            # No specific recovery actions, use exponential backoff
            decision['should_retry'] = True
            decision['retry_delay'] = min(2 ** error_context.retry_count, 300)  # Max 5 minutes
            decision['estimated_recovery_time'] = decision['retry_delay']
        else:
            # Use specific recovery actions
            decision['should_retry'] = True
            decision['recovery_actions'] = [action.name for action in recovery_actions]
            decision['requires_approval'] = any(action.requires_approval for action in recovery_actions)

            # Estimate recovery time
            total_time = sum(action.backoff_seconds for action in recovery_actions)
            decision['estimated_recovery_time'] = total_time

        return decision

    def _execute_recovery(self, error_context: ErrorContext) -> bool:
        """Execute recovery actions"""
        recovery_actions = self._recovery_strategies.get(error_context.category, [])

        for action in recovery_actions:
            if action.max_attempts > 0:
                self.logger.info(f"Attempting recovery action: {action.name}")

                success = action.execute(error_context)
                if success:
                    self.logger.info(f"Recovery action {action.name} succeeded")
                    return True
                else:
                    self.logger.warning(f"Recovery action {action.name} failed")

                # Wait before next action
                if action.backoff_seconds > 0:
                    time.sleep(action.backoff_seconds)

        return False

    def _send_alert(self, error_context: ErrorContext, recovery_decision: Dict[str, Any]):
        """Send alerts for critical errors"""
        alert_message = {
            'timestamp': datetime.now().isoformat(),
            'severity': error_context.severity.value,
            'task_id': error_context.task_id,
            'task_name': error_context.task_name,
            'error_message': error_context.error_message,
            'category': error_context.category.value,
            'recovery_decision': recovery_decision
        }

        for channel in self.alert_channels:
            try:
                if channel == 'email':
                    self._send_email_alert(alert_message)
                elif channel == 'slack':
                    self._send_slack_alert(alert_message)
                elif channel == 'webhook':
                    self._send_webhook_alert(alert_message)
                else:
                    self.logger.warning(f"Unknown alert channel: {channel}")
            except Exception as e:
                self.logger.error(f"Failed to send alert to {channel}: {e}")

    def _send_email_alert(self, alert_message: Dict[str, Any]):
        """Send email alert (placeholder)"""
        self.logger.info(f"Email alert: {alert_message}")

    def _send_slack_alert(self, alert_message: Dict[str, Any]):
        """Send Slack alert (placeholder)"""
        self.logger.info(f"Slack alert: {alert_message}")

    def _send_webhook_alert(self, alert_message: Dict[str, Any]):
        """Send webhook alert (placeholder)"""
        self.logger.info(f"Webhook alert: {alert_message}")

    # Recovery action implementations
    def _wait_for_resources(self, error_context: ErrorContext) -> bool:
        """Wait for resources to become available"""
        # Placeholder - would integrate with resource manager
        time.sleep(5)
        return True

    def _scale_resources(self, error_context: ErrorContext) -> bool:
        """Scale system resources"""
        # Placeholder - would integrate with cloud provider
        return False

    def _validate_data_integrity(self, error_context: ErrorContext) -> bool:
        """Validate and repair data integrity"""
        # Placeholder - would implement data validation logic
        return True

    def _use_fallback_data_source(self, error_context: ErrorContext) -> bool:
        """Switch to fallback data source"""
        # Placeholder - would implement fallback logic
        return False

    def _reload_model(self, error_context: ErrorContext) -> bool:
        """Reload model from cache/storage"""
        # Placeholder - would implement model reloading
        return True

    def _use_fallback_model(self, error_context: ErrorContext) -> bool:
        """Switch to backup model"""
        # Placeholder - would implement model fallback
        return False

    def _increase_timeout(self, error_context: ErrorContext) -> bool:
        """Increase timeout and retry"""
        # Placeholder - would modify task timeout
        return True

    def _optimize_task_execution(self, error_context: ErrorContext) -> bool:
        """Optimize task for better performance"""
        # Placeholder - would implement task optimization
        return False

    def _restart_service(self, error_context: ErrorContext) -> bool:
        """Restart affected service"""
        # Placeholder - would implement service restart
        return True

    def _failover_to_backup(self, error_context: ErrorContext) -> bool:
        """Failover to backup system"""
        # Placeholder - would implement failover logic
        return False

    def _reload_configuration(self, error_context: ErrorContext) -> bool:
        """Reload configuration from source"""
        # Placeholder - would reload config
        return True

    def _use_default_configuration(self, error_context: ErrorContext) -> bool:
        """Fallback to default configuration"""
        # Placeholder - would use default config
        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self._error_lock:
            return {
                'total_errors': self._error_counts['total'],
                'by_category': self._error_counts['by_category'].copy(),
                'by_severity': self._error_counts['by_severity'].copy(),
                'by_task': self._error_counts['by_task'].copy(),
                'recent_errors': [
                    error.to_dict() for error in self._error_history[-10:]
                ]
            }

    def clear_error_history(self):
        """Clear error history"""
        with self._error_lock:
            self._error_history.clear()
            self._error_counts = {
                'total': 0,
                'by_category': {},
                'by_severity': {},
                'by_task': {}
            }


__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "RecoveryAction",
    "ErrorHandler"
]