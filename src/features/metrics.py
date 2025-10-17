# src/features/metrics.py
"""
Performance metrics and monitoring for feature engineering
"""

import time
import psutil
import tracemalloc
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils import logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: Optional[int] = None
    memory_end: Optional[int] = None
    memory_delta: Optional[int] = None
    cpu_percent: Optional[float] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    features_created: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, **kwargs):
        """Mark operation as complete and calculate metrics"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if self.memory_start is not None:
            self.memory_end = tracemalloc.get_traced_memory()[0]
            self.memory_delta = self.memory_end - self.memory_start

        try:
            self.cpu_percent = psutil.cpu_percent(interval=None)
        except:
            pass

        self.metadata.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = {
            'operation': self.operation,
            'duration_seconds': self.duration,
            'timestamp': datetime.now().isoformat(),
            'performance_metric': True
        }

        if self.memory_delta is not None:
            result['memory_delta_mb'] = self.memory_delta / (1024 * 1024)

        if self.cpu_percent is not None:
            result['cpu_percent'] = self.cpu_percent

        if self.input_shape:
            result['input_rows'] = self.input_shape[0]
            result['input_cols'] = self.input_shape[1]

        if self.output_shape:
            result['output_rows'] = self.output_shape[0]
            result['output_cols'] = self.output_shape[1]

        if self.features_created:
            result['features_created'] = self.features_created

        if self.error:
            result['error'] = self.error

        result.update(self.metadata)
        return result


class FeatureEngineeringMonitor:
    """Monitor for feature engineering operations"""

    def __init__(self, name: str = "feature_engineering"):
        self.name = name
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_tracing_memory = False

    def start_memory_tracing(self):
        """Start memory tracing"""
        if not self.is_tracing_memory:
            tracemalloc.start()
            self.is_tracing_memory = True

    def stop_memory_tracing(self):
        """Stop memory tracing"""
        if self.is_tracing_memory:
            tracemalloc.stop()
            self.is_tracing_memory = False

    @contextmanager
    def monitor_operation(self,
                         operation: str,
                         input_data: Optional[pd.DataFrame] = None,
                         **metadata):
        """Context manager for monitoring operations"""
        metrics = PerformanceMetrics(
            operation=f"{self.name}.{operation}",
            start_time=time.time(),
            metadata=metadata
        )

        if input_data is not None:
            metrics.input_shape = input_data.shape

        # Start memory tracing if not already started
        if not self.is_tracing_memory:
            self.start_memory_tracing()
            should_stop_tracing = True
        else:
            should_stop_tracing = False

        try:
            metrics.memory_start = tracemalloc.get_traced_memory()[0]
        except:
            metrics.memory_start = None

        try:
            yield metrics
        except Exception as e:
            metrics.error = str(e)
            raise
        finally:
            metrics.complete()
            self.metrics_history.append(metrics)

            # Log performance metrics
            logger.info(f"Performance: {operation}",
                       extra={"extra_fields": metrics.to_dict()})

            # Stop tracing if we started it
            if should_stop_tracing:
                self.stop_memory_tracing()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics_history:
            return {}

        durations = [m.duration for m in self.metrics_history if m.duration is not None]
        memory_deltas = [m.memory_delta for m in self.metrics_history if m.memory_delta is not None]

        summary = {
            'total_operations': len(self.metrics_history),
            'total_duration': sum(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'total_memory_delta': sum(memory_deltas) / (1024 * 1024) if memory_deltas else 0,  # MB
            'operations_with_errors': sum(1 for m in self.metrics_history if m.error is not None)
        }

        # Group by operation type
        operation_stats = {}
        for metrics in self.metrics_history:
            op = metrics.operation
            if op not in operation_stats:
                operation_stats[op] = []
            if metrics.duration is not None:
                operation_stats[op].append(metrics.duration)

        summary['operation_breakdown'] = {
            op: {
                'count': len(durations),
                'avg_duration': np.mean(durations),
                'total_duration': sum(durations)
            }
            for op, durations in operation_stats.items()
        }

        return summary

    def log_performance_report(self):
        """Log a comprehensive performance report"""
        summary = self.get_performance_summary()

        logger.info("Feature Engineering Performance Report",
                   extra={"extra_fields": {
                       "performance_report": True,
                       **summary
                   }})

    def check_performance_thresholds(self,
                                   max_duration: Optional[float] = None,
                                   max_memory_mb: Optional[float] = None) -> List[str]:
        """Check if performance metrics exceed thresholds"""
        warnings = []

        for metrics in self.metrics_history:
            if max_duration and metrics.duration and metrics.duration > max_duration:
                warnings.append(f"Operation {metrics.operation} exceeded duration threshold: {metrics.duration:.2f}s > {max_duration}s")

            if max_memory_mb and metrics.memory_delta:
                memory_mb = metrics.memory_delta / (1024 * 1024)
                if memory_mb > max_memory_mb:
                    warnings.append(f"Operation {metrics.operation} exceeded memory threshold: {memory_mb:.2f}MB > {max_memory_mb}MB")

        return warnings


class HealthChecker:
    """Health checker for feature engineering components"""

    def __init__(self, name: str = "feature_engineering"):
        self.name = name
        self.last_check = None
        self.health_status = "unknown"

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_check = datetime.now()

        health_data = {
            'component': self.name,
            'timestamp': self.last_check.isoformat(),
            'status': 'healthy',
            'checks': {}
        }

        # Memory usage check
        try:
            memory = psutil.virtual_memory()
            health_data['checks']['memory_usage'] = {
                'percent': memory.percent,
                'available_mb': memory.available / (1024 * 1024),
                'status': 'healthy' if memory.percent < 90 else 'warning'
            }
        except Exception as e:
            health_data['checks']['memory_usage'] = {'error': str(e), 'status': 'error'}

        # CPU usage check
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            health_data['checks']['cpu_usage'] = {
                'percent': cpu_percent,
                'status': 'healthy' if cpu_percent < 80 else 'warning'
            }
        except Exception as e:
            health_data['checks']['cpu_usage'] = {'error': str(e), 'status': 'error'}

        # Disk space check
        try:
            disk = psutil.disk_usage('/')
            health_data['checks']['disk_space'] = {
                'free_mb': disk.free / (1024 * 1024),
                'percent_used': disk.percent,
                'status': 'healthy' if disk.percent < 90 else 'warning'
            }
        except Exception as e:
            health_data['checks']['disk_space'] = {'error': str(e), 'status': 'error'}

        # Determine overall status
        statuses = [check.get('status', 'unknown') for check in health_data['checks'].values()]
        if 'error' in statuses:
            health_data['status'] = 'error'
        elif 'warning' in statuses:
            health_data['status'] = 'warning'
        else:
            health_data['status'] = 'healthy'

        self.health_status = health_data['status']

        # Log health check
        logger.info("Health Check Completed",
                   extra={"extra_fields": {
                       "health_check": True,
                       **health_data
                   }})

        return health_data

    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self.health_status == 'healthy'

    def get_health_status(self) -> str:
        """Get current health status"""
        return self.health_status


# Global instances
feature_monitor = FeatureEngineeringMonitor()
health_checker = HealthChecker()


def monitor_feature_operation(operation_name: str) -> Callable:
    """Decorator for monitoring feature engineering operations"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract DataFrame from args/kwargs if present
            input_data = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    input_data = arg
                    break

            if input_data is None:
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame):
                        input_data = value
                        break

            with feature_monitor.monitor_operation(
                operation_name,
                input_data=input_data,
                function=func.__name__
            ) as metrics:
                result = func(*args, **kwargs)

                # Capture output information
                if isinstance(result, pd.DataFrame):
                    metrics.output_shape = result.shape
                    # Try to count features created (approximate)
                    if input_data is not None:
                        metrics.features_created = result.shape[1] - input_data.shape[1]

                return result

        return wrapper
    return decorator


# Convenience functions
def log_performance_metrics(operation: str, duration: float, **metadata):
    """Log performance metrics (backward compatibility)"""
    logger.info(f"Performance: {operation}",
               extra={"extra_fields": {
                   "operation": operation,
                   "duration_seconds": duration,
                   "performance_metric": True,
                   **metadata
               }})


def check_system_health() -> bool:
    """Quick health check"""
    return health_checker.is_healthy()


__all__ = [
    "PerformanceMetrics",
    "FeatureEngineeringMonitor",
    "HealthChecker",
    "feature_monitor",
    "health_checker",
    "monitor_feature_operation",
    "log_performance_metrics",
    "check_system_health"
]