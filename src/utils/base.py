# src/utils/base.py
"""
Base utilities and helper functions
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
import time
import functools
from contextlib import contextmanager
import hashlib
import json

from .logging import logger


class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "operation", log_start: bool = True):
        self.name = name
        self.log_start = log_start
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.log_start:
            logger.info(f"Starting {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed {self.name} in {duration:.2f}s")


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


class DataValidator:
    """Utility class for data validation"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                         min_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate DataFrame structure

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            results['valid'] = False
            results['errors'].append("Input must be a pandas DataFrame")
            return results

        # Check shape
        results['info']['shape'] = df.shape

        if df.empty:
            results['valid'] = False
            results['errors'].append("DataFrame is empty")
            return results

        if min_rows and len(df) < min_rows:
            results['valid'] = False
            results['errors'].append(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                results['valid'] = False
                results['errors'].append(f"Missing required columns: {missing_columns}")

        # Check for null values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
            results['warnings'].append(".2f")

            # Columns with high null percentage
            high_null_cols = null_counts[null_counts / len(df) > 0.5]
            if not high_null_cols.empty:
                results['warnings'].append(f"Columns with >50% nulls: {list(high_null_cols.index)}")

        # Data types info
        results['info']['dtypes'] = df.dtypes.to_dict()

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        results['info']['memory_mb'] = memory_mb

        if memory_mb > 1000:  # > 1GB
            results['warnings'].append(".1f")

        return results

    @staticmethod
    def validate_target_variable(y: pd.Series, problem_type: str = "classification") -> Dict[str, Any]:
        """
        Validate target variable

        Args:
            y: Target variable
            problem_type: Type of ML problem

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        if not isinstance(y, pd.Series):
            results['valid'] = False
            results['errors'].append("Target must be a pandas Series")
            return results

        if y.empty:
            results['valid'] = False
            results['errors'].append("Target series is empty")
            return results

        # Check for null values
        null_count = y.isnull().sum()
        if null_count > 0:
            results['valid'] = False
            results['errors'].append(f"Target contains {null_count} null values")

        # Problem-specific validation
        if problem_type == "classification":
            unique_values = y.unique()
            n_classes = len(unique_values)
            results['info']['n_classes'] = n_classes
            results['info']['classes'] = unique_values.tolist()

            if n_classes < 2:
                results['valid'] = False
                results['errors'].append("Classification requires at least 2 classes")

            # Class balance check
            class_counts = y.value_counts()
            minority_ratio = class_counts.min() / class_counts.max()

            if minority_ratio < 0.01:  # Less than 1%
                results['warnings'].append(".4f"
                                          "Consider addressing class imbalance")

        elif problem_type == "regression":
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(y):
                results['valid'] = False
                results['errors'].append("Regression target must be numeric")

            # Check variance
            if y.var() == 0:
                results['valid'] = False
                results['errors'].append("Target has zero variance")

        return results


class CacheManager:
    """Simple cache manager for expensive operations"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.cache = {}
        self.logger = logger.getChild("cache_manager")

    def get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key
        """
        # Convert args to strings
        arg_strings = [str(arg) for arg in args]

        # Sort kwargs for consistent hashing
        kwarg_strings = [f"{k}:{v}" for k, v in sorted(kwargs.items())]

        # Combine and hash
        content = "|".join(arg_strings + kwarg_strings)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache

        Args:
            key: Cache key

        Returns:
            Cached item or None
        """
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache

        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self.logger.debug(f"Cached item with key: {key}")

    @contextmanager
    def cached_operation(self, operation_name: str, *args, **kwargs):
        """
        Context manager for cached operations

        Args:
            operation_name: Name of the operation
            *args: Arguments to generate cache key
            **kwargs: Keyword arguments

        Yields:
            Cached result or None
        """
        cache_key = self.get_cache_key(operation_name, *args, **kwargs)
        cached_result = self.get(cache_key)

        if cached_result is not None:
            self.logger.info(f"Using cached result for {operation_name}")
            yield cached_result
            return

        # No cache hit, yield None and cache the result
        yield None

        # Note: The actual caching happens after the context manager
        # The calling code should check if the yield was None and then cache the result

    def clear_cache(self) -> None:
        """Clear all cached items"""
        self.cache.clear()
        self.logger.info("Cache cleared")


class MemoryMonitor:
    """Monitor memory usage"""

    def __init__(self):
        self.logger = logger.getChild("memory_monitor")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage

        Returns:
            Memory usage information
        """
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Memory info
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # System memory
        system_memory = psutil.virtual_memory()

        return {
            'process_rss_mb': memory_info.rss / 1024 / 1024,
            'process_vms_mb': memory_info.vms / 1024 / 1024,
            'process_percent': memory_percent,
            'system_total_mb': system_memory.total / 1024 / 1024,
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_percent': system_memory.percent
        }

    def log_memory_usage(self, message: str = "Memory usage") -> None:
        """
        Log current memory usage

        Args:
            message: Log message prefix
        """
        try:
            usage = self.get_memory_usage()
            self.logger.info(f"{message}: Process={usage['process_rss_mb']:.1f}MB "
                           f"({usage['process_percent']:.1f}%), "
                           f"System={usage['system_percent']:.1f}%")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


class DataHasher:
    """Generate hashes for data integrity checks"""

    @staticmethod
    def hash_dataframe(df: pd.DataFrame, include_index: bool = False) -> str:
        """
        Generate hash for DataFrame

        Args:
            df: DataFrame to hash
            include_index: Whether to include index in hash

        Returns:
            Hash string
        """
        # Convert to string representation
        if include_index:
            content = str(df.to_dict())
        else:
            content = str(df.to_dict('records'))

        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def hash_series(series: pd.Series) -> str:
        """
        Generate hash for Series

        Args:
            series: Series to hash

        Returns:
            Hash string
        """
        content = str(series.to_dict())
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """
        Generate hash for dictionary

        Args:
            data: Dictionary to hash

        Returns:
            Hash string
        """
        # Sort keys for consistent hashing
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and pandas objects"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string

    Args:
        obj: Object to serialize
        **kwargs: Additional json.dumps arguments

    Returns:
        JSON string
    """
    return json.dumps(obj, cls=SafeEncoder, **kwargs)


__all__ = [
    "Timer",
    "time_function",
    "DataValidator",
    "CacheManager",
    "MemoryMonitor",
    "DataHasher",
    "SafeEncoder",
    "safe_json_dumps"
]