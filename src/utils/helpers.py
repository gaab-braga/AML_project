# src/utils/helpers.py
"""
Utility functions and helpers
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time
from datetime import datetime, timedelta


def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique ID with optional prefix

    Args:
        prefix: Optional prefix for the ID
        length: Length of random part

    Returns:
        Unique ID string
    """
    timestamp = str(int(time.time() * 1000000))  # Microsecond precision
    random_part = hashlib.md5(timestamp.encode()).hexdigest()[:length]

    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def calculate_hash(data: Any) -> str:
    """
    Calculate SHA256 hash of data

    Args:
        data: Data to hash (will be JSON serialized if not string)

    Returns:
        SHA256 hash string
    """
    if not isinstance(data, str):
        data = json.dumps(data, sort_keys=True, default=str)

    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return ".1f"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return ".1f"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries

    Args:
        base: Base dictionary
        override: Dictionary to merge on top

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in MB

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0

    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def validate_email(email: str) -> bool:
    """
    Basic email validation

    Args:
        email: Email address to validate

    Returns:
        True if valid email format
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split list into batches

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_memory_usage_mb() -> float:
    """
    Get current memory usage in MB

    Returns:
        Memory usage in MB
    """
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


def retry_with_backoff(func, max_attempts: int = 3, backoff_factor: float = 2.0, max_delay: float = 60.0):
    """
    Retry function with exponential backoff

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        backoff_factor: Backoff multiplier
        max_delay: Maximum delay between attempts

    Returns:
        Function result
    """
    import time
    import random

    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:
                delay = min(backoff_factor ** attempt + random.uniform(0, 1), max_delay)
                time.sleep(delay)

    raise last_exception


class Timer:
    """Context manager for timing operations"""

    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if self.description:
            print(".2f")
        else:
            print(".2f")

    @property
    def duration(self) -> float:
        """Get elapsed time"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


__all__ = [
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