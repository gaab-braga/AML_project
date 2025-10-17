# Data processing module
# Handles data ingestion, preprocessing, and validation

from .base import (
    DataProcessor,
    DataLoader,
    DataValidator,
    DataTransformer,
    DataPipeline
)
from .loaders import (
    CSVLoader,
    DatabaseLoader,
    JSONLoader,
    ParquetLoader
)

__all__ = [
    # Base classes
    "DataProcessor",
    "DataLoader",
    "DataValidator",
    "DataTransformer",
    "DataPipeline",

    # Concrete implementations
    "CSVLoader",
    "DatabaseLoader",
    "JSONLoader",
    "ParquetLoader"
]