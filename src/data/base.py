# src/data/base.py
"""
Base classes and interfaces for data processing
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pandas as pd

from ..utils import logger
from ..config import settings


class DataProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger.getChild(f"data.{name}")

    @abstractmethod
    def process(self, data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """
        Process input data

        Args:
            data: Input data (DataFrame, file path, or string)

        Returns:
            Processed DataFrame
        """
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data

        Args:
            data: Input data to validate

        Returns:
            True if valid
        """
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra={"extra_fields": kwargs})

    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra={"extra_fields": kwargs})


class DataLoader(DataProcessor):
    """Base class for data loading"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.file_path = None

    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from source"""
        pass

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data

        Args:
            data: DataFrame to validate

        Returns:
            Validation results
        """
        from ..utils.base import DataValidator as UtilsDataValidator

        # Use utility data validator
        return UtilsDataValidator.validate_dataframe(data)

    def process(self, data: Any) -> pd.DataFrame:
        """Process by loading data"""
        if isinstance(data, (str, Path)):
            return self.load_data(data)
        else:
            raise ValueError("DataLoader requires file path as input")


class DataValidator(DataProcessor):
    """Base class for data validation"""

    def __init__(self, name: str, rules: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.rules = rules or {}

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame

        Args:
            data: DataFrame to validate

        Returns:
            Validation results dictionary
        """
        pass

    def process(self, data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """Process by validating data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("DataValidator requires DataFrame input")

        validation_results = self.validate(data)

        if not validation_results.get("valid", True):
            errors = validation_results.get("errors", [])
            if errors:
                error_msg = f"Data validation failed: {errors[:5]}"  # Show first 5 errors
                self.log_error(error_msg, validation_errors=len(errors))

                if settings.data.fail_on_missing:
                    raise ValueError(error_msg)

        return data

    def validate_input(self, data: Any) -> bool:
        """Validate input is DataFrame"""
        return isinstance(data, pd.DataFrame)


class DataTransformer(DataProcessor):
    """Base class for data transformation"""

    def __init__(self, name: str, transformations: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.transformations = transformations or []

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame"""
        pass

    def process(self, data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
        """Process by transforming data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("DataTransformer requires DataFrame input")

        self.log_info("Starting data transformation", rows=data.shape[0], cols=data.shape[1])

        transformed_data = self.transform(data)

        self.log_info("Data transformation completed",
                     original_shape=data.shape,
                     final_shape=transformed_data.shape)

        return transformed_data

    def validate_input(self, data: Any) -> bool:
        """Validate input is DataFrame"""
        return isinstance(data, pd.DataFrame)


class DataPipeline:
    """Data processing pipeline"""

    def __init__(self, name: str, processors: List[DataProcessor]):
        self.name = name
        self.processors = processors
        self.logger = logger.getChild(f"pipeline.{name}")

    def execute(self, input_data: Any) -> pd.DataFrame:
        """
        Execute pipeline

        Args:
            input_data: Initial input data

        Returns:
            Final processed DataFrame
        """
        self.logger.info(f"Starting pipeline execution: {self.name}")

        current_data = input_data

        for i, processor in enumerate(self.processors):
            try:
                self.logger.info(f"Executing processor {i+1}/{len(self.processors)}: {processor.name}")

                # Validate input
                if not processor.validate_input(current_data):
                    raise ValueError(f"Invalid input for processor {processor.name}")

                # Process data
                current_data = processor.process(current_data)

                self.logger.info(f"Processor {processor.name} completed successfully")

            except Exception as e:
                error_msg = f"Pipeline failed at processor {processor.name}: {str(e)}"
                self.logger.error(error_msg, processor=processor.name, step=i+1)
                raise

        self.logger.info(f"Pipeline execution completed: {self.name}")
        return current_data

    def validate_pipeline(self) -> List[str]:
        """
        Validate pipeline configuration

        Returns:
            List of validation errors
        """
        errors = []

        if not self.processors:
            errors.append("Pipeline has no processors")
            return errors

        # Check processor compatibility
        for i, processor in enumerate(self.processors[:-1]):
            next_processor = self.processors[i + 1]

            # This is a basic check - in practice, you'd have more sophisticated validation
            if not hasattr(processor, 'process') or not hasattr(next_processor, 'process'):
                errors.append(f"Processor {processor.name} or {next_processor.name} missing process method")

        return errors


__all__ = [
    "DataProcessor",
    "DataLoader",
    "DataValidator",
    "DataTransformer",
    "DataPipeline"
]