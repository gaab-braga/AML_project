# src/interfaces/base.py
"""
Base interfaces and protocols for the AML platform
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, Protocol, runtime_checkable
import pandas as pd
import numpy as np

from ..utils import logger


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """Protocol for data loading components"""

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from source

        Args:
            source: Data source identifier
            **kwargs: Additional loading parameters

        Returns:
            Loaded DataFrame
        """
        ...

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data

        Args:
            data: DataFrame to validate

        Returns:
            Validation results
        """
        ...


@runtime_checkable
class DataProcessorProtocol(Protocol):
    """Protocol for data processing components"""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data

        Args:
            data: Input DataFrame

        Returns:
            Processed DataFrame
        """
        ...

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            Processing statistics
        """
        ...


@runtime_checkable
class FeatureEngineerProtocol(Protocol):
    """Protocol for feature engineering components"""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from data

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with new features
        """
        ...

    def get_feature_names(self) -> List[str]:
        """
        Get names of created features

        Returns:
            List of feature names
        """
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for machine learning models"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model

        Args:
            X: Training features
            y: Training target
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Test features

        Returns:
            Predictions
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Test features

        Returns:
            Class probabilities
        """
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for evaluation components"""

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate predictions

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Evaluation results
        """
        ...


@runtime_checkable
class PipelineStepProtocol(Protocol):
    """Protocol for pipeline steps"""

    def execute(self, input_data: Any) -> Any:
        """
        Execute the pipeline step

        Args:
            input_data: Input data

        Returns:
            Output data
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """
        Get step execution status

        Returns:
            Status information
        """
        ...


class ComponentFactory(ABC):
    """Abstract factory for creating platform components"""

    @abstractmethod
    def create_data_loader(self, name: str, config: Dict[str, Any]) -> DataLoaderProtocol:
        """
        Create a data loader component

        Args:
            name: Component name
            config: Component configuration

        Returns:
            Data loader instance
        """
        pass

    @abstractmethod
    def create_data_processor(self, name: str, config: Dict[str, Any]) -> DataProcessorProtocol:
        """
        Create a data processor component

        Args:
            name: Component name
            config: Component configuration

        Returns:
            Data processor instance
        """
        pass

    @abstractmethod
    def create_feature_engineer(self, name: str, config: Dict[str, Any]) -> FeatureEngineerProtocol:
        """
        Create a feature engineer component

        Args:
            name: Component name
            config: Component configuration

        Returns:
            Feature engineer instance
        """
        pass

    @abstractmethod
    def create_model(self, name: str, config: Dict[str, Any]) -> ModelProtocol:
        """
        Create a model component

        Args:
            name: Component name
            config: Component configuration

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def create_evaluator(self, name: str, config: Dict[str, Any]) -> EvaluatorProtocol:
        """
        Create an evaluator component

        Args:
            name: Component name
            config: Component configuration

        Returns:
            Evaluator instance
        """
        pass


class ComponentRegistry:
    """Registry for platform components"""

    def __init__(self):
        self._data_loaders = {}
        self._data_processors = {}
        self._feature_engineers = {}
        self._models = {}
        self._evaluators = {}
        self.logger = logger.getChild("component_registry")

    def register_data_loader(self, name: str, loader_class: type) -> None:
        """
        Register a data loader class

        Args:
            name: Loader name
            loader_class: Loader class
        """
        self._data_loaders[name] = loader_class
        self.logger.info(f"Registered data loader: {name}")

    def register_data_processor(self, name: str, processor_class: type) -> None:
        """
        Register a data processor class

        Args:
            name: Processor name
            processor_class: Processor class
        """
        self._data_processors[name] = processor_class
        self.logger.info(f"Registered data processor: {name}")

    def register_feature_engineer(self, name: str, engineer_class: type) -> None:
        """
        Register a feature engineer class

        Args:
            name: Engineer name
            engineer_class: Engineer class
        """
        self._feature_engineers[name] = engineer_class
        self.logger.info(f"Registered feature engineer: {name}")

    def register_model(self, name: str, model_class: type) -> None:
        """
        Register a model class

        Args:
            name: Model name
            model_class: Model class
        """
        self._models[name] = model_class
        self.logger.info(f"Registered model: {name}")

    def register_evaluator(self, name: str, evaluator_class: type) -> None:
        """
        Register an evaluator class

        Args:
            name: Evaluator name
            evaluator_class: Evaluator class
        """
        self._evaluators[name] = evaluator_class
        self.logger.info(f"Registered evaluator: {name}")

    def get_data_loader(self, name: str, config: Optional[Dict[str, Any]] = None) -> DataLoaderProtocol:
        """
        Get a data loader instance

        Args:
            name: Loader name
            config: Configuration

        Returns:
            Data loader instance
        """
        if name not in self._data_loaders:
            raise ValueError(f"Data loader '{name}' not registered")

        config = config or {}
        return self._data_loaders[name](**config)

    def get_data_processor(self, name: str, config: Optional[Dict[str, Any]] = None) -> DataProcessorProtocol:
        """
        Get a data processor instance

        Args:
            name: Processor name
            config: Configuration

        Returns:
            Data processor instance
        """
        if name not in self._data_processors:
            raise ValueError(f"Data processor '{name}' not registered")

        config = config or {}
        return self._data_processors[name](**config)

    def get_feature_engineer(self, name: str, config: Optional[Dict[str, Any]] = None) -> FeatureEngineerProtocol:
        """
        Get a feature engineer instance

        Args:
            name: Engineer name
            config: Configuration

        Returns:
            Feature engineer instance
        """
        if name not in self._feature_engineers:
            raise ValueError(f"Feature engineer '{name}' not registered")

        config = config or {}
        return self._feature_engineers[name](**config)

    def get_model(self, name: str, config: Optional[Dict[str, Any]] = None) -> ModelProtocol:
        """
        Get a model instance

        Args:
            name: Model name
            config: Configuration

        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered")

        config = config or {}
        return self._models[name](**config)

    def get_evaluator(self, name: str, config: Optional[Dict[str, Any]] = None) -> EvaluatorProtocol:
        """
        Get an evaluator instance

        Args:
            name: Evaluator name
            config: Configuration

        Returns:
            Evaluator instance
        """
        if name not in self._evaluators:
            raise ValueError(f"Evaluator '{name}' not registered")

        config = config or {}
        return self._evaluators[name](**config)

    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components"""
        return {
            'data_loaders': list(self._data_loaders.keys()),
            'data_processors': list(self._data_processors.keys()),
            'feature_engineers': list(self._feature_engineers.keys()),
            'models': list(self._models.keys()),
            'evaluators': list(self._evaluators.keys())
        }


class PluginManager:
    """Manager for loading and managing plugins"""

    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.loaded_plugins = {}
        self.logger = logger.getChild("plugin_manager")

    def load_plugin(self, plugin_path: str) -> None:
        """
        Load a plugin from path

        Args:
            plugin_path: Path to plugin module
        """
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin from {plugin_path}")

            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)

            # Check if plugin has register_components function
            if hasattr(plugin_module, 'register_components'):
                plugin_module.register_components(self.registry)
                self.loaded_plugins[plugin_path] = plugin_module
                self.logger.info(f"Loaded plugin: {plugin_path}")
            else:
                self.logger.warning(f"Plugin {plugin_path} has no register_components function")

        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_path}: {str(e)}")
            raise

    def unload_plugin(self, plugin_path: str) -> None:
        """
        Unload a plugin

        Args:
            plugin_path: Path to plugin
        """
        if plugin_path in self.loaded_plugins:
            del self.loaded_plugins[plugin_path]
            self.logger.info(f"Unloaded plugin: {plugin_path}")
        else:
            self.logger.warning(f"Plugin {plugin_path} not loaded")

    def list_plugins(self) -> List[str]:
        """List loaded plugins"""
        return list(self.loaded_plugins.keys())


# Global registry instance
component_registry = ComponentRegistry()

# Global plugin manager instance
plugin_manager = PluginManager(component_registry)


__all__ = [
    "DataLoaderProtocol",
    "DataProcessorProtocol",
    "FeatureEngineerProtocol",
    "ModelProtocol",
    "EvaluatorProtocol",
    "PipelineStepProtocol",
    "ComponentFactory",
    "ComponentRegistry",
    "PluginManager",
    "component_registry",
    "plugin_manager"
]