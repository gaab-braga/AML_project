# src/orchestration/base.py
"""
Base classes and interfaces for pipeline orchestration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..utils import logger
from ..config import settings


class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger.getChild(f"pipeline_step.{name}")
        self.execution_time = 0.0
        self.is_completed = False
        self.output = None
        self.error = None

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Execute the pipeline step

        Args:
            input_data: Input data for the step

        Returns:
            Output data from the step
        """
        pass

    def run(self, input_data: Any) -> Any:
        """
        Run the step with timing and error handling

        Args:
            input_data: Input data

        Returns:
            Step output
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting pipeline step: {self.name}")
            self.output = self.execute(input_data)
            self.is_completed = True
            self.logger.info(f"Completed pipeline step: {self.name}")

        except Exception as e:
            error_msg = f"Pipeline step {self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error = str(e)
            raise

        finally:
            self.execution_time = time.time() - start_time
            self.logger.info(f"Step {self.name} execution time: {self.execution_time:.2f}s")

        return self.output

    def get_status(self) -> Dict[str, Any]:
        """Get step execution status"""
        return {
            'name': self.name,
            'completed': self.is_completed,
            'execution_time': self.execution_time,
            'error': self.error,
            'has_output': self.output is not None
        }


class DataProcessingStep(PipelineStep):
    """Base class for data processing steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.input_shape = None
        self.output_shape = None

    def execute(self, input_data: Any) -> Any:
        """Execute data processing"""
        if isinstance(input_data, pd.DataFrame):
            self.input_shape = input_data.shape
            self.logger.info(f"Input shape: {self.input_shape}")

        result = self.process_data(input_data)

        if isinstance(result, pd.DataFrame):
            self.output_shape = result.shape
            self.logger.info(f"Output shape: {self.output_shape}")

        return result

    @abstractmethod
    def process_data(self, data: Any) -> Any:
        """
        Process the data

        Args:
            data: Input data

        Returns:
            Processed data
        """
        pass


class FeatureEngineeringStep(PipelineStep):
    """Base class for feature engineering steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.created_features = []

    def execute(self, input_data: Any) -> Any:
        """Execute feature engineering"""
        result = self.engineer_features(input_data)

        # Track created features
        if hasattr(result, 'columns') and hasattr(input_data, 'columns'):
            new_features = set(result.columns) - set(input_data.columns)
            self.created_features = list(new_features)
            self.logger.info(f"Created {len(self.created_features)} new features")

        return result

    @abstractmethod
    def engineer_features(self, data: Any) -> Any:
        """
        Engineer features

        Args:
            data: Input data

        Returns:
            Data with engineered features
        """
        pass


class ModelingStep(PipelineStep):
    """Base class for modeling steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.model = None
        self.is_fitted = False

    def execute(self, input_data: Any) -> Any:
        """Execute modeling"""
        if isinstance(input_data, dict) and 'X_train' in input_data and 'y_train' in input_data:
            # Training phase
            self.model = self.train_model(input_data['X_train'], input_data['y_train'])
            self.is_fitted = True
            self.logger.info("Model training completed")

            # Return model and training results
            return {
                'model': self.model,
                'training_data': input_data,
                'is_fitted': True
            }

        elif isinstance(input_data, dict) and 'X_test' in input_data and self.is_fitted:
            # Prediction phase
            predictions = self.make_predictions(input_data['X_test'])
            self.logger.info("Model predictions completed")
            return predictions

        else:
            raise ValueError("Invalid input for modeling step")

    @abstractmethod
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train the model

        Args:
            X: Training features
            y: Training target

        Returns:
            Trained model
        """
        pass

    @abstractmethod
    def make_predictions(self, X: pd.DataFrame) -> Any:
        """
        Make predictions

        Args:
            X: Test features

        Returns:
            Predictions
        """
        pass


class EvaluationStep(PipelineStep):
    """Base class for evaluation steps"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.metrics = {}

    def execute(self, input_data: Any) -> Any:
        """Execute evaluation"""
        if not isinstance(input_data, dict):
            raise ValueError("Evaluation step expects dictionary input")

        results = self.evaluate_model(input_data)
        self.metrics = results
        self.logger.info("Model evaluation completed")

        return results

    @abstractmethod
    def evaluate_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance

        Args:
            data: Dictionary containing model, predictions, and true labels

        Returns:
            Evaluation results
        """
        pass


class PipelineOrchestrator:
    """Orchestrator for pipeline execution"""

    def __init__(self, name: str, steps: List[PipelineStep], config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.steps = steps
        self.config = config or {}
        self.logger = logger.getChild(f"orchestrator.{name}")
        self.execution_results = {}
        self.is_parallel = config.get('parallel_execution', False)
        self.max_workers = config.get('max_workers', 4)

    def execute_pipeline(self, initial_data: Any) -> Dict[str, Any]:
        """
        Execute the entire pipeline

        Args:
            initial_data: Initial input data

        Returns:
            Pipeline execution results
        """
        self.logger.info(f"Starting pipeline execution: {self.name}")

        start_time = time.time()
        current_data = initial_data

        try:
            if self.is_parallel:
                results = self._execute_parallel(current_data)
            else:
                results = self._execute_sequential(current_data)

            total_time = time.time() - start_time
            self.logger.info(f"Pipeline execution completed in {total_time:.2f}s")

            return {
                'success': True,
                'results': results,
                'total_time': total_time,
                'step_results': self.execution_results
            }

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'total_time': total_time,
                'step_results': self.execution_results
            }

    def _execute_sequential(self, initial_data: Any) -> Any:
        """Execute steps sequentially"""
        current_data = initial_data

        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")

            try:
                current_data = step.run(current_data)
                self.execution_results[step.name] = step.get_status()

            except Exception as e:
                self.execution_results[step.name] = step.get_status()
                raise

        return current_data

    def _execute_parallel(self, initial_data: Any) -> Any:
        """Execute steps in parallel where possible"""
        # For now, implement simple parallel execution
        # In a more advanced implementation, this would analyze dependencies

        self.logger.info("Executing pipeline in parallel mode")

        # Group independent steps (simplified - assumes all steps are independent)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_step = {}

            for step in self.steps:
                future = executor.submit(step.run, initial_data)
                future_to_step[future] = step

            results = {}
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()
                    results[step.name] = result
                    self.execution_results[step.name] = step.get_status()
                    self.logger.info(f"Step {step.name} completed successfully")

                except Exception as e:
                    self.execution_results[step.name] = step.get_status()
                    self.logger.error(f"Step {step.name} failed: {str(e)}")
                    raise

        # Combine results (simplified - assumes results can be merged)
        final_result = {}
        for step_name, result in results.items():
            if isinstance(result, dict):
                final_result.update(result)
            else:
                final_result[step_name] = result

        return final_result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        total_steps = len(self.steps)
        completed_steps = sum(1 for step in self.steps if step.is_completed)
        failed_steps = sum(1 for step in self.steps if step.error is not None)

        return {
            'pipeline_name': self.name,
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'success_rate': completed_steps / total_steps if total_steps > 0 else 0,
            'step_details': {step.name: step.get_status() for step in self.steps}
        }

    def validate_pipeline(self) -> List[str]:
        """Validate pipeline configuration"""
        errors = []

        if not self.steps:
            errors.append("Pipeline has no steps")
            return errors

        # Check for duplicate step names
        step_names = [step.name for step in self.steps]
        duplicates = set([name for name in step_names if step_names.count(name) > 1])
        if duplicates:
            errors.append(f"Duplicate step names found: {duplicates}")

        # Validate step configurations
        for step in self.steps:
            if hasattr(step, 'validate'):
                step_errors = step.validate()
                if step_errors:
                    errors.extend([f"Step {step.name}: {err}" for err in step_errors])

        return errors


class DAGOrchestrator(PipelineOrchestrator):
    """Directed Acyclic Graph orchestrator for complex pipelines"""

    def __init__(self, name: str, steps: List[PipelineStep], dependencies: Dict[str, List[str]],
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(name, steps, config)
        self.dependencies = dependencies  # step_name -> list of prerequisite step names
        self.step_dict = {step.name: step for step in steps}

    def _execute_sequential(self, initial_data: Any) -> Any:
        """Execute steps in dependency order"""
        # Topological sort
        execution_order = self._topological_sort()
        self.logger.info(f"Execution order: {execution_order}")

        results = {}
        current_data = initial_data

        for step_name in execution_order:
            step = self.step_dict[step_name]
            self.logger.info(f"Executing step: {step_name}")

            # Prepare input data (combine results from dependencies)
            step_input = self._prepare_step_input(step_name, results, current_data)

            try:
                step_result = step.run(step_input)
                results[step_name] = step_result
                self.execution_results[step_name] = step.get_status()

            except Exception as e:
                self.execution_results[step_name] = step.get_status()
                raise

        return results

    def _topological_sort(self) -> List[str]:
        """Perform topological sort of steps based on dependencies"""
        # Kahn's algorithm
        in_degree = {step.name: 0 for step in self.steps}
        for step_name, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[step_name] += 1

        queue = [step_name for step_name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Find steps that depend on current
            for step_name, deps in self.dependencies.items():
                if current in deps:
                    in_degree[step_name] -= 1
                    if in_degree[step_name] == 0:
                        queue.append(step_name)

        if len(result) != len(self.steps):
            raise ValueError("Circular dependency detected in pipeline")

        return result

    def _prepare_step_input(self, step_name: str, results: Dict[str, Any], initial_data: Any) -> Any:
        """Prepare input data for a step based on its dependencies"""
        deps = self.dependencies.get(step_name, [])

        if not deps:
            # No dependencies, use initial data
            return initial_data

        # Combine results from dependencies
        combined_input = {}

        for dep in deps:
            if dep in results:
                dep_result = results[dep]
                if isinstance(dep_result, dict):
                    combined_input.update(dep_result)
                else:
                    combined_input[dep] = dep_result

        # If no dependency results, use initial data
        if not combined_input:
            return initial_data

        return combined_input


__all__ = [
    "PipelineStep",
    "DataProcessingStep",
    "FeatureEngineeringStep",
    "ModelingStep",
    "EvaluationStep",
    "PipelineOrchestrator",
    "DAGOrchestrator"
]