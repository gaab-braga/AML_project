# src/orchestration/__init__.py
"""
Orchestration module for AML pipeline
"""

from .base import (
    PipelineStep,
    DataProcessingStep,
    FeatureEngineeringStep,
    ModelingStep,
    EvaluationStep,
    PipelineOrchestrator,
    DAGOrchestrator
)
from .task_definition import Task, DAG
from .dag_engine import DAGEngine
from .parallel_executor import ParallelExecutor
from .resource_manager import ResourceManager
from .error_handler import ErrorHandler

__all__ = [
    # Legacy classes
    "PipelineStep",
    "DataProcessingStep",
    "FeatureEngineeringStep",
    "ModelingStep",
    "EvaluationStep",
    "PipelineOrchestrator",
    "DAGOrchestrator",
    # New orchestration classes
    "Task",
    "DAG",
    "DAGEngine",
    "ParallelExecutor",
    "ResourceManager",
    "ErrorHandler"
]