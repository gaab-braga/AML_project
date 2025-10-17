"""
AML Pipeline MLflow Integration Module

This module provides comprehensive MLflow integration for the AML Pipeline,
including experiment tracking, model registry, and automated logging.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import configparser

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMLPipelineMLflow:
    """
    MLflow integration class for AML Pipeline operations.

    Provides comprehensive experiment tracking, model versioning,
    and automated logging capabilities.
    """

    def __init__(self, config_path: str = "./mlops/mlops-config.ini"):
        """
        Initialize MLflow integration.

        Args:
            config_path: Path to MLops configuration file
        """
        self.config = self._load_config(config_path)
        self.client = MlflowClient()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config.get('mlflow', 'tracking_uri'))

        # Enable autologging if configured
        if self.config.getboolean('experiment_tracking', 'enable_autologging', fallback=True):
            mlflow.autolog()

        logger.info("AML Pipeline MLflow integration initialized")

    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load MLops configuration."""
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def create_experiment(self, experiment_name: str, description: str = "") -> str:
        """
        Create a new MLflow experiment.

        Args:
            experiment_name: Name of the experiment
            description: Optional experiment description

        Returns:
            Experiment ID
        """
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=self.config.get('mlflow', 'artifact_store_uri')
            )

            # Set experiment tags
            mlflow.set_experiment_tag("description", description)
            mlflow.set_experiment_tag("created_by", "aml_pipeline")
            mlflow.set_experiment_tag("created_at", datetime.now().isoformat())

            logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id

        except MlflowException as e:
            if "already exists" in str(e):
                logger.warning(f"Experiment {experiment_name} already exists")
                return mlflow.get_experiment_by_name(experiment_name).experiment_id
            else:
                raise

    def start_run(self, experiment_name: str, run_name: str = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name

        Returns:
            Active MLflow run
        """
        # Set experiment
        if mlflow.get_experiment_by_name(experiment_name) is None:
            self.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)

        # Start run
        run = mlflow.start_run(run_name=run_name)

        # Set run tags
        mlflow.set_tag("pipeline", "aml_pipeline")
        mlflow.set_tag("started_at", datetime.now().isoformat())

        logger.info(f"Started run: {run.info.run_id}")
        return run

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            parameters: Dictionary of parameters to log
        """
        for key, value in parameters.items():
            mlflow.log_param(key, value)

        logger.info(f"Logged {len(parameters)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

        logger.info(f"Logged {len(metrics)} metrics")

    def log_model(self, model: Any, artifact_path: str = "model",
                  flavor: str = "auto") -> None:
        """
        Log a model to the current run.

        Args:
            model: The model object to log
            artifact_path: Path within the run's artifact directory
            flavor: Model flavor ('auto', 'sklearn', 'xgboost', 'lightgbm', 'catboost')
        """
        if flavor == "auto":
            # Auto-detect model type
            model_type = type(model).__name__.lower()
            if "xgb" in model_type:
                flavor = "xgboost"
            elif "lgb" in model_type or "lightgbm" in model_type:
                flavor = "lightgbm"
            elif "cat" in model_type:
                flavor = "catboost"
            else:
                flavor = "sklearn"

        # Log model based on flavor
        if flavor == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        elif flavor == "catboost":
            mlflow.catboost.log_model(model, artifact_path)
        else:
            # Fallback to sklearn
            mlflow.sklearn.log_model(model, artifact_path)

        logger.info(f"Logged model with flavor: {flavor}")

    def log_artifacts(self, local_dir: str, artifact_path: str = None) -> None:
        """
        Log artifacts to the current run.

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within the run's artifact directory
        """
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.info(f"Logged artifacts from: {local_dir}")

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """
        Log a single artifact to the current run.

        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the run's artifact directory
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def register_model(self, model_name: str, model_uri: str = None) -> str:
        """
        Register a model in the model registry.

        Args:
            model_name: Name of the model
            model_uri: URI of the model to register (if None, uses current run)

        Returns:
            Model version
        """
        if model_uri is None:
            # Use model from current run
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

        try:
            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(model_name)
                logger.info(f"Created registered model: {model_name}")
            except MlflowException:
                logger.info(f"Registered model {model_name} already exists")

            # Create model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None
            )

            logger.info(f"Registered model version: {model_version.version}")
            return model_version.version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str,
                              stage: str) -> None:
        """
        Transition a model version to a different stage.

        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage ('development', 'staging', 'production', 'archived')
        """
        valid_stages = self.config.get('model_registry', 'stages').strip('[]').replace(' ', '').split(',')

        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Valid stages: {valid_stages}")

        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

        logger.info(f"Transitioned model {model_name} v{version} to stage: {stage}")

    def get_model_uri(self, model_name: str, version: str = None,
                      stage: str = None) -> str:
        """
        Get the URI of a registered model.

        Args:
            model_name: Name of the registered model
            version: Specific version (optional)
            stage: Stage to get the latest version from (optional)

        Returns:
            Model URI
        """
        if version:
            model_version = self.client.get_model_version(model_name, version)
        elif stage:
            model_version = self.client.get_latest_versions(model_name, stages=[stage])[0]
        else:
            # Get latest production version
            model_version = self.client.get_latest_versions(model_name, stages=["production"])[0]

        return f"models:/{model_name}/{model_version.version}"

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from the registry.

        Args:
            model_uri: URI of the model to load

        Returns:
            Loaded model object
        """
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from: {model_uri}")
        return model

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment information
        """
        experiments = self.client.list_experiments(ViewType.ALL)
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location
            }
            for exp in experiments
        ]

    def list_runs(self, experiment_id: str = None) -> List[Dict[str, Any]]:
        """
        List runs for an experiment.

        Args:
            experiment_id: Experiment ID (if None, uses current experiment)

        Returns:
            List of run information
        """
        if experiment_id is None and mlflow.get_experiment_by_name(
            self.config.get('experiment_tracking', 'experiment_name')
        ):
            experiment_id = mlflow.get_experiment_by_name(
                self.config.get('experiment_tracking', 'experiment_name')
            ).experiment_id

        if experiment_id:
            runs = self.client.search_runs(experiment_ids=[experiment_id])
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params)
                }
                for run in runs
            ]
        else:
            return []

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of registered model information
        """
        models = self.client.list_registered_models()
        return [
            {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description
            }
            for model in models
        ]

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


# Convenience functions for quick usage
def quick_experiment(experiment_name: str, run_name: str = None):
    """
    Quick experiment context manager.

    Usage:
        with quick_experiment("my_experiment"):
            # Your ML code here
            mlflow.log_param("param", value)
            mlflow.log_metric("metric", value)
    """
    mlflow_integration = AMLPipelineMLflow()

    class QuickExperiment:
        def __enter__(self):
            self.run = mlflow_integration.start_run(experiment_name, run_name)
            return self.run

        def __exit__(self, exc_type, exc_val, exc_tb):
            mlflow_integration.end_run()

    return QuickExperiment()


def log_model_metrics(model_name: str, metrics: Dict[str, float],
                     parameters: Dict[str, Any] = None) -> None:
    """
    Quick function to log model metrics.

    Args:
        model_name: Name for the model/run
        metrics: Dictionary of metrics
        parameters: Dictionary of parameters
    """
    with quick_experiment("model_evaluation", model_name):
        if parameters:
            mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)


# Export main class
__all__ = ['AMLPipelineMLflow', 'quick_experiment', 'log_model_metrics']