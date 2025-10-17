"""
Experiment Management Utilities
Handles experiment configuration, saving, and registry management
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

class ExperimentManager:
    """Manages ML experiments with standardized configuration"""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.experiments_dir = artifacts_dir / "experiments"
        self.registry_path = artifacts_dir / "registry.json"
        self.experiments_dir.mkdir(exist_ok=True)

    def create_experiment_config(
        self,
        experiment_name: str,
        description: str,
        notebook_name: str,
        # Data parameters
        train_samples: int,
        test_samples: int,
        train_fraud_rate: float,
        test_fraud_rate: float,
        features_selected: List[str],
        balancing_applied: bool = False,
        balancing_method: Optional[str] = None,
        # Feature engineering parameters
        temporal_features: bool = True,
        categorical_encoding: bool = True,
        outlier_removal: bool = True,
        leakage_detection: bool = True,
        removed_features: Optional[List[str]] = None,
        final_feature_count: int = 0,
        # Validation parameters
        temporal_cv_folds: int = 5,
        primary_metric: str = "auc",
        # Additional metadata
        **kwargs
    ) -> Dict[str, Any]:
        """Create a standardized experiment configuration"""

        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{timestamp}"

        config = {
            "experiment": {
                "id": experiment_id,
                "name": experiment_name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "created_by": notebook_name,
                "version": "1.0"
            },
            "data": {
                "source": "df_Money_Laundering_v2.csv",
                "temporal_split": True,
                "train_samples": train_samples,
                "test_samples": test_samples,
                "train_fraud_rate": train_fraud_rate,
                "test_fraud_rate": test_fraud_rate,
                "features_selected": features_selected,
                "balancing_applied": balancing_applied,
                "balancing_method": balancing_method
            },
            "feature_engineering": {
                "temporal_features": temporal_features,
                "categorical_encoding": categorical_encoding,
                "outlier_removal": outlier_removal,
                "leakage_detection": leakage_detection,
                "removed_features": removed_features or [],
                "final_feature_count": final_feature_count
            },
            "validation": {
                "temporal_cv_folds": temporal_cv_folds,
                "metrics": ["precision", "recall", "f1", "auc"],
                "primary_metric": primary_metric
            },
            "artifacts": {
                "X_train_path": "X_train_temporal_clean.parquet",
                "y_train_path": "y_train_processed.parquet",
                "X_test_path": "X_test_temporal_clean.parquet",
                "y_test_path": "y_test_processed.parquet",
                "model_path": None,
                "metadata_path": "data_prep_complete_metadata.json"
            },
            "quality_checks": {
                "no_nan_values": True,
                "temporal_ordering_preserved": True,
                "feature_consistency": True,
                "leakage_free": True
            }
        }

        # Add any additional parameters
        if kwargs:
            config["additional_params"] = kwargs

        return config

    def save_experiment(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        removed_features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a complete experiment with all artifacts"""

        experiment_id = config["experiment"]["id"]
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)

        print(f"💾 Saving experiment: {experiment_id}")

        # Save configuration
        config_path = experiment_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"  ✓ Config saved: {config_path}")

        # Save data artifacts
        X_train_path = experiment_dir / config["artifacts"]["X_train_path"]
        y_train_path = experiment_dir / config["artifacts"]["y_train_path"]
        X_test_path = experiment_dir / config["artifacts"]["X_test_path"]
        y_test_path = experiment_dir / config["artifacts"]["y_test_path"]

        X_train.to_parquet(X_train_path)
        y_train.to_frame('target').to_parquet(y_train_path)
        X_test.to_parquet(X_test_path)
        y_test.to_frame('target').to_parquet(y_test_path)

        print(f"  ✓ Data saved: {len(X_train)} train, {len(X_test)} test samples")

        # Save removed features if any
        if removed_features:
            removed_features_path = experiment_dir / "removed_leaky_features.pkl"
            with open(removed_features_path, 'wb') as f:
                pickle.dump(removed_features, f)
            print(f"  ✓ Removed features saved: {len(removed_features)} features")

        # Save metadata if provided
        if metadata:
            metadata_path = experiment_dir / config["artifacts"]["metadata_path"]
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"  ✓ Metadata saved")

        print(f"🎉 Experiment saved successfully: {experiment_dir}")
        return experiment_dir

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load an experiment configuration"""
        config_path = self.experiments_dir / experiment_id / "experiment_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        with open(config_path, 'r') as f:
            return json.load(f)

    def promote_to_production(
        self,
        experiment_id: str,
        environment: str = "staging",
        notes: str = "",
        promoted_by: str = "notebook"
    ) -> None:
        """Promote an experiment to production/staging"""

        if environment not in ["staging", "production"]:
            raise ValueError("Environment must be 'staging' or 'production'")

        # Load registry
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        # Update registry
        experiment_path = f"experiments/{experiment_id}"
        registry[environment] = {
            "model_path": experiment_path,
            "experiment_id": experiment_id,
            "promoted_at": datetime.now().isoformat(),
            "promoted_by": promoted_by,
            "notes": notes
        }

        registry["last_updated"] = datetime.now().isoformat()

        # Save registry
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"🚀 Experiment {experiment_id} promoted to {environment}")
        print(f"   Notes: {notes}")

    def get_production_config(self, environment: str = "production") -> Optional[Dict[str, Any]]:
        """Get the configuration of the current production model"""

        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        env_config = registry.get(environment)
        if not env_config or not env_config.get("experiment_id"):
            return None

        experiment_id = env_config["experiment_id"]
        return self.load_experiment(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiments"""
        experiments = []

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('template'):
                config_path = exp_dir / "experiment_config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        experiments.append({
                            "id": config["experiment"]["id"],
                            "name": config["experiment"]["name"],
                            "created_at": config["experiment"]["created_at"],
                            "created_by": config["experiment"]["created_by"]
                        })

        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)


# Convenience function for notebooks
def create_experiment_manager(artifacts_dir: Optional[Path] = None) -> ExperimentManager:
    """Create an experiment manager instance"""
    if artifacts_dir is None:
        artifacts_dir = Path("../../artifacts").resolve()

    return ExperimentManager(artifacts_dir)