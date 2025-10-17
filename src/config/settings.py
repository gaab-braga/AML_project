# src/config/settings.py
"""
Unified Configuration Management
Uses Pydantic for validation and type safety
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import os
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataConfig(BaseSettings):
    """Data processing configuration"""
    raw_path: Path = Field(default=Path("data"), description="Raw data directory")
    processed_path: Path = Field(default=Path("artifacts/processed"), description="Processed data directory")
    models_path: Path = Field(default=Path("artifacts/models"), description="Models directory")

    validation_enabled: bool = Field(default=True, description="Enable data validation")
    fail_on_missing: bool = Field(default=True, description="Fail if data is missing")
    max_missing_percentage: float = Field(default=0.05, ge=0.0, le=1.0, description="Max missing data percentage")

    class Config:
        env_prefix = "DATA_"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration"""
    anti_leakage_enabled: bool = Field(default=True, description="Enable anti-leakage validation")
    temporal_validation: bool = Field(default=True, description="Enable temporal validation")

    selection_method: str = Field(default="permutation_importance", description="Feature selection method")
    selection_threshold: float = Field(default=0.001, ge=0.0, le=1.0, description="Feature selection threshold")

    scaling_method: str = Field(default="robust", description="Feature scaling method")
    encoding_method: str = Field(default="frequency", description="Categorical encoding method")

    class Config:
        env_prefix = "FEATURES_"


class ModelingConfig(BaseSettings):
    """ML modeling configuration"""
    algorithms: List[str] = Field(
        default=["lightgbm", "xgboost", "catboost"],
        description="ML algorithms to use"
    )

    ensemble_method: str = Field(default="stacking", description="Ensemble method")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")
    meta_algorithm: str = Field(default="xgboost", description="Meta-algorithm for stacking")

    calibration_method: str = Field(default="isotonic", description="Model calibration method")
    optimize_threshold: bool = Field(default=True, description="Optimize decision threshold")

    class Config:
        env_prefix = "MODELING_"


class OrchestrationConfig(BaseSettings):
    """DAG orchestration configuration"""
    max_parallel_tasks: int = Field(default=4, ge=1, le=16, description="Maximum parallel tasks")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Task retry attempts")
    task_timeout_seconds: Optional[int] = Field(default=3600, ge=60, description="Task timeout")

    cache_memory_limit_mb: int = Field(default=2048, ge=128, description="Cache memory limit")
    cache_disk_path: Path = Field(default=Path("./cache"), description="Cache disk path")
    cache_redis_url: Optional[str] = Field(default=None, description="Redis cache URL")
    cache_ttl_hours: int = Field(default=24, ge=1, description="Cache TTL hours")

    class Config:
        env_prefix = "ORCHESTRATION_"


class EvaluationConfig(BaseSettings):
    """Model evaluation configuration"""
    metrics: List[str] = Field(
        default=["precision", "recall", "f1", "auc_pr", "auc_roc"],
        description="Evaluation metrics"
    )

    validation_method: str = Field(default="temporal", description="Validation method")
    n_splits: int = Field(default=5, ge=2, le=10, description="Number of validation splits")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")

    calibration_bins: int = Field(default=10, ge=5, le=20, description="Calibration bins")

    class Config:
        env_prefix = "EVALUATION_"


class InterfaceConfig(BaseSettings):
    """Interface configuration"""
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1000, le=9999, description="API port")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    api_key_required: bool = Field(default=True, description="Require API key")

    web_enabled: bool = Field(default=True, description="Enable web dashboard")
    web_port: int = Field(default=8501, ge=1000, le=9999, description="Web dashboard port")
    web_theme: str = Field(default="light", description="Web theme")

    cli_auto_complete: bool = Field(default=True, description="Enable CLI auto-complete")

    class Config:
        env_prefix = "INTERFACE_"


class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration"""
    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, ge=1000, le=9999, description="Prometheus port")

    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_file: Optional[Path] = Field(default=Path("logs/aml_platform.log"), description="Log file path")

    alert_channels: List[str] = Field(default=["email", "slack"], description="Alert channels")
    alert_rules: Dict[str, Any] = Field(default_factory=dict, description="Alert rules")

    class Config:
        env_prefix = "MONITORING_"


class AMLSettings(BaseSettings):
    """Main application settings"""
    project_name: str = Field(default="aml_detection_platform", description="Project name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    interfaces: InterfaceConfig = Field(default_factory=InterfaceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    class Config:
        env_prefix = "AML_"
        case_sensitive = False

    @validator("data", "features", "modeling", "orchestration", "evaluation", "interfaces", "monitoring", pre=True)
    def validate_sub_configs(cls, v):
        """Ensure sub-configs are properly instantiated"""
        if isinstance(v, dict):
            return v
        return {}

    def get_config_file_path(self) -> Path:
        """Get configuration file path based on environment"""
        config_dir = Path("config")
        env_file = f"settings.{self.environment.value}.yaml"
        default_file = "settings.yaml"

        # Try environment-specific config first
        env_config = config_dir / env_file
        if env_config.exists():
            return env_config

        # Fall back to default config
        default_config = config_dir / default_file
        if default_config.exists():
            return default_config

        # Return default path (will be created)
        return config_dir / default_file


# Global settings instance
settings = AMLSettings()


def load_settings(config_file: Optional[Path] = None) -> AMLSettings:
    """
    Load settings from file or environment variables

    Args:
        config_file: Optional path to YAML config file

    Returns:
        AMLSettings: Loaded and validated settings
    """
    global settings

    if config_file and config_file.exists():
        # Load from YAML file
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            settings = AMLSettings(**config_data)
        except ImportError:
            # YAML not available, use environment variables only
            settings = AMLSettings()
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
            settings = AMLSettings()
    else:
        # Load from environment variables only
        settings = AMLSettings()

    return settings


def save_settings(config_file: Optional[Path] = None) -> None:
    """
    Save current settings to YAML file

    Args:
        config_file: Optional path to save config file
    """
    if not config_file:
        config_file = settings.get_config_file_path()

    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml
        config_data = settings.dict()
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        print("Warning: PyYAML not available, cannot save config to file")


# Initialize settings on import
def init_settings(config_file: Optional[Path] = None) -> AMLSettings:
    """Initialize settings with optional config file"""
    return load_settings(config_file)


# Export main settings object
__all__ = ["AMLSettings", "settings", "load_settings", "save_settings", "init_settings"]