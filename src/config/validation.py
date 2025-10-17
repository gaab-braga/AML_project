# src/config/validation.py
"""
Configuration validation utilities
"""

from typing import Dict, Any, List
from pathlib import Path
import os
from .settings import AMLSettings, Environment


def validate_config_file(config_file: Path) -> List[str]:
    """
    Validate configuration file structure and values

    Args:
        config_file: Path to configuration file

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not config_file.exists():
        errors.append(f"Configuration file not found: {config_file}")
        return errors

    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            errors.append("Configuration file is empty")
            return errors

        # Validate required sections
        required_sections = [
            "data", "features", "modeling", "orchestration",
            "evaluation", "interfaces", "monitoring"
        ]

        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: {section}")

        # Validate data paths
        if "data" in config_data:
            data_config = config_data["data"]
            path_fields = ["raw_path", "processed_path", "models_path"]

            for field in path_fields:
                if field in data_config:
                    path_str = data_config[field]
                    if isinstance(path_str, str):
                        path = Path(path_str)
                        if not path.parent.exists():
                            errors.append(f"Parent directory does not exist for {field}: {path.parent}")

        # Validate modeling algorithms
        if "modeling" in config_data:
            modeling_config = config_data["modeling"]
            if "algorithms" in modeling_config:
                algorithms = modeling_config["algorithms"]
                valid_algorithms = ["lightgbm", "xgboost", "catboost", "random_forest", "logistic_regression"]

                for alg in algorithms:
                    if alg not in valid_algorithms:
                        errors.append(f"Invalid algorithm: {alg}. Valid options: {valid_algorithms}")

        # Validate orchestration settings
        if "orchestration" in config_data:
            orch_config = config_data["orchestration"]
            if "max_parallel_tasks" in orch_config:
                max_parallel = orch_config["max_parallel_tasks"]
                if not isinstance(max_parallel, int) or not (1 <= max_parallel <= 16):
                    errors.append("max_parallel_tasks must be integer between 1 and 16")

    except Exception as e:
        errors.append(f"Error parsing configuration file: {e}")

    return errors


def validate_environment_variables() -> List[str]:
    """
    Validate environment variables for configuration

    Returns:
        List of validation errors
    """
    errors = []

    # Check critical environment variables
    critical_vars = [
        ("AML_PROJECT_NAME", "Project name"),
        ("AML_ENVIRONMENT", "Deployment environment"),
    ]

    for var_name, description in critical_vars:
        if not os.getenv(var_name):
            errors.append(f"Missing environment variable: {var_name} ({description})")

    # Validate environment value
    env_value = os.getenv("AML_ENVIRONMENT", "").upper()
    if env_value and env_value not in [e.value.upper() for e in Environment]:
        valid_envs = [e.value for e in Environment]
        errors.append(f"Invalid AML_ENVIRONMENT: {env_value}. Valid options: {valid_envs}")

    return errors


def create_default_config(config_file: Path, environment: Environment = Environment.DEVELOPMENT) -> None:
    """
    Create default configuration file

    Args:
        config_file: Path to create config file
        environment: Target environment
    """
    from .settings import AMLSettings

    # Create default settings
    default_settings = AMLSettings(environment=environment)

    # Adjust settings based on environment
    if environment == Environment.PRODUCTION:
        default_settings.orchestration.max_parallel_tasks = 8
        default_settings.monitoring.log_level = "WARNING"
        default_settings.interfaces.api_key_required = True
    elif environment == Environment.STAGING:
        default_settings.orchestration.max_parallel_tasks = 4
        default_settings.monitoring.log_level = "INFO"

    # Save to file
    from .settings import save_settings
    save_settings(config_file)

    print(f"Default configuration created: {config_file}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Configuration to override

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_config_summary(settings: AMLSettings) -> Dict[str, Any]:
    """
    Get configuration summary for logging

    Args:
        settings: AML settings object

    Returns:
        Summary dictionary
    """
    return {
        "project": f"{settings.project_name} v{settings.version}",
        "environment": settings.environment.value,
        "data_paths": {
            "raw": str(settings.data.raw_path),
            "processed": str(settings.data.processed_path),
            "models": str(settings.data.models_path),
        },
        "modeling": {
            "algorithms": settings.modeling.algorithms,
            "ensemble": settings.modeling.ensemble_method,
            "cv_folds": settings.modeling.cv_folds,
        },
        "orchestration": {
            "max_parallel": settings.orchestration.max_parallel_tasks,
            "cache_memory": f"{settings.orchestration.cache_memory_limit_mb}MB",
            "cache_ttl": f"{settings.orchestration.cache_ttl_hours}h",
        },
        "interfaces": {
            "api": f"{settings.interfaces.api_host}:{settings.interfaces.api_port}",
            "web": f"enabled={settings.interfaces.web_enabled}",
        },
        "monitoring": {
            "prometheus": f"enabled={settings.monitoring.enable_prometheus}",
            "log_level": settings.monitoring.log_level.value,
        }
    }


__all__ = [
    "validate_config_file",
    "validate_environment_variables",
    "create_default_config",
    "merge_configs",
    "get_config_summary"
]