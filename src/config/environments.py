# src/config/environments.py
"""
Environment-specific configuration management
"""

from typing import Dict, Any
from pathlib import Path
from .settings import AMLSettings, Environment, load_settings


def get_environment_config(environment: Environment) -> Dict[str, Any]:
    """
    Get environment-specific configuration overrides

    Args:
        environment: Target environment

    Returns:
        Dictionary with environment-specific settings
    """
    base_config = {
        "environment": environment.value,
    }

    if environment == Environment.DEVELOPMENT:
        config = {
            **base_config,
            "orchestration": {
                "max_parallel_tasks": 2,
                "cache_memory_limit_mb": 1024,
                "task_timeout_seconds": 1800,  # 30 minutes
            },
            "monitoring": {
                "log_level": "DEBUG",
                "enable_prometheus": False,
            },
            "interfaces": {
                "api_key_required": False,
                "web_enabled": True,
            }
        }

    elif environment == Environment.STAGING:
        config = {
            **base_config,
            "orchestration": {
                "max_parallel_tasks": 4,
                "cache_memory_limit_mb": 2048,
                "task_timeout_seconds": 3600,  # 1 hour
            },
            "monitoring": {
                "log_level": "INFO",
                "enable_prometheus": True,
                "alert_channels": ["email"],
            },
            "interfaces": {
                "api_key_required": True,
                "web_enabled": True,
            }
        }

    elif environment == Environment.PRODUCTION:
        config = {
            **base_config,
            "orchestration": {
                "max_parallel_tasks": 8,
                "cache_memory_limit_mb": 4096,
                "task_timeout_seconds": 7200,  # 2 hours
            },
            "monitoring": {
                "log_level": "WARNING",
                "enable_prometheus": True,
                "alert_channels": ["email", "slack"],
                "alert_rules": {
                    "task_failure_threshold": 5,
                    "pipeline_timeout_hours": 4,
                }
            },
            "interfaces": {
                "api_key_required": True,
                "web_enabled": True,
            },
            "data": {
                "validation_enabled": True,
                "fail_on_missing": True,
            }
        }

    else:
        raise ValueError(f"Unknown environment: {environment}")

    return config


def load_environment_settings(environment: Environment,
                             config_file: Path = None) -> AMLSettings:
    """
    Load settings for specific environment

    Args:
        environment: Target environment
        config_file: Optional config file path

    Returns:
        AMLSettings configured for environment
    """
    # Load base settings
    if config_file and config_file.exists():
        settings = load_settings(config_file)
    else:
        settings = AMLSettings()

    # Apply environment-specific overrides
    env_config = get_environment_config(environment)

    # Update settings with environment config
    for key, value in env_config.items():
        if hasattr(settings, key):
            if isinstance(value, dict):
                # Update nested attributes
                sub_obj = getattr(settings, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_obj, sub_key):
                        setattr(sub_obj, sub_key, sub_value)
            else:
                setattr(settings, key, value)

    # Ensure environment is set correctly
    settings.environment = environment

    return settings


def create_environment_config_files(base_dir: Path = Path("config")) -> None:
    """
    Create configuration files for all environments

    Args:
        base_dir: Base configuration directory
    """
    from .validation import create_default_config

    base_dir.mkdir(exist_ok=True)

    for env in Environment:
        config_file = base_dir / f"settings.{env.value}.yaml"
        if not config_file.exists():
            create_default_config(config_file, env)
            print(f"Created {env.value} config: {config_file}")


def validate_environment_setup(environment: Environment) -> Dict[str, Any]:
    """
    Validate environment setup and requirements

    Args:
        environment: Environment to validate

    Returns:
        Validation results dictionary
    """
    results = {
        "environment": environment.value,
        "valid": True,
        "issues": [],
        "recommendations": []
    }

    # Check configuration file
    config_dir = Path("config")
    config_file = config_dir / f"settings.{environment.value}.yaml"
    default_config = config_dir / "settings.yaml"

    if not config_file.exists() and not default_config.exists():
        results["issues"].append("No configuration file found")
        results["recommendations"].append("Run create_environment_config_files() to create config files")

    # Check required directories
    from .settings import AMLSettings
    settings = AMLSettings()

    required_dirs = [
        settings.data.raw_path,
        settings.data.processed_path,
        settings.data.models_path,
        settings.orchestration.cache_disk_path,
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            results["issues"].append(f"Required directory missing: {dir_path}")
            results["recommendations"].append(f"Create directory: mkdir -p {dir_path}")

    # Environment-specific checks
    if environment == Environment.PRODUCTION:
        if not settings.orchestration.cache_redis_url:
            results["issues"].append("Redis cache URL not configured for production")
            results["recommendations"].append("Set ORCHESTRATION_CACHE_REDIS_URL environment variable")

        if not settings.monitoring.enable_prometheus:
            results["issues"].append("Prometheus monitoring disabled in production")
            results["recommendations"].append("Enable monitoring for production environment")

    elif environment == Environment.DEVELOPMENT:
        if settings.orchestration.max_parallel_tasks > 4:
            results["recommendations"].append("Consider reducing parallel tasks for development")

    # Update validity
    results["valid"] = len(results["issues"]) == 0

    return results


def setup_environment(environment: Environment,
                     config_dir: Path = Path("config"),
                     create_dirs: bool = True) -> AMLSettings:
    """
    Complete environment setup

    Args:
        environment: Target environment
        config_dir: Configuration directory
        create_dirs: Whether to create required directories

    Returns:
        Configured AMLSettings
    """
    print(f"Setting up environment: {environment.value}")

    # Create config files if needed
    create_environment_config_files(config_dir)

    # Load environment settings
    config_file = config_dir / f"settings.{environment.value}.yaml"
    settings = load_environment_settings(environment, config_file)

    # Create required directories
    if create_dirs:
        dirs_to_create = [
            settings.data.raw_path,
            settings.data.processed_path,
            settings.data.models_path,
            settings.orchestration.cache_disk_path,
            Path("logs"),
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")

    # Validate setup
    validation = validate_environment_setup(environment)
    if not validation["valid"]:
        print("Setup validation issues:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
        print("Recommendations:")
        for rec in validation["recommendations"]:
            print(f"  - {rec}")

    print(f"Environment setup complete for: {environment.value}")
    return settings


__all__ = [
    "get_environment_config",
    "load_environment_settings",
    "create_environment_config_files",
    "validate_environment_setup",
    "setup_environment"
]