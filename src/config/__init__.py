# Configuration module
# Handles unified configuration management with Pydantic

from .settings import AMLSettings, settings, load_settings, save_settings, init_settings
from .validation import (
    validate_config_file,
    validate_environment_variables,
    create_default_config,
    merge_configs,
    get_config_summary
)
from .environments import (
    get_environment_config,
    load_environment_settings,
    create_environment_config_files,
    validate_environment_setup,
    setup_environment
)

__all__ = [
    # Main settings
    "AMLSettings",
    "settings",
    "load_settings",
    "save_settings",
    "init_settings",

    # Validation
    "validate_config_file",
    "validate_environment_variables",
    "create_default_config",
    "merge_configs",
    "get_config_summary",

    # Environments
    "get_environment_config",
    "load_environment_settings",
    "create_environment_config_files",
    "validate_environment_setup",
    "setup_environment",
]