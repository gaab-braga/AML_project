# Interfaces module
# Handles CLI, Web dashboard, REST API, and SDK

from .base import (
    DataLoaderProtocol,
    DataProcessorProtocol,
    FeatureEngineerProtocol,
    ModelProtocol,
    EvaluatorProtocol,
    PipelineStepProtocol,
    ComponentFactory,
    ComponentRegistry,
    PluginManager,
    component_registry,
    plugin_manager
)

# CLI Interface
from .cli import cli, CLIContext

# Web Dashboard
from .web_app import main as run_dashboard

# REST API
from .api import app as api_app, main as run_api

__all__ = [
    # Base components
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
    "plugin_manager",

    # CLI Interface
    "cli",
    "CLIContext",

    # Web Dashboard
    "run_dashboard",

    # REST API
    "api_app",
    "run_api"
]