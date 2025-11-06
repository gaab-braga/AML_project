"""
Centralized configuration management.
"""
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Singleton configuration loader."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load()
    
    def load(self, config_path: str = "config/pipeline_config.yaml"):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default
    
    @property
    def data_path(self) -> Path:
        return Path(self.get('paths.data', 'data/'))
    
    @property
    def model_path(self) -> Path:
        return Path(self.get('paths.model', 'models/'))
    
    @property
    def artifacts_path(self) -> Path:
        return Path(self.get('paths.artifacts', 'artifacts/'))


config = Config()
