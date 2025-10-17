# src/features/versioning.py
"""
Feature versioning and caching system
"""

import hashlib
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils import logger
from .metrics import feature_monitor


class FeatureVersion:
    """Represents a version of feature engineering configuration"""

    def __init__(self,
                 name: str,
                 config: Dict[str, Any],
                 feature_names: List[str],
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config
        self.feature_names = feature_names
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.version_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of the feature version"""
        # Create a deterministic representation
        version_data = {
            'name': self.name,
            'config': self._normalize_config(self.config),
            'feature_names': sorted(self.feature_names),
            'metadata': self._normalize_metadata(self.metadata)
        }

        # Convert to JSON string with sorted keys
        json_str = json.dumps(version_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config for hashing"""
        # Remove non-deterministic elements
        normalized = {}
        for key, value in config.items():
            if key not in ['random_state', 'timestamp', 'run_id']:
                if isinstance(value, dict):
                    normalized[key] = self._normalize_config(value)
                elif isinstance(value, (int, float, str, bool, list)):
                    normalized[key] = value
        return normalized

    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata for hashing"""
        normalized = {}
        for key, value in metadata.items():
            if key not in ['timestamp', 'created_at', 'execution_time']:
                if isinstance(value, (int, float, str, bool, list)):
                    normalized[key] = value
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'version_hash': self.version_hash,
            'config': self.config,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVersion':
        """Create from dictionary"""
        version = cls(
            name=data['name'],
            config=data['config'],
            feature_names=data['feature_names'],
            metadata=data.get('metadata', {})
        )
        version.created_at = datetime.fromisoformat(data['created_at'])
        return version


class FeatureCache:
    """Cache for feature engineering results"""

    def __init__(self, cache_dir: Path = Path("artifacts/cache/features")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # Simple in-memory cache

    def _get_cache_key(self, data_hash: str, version_hash: str) -> str:
        """Generate cache key"""
        return f"{data_hash}_{version_hash}"

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of input data"""
        # Use shape and column names for simplicity
        # In production, might want more sophisticated hashing
        data_repr = f"{data.shape}_{sorted(data.columns.tolist())}"
        return hashlib.sha256(data_repr.encode()).hexdigest()[:16]

    def get(self, data: pd.DataFrame, version: FeatureVersion) -> Optional[pd.DataFrame]:
        """Get cached features if available"""
        data_hash = self._compute_data_hash(data)
        cache_key = self._get_cache_key(data_hash, version.version_hash)

        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.info(f"Cache hit (memory): {cache_key}")
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Cache hit (disk): {cache_key}")
                # Store in memory cache too
                self.memory_cache[cache_key] = cached_data
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")

        return None

    def put(self, data: pd.DataFrame, features: pd.DataFrame, version: FeatureVersion) -> None:
        """Cache feature engineering results"""
        data_hash = self._compute_data_hash(data)
        cache_key = self._get_cache_key(data_hash, version.version_hash)

        # Store in memory cache
        self.memory_cache[cache_key] = features.copy()

        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features.copy(), f)
            logger.info(f"Cached features: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        cleared = 0

        # Clear memory cache
        if pattern:
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
                cleared += 1
        else:
            cleared = len(self.memory_cache)
            self.memory_cache.clear()

        # Clear disk cache
        if pattern:
            for cache_file in self.cache_dir.glob(f"*{pattern}*.pkl"):
                try:
                    cache_file.unlink()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {cleared} cache entries")
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        disk_files = list(self.cache_dir.glob("*.pkl"))

        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(disk_files),
            'total_cache_size_mb': sum(f.stat().st_size for f in disk_files) / (1024 * 1024)
        }


class FeatureVersionManager:
    """Manages feature engineering versions and caching"""

    def __init__(self, versions_dir: Path = Path("artifacts/versions")):
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, FeatureVersion] = {}
        self.cache = FeatureCache()
        self.current_version: Optional[FeatureVersion] = None

        # Load existing versions
        self._load_versions()

    def _load_versions(self):
        """Load existing versions from disk"""
        for version_file in self.versions_dir.glob("*.json"):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                version = FeatureVersion.from_dict(data)
                self.versions[version.name] = version
            except Exception as e:
                logger.warning(f"Failed to load version file {version_file}: {e}")

    def create_version(self,
                      name: str,
                      config: Dict[str, Any],
                      feature_names: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> FeatureVersion:
        """Create a new feature version"""
        version = FeatureVersion(name, config, feature_names, metadata)
        self.versions[name] = version

        # Save to disk
        version_file = self.versions_dir / f"{name}.json"
        try:
            with open(version_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2, default=str)
            logger.info(f"Created feature version: {name} ({version.version_hash})")
        except Exception as e:
            logger.error(f"Failed to save version file {version_file}: {e}")

        return version

    def get_version(self, name: str) -> Optional[FeatureVersion]:
        """Get a version by name"""
        return self.versions.get(name)

    def list_versions(self) -> List[str]:
        """List all version names"""
        return list(self.versions.keys())

    def set_current_version(self, name: str) -> bool:
        """Set the current active version"""
        if name in self.versions:
            self.current_version = self.versions[name]
            logger.info(f"Set current version to: {name}")
            return True
        else:
            logger.error(f"Version not found: {name}")
            return False

    def get_current_version(self) -> Optional[FeatureVersion]:
        """Get the current active version"""
        return self.current_version

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        comparison = {
            "version1": v1.name,
            "version2": v2.name,
            "hash1": v1.version_hash,
            "hash2": v2.version_hash,
            "same_hash": v1.version_hash == v2.version_hash,
            "features_v1": len(v1.feature_names),
            "features_v2": len(v2.feature_names),
            "common_features": len(set(v1.feature_names) & set(v2.feature_names)),
            "unique_to_v1": list(set(v1.feature_names) - set(v2.feature_names)),
            "unique_to_v2": list(set(v2.feature_names) - set(v1.feature_names))
        }

        return comparison

    def validate_version_compatibility(self,
                                     data: pd.DataFrame,
                                     version: FeatureVersion) -> Dict[str, Any]:
        """Validate if data is compatible with a version"""
        validation = {
            "compatible": True,
            "issues": [],
            "warnings": []
        }

        # Check if required columns exist
        required_cols = set(version.config.get('required_columns', []))
        available_cols = set(data.columns)

        missing_cols = required_cols - available_cols
        if missing_cols:
            validation["compatible"] = False
            validation["issues"].append(f"Missing required columns: {missing_cols}")

        # Check data types if specified
        expected_types = version.config.get('expected_dtypes', {})
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if actual_type != expected_type:
                    validation["warnings"].append(
                        f"Column {col} type mismatch: expected {expected_type}, got {actual_type}"
                    )

        return validation

    def get_features_with_caching(self,
                                 data: pd.DataFrame,
                                 version: FeatureVersion,
                                 feature_engineer) -> pd.DataFrame:
        """Get features with caching support"""
        # Check cache first
        cached_features = self.cache.get(data, version)
        if cached_features is not None:
            return cached_features

        # Validate compatibility
        validation = self.validate_version_compatibility(data, version)
        if not validation["compatible"]:
            raise ValueError(f"Version incompatibility: {validation['issues']}")

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)

        # Generate features
        with feature_monitor.monitor_operation(
            f"version_{version.name}_feature_generation",
            data
        ) as metrics:
            features = feature_engineer.create_features(data)
            metrics.output_shape = features.shape
            metrics.features_created = features.shape[1] - data.shape[1]

        # Cache the results
        self.cache.put(data, features, version)

        return features

    def get_version_stats(self) -> Dict[str, Any]:
        """Get statistics about versions"""
        if not self.versions:
            return {}

        created_dates = [v.created_at for v in self.versions.values()]

        return {
            'total_versions': len(self.versions),
            'oldest_version': min(created_dates).isoformat(),
            'newest_version': max(created_dates).isoformat(),
            'versions_by_month': self._group_versions_by_month(),
            'cache_stats': self.cache.get_stats()
        }

    def _group_versions_by_month(self) -> Dict[str, int]:
        """Group versions by creation month"""
        monthly_counts = {}
        for version in self.versions.values():
            month_key = version.created_at.strftime("%Y-%m")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        return monthly_counts

    def cleanup_old_versions(self, keep_recent: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones"""
        if len(self.versions) <= keep_recent:
            return 0

        # Sort versions by creation date (newest first)
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )

        # Remove old versions
        removed = 0
        for name, version in sorted_versions[keep_recent:]:
            # Remove from memory
            del self.versions[name]

            # Remove from disk
            version_file = self.versions_dir / f"{name}.json"
            try:
                version_file.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove version file {version_file}: {e}")

        logger.info(f"Cleaned up {removed} old versions")
        return removed


# Global instance
version_manager = FeatureVersionManager()


def create_feature_version(name: str,
                          config: Dict[str, Any],
                          feature_names: List[str],
                          **metadata) -> FeatureVersion:
    """Convenience function to create a feature version"""
    return version_manager.create_version(name, config, feature_names, metadata)


def get_cached_features(data: pd.DataFrame,
                       version_name: str,
                       feature_engineer) -> pd.DataFrame:
    """Convenience function to get features with caching"""
    version = version_manager.get_version(version_name)
    if not version:
        raise ValueError(f"Version not found: {version_name}")

    return version_manager.get_features_with_caching(data, version, feature_engineer)


__all__ = [
    "FeatureVersion",
    "FeatureCache",
    "FeatureVersionManager",
    "version_manager",
    "create_feature_version",
    "get_cached_features"
]