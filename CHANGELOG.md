# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [1.0.0] - 2025-11-06

### Added
- Complete refactoring from notebooks to production code
- Clean architecture with `entrypoints/` and `src/` separation
- FastAPI REST API with endpoints: `/predict`, `/predict/batch`, `/health`
- CLI interface with Typer: `train`, `predict`, `evaluate`, `serve`
- Batch processing script for scheduled jobs
- Comprehensive test suite (37 tests) with pytest
- Docker containerization with `Dockerfile` and `docker-compose.yml`
- CI/CD pipelines with GitHub Actions
- Production monitoring service with metrics and alerting
- Complete documentation in `docs/` directory
- Configuration management via YAML

### Changed
- Migrated from notebook-based workflow to modular Python packages
- Replaced Flask API with FastAPI for better performance
- Consolidated scripts into `src/` modules and CLI commands
- Organized documentation into `docs/` structure

### Removed
- Legacy `api/` directory (moved to `_legacy/`)
- Legacy `scripts/` directory (moved to `_legacy/`)
- Duplicate code and configurations
- Unused cache and temporary files

### Implementation Details
- Temporal split (80/20) for time-series validation
- Model registry supporting XGBoost, LightGBM, Random Forest
- Feature engineering simplified for pre-aggregated data
- Centralized logging and configuration
- Health checks and monitoring endpoints

### Documentation
- Quick Start Guide
- Deployment Guide
- Commands Cheat Sheet
- Implementation Status Report
- Migration Guide from notebooks
- Full API documentation (auto-generated)

---

## Version History

- **1.0.0** (2025-11-06): Initial production-ready release
- **0.1.0** (2024-XX-XX): Notebook-based prototype

[Unreleased]: https://github.com/gaab-braga/AML_project/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gaab-braga/AML_project/releases/tag/v1.0.0
