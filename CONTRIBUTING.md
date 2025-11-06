# Contributing to AML Detection System

Thank you for your interest in contributing!

## Quick Start for Contributors

1. **Fork** the repository
2. **Clone** your fork
3. **Create** a feature branch: `git checkout -b feature/your-feature`
4. **Make** your changes
5. **Test** your changes: `pytest tests/ -v`
6. **Commit** with clear messages: `git commit -m "feat: add feature"`
7. **Push** to your fork: `git push origin feature/your-feature`
8. **Open** a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/gaab-braga/AML_project.git
cd AML_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Keep functions small and focused
- No emojis in code (only in documentation)

### Clean Code Principles
- **KISS**: Keep it simple
- **DRY**: Don't repeat yourself
- **Single Responsibility**: One function, one purpose
- **Clear Naming**: Descriptive variable/function names

### Example
```python
# Good
def calculate_fraud_score(transaction: dict) -> float:
    """Calculate fraud risk score for a transaction."""
    amount = transaction.get('amount', 0)
    return min(amount / 10000, 1.0)

# Bad
def calc(t):
    a = t.get('amount', 0)
    return min(a/10000,1.0)
```

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Use pytest fixtures from `tests/conftest.py`
- Test naming: `test_<feature>_<scenario>()`

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_api.py::test_predict_endpoint -v
```

## Commit Messages

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat: add batch prediction endpoint
fix: handle missing values in preprocessing
docs: update deployment guide
test: add integration tests for API
```

## Project Structure

```
entrypoints/  - Application interfaces (CLI, API, Batch)
src/          - Core business logic
  ├── data/      - Data loading & preprocessing
  ├── features/  - Feature engineering
  ├── models/    - Model training & inference
  └── monitoring/- Production monitoring
tests/        - Test suite
docs/         - Documentation
```

## Code Review Process

Pull requests will be reviewed for:
1. **Functionality**: Does it work as intended?
2. **Tests**: Are there adequate tests?
3. **Code Quality**: Is it clean and maintainable?
4. **Documentation**: Is it well documented?
5. **Breaking Changes**: Does it break existing functionality?

## Reporting Issues

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant logs or error messages

## Feature Requests

When suggesting features:
- Explain the use case
- Describe expected behavior
- Consider impact on existing functionality
- Provide examples if possible

## Documentation

- Update relevant docs when changing functionality
- Add docstrings to new functions/classes
- Update [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md) if adding new docs
- Keep README.md up to date

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Questions?

Open an issue or contact the maintainers.

---

**Happy Contributing!**
