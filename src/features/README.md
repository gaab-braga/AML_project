# src/features/ - Feature Engineering Module

## Overview

The `src/features/` module provides production-ready feature engineering utilities with anti-leakage validation and temporal feature creation. This module was refactored from exploratory notebook functions to ensure code quality, reusability, and maintainability.

## Architecture

```
src/features/
├── __init__.py          # Module exports and imports
├── base.py              # Base classes for feature engineering
├── temporal.py          # Temporal feature engineering (safe)
├── leakage_detector.py  # Data leakage detection and removal
└── statistical.py       # Statistical feature engineering
```

## Key Features

- ✅ **Type Hints**: Complete type annotations for better IDE support
- ✅ **Input Validation**: Robust parameter validation with descriptive errors
- ✅ **Error Handling**: Comprehensive exception handling with logging
- ✅ **Documentation**: Detailed docstrings with examples
- ✅ **Anti-Leakage**: Built-in data leakage detection and prevention
- ✅ **Temporal Safety**: Time-aware feature engineering without future leakage
- ✅ **Testing**: Comprehensive unit test coverage

## Quick Start

```python
from src.features import (
    remove_leaky_features,
    detect_data_leakage_features,
    create_temporal_features_safe,
    get_temporal_cv_splits,
    evaluate_temporal_cv
)

# Detect data leakage
leakage_results = detect_data_leakage_features(X_train, X_test, y_train)

# Remove leaky features
X_clean, removed = remove_leaky_features(X_train, leakage_results['leakage_candidates'])

# Create safe temporal features
X_temporal = create_temporal_features_safe(X_clean, time_col='timestamp')

# Get temporal CV splits
cv_splits = get_temporal_cv_splits(X_temporal, y_train, n_splits=5)

# Evaluate with temporal CV
results = evaluate_temporal_cv(model, X_temporal, y_train, cv_splits)
```

## API Reference

### Data Leakage Detection

#### `detect_data_leakage_features(X_train, X_test, y_train, logger=None)`

Detects potential data leakage by analyzing feature distributions between train and test sets.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `X_test` (pd.DataFrame): Test features
- `y_train` (pd.Series): Training target
- `logger` (logging.Logger, optional): Logger instance

**Returns:**
- `Dict`: Analysis results with suspicious features and risk levels

**Example:**
```python
results = detect_data_leakage_features(X_train, X_test, y_train)
print(f"Suspicious features: {results['suspicious_features_count']}")
```

#### `remove_leaky_features(X, leakage_candidates, logger=None)`

Removes features identified as potentially leaky.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `leakage_candidates` (List[Dict]): Leakage analysis results
- `logger` (logging.Logger, optional): Logger instance

**Returns:**
- `Tuple[pd.DataFrame, List[str]]`: Cleaned features and removed feature names

**Example:**
```python
X_clean, removed = remove_leaky_features(X_train, leakage_candidates)
print(f"Removed {len(removed)} features")
```

### Temporal Feature Engineering

#### `create_temporal_features_safe(df, time_col='Timestamp', group_cols=None, windows=None, logger=None)`

Creates temporal aggregation features without data leakage by using only historical data.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with temporal column
- `time_col` (str): Name of timestamp column (default: 'Timestamp')
- `group_cols` (List[str], optional): Columns to group by (default: ['Account', 'From Bank', 'To Bank'])
- `windows` (List[int], optional): Time windows in days (default: [7, 30, 90])
- `logger` (logging.Logger, optional): Logger instance

**Returns:**
- `pd.DataFrame`: DataFrame with temporal features added

**Features Created:**
- Rolling sums, means, counts, and standard deviations
- Grouped by specified columns over time windows
- NaN values filled with 0 for periods with insufficient history

**Example:**
```python
X_temporal = create_temporal_features_safe(
    df=X_train,
    time_col='timestamp',
    group_cols=['account', 'bank'],
    windows=[7, 30, 90]
)
# Creates features like: account_amount_sum_7d, bank_amount_mean_30d, etc.
```

#### `create_temporal_aggregations_safe(df, time_col='Timestamp', group_cols=None, windows=None, logger=None)`

Alias for `create_temporal_features_safe` for backward compatibility.

### Temporal Cross-Validation

#### `get_temporal_cv_splits(X, y, n_splits=5, test_size=30, gap=0, logger=None)`

Creates temporal cross-validation splits that respect time ordering.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix (sorted by time)
- `y` (pd.Series): Target variable
- `n_splits` (int): Number of CV splits (default: 5)
- `test_size` (int): Size of test set in each fold (default: 30)
- `gap` (int): Samples to skip between train/test (default: 0)
- `logger` (logging.Logger, optional): Logger instance

**Returns:**
- `List[Tuple[np.ndarray, np.ndarray]]`: List of (train_idx, test_idx) tuples

**Example:**
```python
cv_splits = get_temporal_cv_splits(X_train, y_train, n_splits=5, test_size=20)
for i, (train_idx, test_idx) in enumerate(cv_splits):
    print(f"Fold {i+1}: {len(train_idx)} train, {len(test_idx)} test")
```

#### `evaluate_temporal_cv(model, X, y, cv_splits, metrics=None, logger=None)`

Evaluates model using temporal cross-validation splits.

**Parameters:**
- `model`: Scikit-learn compatible model
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `cv_splits` (List[Tuple]): CV splits from `get_temporal_cv_splits`
- `metrics` (List[str], optional): Metrics to compute (default: ['recall', 'precision', 'f1', 'accuracy'])
- `logger` (logging.Logger, optional): Logger instance

**Returns:**
- `Dict`: Comprehensive evaluation results with fold details and summaries

**Example:**
```python
results = evaluate_temporal_cv(model, X_train, y_train, cv_splits)
print(f"Mean F1: {results['f1_summary']['test_mean']:.3f}")
print(f"Overfitting: {results['f1_summary']['overfitting_mean']:.3f}")
```

## Migration Guide

### From Notebook Functions to Module

**Before (Notebook):**
```python
# Inline function definitions
def detect_data_leakage_features(X_train, X_test, y_train):
    # 50+ lines of code...
    pass

# Direct function calls
results = detect_data_leakage_features(X_train, X_test, y_train)
```

**After (Module):**
```python
# Import from module
from src.features import detect_data_leakage_features

# Same function calls - no workflow changes
results = detect_data_leakage_features(X_train, X_test, y_train)
```

### Key Benefits of Migration

1. **Code Quality**: Type hints, validation, error handling
2. **Reusability**: Functions available across notebooks
3. **Maintainability**: Centralized, tested code
4. **Documentation**: Comprehensive API docs
5. **Testing**: Unit test coverage
6. **Performance**: Optimized implementations

## Error Handling

All functions include comprehensive error handling:

```python
from src.features import create_temporal_features_safe

try:
    X_temporal = create_temporal_features_safe(df, time_col='timestamp')
except ValueError as e:
    print(f"Validation error: {e}")
except TypeError as e:
    print(f"Type error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Logging

Functions accept optional logger instances for detailed logging:

```python
import logging

logger = logging.getLogger(__name__)
results = detect_data_leakage_features(X_train, X_test, y_train, logger=logger)
```

## Testing

Run the test suite:

```bash
# From project root
python -m pytest tests/test_features_*.py -v

# Or specific test files
python -m pytest tests/test_features_leakage.py tests/test_features_temporal.py -v
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

## Contributing

1. Follow existing code patterns and type hints
2. Add comprehensive docstrings with examples
3. Include unit tests for new functionality
4. Update this documentation

## Version History

- **v1.0.0**: Initial refactoring from notebook functions
- **v1.1.0**: Added temporal cross-validation functions
- **v1.2.0**: Enhanced error handling and logging
- **v1.3.0**: Added comprehensive unit tests