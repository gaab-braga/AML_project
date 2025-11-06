# Tests

Created test suite following notebook patterns.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_preprocessing.py -v
```

## Test Structure

- `conftest.py`: Shared fixtures (sample data, config)
- `test_data_preprocessing.py`: Data cleaning and temporal split validation
- `test_models_train.py`: Model training and serialization

## Key Test Coverage

1. **Data Preprocessing**
   - Duplicate removal
   - Missing value handling
   - Infinite value handling
   - Temporal split chronological order
   - No data leakage (train before test)
   
2. **Model Training**
   - Model instantiation (XGBoost, LightGBM, RandomForest)
   - Training pipeline
   - Model serialization and deserialization

## Future Tests

- Feature engineering validation
- Prediction consistency
- API endpoint testing
- Monitoring service testing
