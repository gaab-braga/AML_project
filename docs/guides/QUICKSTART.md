# Quick Start Guide

Get up and running in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/gaab-braga/AML_project.git
cd AML_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Train Your First Model

```bash
# Using CLI
python -m entrypoints.cli train --model-name xgboost

# Expected output:
# INFO - Loading data from data/processed/features_with_patterns.parquet
# INFO - Loaded data: (123456, 45)
# INFO - Training xgboost model
# INFO - Model trained successfully
# INFO - Model saved to models/xgboost_model.pkl
```

## Make Predictions

### Option 1: CLI

```bash
python -m entrypoints.cli predict --input data/test.csv --output predictions.csv
```

### Option 2: API

```bash
# Start API server
python -m entrypoints.api

# In another terminal, make a request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500,
    "payment_format": 2,
    "hour": 14,
    "day_of_week": 3,
    "transaction_count": 5
  }'
```

### Option 3: Python Script

```python
from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data
from src.features.engineering import build_features
from src.models.train import load_model
from src.models.predict import predict

# Load and prepare data
df = load_raw_data()
df_clean = clean_data(df)
df_features = build_features(df_clean)

# Load model
model = load_model("models/xgboost_model.pkl")

# Predict
predictions = predict(df_features, model, return_proba=True)
print(f"Suspicious transactions: {(predictions > 0.5).sum()}")
```

## Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_api.py::test_health_endpoint -v
```

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# Test
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Next Steps

- Read [Migration Guide](notebooks/MIGRATION_GUIDE.md) to understand code structure
- Check [Deployment Guide](DEPLOYMENT.md) for production deployment
- Explore [API Documentation](http://localhost:8000/docs) when API is running
- Review [ROADMAP](ROADMAP_REFACTORING.md) for implementation details

## Troubleshooting

### "Module not found"
```bash
# Ensure you're in project root
cd AML_project

# Reinstall dependencies
pip install -r requirements.txt
```

### "Model file not found"
```bash
# Train a model first
python -m entrypoints.cli train --model-name xgboost
```

### "Data file not found"
```bash
# Ensure data exists
ls data/processed/features_with_patterns.parquet

# If missing, check notebooks/01_Data_Ingestion_EDA.ipynb
```
