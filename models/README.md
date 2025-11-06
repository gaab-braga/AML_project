# Models Directory

This directory stores trained ML models for the AML detection system.

## Model Files (Not in Git)

Model files (`.pkl`, `.joblib`, `.pt`) are excluded from version control due to their size. They are stored in `.gitignore`.

## Training a Model

To train and save a model:

```python
from src.models.train import train_model
from src.data.load import load_data

# Load your data
X_train, y_train = load_data()

# Train the model
model = train_model(X_train, y_train)

# Model is automatically saved to models/model.pkl
```

## Using in Docker

To include a model in Docker:

1. Train your model locally
2. Ensure `models/model.pkl` exists
3. Build Docker image: `docker build -t aml-project .`
4. The model will be copied into the container

## Model Files Expected

- `model.pkl` - Main production model
- `ssl_encoder_best.pt` - SSL encoder (PyTorch)
- `metadata.yaml` - Model metadata

## API Behavior

The API will start even if no model is present, but prediction endpoints will return HTTP 503 errors until a model is loaded.
