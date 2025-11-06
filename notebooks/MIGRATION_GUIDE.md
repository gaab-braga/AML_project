# Migration Guide: Notebooks → Refactored Code

Guia de migração dos notebooks para o código refatorado.

## Overview

| Notebook | Módulo Refatorado | Função Principal |
|----------|-------------------|------------------|
| 01_Data_Ingestion_EDA.ipynb | `src.data.loader` | `load_raw_data()` |
| 01_Data_Ingestion_EDA.ipynb | `src.data.preprocessing` | `clean_data()` |
| 03_Model_Selection_Tuning.ipynb | `src.data.preprocessing` | `temporal_split()` |
| 03_Model_Selection_Tuning.ipynb | `src.features.engineering` | `build_features()` |
| 03_Model_Selection_Tuning.ipynb | `src.models.train` | `train_model()` |
| 04_Ensemble_Modeling.ipynb | `src.models.train` | Model registry |
| 05_Model_Interpretation.ipynb | `src.models.evaluate` | `evaluate_model()` |

## Mapping: Old → New

### Data Loading

**Before (Notebook):**
```python
df_patterns = pd.read_parquet('data/processed/features_with_patterns.parquet')
```

**After (Refactored):**
```python
from src.data.loader import load_raw_data
df = load_raw_data()
```

### Data Cleaning

**Before (Notebook):**
```python
df_clean = df_patterns.drop_duplicates()
df_clean = df_clean.fillna(0)
df_clean = df_clean.replace([np.inf, -np.inf], 0)
```

**After (Refactored):**
```python
from src.data.preprocessing import clean_data
df_clean = clean_data(df)
```

### Temporal Split

**Before (Notebook):**
```python
df_sorted = df_patterns.sort_values('timestamp').reset_index(drop=True)

# Convert datetime to numeric
for col in df_sorted.select_dtypes(include=['datetime64']).columns:
    df_sorted[col] = df_sorted[col].astype('int64') // 10**9

# Select numeric features
numeric_cols = [col for col in feature_cols if df_sorted[col].dtype in ['int64', 'float64']]
X = df_sorted[numeric_cols]
y = df_sorted['is_laundering']

# Split 80/20
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

**After (Refactored):**
```python
from src.data.preprocessing import temporal_split
X_train, X_test, y_train, y_test = temporal_split(df_clean, 'is_laundering')
```

### Model Training

**Before (Notebook):**
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42
)
model.fit(X_train, y_train)
```

**After (Refactored):**
```python
from src.models.train import train_model
model = train_model(X_train, y_train, model_name='xgboost')
```

### Prediction

**Before (Notebook):**
```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

**After (Refactored):**
```python
from src.models.predict import predict
y_pred = predict(X_test, model, return_proba=False)
y_proba = predict(X_test, model, return_proba=True)
```

### Evaluation

**Before (Notebook):**
```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
```

**After (Refactored):**
```python
from src.models.evaluate import evaluate_model, print_evaluation_summary
metrics = evaluate_model(y_test, y_pred, y_proba)
print_evaluation_summary(metrics)
```

## Benefits

1. **Menos código duplicado**: DRY principle
2. **Testável**: Pytest fixtures
3. **Configurável**: YAML configs
4. **CLI & API**: Pronto para produção
5. **Logging**: Rastreamento centralizado

## Next Steps

1. Substitua células do notebook por imports do `src/`
2. Use `entrypoints/cli.py` para treinar: `python -m entrypoints.cli train`
3. Use `entrypoints/api.py` para servir: `python -m entrypoints.api`
4. Execute testes: `pytest tests/ -v`
