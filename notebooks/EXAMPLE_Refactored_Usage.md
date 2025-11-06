# Exemplo: Usando o Pipeline Refatorado

Este notebook demonstra como usar os módulos refatorados.

## Imports

```python
from src.config import config
from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data, temporal_split
from src.features.engineering import build_features
from src.models.train import train_model, save_model
from src.models.predict import predict, predict_batch
from src.models.evaluate import evaluate_model, print_evaluation_summary
from src.monitoring.service import AMLMonitor
```

## 1. Load Data

```python
# Carrega dados processados
df = load_raw_data()
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

## 2. Preprocess

```python
# Limpa dados
df_clean = clean_data(df)
print(f"Cleaned shape: {df_clean.shape}")
```

## 3. Features

```python
# Prepara features
df_features = build_features(df_clean)
print(f"Features shape: {df_features.shape}")
```

## 4. Split

```python
# Split temporal (80/20)
target_col = config.get('model.target_column')
X_train, X_test, y_train, y_test = temporal_split(df_features, target_col)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")
```

## 5. Train

```python
# Treina modelo
model = train_model(X_train, y_train, model_name='xgboost')

# Salva modelo
save_model(model, "models/example_model.pkl")
```

## 6. Predict

```python
# Predições
y_pred = model.predict(X_test)
y_proba = predict(X_test, model, return_proba=True)

print(f"Predictions shape: {y_pred.shape}")
print(f"Suspicious transactions: {y_pred.sum()} ({y_pred.mean()*100:.2f}%)")
```

## 7. Evaluate

```python
# Avalia modelo
metrics = evaluate_model(y_test, y_pred, y_proba)
print_evaluation_summary(metrics)
```

## 8. Monitoring

```python
# Monitoramento
monitor = AMLMonitor()
metrics = monitor.collect_metrics(y_test, y_proba[:, 1], X_test, latency_ms=50)
alerts = monitor.check_alerts(metrics)

print(f"Alerts: {len(alerts)}")
for alert in alerts:
    print(f"  - {alert['metric']}: {alert['current_value']:.4f}")
```

## Conclusão

O código refatorado:
- **Modular**: Cada função tem responsabilidade única
- **Testável**: Fixtures pytest para validação
- **Limpo**: Sem over-engineering, objetivo
- **Notebook-compatible**: Mantém workflow dos notebooks originais
