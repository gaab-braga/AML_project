# Deployment Guide

Guide for deploying the AML Detection System.

## Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ (for local development)
- Model artifacts in `models/` directory
- Data in `data/processed/`

## Local Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health

# Stop services
docker-compose down
```

### Option 2: Local Python

```bash
# Activate environment
conda activate aml

# Start API
python -m entrypoints.api

# Or use CLI
python -m entrypoints.cli train --model-name xgboost
```

## Testing Deployment

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific module
pytest tests/test_api.py -v

# Test Docker container
docker run -d -p 8000:8000 aml-project:latest
curl http://localhost:8000/health
```

## API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "payment_format": 2, "hour": 14}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"amount": 1500, "payment_format": 2}]}'
```

## Production Considerations

### Environment Variables

```bash
# Set in docker-compose.yml or .env
LOG_LEVEL=INFO
MODEL_PATH=/app/models/model.pkl
DATA_PATH=/app/data/processed
```

### Monitoring

```python
# Integrate with monitoring service
from src.monitoring.service import AMLMonitor

monitor = AMLMonitor()
metrics = monitor.collect_metrics(y_true, y_proba, features, latency_ms)
alerts = monitor.check_alerts(metrics)
```

### Security

1. **API Authentication** (TODO):
   - Add JWT token authentication
   - Use API keys for batch endpoints

2. **HTTPS**:
   - Use reverse proxy (nginx) with SSL certificates
   - Configure in production environment

3. **Rate Limiting**:
   - Add rate limiting middleware
   - Configure per-endpoint limits

### Scaling

```yaml
# docker-compose.yml with scaling
services:
  api:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
```

## CI/CD Pipeline

GitHub Actions workflows configured:
- `.github/workflows/test.yml` - Run tests on push
- `.github/workflows/docker.yml` - Build Docker image

## Troubleshooting

### API not responding
```bash
# Check logs
docker-compose logs api

# Check container status
docker ps

# Restart
docker-compose restart api
```

### Model not loading
```bash
# Ensure model file exists
ls -lh models/

# Check permissions
chmod 644 models/model.pkl
```

### Memory issues
```bash
# Increase Docker memory limit
# Edit Docker Desktop settings or docker-compose.yml
```
