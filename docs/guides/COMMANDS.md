# 🛠️ Common Commands Cheat Sheet

Quick reference for frequently used commands.

---

## 🐍 Python Environment

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Upgrade pip
python -m pip install --upgrade pip
```

### Conda (if using)
```bash
# Create environment
conda create -n aml python=3.10

# Activate
conda activate aml

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Specific test file
python -m pytest tests/test_api.py -v

# Specific test
python -m pytest tests/test_api.py::test_health_endpoint -v

# Stop on first failure
python -m pytest tests/ -x

# Show print statements
python -m pytest tests/ -s

# Run only failed tests
python -m pytest tests/ --lf
```

### Using Makefile
```bash
# Run tests
make test

# Run with coverage
make test-cov
```

---

## 🎮 CLI Commands

### Training
```bash
# Train with default (xgboost)
python -m entrypoints.cli train

# Train specific model
python -m entrypoints.cli train --model-name lightgbm
python -m entrypoints.cli train --model-name random_forest

# With custom output
python -m entrypoints.cli train --output models/my_model.pkl
```

### Prediction
```bash
# Predict from file
python -m entrypoints.cli predict --input data/test.csv --output predictions.csv

# Show help
python -m entrypoints.cli predict --help
```

### Evaluation
```bash
# Evaluate model
python -m entrypoints.cli evaluate

# With specific model
python -m entrypoints.cli evaluate --model-path models/my_model.pkl
```

### Serve API
```bash
# Start API server
python -m entrypoints.cli serve

# Custom port
python -m entrypoints.cli serve --port 9000
```

---

## 🌐 API Commands

### Start Server
```bash
# Production
python -m entrypoints.api

# Development (auto-reload)
uvicorn entrypoints.api:app --reload

# Custom host/port
uvicorn entrypoints.api:app --host 0.0.0.0 --port 9000

# Multiple workers
uvicorn entrypoints.api:app --workers 4
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Root
curl http://localhost:8000/

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "payment_format": 2, "hour": 14, "day_of_week": 3, "transaction_count": 5}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"amount": 1500, "payment_format": 2, "hour": 14}, {"amount": 5000, "payment_format": 3, "hour": 22}]}'

# API documentation (open in browser)
# http://localhost:8000/docs
```

### With PowerShell
```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get

# Prediction
$body = @{
    amount = 1500
    payment_format = 2
    hour = 14
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```

---

## 🐳 Docker Commands

### Build
```bash
# Build image
docker build -t aml-project:latest .

# Build with no cache
docker build --no-cache -t aml-project:latest .

# Check image size
docker images aml-project:latest
```

### Run
```bash
# Run container
docker run -d -p 8000:8000 --name aml_api aml-project:latest

# Run with volume mounts
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name aml_api aml-project:latest

# Run interactively
docker run -it --rm aml-project:latest /bin/bash
```

### Manage
```bash
# List containers
docker ps
docker ps -a

# Stop container
docker stop aml_api

# Remove container
docker rm aml_api

# View logs
docker logs aml_api
docker logs -f aml_api  # follow

# Execute command in container
docker exec -it aml_api python -m entrypoints.cli --help
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
docker-compose logs -f api

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Scale service
docker-compose up -d --scale api=3
```

---

## 📦 Batch Processing

```bash
# Run batch processing
python -m entrypoints.batch --input data/transactions.csv --output data/predictions

# With specific date
python -m entrypoints.batch --input data/transactions.csv --date 2024-01-15

# Show help
python -m entrypoints.batch --help
```

---

## 🔍 Development

### Code Quality
```bash
# Format code (if black installed)
black src/ entrypoints/ tests/

# Lint code (if flake8 installed)
flake8 src/ entrypoints/ --max-line-length=120

# Type checking (if mypy installed)
mypy src/ entrypoints/
```

### Debugging
```bash
# Run with debugger
python -m pdb -m entrypoints.cli train

# Interactive Python
python -i -c "from src.config import config; print(config.get('model.target_column'))"

# Check imports
python -c "from src.data.loader import load_raw_data; print('OK')"
```

---

## 📊 Monitoring

### View Logs
```bash
# Tail logs
tail -f logs/aml_pipeline.log

# Search logs
grep "ERROR" logs/aml_pipeline.log

# Count errors
grep -c "ERROR" logs/aml_pipeline.log
```

### Metrics
```python
# In Python
from src.monitoring.service import AMLMonitor

monitor = AMLMonitor()
report = monitor.get_health_report()
print(report)
```

---

## 🗂️ File Operations

### Data
```bash
# Check data file
ls -lh data/processed/features_with_patterns.parquet

# Count records (requires pandas)
python -c "import pandas as pd; df = pd.read_parquet('data/processed/features_with_patterns.parquet'); print(len(df))"
```

### Models
```bash
# List models
ls -lh models/

# Check model file
file models/xgboost_model.pkl
```

### Artifacts
```bash
# List artifacts
ls -lh artifacts/

# View JSON artifact
cat artifacts/final_evaluation_results.json | python -m json.tool
```

---

## 🧹 Cleanup

```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clean pytest cache
rm -rf .pytest_cache/

# Clean coverage
rm -rf htmlcov/
rm .coverage

# Clean logs (be careful!)
rm logs/*.log

# Using Makefile
make clean
```

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000           # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>           # Linux/Mac
taskkill /F /PID <PID>  # Windows
```

### Permission Denied
```bash
# Fix file permissions
chmod +x entrypoints/*.py
chmod 644 models/*.pkl
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"              # PowerShell
```

---

## 📝 Git Commands

```bash
# Status
git status

# Add files
git add .

# Commit
git commit -m "feat: implementation complete"

# Push
git push origin main

# Create branch
git checkout -b feature/new-feature

# View changes
git diff
git diff --cached
```

---

## 🎯 Quick Validation

```bash
# Full validation pipeline
python -m pytest tests/ -v && \
python -m entrypoints.cli train --model-name random_forest && \
python -m entrypoints.cli evaluate && \
echo "✅ All validations passed!"
```

---

## 📖 Documentation

```bash
# View README
cat README.md

# Open API docs (start server first)
# Then open http://localhost:8000/docs in browser

# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html
```

---

## 💡 Tips

- Use `--help` on any command to see options
- Check logs/ directory for debugging
- Use Docker for consistent environments
- Run tests before committing
- Keep virtual environment active

---

**Last Updated:** November 6, 2025
