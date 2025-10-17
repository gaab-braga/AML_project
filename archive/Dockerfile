# ==============================================================================
# Multi-stage Dockerfile para API de Scoring
# ==============================================================================
# 
# Build otimizado com camadas mínimas e imagem slim.
# Autor: Time de Data Science
# Data: Outubro 2025
#

# ==============================================================================
# Stage 1: Base com dependências do sistema
# ==============================================================================
FROM python:3.10-slim as base

# Metadados
LABEL maintainer="datascience@empresa.com"
LABEL version="1.0.0"
LABEL description="API de Scoring AML com FastAPI"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Build - Instalar dependências Python
# ==============================================================================
FROM base as builder

# Diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Instalar dependências adicionais para API
RUN pip install --user --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    prometheus-client==0.19.0

# ==============================================================================
# Stage 3: Production - Imagem final slim
# ==============================================================================
FROM python:3.10-slim as production

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    MODEL_PATH=/app/models/best_model_tuned.pkl \
    LOG_LEVEL=info \
    WORKERS=1 \
    PORT=8000

# Instalar apenas runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /app /app/models /app/data /app/logs && \
    chown -R apiuser:apiuser /app

# Copiar dependências Python do builder
COPY --from=builder --chown=apiuser:apiuser /root/.local /root/.local

# Diretório de trabalho
WORKDIR /app

# Copiar código da aplicação
COPY --chown=apiuser:apiuser api/ ./api/
COPY --chown=apiuser:apiuser utils/ ./utils/
COPY --chown=apiuser:apiuser config.yaml ./

# Copiar modelo e dados (se existirem)
COPY --chown=apiuser:apiuser models/ ./models/
COPY --chown=apiuser:apiuser data/label_encoders.pkl ./data/
COPY --chown=apiuser:apiuser data/freq_mappings.pkl ./data/

# Criar arquivo __init__.py vazio
RUN touch api/__init__.py

# Mudar para usuário não-root
USER apiuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]

# ==============================================================================
# Build & Run Instructions
# ==============================================================================
#
# Build:
#   docker build -t aml-api:latest .
#
# Run (development):
#   docker run -p 8000:8000 aml-api:latest
#
# Run (production com variáveis de ambiente):
#   docker run -p 8000:8000 \
#     -e LOG_LEVEL=warning \
#     -e WORKERS=4 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/logs:/app/logs \
#     aml-api:latest
#
# Run com docker-compose:
#   docker-compose up -d
#
# ==============================================================================
