"""
FastAPI REST API para Inferência de Modelos AML
================================================

API de alta performance para scoring de transações em tempo real.

Endpoints:
----------
- POST /predict: Scoring individual
- POST /batch_predict: Scoring em lote
- GET /health: Health check
- GET /metrics: Métricas Prometheus
- GET /model/info: Informações do modelo

Performance Targets:
--------------------
- Latência p50: < 50ms
- Latência p95: < 200ms
- Latência p99: < 500ms
- Throughput: > 1000 req/s

Autor: Time de Data Science
Data: Outubro 2025
Fase: 5 - Real-time & API
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

from .models import (
    Transaction, TransactionBatch, PredictionResponse, BatchPredictionResponse, 
    ModelInfo, HealthResponse, DashboardMetrics, ValidationResult, AuditLog, ConfigItem,
    LoginRequest, TokenResponse, UserInfo
)
from .model_loader import ModelService
from .feature_service import FeatureService
from .cache_service import get_cache_service
from .database_service import get_database_service
from .security_service import SecurityService, get_current_user, get_client_ip

# Novos imports para dashboard API
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="AML Detection API",
    description="API de detecção de lavagem de dinheiro em tempo real",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (ajustar origins conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção: especificar origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrumentação Prometheus
Instrumentator().instrument(app).expose(app)

# Serviços (singleton)
model_service = ModelService()
feature_service = FeatureService()
cache_service = get_cache_service()
db_service = get_database_service()
security_service = get_security_service()


# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Inicialização da API."""
    logger.info("🚀 Iniciando AML Detection API...")
    
    # Carregar modelo do registry
    try:
        model_service.load_model_from_registry(environment="production")
        logger.info(f"✅ Modelo carregado: {model_service.model_version}")
        logger.info(f"   Experimento: {getattr(model_service, 'experiment_id', 'N/A')}")
        logger.info(f"   ROC-AUC: {model_service.training_metrics.get('roc_auc', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise
    
    logger.info("✅ API pronta para receber requisições")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown graceful."""
    logger.info("🛑 Desligando AML Detection API...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "AML Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Verifica:
    - API está respondendo
    - Modelo está carregado
    - Latência de inferência
    """
    start_time = time.time()
    
    # Verificar se modelo está carregado
    model_loaded = model_service.model is not None
    
    # Inferência de teste (se modelo carregado)
    test_latency = None
    if model_loaded:
        try:
            # Criar transação fictícia
            test_tx = pd.DataFrame([{
                'Amount': 1000.0,
                'Hour': 10,
                'Day': 1,
                'From_Bank': 1,
                'To_Bank': 2,
                'Account': 1234,
                'Account_1': 5678
            }])
            
            test_start = time.time()
            _ = model_service.predict(test_tx)
            test_latency = (time.time() - test_start) * 1000  # ms
            
        except Exception as e:
            logger.warning(f"Teste de inferência falhou: {e}")
    
    health_status = "healthy" if model_loaded else "unhealthy"
    
    response_time = (time.time() - start_time) * 1000  # ms
    
    return HealthResponse(
        status=health_status,
        model_loaded=model_loaded,
        model_version=model_service.model_version,
        response_time_ms=response_time,
        test_inference_latency_ms=test_latency
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """
    Retorna informações do modelo em produção.
    """
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    return ModelInfo(
        model_version=model_service.model_version,
        model_type=model_service.model_type,
        loaded_at=model_service.loaded_at,
        feature_count=model_service.n_features,
        model_path=model_service.model_path
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: Transaction):
    """
    Scoring individual de transação.
    
    Parameters
    ----------
    transaction : Transaction
        Dados da transação
        
    Returns
    -------
    prediction : PredictionResponse
        Score de risco, decisão, explicação
    """
    start_time = time.time()
    
    try:
        # 1. Verificar cache primeiro
        transaction_dict = transaction.dict()
        cached_result = cache_service.get_cached_prediction(transaction_dict)
        
        if cached_result:
            logger.info(f"✅ Cache hit para transação {transaction.transaction_id}")
            # Atualizar apenas timestamp e latency
            cached_result["timestamp"] = datetime.now().isoformat()
            cached_result["latency_ms"] = (time.time() - start_time) * 1000
            return PredictionResponse(**cached_result)
        
        # 2. Feature Engineering
        features_df = feature_service.engineer_features(transaction_dict)
        
        # 3. Validação
        if not feature_service.validate_features(features_df):
            raise HTTPException(
                status_code=400,
                detail="Features inválidas ou faltando"
            )
        
        # 4. Predição
        risk_score = model_service.predict(features_df)[0]
        
        # 5. Decisão (threshold configurável)
        threshold = 0.5  # TODO: carregar de config
        decision = "ALERT" if risk_score >= threshold else "APPROVE"
        
        # 6. Explicação (top features)
        top_features = model_service.explain_prediction(features_df)
        
        # 7. Latência
        latency_ms = (time.time() - start_time) * 1000
        
        # 8. Salvar transação processada no banco (para métricas do dashboard)
        transaction_record = {
            "transaction_id": transaction.transaction_id,
            "timestamp": transaction.timestamp.isoformat() if hasattr(transaction.timestamp, 'isoformat') else str(transaction.timestamp),
            "amount": transaction.amount,
            "from_bank": transaction.from_bank,
            "to_bank": transaction.to_bank,
            "account": transaction.account,
            "account_1": transaction.account_1,
            "risk_score": float(risk_score),
            "decision": decision,
            "model_version": model_service.model_version,
            "latency_ms": latency_ms,
            "features": features_df.to_dict('records')[0] if not features_df.empty else {}
        }
        
        # Salvar em background (não bloquear resposta)
        try:
            db_service.insert_transaction(transaction_record)
        except Exception as e:
            logger.warning(f"Erro ao salvar transação no banco: {e}")
        
        # 9. Preparar resultado
        result = {
            "transaction_id": transaction.transaction_id,
            "risk_score": float(risk_score),
            "decision": decision,
            "top_features": top_features,
            "model_version": model_service.model_version,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # 10. Cache do resultado (exceto para transações muito recentes)
        if latency_ms > 10:  # Só cache se levou tempo para processar
            cache_service.cache_prediction(transaction_dict, result)
        
        logger.info(
            f"Prediction: tx_id={transaction.transaction_id}, "
            f"score={risk_score:.4f}, decision={decision}, latency={latency_ms:.1f}ms"
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(batch: TransactionBatch):
    """
    Scoring em lote (batch).
    
    Parameters
    ----------
    batch : TransactionBatch
        Lista de transações
        
    Returns
    -------
    predictions : BatchPredictionResponse
        Predições para cada transação
    """
    start_time = time.time()
    
    try:
        # Converter para DataFrame
        transactions_list = [tx.dict() for tx in batch.transactions]
        
        # Feature Engineering em batch
        features_df = pd.DataFrame([
            feature_service.engineer_features(tx)
            for tx in transactions_list
        ])
        
        # Predições
        risk_scores = model_service.predict(features_df)
        
        # Decisões
        threshold = 0.5
        decisions = ["ALERT" if score >= threshold else "APPROVE" for score in risk_scores]
        
        # Montar respostas
        predictions = []
        for i, tx in enumerate(batch.transactions):
            predictions.append({
                "transaction_id": tx.transaction_id,
                "risk_score": float(risk_scores[i]),
                "decision": decisions[i]
            })
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch prediction: {len(predictions)} transações, "
            f"latency={latency_ms:.1f}ms, "
            f"avg_latency_per_tx={latency_ms/len(predictions):.1f}ms"
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            alert_count=sum(1 for p in predictions if p['decision'] == 'ALERT'),
            model_version=model_service.model_version,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro no batch prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro no batch: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Endpoint para métricas customizadas (além do Prometheus).
    """
    return {
        "model_version": model_service.model_version,
        "total_predictions": model_service.prediction_count,
        "uptime_seconds": time.time() - model_service.startup_time,
        "avg_latency_ms": model_service.get_avg_latency()
    }


@app.get("/cache/status", tags=["Monitoring"])
async def cache_status():
    """
    Retorna status do sistema de cache.
    """
    try:
        cache_stats = cache_service.get_stats()
        
        return {
            "cache_status": "healthy" if cache_stats["redis_available"] else "memory_fallback",
            "redis_available": cache_stats["redis_available"],
            "memory_cache_size": cache_stats["memory_cache_size"],
            "redis_stats": cache_stats.get("redis_stats", {}),
            "cache_hit_ratio": "N/A",  # TODO: implementar métricas de hit ratio
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter status do cache: {e}")
        return {
            "cache_status": "error",
            "error": str(e)
        }


# ============================================================================
# DASHBOARD API ENDPOINTS
# ============================================================================

@app.get("/dashboard/metrics", response_model=DashboardMetrics, tags=["Dashboard"])
async def get_dashboard_metrics():
    """
    Retorna métricas principais do dashboard AML.
    
    Inclui contadores de transações, alertas, distribuição de risco,
    métricas de performance e saúde do sistema.
    """
    try:
        # 1. Verificar cache primeiro
        cached_metrics = cache_service.get_cached_dashboard_metrics()
        if cached_metrics:
            logger.info("✅ Cache hit para métricas do dashboard")
            cached_metrics["last_updated"] = datetime.now().isoformat()
            return DashboardMetrics(**cached_metrics)
        
        # 2. Buscar métricas otimizadas do banco
        db_metrics = db_service.get_dashboard_metrics_optimized()
        
        # Performance metrics (simulado - em produção viria do monitoring)
        performance_metrics = {
            "accuracy": 0.94,
            "precision": 0.89,
            "recall": 0.91,
            "f1_score": 0.90
        }
        
        # System health
        system_health = {
            "api": "healthy",
            "database": "healthy",
            "model": "healthy" if model_service.model else "unhealthy",
            "cache": "healthy" if cache_service.redis_available else "memory_fallback"
        }
        
        # 3. Preparar resultado
        metrics = {
            **db_metrics,
            "performance_metrics": performance_metrics,
            "system_health": system_health,
            "last_updated": datetime.now().isoformat()
        }
        
        # 4. Cache do resultado
        cache_service.cache_dashboard_metrics(metrics)
        
        logger.info(f"📊 Métricas calculadas: {db_metrics['total_transactions']} transações, {db_metrics['alerts_today']} alertas")
        
        return DashboardMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Erro ao buscar métricas do dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/dashboard/validation-results", response_model=List[ValidationResult], tags=["Dashboard"])
async def get_validation_results(limit: int = 50):
    """
    Retorna resultados recentes de validação de dados.
    
    Parameters
    ----------
    limit : int
        Número máximo de resultados (default: 50)
    """
    try:
        # Query otimizada
        query = """
            SELECT validation_id, timestamp, data_type, status, issues_found, details
            FROM validation_results
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        results = db_service.execute_query(query, (limit,))
        
        validation_results = []
        for row in results:
            validation_results.append(ValidationResult(
                validation_id=row[0],
                timestamp=row[1],
                data_type=row[2],
                status=row[3],
                issues_found=row[4],
                details=json.loads(row[5]) if row[5] else None
            ))
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Erro ao buscar resultados de validação: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/dashboard/audit-logs", response_model=List[AuditLog], tags=["Dashboard"])
async def get_audit_logs(
    limit: int = 100,
    user: Optional[str] = None,
    action: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Retorna logs de auditoria do sistema.
    
    Parameters
    ----------
    limit : int
        Número máximo de logs (default: 100)
    user : str, optional
        Filtrar por usuário
    action : str, optional
        Filtrar por ação
    start_date : str, optional
        Data inicial (YYYY-MM-DD)
    end_date : str, optional
        Data final (YYYY-MM-DD)
    """
    try:
        # Construir query dinamicamente
        query = """
            SELECT id, timestamp, user, action, resource, details, ip_address
            FROM audit_log
            WHERE 1=1
        """
        params = []
        
        if user:
            query += " AND user = ?"
            params.append(user)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = db_service.execute_query(query, tuple(params))
        
        logs = []
        for row in results:
            logs.append(AuditLog(
                log_id=row[0],
                timestamp=row[1],
                user=row[2],
                action=row[3],
                resource=row[4],
                details=json.loads(row[5]) if row[5] else None,
                ip_address=row[6]
            ))
        
        return logs
        
    except Exception as e:
        logger.error(f"Erro ao buscar logs de auditoria: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/dashboard/config", response_model=List[ConfigItem], tags=["Dashboard"])
async def get_config():
    """
    Retorna configurações do sistema.
    """
    try:
        query = """
            SELECT key, value, description, last_updated, updated_by
            FROM config
            ORDER BY key
        """
        
        results = db_service.execute_query(query)
        
        configs = []
        for row in results:
            configs.append(ConfigItem(
                key=row[0],
                value=json.loads(row[1]) if row[1] else None,
                description=row[2],
                last_updated=row[3],
                updated_by=row[4]
            ))
        
        return configs
        
    except Exception as e:
        logger.error(f"Erro ao buscar configurações: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.put("/dashboard/config/{key}", tags=["Dashboard"])
async def update_config(key: str, value: Any, updated_by: Optional[str] = None):
    """
    Atualiza uma configuração do sistema.
    
    Parameters
    ----------
    key : str
        Chave da configuração
    value : Any
        Novo valor
    updated_by : str, optional
        Usuário que fez a atualização
    """
    try:
        # Verificar se configuração existe
        check_query = "SELECT COUNT(*) FROM config WHERE key = ?"
        exists = db_service.execute_query(check_query, (key,))[0][0] > 0
        
        if exists:
            # Update
            update_query = """
                UPDATE config 
                SET value = ?, last_updated = ?, updated_by = ?
                WHERE key = ?
            """
            db_service.execute_query(update_query, (json.dumps(value), datetime.now().isoformat(), updated_by, key), fetch=False)
        else:
            # Insert
            insert_query = """
                INSERT INTO config (key, value, description, last_updated, updated_by)
                VALUES (?, ?, ?, ?, ?)
            """
            db_service.execute_query(insert_query, (key, json.dumps(value), f"Config {key}", datetime.now().isoformat(), updated_by), fetch=False)
        
        # Invalidar cache de configurações
        cache_service.clear_prefix("config")
        
        return {"message": f"Configuração '{key}' atualizada com sucesso"}
        
    except Exception as e:
        logger.error(f"Erro ao atualizar configuração: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/dashboard/transactions", tags=["Dashboard"])
async def get_transactions(
    limit: int = 100,
    offset: int = 0,
    risk_threshold: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Retorna transações processadas (com paginação).
    
    Parameters
    ----------
    limit : int
        Número máximo de transações (default: 100)
    offset : int
        Offset para paginação (default: 0)
    risk_threshold : float, optional
        Filtrar por score de risco mínimo
    start_date : str, optional
        Data inicial (YYYY-MM-DD)
    end_date : str, optional
        Data final (YYYY-MM-DD)
    """
    try:
        # Usar método otimizado do database service
        result = db_service.get_transactions_paginated(
            limit=limit,
            offset=offset,
            risk_threshold=risk_threshold,
            start_date=start_date,
            end_date=end_date
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao buscar transações: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: LoginRequest):
    """
    Autenticação de usuário e geração de token JWT.
    
    Parameters
    ----------
    credentials : LoginRequest
        Credenciais do usuário (username e password)
        
    Returns
    -------
    token : TokenResponse
        Token JWT e informações do usuário
    """
    try:
        # Autenticar usuário
        user = security_service.authenticate_user(credentials.username, credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciais inválidas"
            )
        
        # Gerar token JWT
        token_data = {
            "sub": user["id"],
            "username": user["username"],
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=8)  # 8 horas
        }
        
        token = security_service.create_jwt_token(token_data)
        
        # Log de auditoria
        security_service.audit_log(
            user_id=user["id"],
            action="login",
            resource="authentication",
            details=f"Login via API - usuário: {user['username']}",
            ip_address=get_client_ip(request)
        )
        
        logger.info(f"✅ Login API bem-sucedido: {user['username']}")
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=user,
            expires_in=28800  # 8 horas em segundos
        )
        
    except Exception as e:
        logger.error(f"Erro no login API: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor")


@app.post("/auth/logout", tags=["Authentication"])
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Logout do usuário (invalidação do token).
    
    Parameters
    ----------
    current_user : dict
        Usuário autenticado (via token JWT)
    """
    try:
        # Adicionar token à blacklist (se implementado)
        # Por enquanto, apenas log de auditoria
        
        security_service.audit_log(
            user_id=current_user["id"],
            action="logout",
            resource="authentication",
            details=f"Logout via API - usuário: {current_user['username']}",
            ip_address=get_client_ip(request)
        )
        
        logger.info(f"✅ Logout API: {current_user['username']}")
        
        return {"message": "Logout realizado com sucesso"}
        
    except Exception as e:
        logger.error(f"Erro no logout API: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor")


@app.get("/auth/me", response_model=UserInfo, tags=["Authentication"])
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Retorna informações do usuário autenticado.
    
    Parameters
    ----------
    current_user : dict
        Usuário autenticado (via token JWT)
        
    Returns
    -------
    user_info : UserInfo
        Informações completas do usuário
    """
    try:
        # Buscar informações atualizadas do usuário
        user_info = security_service.get_user_by_id(current_user["id"])
        
        if not user_info:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        return UserInfo(**user_info)
        
    except Exception as e:
        logger.error(f"Erro ao buscar informações do usuário: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor")


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Renova o token JWT do usuário.
    
    Parameters
    ----------
    current_user : dict
        Usuário autenticado (via token JWT)
        
    Returns
    -------
    token : TokenResponse
        Novo token JWT
    """
    try:
        # Gerar novo token
        token_data = {
            "sub": current_user["id"],
            "username": current_user["username"],
            "role": current_user["role"],
            "exp": datetime.utcnow() + timedelta(hours=8)
        }
        
        new_token = security_service.create_jwt_token(token_data)
        
        # Log de auditoria
        security_service.audit_log(
            user_id=current_user["id"],
            action="token_refresh",
            resource="authentication",
            details=f"Token renovado - usuário: {current_user['username']}",
            ip_address=get_client_ip(request)
        )
        
        logger.info(f"✅ Token renovado: {current_user['username']}")
        
        return TokenResponse(
            access_token=new_token,
            token_type="bearer",
            user=current_user,
            expires_in=28800
        )
        
    except Exception as e:
        logger.error(f"Erro na renovação de token: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor")


# ============================================================================
# MAIN (para execução local)
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Apenas em desenvolvimento
        log_level="info"
    )
