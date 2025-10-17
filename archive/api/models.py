"""
Pydantic Models para API
=========================

Schemas de request/response validados com Pydantic.

Autor: Time de Data Science
Data: Outubro 2025
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime


class Transaction(BaseModel):
    """Schema de uma transação individual."""
    
    transaction_id: str = Field(..., description="ID único da transação")
    timestamp: datetime = Field(..., description="Timestamp da transação")
    amount: float = Field(..., gt=0, description="Valor da transação (> 0)")
    from_bank: int = Field(..., description="Banco de origem")
    to_bank: int = Field(..., description="Banco de destino")
    account: int = Field(..., description="Conta de origem")
    account_1: int = Field(..., description="Conta de destino")
    receiving_currency: str = Field(default="USD", description="Moeda")
    payment_currency: str = Field(default="USD", description="Moeda de pagamento")
    payment_format: str = Field(default="ACH", description="Formato de pagamento")
    
    # Campos opcionais
    country: Optional[str] = Field(None, description="País")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hora (0-23)")
    day: Optional[int] = Field(None, ge=1, le=31, description="Dia (1-31)")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "timestamp": "2025-10-01T10:30:00",
                "amount": 5000.50,
                "from_bank": 1,
                "to_bank": 2,
                "account": 1001,
                "account_1": 2002,
                "receiving_currency": "USD",
                "payment_currency": "USD",
                "payment_format": "ACH",
                "country": "US",
                "hour": 10,
                "day": 1
            }
        }


class TransactionBatch(BaseModel):
    """Batch de transações."""
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "tx_1",
                        "timestamp": "2025-10-01T10:00:00",
                        "amount": 1000.0,
                        "from_bank": 1,
                        "to_bank": 2,
                        "account": 1001,
                        "account_1": 2001
                    },
                    {
                        "transaction_id": "tx_2",
                        "timestamp": "2025-10-01T10:01:00",
                        "amount": 2500.0,
                        "from_bank": 3,
                        "to_bank": 4,
                        "account": 3001,
                        "account_1": 4001
                    }
                ]
            }
        }


class FeatureImportance(BaseModel):
    """Importância de uma feature."""
    feature: str
    importance: float
    contribution: Optional[float] = None  # SHAP value


class PredictionResponse(BaseModel):
    """Response de predição individual."""
    
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Score de risco (0-1)")
    decision: str = Field(..., description="ALERT ou APPROVE")
    top_features: Optional[List[Dict[str, Any]]] = Field(None, description="Top features importantes")
    explanation: Optional[str] = Field(None, description="Explicação textual")
    model_version: str = Field(..., description="Versão do modelo")
    latency_ms: float = Field(..., description="Latência da predição (ms)")
    timestamp: str = Field(..., description="Timestamp da predição")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "risk_score": 0.85,
                "decision": "ALERT",
                "top_features": [
                    {"feature": "amount", "importance": 0.35},
                    {"feature": "hour", "importance": 0.18}
                ],
                "explanation": "Alto risco devido a: amount elevado (0.35), hora incomum (0.18)",
                "model_version": "v2.1.0",
                "latency_ms": 45.2,
                "timestamp": "2025-10-01T10:30:01"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response de predição em batch."""
    
    predictions: List[Dict[str, Any]]
    total_count: int
    alert_count: int
    model_version: str
    latency_ms: float
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"transaction_id": "tx_1", "risk_score": 0.25, "decision": "APPROVE"},
                    {"transaction_id": "tx_2", "risk_score": 0.92, "decision": "ALERT"}
                ],
                "total_count": 2,
                "alert_count": 1,
                "model_version": "v2.1.0",
                "latency_ms": 120.5,
                "timestamp": "2025-10-01T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="healthy ou unhealthy")
    model_loaded: bool
    model_version: Optional[str] = None
    response_time_ms: float
    test_inference_latency_ms: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "v2.1.0",
                "response_time_ms": 12.3,
                "test_inference_latency_ms": 8.5
            }
        }


class ModelInfo(BaseModel):
    """Informações do modelo."""
    
    model_version: str
    model_type: str
    loaded_at: str
    feature_count: int
    model_path: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "model_version": "v2.1.0",
                "model_type": "LightGBM",
                "loaded_at": "2025-10-01T09:00:00",
                "feature_count": 65,
                "model_path": "models/best_model_tuned.pkl"
            }
        }


# ============================================================================
# DASHBOARD API MODELS
# ============================================================================

class DashboardMetrics(BaseModel):
    """Métricas do dashboard AML."""
    
    total_transactions: int
    alerts_today: int
    risk_distribution: Dict[str, int]
    performance_metrics: Dict[str, float]
    system_health: Dict[str, str]
    last_updated: str
    
    class Config:
        schema_extra = {
            "example": {
                "total_transactions": 15420,
                "alerts_today": 23,
                "risk_distribution": {"low": 12000, "medium": 2500, "high": 920},
                "performance_metrics": {"accuracy": 0.94, "precision": 0.89, "recall": 0.91},
                "system_health": {"api": "healthy", "database": "healthy", "model": "healthy"},
                "last_updated": "2025-10-01T14:30:00"
            }
        }


class ValidationResult(BaseModel):
    """Resultado de validação."""
    
    validation_id: str
    timestamp: str
    data_type: str
    status: str
    issues_found: int
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "validation_id": "val_12345",
                "timestamp": "2025-10-01T10:00:00",
                "data_type": "transactions",
                "status": "passed",
                "issues_found": 0,
                "details": {"checks": ["schema", "ranges", "consistency"]}
            }
        }


class AuditLog(BaseModel):
    """Log de auditoria."""
    
    log_id: int
    timestamp: str
    user: str
    action: str
    resource: str
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "log_id": 12345,
                "timestamp": "2025-10-01T14:30:00",
                "user": "analyst@example.com",
                "action": "view_dashboard",
                "resource": "aml_dashboard",
                "details": {"page": "overview", "filters": {"date_range": "7d"}},
                "ip_address": "192.168.1.100"
            }
        }


class ConfigItem(BaseModel):
    """Item de configuração."""
    
    key: str
    value: Any
    description: Optional[str] = None
    last_updated: str
    updated_by: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "key": "alert_threshold",
                "value": 0.8,
                "description": "Threshold for triggering AML alerts",
                "last_updated": "2025-10-01T09:00:00",
                "updated_by": "admin@example.com"
            }
        }


# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class LoginRequest(BaseModel):
    """Request de login."""
    
    username: str = Field(..., min_length=1, max_length=100, description="Nome de usuário")
    password: str = Field(..., min_length=1, max_length=255, description="Senha")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "analyst@example.com",
                "password": "secure_password_123"
            }
        }


class TokenResponse(BaseModel):
    """Response com token JWT."""
    
    access_token: str = Field(..., description="Token JWT de acesso")
    token_type: str = Field(default="bearer", description="Tipo do token")
    user: Dict[str, Any] = Field(..., description="Informações do usuário")
    expires_in: int = Field(..., description="Tempo de expiração em segundos")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "user": {
                    "id": 1,
                    "username": "analyst@example.com",
                    "role": "analyst",
                    "full_name": "John Analyst"
                },
                "expires_in": 28800
            }
        }


class UserInfo(BaseModel):
    """Informações completas do usuário."""
    
    id: int
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None
    permissions: List[str] = []
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "username": "analyst@example.com",
                "email": "analyst@example.com",
                "full_name": "John Analyst",
                "role": "analyst",
                "is_active": True,
                "created_at": "2025-01-01T00:00:00",
                "last_login": "2025-10-01T09:00:00",
                "permissions": ["view_dashboard", "view_transactions", "create_alerts"]
            }
        }
