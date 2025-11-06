"""
FastAPI production API for AML detection.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

from src.models.predict import predict, load_model
from src.features.engineering import build_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="AML Detection API",
    description="Anti-Money Laundering detection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        logger.info("Loading model")
        model = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.warning("API will start without model (predictions will fail)")
        model = None


class Transaction(BaseModel):
    """Transaction schema."""
    account_id: str
    amount: float = Field(..., gt=0)
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "ACC123456",
                "amount": 15000.50,
                "timestamp": "2025-11-06T14:30:00"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    transaction_id: str
    is_suspicious: bool
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str


@app.get("/")
async def root():
    """Health check."""
    return {"status": "online", "service": "AML Detection API"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """Predict single transaction."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Cannot make predictions."
        )
    
    df = pd.DataFrame([transaction.dict()])
    df_features = build_features(df)
    
    risk_score = float(predict(df_features, model, return_proba=True)[0])
    
    if risk_score < 0.3:
        risk_level = "LOW"
    elif risk_score < 0.6:
        risk_level = "MEDIUM"
    elif risk_score < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    return PredictionResponse(
        transaction_id=transaction.account_id,
        is_suspicious=risk_score > 0.5,
        risk_score=risk_score,
        risk_level=risk_level
    )


@app.post("/predict/batch")
async def predict_batch_endpoint(transactions: List[Transaction]):
    """Predict multiple transactions."""
    
    results = []
    for txn in transactions:
        result = await predict_transaction(txn)
        results.append(result)
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
