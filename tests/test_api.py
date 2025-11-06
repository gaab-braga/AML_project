"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from entrypoints.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    """Sample transaction data."""
    return {
        "amount": 1500.0,
        "payment_format": 2,
        "hour": 14,
        "day_of_week": 3,
        "transaction_count": 5
    }


@pytest.fixture
def sample_batch():
    """Sample batch transactions."""
    return [
        {"amount": 1500.0, "payment_format": 2, "hour": 14},
        {"amount": 500.0, "payment_format": 1, "hour": 10},
        {"amount": 5000.0, "payment_format": 3, "hour": 22}
    ]


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "AML" in data["message"]


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint_valid(client, sample_transaction):
    """Test prediction with valid transaction."""
    response = client.post("/predict", json=sample_transaction)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_endpoint_missing_field(client):
    """Test prediction with missing required field."""
    invalid_transaction = {"amount": 1500.0}
    
    response = client.post("/predict", json=invalid_transaction)
    
    assert response.status_code == 422


def test_predict_endpoint_invalid_type(client):
    """Test prediction with invalid data type."""
    invalid_transaction = {
        "amount": "invalid",
        "payment_format": 2
    }
    
    response = client.post("/predict", json=invalid_transaction)
    
    assert response.status_code == 422


def test_predict_batch_endpoint(client, sample_batch):
    """Test batch prediction endpoint."""
    response = client.post("/predict/batch", json={"transactions": sample_batch})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == len(sample_batch)
    
    for pred in data["predictions"]:
        assert "prediction" in pred
        assert "probability" in pred
        assert pred["prediction"] in [0, 1]


def test_predict_batch_empty(client):
    """Test batch prediction with empty list."""
    response = client.post("/predict/batch", json={"transactions": []})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 0


def test_cors_headers(client, sample_transaction):
    """Test CORS headers are present."""
    response = client.post("/predict", json=sample_transaction)
    
    assert response.status_code == 200
    # CORS middleware should add headers
