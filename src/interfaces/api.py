#!/usr/bin/env python3
"""
AML Pipeline REST API

FastAPI-based REST API for AML Pipeline programmatic access and integration.
Provides endpoints for pipeline execution, monitoring, and management.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline_controller import PipelineController
from cache import HierarchicalCache, MemoryCache, DiskCache
from evaluation.metrics import ModelEvaluator
from config.config_loader import ConfigLoader
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Global variables for components
controller: Optional[PipelineController] = None
cache: Optional[HierarchicalCache] = None
evaluator: Optional[ModelEvaluator] = None
config: Optional[Dict[str, Any]] = None


# Pydantic models for request/response
class PipelineRunRequest(BaseModel):
    """Request model for pipeline execution."""
    target: str = Field(..., description="Target environment", example="production")
    mode: str = Field("full", description="Execution mode", example="full")
    config_override: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")

    @validator('target')
    def validate_target(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError('Target must be development, staging, or production')
        return v

    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['full', 'fast', 'custom']:
            raise ValueError('Mode must be full, fast, or custom')
        return v


class PipelineRunResponse(BaseModel):
    """Response model for pipeline execution."""
    execution_id: str = Field(..., description="Unique execution identifier")
    status: str = Field(..., description="Execution status")
    message: str = Field(..., description="Status message")
    started_at: datetime = Field(..., description="Execution start time")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float = Field(..., description="Prediction score/probability")
    confidence: float = Field(..., description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    pipeline_metrics: Dict[str, Any] = Field(..., description="Pipeline performance metrics")
    cache_metrics: Dict[str, Any] = Field(..., description="Cache performance metrics")
    system_metrics: Dict[str, Any] = Field(..., description="System resource metrics")


class StatusResponse(BaseModel):
    """Response model for status checks."""
    status: str = Field(..., description="Overall system status")
    uptime: str = Field(..., description="System uptime")
    version: str = Field(..., description="API version")
    pipeline_status: str = Field(..., description="Pipeline status")
    active_executions: int = Field(..., description="Number of active executions")
    last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")


class CacheOperationRequest(BaseModel):
    """Request model for cache operations."""
    operation: str = Field(..., description="Cache operation type")
    key: Optional[str] = Field(None, description="Cache key for operations")
    value: Optional[Any] = Field(None, description="Value for set operations")
    ttl: Optional[int] = Field(None, description="TTL for set operations")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Service health status")
    version: str = Field(..., description="API version")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global controller, cache, evaluator, config

    try:
        logger.info("Initializing AML Pipeline API...")

        # Load configuration
        config = ConfigLoader.load_config("config/pipeline_config.yaml")

        # Initialize components
        controller = PipelineController(config)

        # Initialize cache
        memory_cache = MemoryCache(max_size=1000, ttl=3600)
        disk_cache = DiskCache(cache_dir="./cache", max_size_mb=500)
        cache = HierarchicalCache(
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            cache_strategy="write-through"
        )

        # Initialize evaluator
        evaluator = ModelEvaluator(config)

        logger.info("AML Pipeline API initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AML Pipeline API...")


# Create FastAPI app
app = FastAPI(
    title="AML Pipeline API",
    description="Enterprise Anti-Money Laundering Pipeline REST API",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """API root endpoint with basic information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AML Pipeline API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2E86AB; }
            .endpoint { background: #f0f0f0; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1 class="header">🔍 AML Pipeline API v2.0.0</h1>
        <p><strong>Enterprise Anti-Money Laundering Pipeline</strong></p>

        <h2>Available Endpoints</h2>

        <div class="endpoint">
            <strong>GET /health</strong> - Health check
        </div>

        <div class="endpoint">
            <strong>GET /status</strong> - System status
        </div>

        <div class="endpoint">
            <strong>POST /run</strong> - Execute pipeline
        </div>

        <div class="endpoint">
            <strong>POST /predict</strong> - Make predictions
        </div>

        <div class="endpoint">
            <strong>GET /metrics</strong> - Performance metrics
        </div>

        <div class="endpoint">
            <strong>GET /cache</strong> - Cache statistics
        </div>

        <div class="endpoint">
            <strong>POST /cache</strong> - Cache operations
        </div>

        <h2>Documentation</h2>
        <p><a href="/docs">Interactive API Documentation (Swagger UI)</a></p>
        <p><a href="/redoc">ReDoc Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        services_status = {}

        # Check pipeline controller
        if controller:
            services_status["pipeline_controller"] = "healthy"
        else:
            services_status["pipeline_controller"] = "unhealthy"

        # Check cache
        if cache:
            services_status["cache"] = "healthy"
        else:
            services_status["cache"] = "unhealthy"

        # Check evaluator
        if evaluator:
            services_status["evaluator"] = "healthy"
        else:
            services_status["evaluator"] = "unhealthy"

        # Overall status
        overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            services=services_status,
            version="2.0.0"
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    try:
        if not controller:
            raise HTTPException(status_code=503, detail="Pipeline controller not initialized")

        status_info = controller.get_status()

        return StatusResponse(
            status="operational",
            uptime=status_info.get('uptime', 'unknown'),
            version="2.0.0",
            pipeline_status=status_info.get('pipeline_status', 'unknown'),
            active_executions=status_info.get('active_executions', 0),
            last_execution=None  # TODO: Implement last execution tracking
        )

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


@app.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Execute the AML pipeline."""
    try:
        if not controller:
            raise HTTPException(status_code=503, detail="Pipeline controller not initialized")

        logger.info(f"Starting pipeline execution: target={request.target}, mode={request.mode}")

        # For now, run synchronously (implement async execution later)
        result = controller.run_pipeline(request.target, request.mode)

        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return PipelineRunResponse(
            execution_id=execution_id,
            status="completed" if result.get('success') else "failed",
            message=result.get('message', 'Pipeline execution completed'),
            started_at=datetime.now()
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make predictions using the trained model."""
    try:
        if not evaluator:
            raise HTTPException(status_code=503, detail="Model evaluator not initialized")

        import time
        start_time = time.time()

        # TODO: Implement actual prediction logic
        # For now, return mock prediction
        prediction = 0.85  # Mock prediction score
        confidence = 0.92  # Mock confidence

        processing_time = time.time() - start_time

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version or "latest",
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    period: str = Query("1h", description="Time period for metrics")
):
    """Get performance metrics."""
    try:
        if not controller:
            raise HTTPException(status_code=503, detail="Pipeline controller not initialized")

        pipeline_metrics = controller.get_metrics()
        cache_metrics = cache.get_stats() if cache else {}

        # Mock system metrics (implement actual system monitoring)
        system_metrics = {
            "cpu_percent": 45.2,
            "memory_percent": 67.8,
            "disk_usage_percent": 54.3
        }

        return MetricsResponse(
            timestamp=datetime.now(),
            pipeline_metrics=pipeline_metrics,
            cache_metrics=cache_metrics,
            system_metrics=system_metrics
        )

    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {e}")


@app.get("/cache")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        if not cache:
            raise HTTPException(status_code=503, detail="Cache not initialized")

        stats = cache.get_stats()
        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats retrieval failed: {e}")


@app.post("/cache")
async def cache_operation(request: CacheOperationRequest):
    """Perform cache operations."""
    try:
        if not cache:
            raise HTTPException(status_code=503, detail="Cache not initialized")

        operation = request.operation.lower()

        if operation == "get" and request.key:
            value = cache.get(request.key)
            return {"operation": "get", "key": request.key, "value": value}

        elif operation == "set" and request.key and request.value is not None:
            cache.set(request.key, request.value, request.ttl)
            return {"operation": "set", "key": request.key, "status": "success"}

        elif operation == "delete" and request.key:
            deleted = cache.delete(request.key)
            return {"operation": "delete", "key": request.key, "deleted": deleted}

        elif operation == "clear":
            cache.clear()
            return {"operation": "clear", "status": "success"}

        else:
            raise HTTPException(status_code=400, detail=f"Invalid cache operation: {operation}")

    except Exception as e:
        logger.error(f"Cache operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {e}")


@app.get("/config")
async def get_config():
    """Get current configuration (sensitive data redacted)."""
    try:
        if not config:
            raise HTTPException(status_code=503, detail="Configuration not loaded")

        # Return redacted config (remove sensitive data)
        redacted_config = json.loads(json.dumps(config))

        # Remove sensitive fields
        sensitive_fields = ['password', 'secret', 'key', 'token']
        def redact_sensitive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(field in key.lower() for field in sensitive_fields):
                        obj[key] = "***REDACTED***"
                    else:
                        redact_sensitive(value)
            elif isinstance(obj, list):
                for item in obj:
                    redact_sensitive(item)

        redact_sensitive(redacted_config)

        return JSONResponse(content=redacted_config)

    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {e}")


@app.post("/config/reload")
async def reload_config():
    """Reload configuration from disk."""
    try:
        global config
        config = ConfigLoader.load_config("config/pipeline_config.yaml")

        # Reinitialize components with new config
        global controller, evaluator
        controller = PipelineController(config)
        evaluator = ModelEvaluator(config)

        return {"status": "success", "message": "Configuration reloaded successfully"}

    except Exception as e:
        logger.error(f"Config reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Config reload failed: {e}")


@app.get("/logs")
async def get_logs(
    lines: int = Query(100, description="Number of log lines to retrieve", ge=1, le=1000),
    level: str = Query("INFO", description="Minimum log level")
):
    """Get recent application logs."""
    try:
        # TODO: Implement actual log retrieval
        # For now, return mock logs
        mock_logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Pipeline initialized"},
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Cache system ready"},
        ]

        return JSONResponse(content={"logs": mock_logs[:lines]})

    except Exception as e:
        logger.error(f"Log retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {e}")


@app.get("/docs")
async def api_docs():
    """Redirect to API documentation."""
    # This will be handled by FastAPI's automatic documentation
    pass


def main():
    """Run the API server."""
    uvicorn.run(
        "src.interfaces.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()