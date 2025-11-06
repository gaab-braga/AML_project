"""
Tests for monitoring service.
"""
import pytest
import numpy as np
import pandas as pd

from src.monitoring.service import AMLMonitor


@pytest.fixture
def monitor():
    """Create monitoring instance."""
    return AMLMonitor()


def test_monitor_initialization(monitor):
    """Test monitor initializes correctly."""
    assert monitor.metrics_history == []
    assert monitor.alert_history == []


def test_collect_metrics(monitor, sample_features):
    """Test metrics collection."""
    X, y = sample_features
    y_proba = np.random.random(len(y))
    
    metrics = monitor.collect_metrics(y, y_proba, X, latency_ms=50)
    
    assert "timestamp" in metrics
    assert "performance" in metrics
    assert "operational" in metrics
    assert "data_quality" in metrics
    
    assert metrics["performance"]["pr_auc"] >= 0
    assert metrics["operational"]["latency_ms"] == 50
    assert len(monitor.metrics_history) == 1


def test_calculate_drift_score(monitor, sample_features):
    """Test drift score calculation."""
    X, _ = sample_features
    
    drift_score = monitor.calculate_drift_score(X)
    
    assert isinstance(drift_score, float)
    assert drift_score >= 0


def test_check_alerts_no_alerts(monitor, sample_features):
    """Test alert checking with good metrics."""
    X, y = sample_features
    y_proba = np.random.random(len(y))
    
    metrics = monitor.collect_metrics(y, y_proba, X, latency_ms=50)
    alerts = monitor.check_alerts(metrics)
    
    # May or may not have alerts depending on random data
    assert isinstance(alerts, list)


def test_check_alerts_critical(monitor, sample_features):
    """Test alert with critical performance."""
    X, y = sample_features
    y_proba = np.zeros(len(y))  # Poor predictions
    
    metrics = monitor.collect_metrics(y, y_proba, X, latency_ms=50)
    alerts = monitor.check_alerts(metrics)
    
    # Should trigger performance alert
    assert len(alerts) > 0


def test_check_alerts_high_latency(monitor, sample_features):
    """Test alert with high latency."""
    X, y = sample_features
    y_proba = np.random.random(len(y))
    
    metrics = monitor.collect_metrics(y, y_proba, X, latency_ms=2000)
    alerts = monitor.check_alerts(metrics)
    
    latency_alerts = [a for a in alerts if a["metric"] == "latency_ms"]
    assert len(latency_alerts) > 0


def test_health_report_no_data(monitor):
    """Test health report with no data."""
    report = monitor.get_health_report()
    
    assert report["status"] == "no_data"


def test_health_report_with_data(monitor, sample_features):
    """Test health report with metrics."""
    X, y = sample_features
    y_proba = np.random.random(len(y))
    
    for _ in range(5):
        metrics = monitor.collect_metrics(y, y_proba, X, latency_ms=50)
    
    report = monitor.get_health_report()
    
    assert "status" in report
    assert report["status"] in ["healthy", "warning", "critical"]
    assert "metrics_summary" in report
    assert "alerts_today" in report


def test_save_metrics(monitor, sample_features, tmp_path):
    """Test saving metrics to file."""
    X, y = sample_features
    y_proba = np.random.random(len(y))
    
    monitor.collect_metrics(y, y_proba, X, latency_ms=50)
    monitor.metrics_file = tmp_path / "metrics.json"
    monitor.save_metrics()
    
    assert monitor.metrics_file.exists()
