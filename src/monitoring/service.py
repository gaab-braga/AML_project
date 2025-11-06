"""
Production monitoring service for AML detection system.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__)


class AMLMonitor:
    """Monitor model performance and data quality in production."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_history = []
        self.metrics_file = Path('logs/metrics_history.json')
        self.alerts_file = Path('logs/alerts_history.json')
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
    def collect_metrics(self, y_true, y_pred_proba, features, latency_ms):
        """
        Collect production metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Prediction probabilities
            features: Input features DataFrame
            latency_ms: Inference latency in milliseconds
            
        Returns:
            dict: Collected metrics
        """
        timestamp = datetime.now()
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
        throughput = len(y_true) / (latency_ms / 1000) if latency_ms > 0 else 0
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'performance': {
                'pr_auc': float(pr_auc),
                'precision': float(optimal_precision),
                'recall': float(optimal_recall)
            },
            'operational': {
                'latency_ms': latency_ms,
                'throughput_per_second': throughput,
                'sample_size': len(y_true)
            },
            'data_quality': {
                'missing_data_rate': float(features.isnull().mean().mean()),
                'data_drift_score': self.calculate_drift_score(features)
            }
        }
        
        self.metrics_history.append(metrics)
        logger.info(f"Metrics collected: PR-AUC={pr_auc:.4f}, Latency={latency_ms:.2f}ms")
        
        return metrics
    
    def calculate_drift_score(self, features):
        """Calculate data drift score."""
        try:
            drift_scores = []
            for col in features.select_dtypes(include=[np.number]).columns[:10]:
                feature_data = features[col].dropna()
                if len(feature_data) > 10:
                    current_mean = feature_data.mean()
                    current_std = feature_data.std()
                    drift = abs(current_mean) / (current_std + 1e-10)
                    drift_scores.append(drift)
            
            return float(np.mean(drift_scores)) if drift_scores else 0.0
        except Exception as e:
            logger.warning(f"Drift calculation error: {e}")
            return 0.0
    
    def check_alerts(self, metrics):
        """Check if metrics violate thresholds."""
        alerts = []
        
        if metrics['performance']['pr_auc'] < 0.25:
            alerts.append({
                'level': 'critical',
                'metric': 'pr_auc',
                'current_value': metrics['performance']['pr_auc'],
                'threshold': 0.25,
                'description': 'Model performance degradation'
            })
        
        if metrics['operational']['latency_ms'] > 1000:
            alerts.append({
                'level': 'warning',
                'metric': 'latency_ms',
                'current_value': metrics['operational']['latency_ms'],
                'threshold': 1000,
                'description': 'High inference latency'
            })
        
        if metrics['data_quality']['missing_data_rate'] > 0.2:
            alerts.append({
                'level': 'warning',
                'metric': 'missing_data_rate',
                'current_value': metrics['data_quality']['missing_data_rate'],
                'threshold': 0.2,
                'description': 'High missing data rate'
            })
        
        for alert in alerts:
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'alert': alert
            })
            logger.warning(f"Alert: {alert['metric']} - {alert['level']}")
        
        return alerts
    
    def get_health_report(self):
        """Generate system health report."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]
        
        report = {
            'status': 'healthy',
            'last_update': recent_metrics[-1]['timestamp'],
            'metrics_summary': {
                'avg_pr_auc': np.mean([m['performance']['pr_auc'] for m in recent_metrics]),
                'avg_latency': np.mean([m['operational']['latency_ms'] for m in recent_metrics]),
                'avg_throughput': np.mean([m['operational']['throughput_per_second'] for m in recent_metrics])
            },
            'alerts_today': len([a for a in self.alert_history
                               if a['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]),
            'drift_status': 'stable'
        }
        
        if report['metrics_summary']['avg_pr_auc'] < 0.25:
            report['status'] = 'critical'
        elif report['alerts_today'] > 5:
            report['status'] = 'warning'
        
        return report
    
    def save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics saved to {self.metrics_file}")
    
    def save_alerts(self):
        """Save alerts to file."""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
        logger.info(f"Alerts saved to {self.alerts_file}")
