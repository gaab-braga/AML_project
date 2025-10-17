#!/usr/bin/env python3
"""
SERVIÇO DE MONITORAMENTO PARA PRODUÇÃO
Sistema de Detecção de AML
"""

import time
import json
import yaml
import logging
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

class AMLMonitor:
    def __init__(self, config_path='config/monitoring_config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.metrics_history = []
        self.alert_history = []

    def load_config(self, config_path):
        """Carrega configuração de monitoramento"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Configura logging"""
        logging.basicConfig(
            filename='logs/aml_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def collect_metrics(self, y_true, y_pred_proba, features, latency_ms):
        """Coleta métricas de performance"""
        timestamp = datetime.now()

        # Calcular métricas
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        # Precision e recall no threshold ótimo (max F1)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]

        # Throughput
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
                'missing_data_rate': features.isnull().mean().mean(),
                'data_drift_score': self.calculate_drift_score(features)
            }
        }

        self.metrics_history.append(metrics)
        self.logger.info(f"Métricas coletadas: PR-AUC={pr_auc:.4f}")

        return metrics

    def calculate_drift_score(self, features):
        """Calcula score de drift estatístico"""
        # Implementação simplificada - em produção usar libraries como alibi-detect
        drift_scores = []

        for feature in self.config['drift_detection']['features_to_monitor']:
            if feature in features.columns:
                # Kolmogorov-Smirnov test simplificado
                feature_data = features[feature].dropna()
                if len(feature_data) > 10:
                    # Comparar com distribuição baseline (simplificado)
                    baseline_mean = 0  # Em produção, carregar baseline real
                    baseline_std = 1
                    current_mean = feature_data.mean()
                    current_std = feature_data.std()

                    # Calcular diferença normalizada
                    drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-10)
                    drift_scores.append(drift)

        return np.mean(drift_scores) if drift_scores else 0.0

    def check_alerts(self, metrics):
        """Verifica se métricas violam thresholds de alerta"""
        alerts = []

        # Verificar métricas de performance
        for metric_name, config in self.config['metrics']['performance'].items():
            current_value = metrics['performance'].get(metric_name, 0)
            threshold = config['threshold']

            if current_value < threshold:
                alerts.append({
                    'level': config['alert_level'],
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'description': config['description']
                })

        # Verificar métricas operacionais
        for metric_name, config in self.config['metrics']['operational'].items():
            current_value = metrics['operational'].get(metric_name, 0)
            threshold = config['threshold']

            # Para latência, alerta se MAIOR que threshold
            if metric_name == 'latency_ms' and current_value > threshold:
                alerts.append({
                    'level': config['alert_level'],
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'description': config['description']
                })
            # Para throughput e error_rate, alerta se MENOR que threshold
            elif metric_name in ['throughput_per_second', 'error_rate'] and current_value < threshold:
                alerts.append({
                    'level': config['alert_level'],
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'description': config['description']
                })

        # Verificar qualidade de dados
        for metric_name, config in self.config['metrics']['data_quality'].items():
            current_value = metrics['data_quality'].get(metric_name, 0)
            threshold = config['threshold']

            if current_value > threshold:
                alerts.append({
                    'level': config['alert_level'],
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'description': config['description']
                })

        return alerts

    def send_alert(self, alert):
        """Envia alerta pelos canais configurados"""
        level = alert['level']
        channels = self.config['alerts']['escalation'].get(level, [])

        message = f"""
        🚨 ALERTA AML - {level.upper()}

        Métrica: {alert['metric']}
        Valor atual: {alert['current_value']:.4f}
        Threshold: {alert['threshold']:.4f}
        Descrição: {alert['description']}

        Timestamp: {datetime.now().isoformat()}
        """

        for channel in channels:
            if channel == 'email' and self.config['alerts']['channels']['email']['enabled']:
                self.send_email_alert(message, level)
            elif channel == 'slack' and self.config['alerts']['channels']['slack']['enabled']:
                self.send_slack_alert(message, level)

        self.alert_history.append({
            'timestamp': datetime.now().isoformat(),
            'alert': alert,
            'channels': channels
        })

        self.logger.warning(f"Alerta enviado: {alert['metric']} - {level}")

    def send_email_alert(self, message, level):
        """Envia alerta por email"""
        try:
            config = self.config['alerts']['channels']['email']

            msg = MIMEMultipart()
            msg['From'] = config['sender']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"AML Alert - {level.upper()}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            # server.login(username, password)  # Configurar credenciais
            server.sendmail(config['sender'], config['recipients'], msg.as_string())
            server.quit()

            self.logger.info("Alerta enviado por email")
        except Exception as e:
            self.logger.error(f"Erro ao enviar email: {e}")

    def send_slack_alert(self, message, level):
        """Envia alerta por Slack"""
        # Implementação do Slack webhook
        pass

    def get_health_report(self):
        """Gera relatório de saúde do sistema"""
        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = self.metrics_history[-10:]  # Últimas 10 medições

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
            'drift_status': 'stable'  # Implementar verificação de drift
        }

        # Verificar se sistema está saudável
        if report['metrics_summary']['avg_pr_auc'] < 0.25:
            report['status'] = 'critical'
        elif report['alerts_today'] > 5:
            report['status'] = 'warning'

        return report

    def save_metrics(self):
        """Salva métricas em arquivo"""
        metrics_file = Path('logs/metrics_history.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

# Exemplo de uso
if __name__ == "__main__":
    monitor = AMLMonitor()

    # Simular coleta de métricas
    import numpy as np
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    features = pd.DataFrame({
        'amount': np.random.exponential(100, 1000),
        'payment_format': np.random.randint(0, 5, 1000),
        'hour': np.random.randint(0, 24, 1000)
    })

    metrics = monitor.collect_metrics(y_true, y_pred_proba, features, latency_ms=50)
    alerts = monitor.check_alerts(metrics)

    for alert in alerts:
        monitor.send_alert(alert)

    print("Monitoramento executado com sucesso")
