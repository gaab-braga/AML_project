"""
Performance Monitoring para Modelos em Produção
================================================

Monitora métricas de performance em tempo real:
- Comparação com baseline
- Detecção de degradação
- Alertas automáticos
- Integração com Prometheus/Grafana

Métricas Monitoradas:
---------------------
- ROC-AUC, PR-AUC, F1-Score
- Recall@K, Precision@K
- Alert Rate, Strike Rate
- Expected Value
- Latência (p50, p95, p99)
- Throughput

Autor: Time de Data Science
Data: Outubro 2025
Fase: 4.2 - Monitoramento
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance em um momento."""
    timestamp: str
    metrics: Dict[str, float]
    sample_size: int
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceMonitor:
    """
    Monitor de performance contínuo.
    
    Exemplo
    -------
    >>> monitor = PerformanceMonitor(baseline_metrics={'roc_auc': 0.92, 'pr_auc': 0.85})
    >>> monitor.log_metrics({'roc_auc': 0.90, 'pr_auc': 0.83}, sample_size=1000)
    >>> alert = monitor.check_degradation()
    >>> monitor.plot_performance_over_time()
    """
    
    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        tracking_file: Optional[str] = None
    ):
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.critical_threshold = critical_threshold
        self.tracking_file = tracking_file
        self.history: List[PerformanceSnapshot] = []
        
        if tracking_file and Path(tracking_file).exists():
            self._load_history()
        
        logger.info(f"📊 PerformanceMonitor inicializado")
        logger.info(f"   Baseline: {baseline_metrics}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        sample_size: int,
        metadata: Optional[Dict] = None
    ):
        """Registra métricas."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            sample_size=sample_size,
            metadata=metadata or {}
        )
        
        self.history.append(snapshot)
        
        if self.tracking_file:
            self._save_history()
        
        logger.info(f"✅ Métricas registradas: {list(metrics.keys())}")
    
    def check_degradation(self) -> Optional[Dict]:
        """
        Verifica degradação de performance.
        
        Returns
        -------
        alert : Dict ou None
            Alerta se houver degradação
        """
        if not self.history:
            return None
        
        latest = self.history[-1]
        degradations = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in latest.metrics:
                current_value = latest.metrics[metric]
                drop = (baseline_value - current_value) / baseline_value
                
                if drop > self.degradation_threshold:
                    severity = 'CRITICAL' if drop > self.critical_threshold else 'WARNING'
                    degradations.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'drop_pct': drop * 100,
                        'severity': severity
                    })
        
        if degradations:
            alert = {
                'timestamp': latest.timestamp,
                'alert_level': 'CRITICAL' if any(d['severity'] == 'CRITICAL' for d in degradations) else 'WARNING',
                'degradations': degradations,
                'recommendation': self._get_degradation_recommendation(degradations)
            }
            
            logger.warning(f"⚠️ Degradação detectada: {len(degradations)} métricas")
            return alert
        
        return None
    
    def _get_degradation_recommendation(self, degradations: List[Dict]) -> str:
        """Gera recomendação."""
        if any(d['severity'] == 'CRITICAL' for d in degradations):
            return "🚨 CRÍTICO: Performance caiu >10%. Retreinar URGENTE ou rollback."
        return "⚠️ ATENÇÃO: Performance caiu >5%. Monitorar e agendar retreinamento."
    
    def plot_performance_over_time(self, metrics: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plota evolução de métricas."""
        if len(self.history) < 2:
            logger.warning("⚠️ Histórico insuficiente")
            return
        
        df = pd.DataFrame([
            {'timestamp': s.timestamp, **s.metrics}
            for s in self.history
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if metrics is None:
            metrics = [c for c in df.columns if c != 'timestamp']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for metric in metrics:
            if metric in df.columns:
                ax.plot(df['timestamp'], df[metric], marker='o', label=metric, linewidth=2)
                
                # Baseline
                if metric in self.baseline_metrics:
                    ax.axhline(self.baseline_metrics[metric], linestyle='--', alpha=0.5, label=f'{metric} baseline')
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Over Time', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"💾 Plot salvo: {save_path}")
        
        plt.show()
    
    def _save_history(self):
        """Salva histórico."""
        data = [s.to_dict() for s in self.history]
        with open(self.tracking_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_history(self):
        """Carrega histórico."""
        with open(self.tracking_file, 'r') as f:
            data = json.load(f)
        self.history = [PerformanceSnapshot(**s) for s in data]
        logger.info(f"📂 Histórico carregado: {len(self.history)} snapshots")


if __name__ == "__main__":
    print("="*80)
    print("TESTE: Performance Monitoring")
    print("="*80)
    
    # Simular monitoramento
    monitor = PerformanceMonitor(
        baseline_metrics={'roc_auc': 0.92, 'pr_auc': 0.85, 'f1': 0.78},
        degradation_threshold=0.05,
        tracking_file='performance_tracking_test.json'
    )
    
    # Semana 1: Performance boa
    monitor.log_metrics({'roc_auc': 0.91, 'pr_auc': 0.84, 'f1': 0.77}, sample_size=1000)
    
    # Semana 2: Degradação leve
    monitor.log_metrics({'roc_auc': 0.88, 'pr_auc': 0.81, 'f1': 0.74}, sample_size=1000)
    
    # Semana 3: Degradação severa
    monitor.log_metrics({'roc_auc': 0.80, 'pr_auc': 0.72, 'f1': 0.65}, sample_size=1000)
    
    # Verificar
    alert = monitor.check_degradation()
    if alert:
        print(f"\n⚠️ Alerta: {alert['alert_level']}")
        print(f"   {alert['recommendation']}")
        for deg in alert['degradations']:
            print(f"   - {deg['metric']}: {deg['baseline']:.3f} → {deg['current']:.3f} (-{deg['drop_pct']:.1f}%)")
    
    monitor.plot_performance_over_time(save_path='performance_test.png')
    
    print("\n✅ Teste concluído!")
