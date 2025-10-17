"""
Callback SSE para enviar métricas de treinamento em tempo real para dashboard Flask.
"""

import time
import requests
import logging

logger = logging.getLogger(__name__)

class SSELiveCallback:
    """Callback para enviar métricas de treinamento em tempo real via SSE."""
    
    def __init__(self, model_name, server_url='http://localhost:5000', update_interval=5):
        self.model_name = model_name
        self.server_url = server_url
        self.update_interval = update_interval
        self.iterations = []
        self.train_auc = []
        self.val_auc = []
        self.oob_scores = []
        self.start_time = time.time()
        
    def send_to_dashboard(self, event_type, data):
        """Envia dados para o dashboard via HTTP POST."""
        try:
            response = requests.post(
                f"{self.server_url}/send_training_data",
                json={'type': event_type, 'data': data},
                timeout=1.0
            )
            if response.status_code == 200:
                logger.info(f"📡 Enviado {event_type} para dashboard")
            else:
                logger.warning(f"⚠️ Falha ao enviar para dashboard: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao conectar dashboard: {e}")
    
    def add_metrics(self, iteration, train_auc=None, val_auc=None, oob_score=None):
        """Adiciona métricas e envia para dashboard se necessário."""
        self.iterations.append(iteration)
        if train_auc is not None:
            self.train_auc.append(train_auc)
        if val_auc is not None:
            self.val_auc.append(val_auc)
        if oob_score is not None:
            self.oob_scores.append(oob_score)
        
        if iteration % self.update_interval == 0 or iteration == 1:
            self.send_update()
    
    def send_start(self, message="Training started"):
        """Envia evento de início."""
        self.send_to_dashboard('start', {'message': message})
    
    def send_update(self):
        """Envia atualização de métricas."""
        data = {
            'iteration': self.iterations[-1] if self.iterations else 0,
            'roc_auc': self.val_auc[-1] if self.val_auc else None,
            'accuracy': self.train_auc[-1] if self.train_auc else None,  # Usando train_auc como proxy
            'loss': 1 - (self.val_auc[-1] if self.val_auc else 0) if self.val_auc else None
        }
        self.send_to_dashboard('update', data)
    
    def send_complete(self, final_metrics=None):
        """Envia evento de conclusão."""
        data = {'message': 'Training completed'}
        if final_metrics:
            data.update(final_metrics)
        self.send_to_dashboard('complete', data)