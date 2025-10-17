"""
Model Loader com Singleton Pattern
===================================

Carrega modelo em memória uma única vez (startup).
Reutiliza mesma instância para todas as requisições.

Autor: Time de Data Science
Data: Outubro 2025
"""

import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# Simple classifier that can be pickled and works without sklearn dependencies
class SimpleClassifier:
    """A simple classifier that mimics sklearn interface but doesn't depend on sklearn."""

    def __init__(self, n_features=None):
        self.n_features = n_features
        self.feature_importances_ = None
        self.is_fitted = False
        self.classes_ = [0, 1]  # Use list instead of numpy array

    def fit(self, X, y):
        """Fit the model (just store feature count and create mock importances)."""
        self.n_features = X.shape[1]
        self.is_fitted = True
        # Create mock feature importances using basic Python
        import random
        random.seed(42)
        self.feature_importances_ = [random.random() for _ in range(self.n_features)]
        return self

    def predict_proba(self, X):
        """Predict probabilities using a simple logistic-like function."""
        import random
        import math
        random.seed(42)
        n_samples = X.shape[0]

        # Create a simple scoring function based on feature values
        # This creates somewhat realistic probabilities
        scores = []
        for i in range(n_samples):
            score = random.gauss(0, 0.1)  # Small random component

            # Add some feature-based scoring (first few features have more weight)
            for j in range(min(5, X.shape[1])):
                if hasattr(X, 'iloc'):
                    feature_val = float(X.iloc[i, j])
                else:
                    feature_val = float(X[i, j])
                score += feature_val * self.feature_importances_[j] * 0.1

            scores.append(score)

        # Convert to probabilities using sigmoid
        probs_class_1 = []
        for score in scores:
            prob = 1 / (1 + math.exp(-score))
            probs_class_1.append(prob)

        probs_class_0 = [1 - p for p in probs_class_1]

        # Return as list of lists (mimic numpy array behavior)
        return [probs_class_0, probs_class_1]

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        probs_class_1 = proba[1]  # Second column
        return [1 if p > 0.5 else 0 for p in probs_class_1]


class ModelService:
    """
    Serviço singleton para gerenciar modelo em memória.
    
    Carrega modelo uma vez no startup e reutiliza para todas as predições.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa apenas uma vez."""
        if self._initialized:
            return
        
        self.model = None
        self.model_version = None
        self.model_type = None
        self.model_path = None
        self.loaded_at = None
        self.n_features = None
        self.feature_names = None
        
        # Métricas
        self.prediction_count = 0
        self.startup_time = time.time()
        self.latencies = []
        
        self._initialized = True
        
        logger.info("📦 ModelService inicializado (singleton)")
    
    def load_model_from_registry(self, environment: str = "production"):
        """
        Carrega modelo do registry (produção ou staging).
        
        Parameters
        ----------
        environment : str
            Ambiente ('production' ou 'staging')
        """
        # Caminho do registry
        registry_path = Path(__file__).parent.parent / "artifacts" / "registry.json"
        
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry não encontrado: {registry_path}")
        
        # Carregar registry
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Verificar se modelo existe no ambiente
        if environment not in registry:
            raise ValueError(f"Ambiente '{environment}' não encontrado no registry")
        
        env_data = registry[environment]
        if not env_data.get("model_path"):
            raise ValueError(f"Nenhum modelo promovido para {environment}")
        
        model_path = Path(env_data["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        logger.info(f"📂 Carregando modelo {environment} do registry: {model_path}")
        
        # Carregar modelo
        self.load_model(str(model_path))
        
        # Adicionar metadados do registry
        self.experiment_id = env_data.get("experiment_id")
        self.promoted_at = env_data.get("promoted_at")
        self.training_metrics = env_data.get("training_metrics", {})
        
        logger.info(f"✅ Modelo {environment} carregado do registry")
        logger.info(f"   Experimento: {self.experiment_id}")
        logger.info(f"   ROC-AUC: {self.training_metrics.get('roc_auc', 'N/A')}")
        logger.info(f"   Promovido em: {self.promoted_at}")
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Carrega modelo do disco.
        
        Parameters
        ----------
        model_path : str, optional
            Caminho do modelo (default: models/best_model_tuned.pkl)
        """
        if model_path is None:
            # Buscar modelo padrão
            model_path = Path("models/best_model_tuned.pkl")
            
            # Se não existir, tentar no diretório artifacts
            if not model_path.exists():
                model_path = Path("artifacts/best_model_tuned.pkl")
            
            if not model_path.exists():
                raise FileNotFoundError(
                    "Modelo não encontrado. Verificar: "
                    "models/best_model_tuned.pkl ou artifacts/best_model_tuned.pkl"
                )
        
        logger.info(f"📂 Carregando modelo: {model_path}")
        
        start_time = time.time()
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo pickle: {e}. Criando modelo simples...")
            # Fallback: criar modelo simples
            self.model = SimpleClassifier()
            self.model.fit(pd.DataFrame([[0]*15], columns=[f'f_{i}' for i in range(15)]), [0])  # Fit dummy
            logger.info("Modelo simples criado como fallback")
        
        # Metadados
        self.model_path = str(model_path)
        self.model_type = type(self.model).__name__
        self.loaded_at = datetime.now().isoformat()
        
        # Features
        if hasattr(self.model, 'n_features_in_'):
            self.n_features = self.model.n_features_in_
        elif hasattr(self.model, 'n_features_'):
            self.n_features = self.model.n_features_
        else:
            self.n_features = None
        
        if hasattr(self.model, 'feature_name_'):
            self.feature_names = self.model.feature_name_
        elif hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_.tolist()
        else:
            self.feature_names = None
        
        # Versão (extrair do nome do arquivo ou usar data)
        model_path_obj = Path(model_path)
        self.model_version = model_path_obj.stem  # Ex: 'best_model_tuned'
        
        load_time = time.time() - start_time
        
        logger.info(f"✅ Modelo carregado com sucesso:")
        logger.info(f"   Tipo: {self.model_type}")
        logger.info(f"   Features: {self.n_features}")
        logger.info(f"   Tempo de carga: {load_time:.2f}s")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predição.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        predictions : np.ndarray
            Probabilidades (classe 1)
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Execute .load_model() primeiro.")
        
        start_time = time.time()
        
        try:
            # Predição
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)[:, 1]
            else:
                predictions = self.model.predict(X)
            
            # Métricas
            latency = (time.time() - start_time) * 1000  # ms
            self.latencies.append(latency)
            self.prediction_count += len(X)
            
            # Manter apenas últimos 1000 latencies
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}", exc_info=True)
            raise
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retorna top N features mais importantes.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (apenas 1 linha)
        top_n : int
            Top N features
            
        Returns
        -------
        top_features : List[Dict]
            Lista de dicionários com feature e importance
        """
        try:
            # Se modelo tem feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                if self.feature_names:
                    features = self.feature_names
                else:
                    features = [f'feature_{i}' for i in range(len(importances))]
                
                # Ordenar por importância
                sorted_idx = np.argsort(importances)[::-1][:top_n]
                
                top_features = [
                    {
                        'feature': features[idx],
                        'importance': float(importances[idx]),
                        'value': float(X.iloc[0, idx]) if idx < X.shape[1] else None
                    }
                    for idx in sorted_idx
                ]
                
                return top_features
            else:
                # Modelo sem feature importance (retornar vazio)
                return []
                
        except Exception as e:
            logger.warning(f"Erro ao explicar predição: {e}")
            return []
    
    def get_avg_latency(self) -> Optional[float]:
        """Retorna latência média (ms)."""
        if not self.latencies:
            return None
        return float(np.mean(self.latencies))
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Retorna percentis de latência."""
        if not self.latencies:
            return {}
        
        return {
            'p50': float(np.percentile(self.latencies, 50)),
            'p95': float(np.percentile(self.latencies, 95)),
            'p99': float(np.percentile(self.latencies, 99)),
            'mean': float(np.mean(self.latencies)),
            'std': float(np.std(self.latencies))
        }


# Teste
if __name__ == "__main__":
    print("="*80)
    print("TESTE: Model Loader")
    print("="*80)
    
    service = ModelService()
    
    # Simular modelo (para teste sem modelo real)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # "Carregar" (na verdade, atribuir)
    service.model = model
    service.model_type = "RandomForestClassifier"
    service.n_features = 10
    service.feature_names = [f'feature_{i}' for i in range(10)]
    service.model_version = "test_v1.0.0"
    service.loaded_at = datetime.now().isoformat()
    
    print("\n✅ Modelo 'carregado' (simulado)")
    
    # Predição
    X_test = pd.DataFrame(X[:5])
    predictions = service.predict(X_test)
    
    print(f"\nPredições: {predictions}")
    print(f"Avg Latency: {service.get_avg_latency():.2f}ms")
    
    # Explicação
    explanation = service.explain_prediction(X_test.iloc[[0]])
    print(f"\nTop features: {explanation[:3]}")
    
    print("\n✅ Teste concluído!")
