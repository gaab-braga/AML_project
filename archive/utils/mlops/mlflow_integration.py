"""
MLflow Integration para Tracking e Model Registry
==================================================

Integra MLflow para:
- Tracking de experimentos (hiperparâmetros, métricas, artifacts)
- Model Registry (versionamento, lifecycle management)
- Promoção de modelos (staging → production)
- Comparação de runs
- Rollback automático

Requisitos:
-----------
pip install mlflow boto3  # boto3 se usar S3 como artifact store

Setup do MLflow Server:
-----------------------
# Local (desenvolvimento)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Produção (exemplo com PostgreSQL + S3)
mlflow server \
    --backend-store-uri postgresql://user:pass@host:5432/mlflow \
    --default-artifact-root s3://mlflow-artifacts/

Autor: Time de Data Science
Data: Outubro 2025
Fase: 4.1 - Versionamento e Registro
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Gerenciador de tracking de experimentos com MLflow.
    
    Parâmetros
    ----------
    tracking_uri : str, optional
        URI do MLflow tracking server (default: local ./mlruns)
    experiment_name : str, default='aml-detection'
        Nome do experimento
    auto_log : bool, default=False
        Ativar autolog do framework (sklearn, lightgbm, etc.)
        
    Exemplo
    -------
    >>> tracker = MLflowTracker(experiment_name='aml-detection')
    >>> 
    >>> with tracker.start_run(run_name='lgbm_v1.2.0'):
    ...     tracker.log_params(best_params)
    ...     tracker.log_metrics(test_metrics)
    ...     tracker.log_model(model, 'model')
    ...     tracker.log_artifacts('artifacts/')
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = 'aml-detection',
        auto_log: bool = False
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"📊 MLflow tracking URI: {tracking_uri}")
        else:
            logger.info("📊 MLflow usando tracking local (./mlruns)")
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        if auto_log:
            mlflow.sklearn.autolog()
            mlflow.lightgbm.autolog()
            mlflow.xgboost.autolog()
            logger.info("✅ Autolog ativado (sklearn, lightgbm, xgboost)")
        
        logger.info(f"🔬 Experimento: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Inicia um run de MLflow (use como context manager).
        
        Parâmetros
        ----------
        run_name : str, optional
            Nome do run
        tags : Dict, optional
            Tags adicionais
            
        Returns
        -------
        mlflow.ActiveRun
            Context manager do run
        """
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_params(self, params: Dict[str, Any]):
        """Log de hiperparâmetros."""
        mlflow.log_params(params)
        logger.info(f"📝 Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log de métricas."""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"📊 Logged {len(metrics)} metrics")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log de métrica individual."""
        mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log de modelo.
        
        Parâmetros
        ----------
        model : Any
            Modelo treinado
        artifact_path : str
            Caminho no MLflow (ex: 'model')
        registered_model_name : str, optional
            Nome para Model Registry
        signature : ModelSignature, optional
            Assinatura do modelo (input/output schema)
        input_example : Any, optional
            Exemplo de input
        """
        # Detectar tipo de modelo
        model_type = type(model).__name__
        
        if 'LGBMClassifier' in model_type or 'LGBMRegressor' in model_type:
            mlflow.lightgbm.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        elif 'XGB' in model_type:
            mlflow.xgboost.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        
        logger.info(f"🤖 Modelo logado: {artifact_path}")
        if registered_model_name:
            logger.info(f"📦 Registrado no Model Registry: {registered_model_name}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log de arquivo individual."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"📁 Artifact logado: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log de diretório completo."""
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.info(f"📁 Artifacts logados: {local_dir}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """Log de dicionário como JSON."""
        mlflow.log_dict(dictionary, filename)
        logger.info(f"📝 Dict logado: {filename}")
    
    def log_figure(self, figure, filename: str):
        """Log de figura matplotlib."""
        mlflow.log_figure(figure, filename)
        logger.info(f"📊 Figura logada: {filename}")
    
    def get_run_id(self) -> str:
        """Retorna ID do run ativo."""
        return mlflow.active_run().info.run_id
    
    def search_runs(
        self,
        filter_string: str = "",
        order_by: List[str] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Busca runs no experimento.
        
        Parâmetros
        ----------
        filter_string : str
            Filtro MLflow (ex: "metrics.roc_auc > 0.9")
        order_by : List[str]
            Ordenação (ex: ["metrics.roc_auc DESC"])
        max_results : int
            Número máximo de resultados
            
        Returns
        -------
        runs_df : pd.DataFrame
            DataFrame com runs
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by or ["metrics.roc_auc DESC"],
            max_results=max_results
        )
        
        logger.info(f"🔍 Encontrados {len(runs)} runs")
        return runs
    
    def get_best_run(self, metric: str = 'roc_auc', mode: str = 'max') -> pd.Series:
        """
        Retorna melhor run baseado em métrica.
        
        Parâmetros
        ----------
        metric : str
            Métrica para comparação
        mode : str
            'max' ou 'min'
            
        Returns
        -------
        best_run : pd.Series
            Run com melhor métrica
        """
        order = "DESC" if mode == 'max' else "ASC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError("Nenhum run encontrado")
        
        best_run = runs.iloc[0]
        logger.info(f"🏆 Melhor run: {best_run['tags.mlflow.runName']} ({metric}={best_run[f'metrics.{metric}']:.4f})")
        
        return best_run


class ModelRegistry:
    """
    Gerenciador de Model Registry do MLflow.
    
    Lifecycle de Modelos:
    ---------------------
    None → Staging → Production → Archived
    
    Parâmetros
    ----------
    tracking_uri : str, optional
        URI do MLflow tracking server
        
    Exemplo
    -------
    >>> registry = ModelRegistry()
    >>> 
    >>> # Registrar modelo
    >>> registry.register_model('runs:/abc123/model', 'aml-detector')
    >>> 
    >>> # Promover para staging
    >>> registry.transition_model('aml-detector', version=1, stage='Staging')
    >>> 
    >>> # Promover para produção
    >>> registry.transition_model('aml-detector', version=1, stage='Production')
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        logger.info("📦 ModelRegistry inicializado")
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict] = None,
        description: Optional[str] = None
    ) -> ModelVersion:
        """
        Registra modelo no Model Registry.
        
        Parâmetros
        ----------
        model_uri : str
            URI do modelo (ex: 'runs:/run_id/model')
        name : str
            Nome do modelo registrado
        tags : Dict, optional
            Tags adicionais
        description : str, optional
            Descrição do modelo
            
        Returns
        -------
        model_version : ModelVersion
            Versão registrada
        """
        # Criar modelo registrado se não existir
        try:
            self.client.create_registered_model(name, tags=tags, description=description)
            logger.info(f"✅ Modelo '{name}' criado no Registry")
        except Exception:
            logger.info(f"ℹ️ Modelo '{name}' já existe no Registry")
        
        # Criar versão
        model_version = mlflow.register_model(model_uri, name)
        
        logger.info(f"📦 Modelo registrado: {name} (versão {model_version.version})")
        
        return model_version
    
    def transition_model(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = False
    ):
        """
        Move modelo para outro stage.
        
        Parâmetros
        ----------
        name : str
            Nome do modelo
        version : int
            Versão do modelo
        stage : str
            Stage destino ('Staging', 'Production', 'Archived', 'None')
        archive_existing_versions : bool
            Arquivar versões antigas no mesmo stage
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        logger.info(f"🔄 Modelo '{name}' v{version} → {stage}")
    
    def get_latest_version(self, name: str, stage: Optional[str] = None) -> ModelVersion:
        """
        Retorna última versão de um modelo.
        
        Parâmetros
        ----------
        name : str
            Nome do modelo
        stage : str, optional
            Filtrar por stage ('Staging', 'Production', etc.)
            
        Returns
        -------
        model_version : ModelVersion
            Última versão
        """
        if stage:
            versions = self.client.get_latest_versions(name, stages=[stage])
        else:
            versions = self.client.search_model_versions(f"name='{name}'")
        
        if not versions:
            raise ValueError(f"Nenhuma versão encontrada para '{name}'")
        
        latest = versions[0]
        logger.info(f"📦 Última versão: {name} v{latest.version} ({latest.current_stage})")
        
        return latest
    
    def load_model(self, name: str, stage: str = 'Production'):
        """
        Carrega modelo de um stage específico.
        
        Parâmetros
        ----------
        name : str
            Nome do modelo
        stage : str
            Stage ('Staging', 'Production')
            
        Returns
        -------
        model : Any
            Modelo carregado
        """
        model_uri = f"models:/{name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"✅ Modelo carregado: {name} ({stage})")
        
        return model
    
    def compare_versions(
        self,
        name: str,
        version_1: int,
        version_2: int,
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Compara métricas entre duas versões.
        
        Parâmetros
        ----------
        name : str
            Nome do modelo
        version_1 : int
            Primeira versão
        version_2 : int
            Segunda versão
        metrics : List[str]
            Métricas para comparar
            
        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame com comparação
        """
        v1 = self.client.get_model_version(name, version_1)
        v2 = self.client.get_model_version(name, version_2)
        
        # Buscar runs
        run1 = self.client.get_run(v1.run_id)
        run2 = self.client.get_run(v2.run_id)
        
        comparison = []
        for metric in metrics:
            val1 = run1.data.metrics.get(metric, np.nan)
            val2 = run2.data.metrics.get(metric, np.nan)
            
            comparison.append({
                'metric': metric,
                f'v{version_1}': val1,
                f'v{version_2}': val2,
                'difference': val2 - val1,
                'improvement_%': ((val2 - val1) / val1 * 100) if val1 > 0 else np.nan
            })
        
        df = pd.DataFrame(comparison)
        
        logger.info(f"📊 Comparação: {name} v{version_1} vs v{version_2}")
        
        return df
    
    def delete_model_version(self, name: str, version: int):
        """Deleta uma versão do modelo."""
        self.client.delete_model_version(name, version)
        logger.info(f"🗑️ Modelo deletado: {name} v{version}")


# Funções auxiliares para logging completo
def log_training_session(
    tracker: MLflowTracker,
    run_name: str,
    model: Any,
    params: Dict,
    metrics: Dict,
    artifacts_dir: Optional[Path] = None,
    register_name: Optional[str] = None,
    tags: Optional[Dict] = None
):
    """
    Log completo de uma sessão de treinamento.
    
    Parâmetros
    ----------
    tracker : MLflowTracker
        Tracker do MLflow
    run_name : str
        Nome do run
    model : Any
        Modelo treinado
    params : Dict
        Hiperparâmetros
    metrics : Dict
        Métricas de avaliação
    artifacts_dir : Path, optional
        Diretório de artifacts
    register_name : str, optional
        Nome para Model Registry
    tags : Dict, optional
        Tags adicionais
    """
    with tracker.start_run(run_name=run_name, tags=tags):
        # Log de parâmetros
        tracker.log_params(params)
        
        # Log de métricas
        tracker.log_metrics(metrics)
        
        # Log de modelo
        tracker.log_model(
            model,
            artifact_path='model',
            registered_model_name=register_name
        )
        
        # Log de artifacts
        if artifacts_dir and artifacts_dir.exists():
            tracker.log_artifacts(str(artifacts_dir))
        
        run_id = tracker.get_run_id()
        logger.info(f"✅ Sessão completa logada: {run_name} (run_id: {run_id})")


# Exemplo de uso
if __name__ == "__main__":
    print("="*80)
    print("TESTE: MLflow Integration")
    print("="*80)
    
    # Setup
    tracker = MLflowTracker(experiment_name='aml-detection-test')
    registry = ModelRegistry()
    
    # Simular treinamento
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    print("\n1. Treinando modelo...")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Métricas
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    # Logging
    print("\n2. Logging no MLflow...")
    with tracker.start_run(run_name='random_forest_v1.0.0', tags={'version': '1.0.0', 'type': 'baseline'}):
        tracker.log_params({
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        })
        
        tracker.log_metrics({
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        })
        
        tracker.log_model(model, 'model', registered_model_name='aml-detector-test')
        
        run_id = tracker.get_run_id()
        print(f"   Run ID: {run_id}")
    
    # Buscar runs
    print("\n3. Buscando runs...")
    runs = tracker.search_runs(max_results=5)
    print(f"   Total de runs: {len(runs)}")
    
    # Model Registry
    print("\n4. Model Registry...")
    try:
        # Transição para Staging
        latest_version = registry.get_latest_version('aml-detector-test')
        registry.transition_model('aml-detector-test', latest_version.version, 'Staging')
        
        # Transição para Production
        registry.transition_model('aml-detector-test', latest_version.version, 'Production', archive_existing_versions=True)
        
        print(f"   Modelo promovido para Production: v{latest_version.version}")
    except Exception as e:
        print(f"   Erro: {e}")
    
    print("\n✅ Teste concluído!")
    print("   Acesse MLflow UI: http://localhost:5000")
