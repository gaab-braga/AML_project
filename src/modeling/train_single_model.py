"""
Função unificada para treinamento de modelos AML individuais.
Consolida toda a lógica de treinamento, cache e avaliação em uma única função.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import time
import logging
import sys
from datetime import datetime

# Configuração centralizada de logging para evitar duplicatas
def setup_logging():
    """Configura logging centralizado para o projeto AML."""
    # Desabilitar logging completamente para reduzir verbosidade
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

# Configurar logging na importação
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Apenas erros

def train_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: Dict[str, Any],
    artifacts_dir: Path,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Treina um modelo AML individual com sistema completo de cache, validação temporal
    e avaliação focada em compliance regulatória.

    Esta função consolida toda a lógica de treinamento que estava espalhada em múltiplas
    células do notebook, seguindo as diretrizes do roadmap de refatoração.

    Args:
        X: Features de entrada (DataFrame)
        y: Target (Series com labels de fraude)
        model_name: Nome do modelo ('xgboost', 'lightgbm', 'random_forest')
        config: Configuração experimental completa
        artifacts_dir: Diretório para salvar artefatos e cache
        force_retrain: Se True, força retreinamento mesmo com cache disponível

    Returns:
        Dict com resultados completos: modelo, métricas CV, avaliação, compliance

    Raises:
        ValueError: Se modelo não suportado ou configuração inválida
    """
    start_time = time.time()

    # Validar entradas
    if model_name not in config['models']:
        raise ValueError(f"Modelo '{model_name}' não encontrado na configuração")

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X deve ser DataFrame e y deve ser Series")

    # Verificar cache primeiro
    if not force_retrain and _check_model_cache(model_name, artifacts_dir):
        try:
            return _load_model_from_cache(model_name, artifacts_dir, X, y)
        except Exception as e:
            pass

    # Importar trainer AML (lazy import para evitar dependências circulares)
    try:
        from src.modeling.aml_modeling import AMLModelTrainer
    except ImportError:
        raise ImportError("AMLModelTrainer não encontrado. Verifique src/modeling/aml_modeling.py")

    # Inicializar trainer
    trainer = AMLModelTrainer(config)

    try:
        # Treinamento com validação temporal
        training_results = trainer.train_with_temporal_cv(X, y, model_name)

        # Avaliação completa
        evaluation_results = trainer.evaluate_model(X, y, model_name)

        # Salvar em cache
        try:
            _save_model_to_cache(model_name, trainer.trained_models[model_name],
                               training_results, evaluation_results, artifacts_dir)
        except Exception as e:
            pass

        # Métricas resumidas para comparação
        cv_metrics = training_results['cv_results']
        threshold_metrics = evaluation_results['threshold_analysis']
        optimal_metrics = threshold_metrics[threshold_metrics['threshold'] ==
                                          evaluation_results['optimal_threshold']].iloc[0]

        # Resultado consolidado
        result = {
            'model_name': model_name,
            'pipeline': trainer.trained_models[model_name],
            'training_results': training_results,
            'evaluation_results': evaluation_results,

            # Métricas resumidas para comparação de modelos
            'comparison_metrics': {
                'model': model_name.upper(),
                'roc_auc_cv': cv_metrics['roc_auc']['mean'],
                'roc_auc_test': evaluation_results['roc_auc'],
                'pr_auc': evaluation_results['pr_auc'],
                'recall_cv': cv_metrics['recall']['mean'],
                'recall_test': optimal_metrics['recall'],
                'precision_test': optimal_metrics['precision'],
                'f1_test': optimal_metrics['f1'],
                'optimal_threshold': evaluation_results['optimal_threshold'],
                'fraud_rate': optimal_metrics['predicted_fraud_rate'],
                'compliant': evaluation_results['regulatory_compliance']['overall_compliant']
            },

            # Metadata
            'training_time_seconds': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'config_used': config['models'][model_name],
            'data_shape': {'X': X.shape, 'y': y.shape}
        }

        training_time = time.time() - start_time

        return result

    except Exception as e:
        raise


def _check_model_cache(model_name: str, artifacts_dir: Path) -> bool:
    """Verifica se modelo e metadata existem no cache."""
    models_dir = artifacts_dir / 'trained_models'
    model_path = models_dir / f'aml_model_{model_name}.pkl'
    metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'
    return model_path.exists() and metadata_path.exists()


def _load_model_from_cache(model_name: str, artifacts_dir: Path, X: pd.DataFrame = None, y: pd.Series = None) -> Dict[str, Any]:
    """Carrega modelo e resultados do cache."""
    import joblib
    import json

    models_dir = artifacts_dir / 'trained_models'
    model_path = models_dir / f'aml_model_{model_name}.pkl'
    metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'

    # Carregar modelo
    model_data = joblib.load(model_path)

    # Carregar metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Reconstruir resultado no formato esperado
    result = {
        'model_name': model_name,
        'pipeline': model_data,
        'training_results': metadata.get('training_results', {}),
        'evaluation_results': metadata.get('evaluation_results', {}),
        'comparison_metrics': _extract_comparison_metrics(metadata, model_name),
        'training_time_seconds': metadata.get('training_time_seconds', 0),
        'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
        'config_used': metadata.get('config_used', {}),
        'data_shape': metadata.get('data_shape', {}),
        'cached': True
    }

    # Se probabilities não estão no evaluation_results, recalcular
    if 'probabilities' not in result['evaluation_results'] and X is not None:
        try:
            y_pred_proba = result['pipeline'].predict_proba(X)[:, 1]
            result['evaluation_results']['probabilities'] = y_pred_proba
        except Exception as e:
            print(f"Warning: Could not recalculate probabilities for cached model {model_name}: {e}")
    elif 'probabilities' in result['evaluation_results']:
        # Se probabilities estão como lista (de JSON), converter para numpy array
        if isinstance(result['evaluation_results']['probabilities'], list):
            result['evaluation_results']['probabilities'] = np.array(result['evaluation_results']['probabilities'])

    return result


def _save_model_to_cache(model_name: str, model: Any, training_results: Dict,
                        evaluation_results: Dict, artifacts_dir: Path) -> None:
    """Salva modelo e metadata no cache."""
    import joblib
    import json

    models_dir = artifacts_dir / 'trained_models'
    models_dir.mkdir(exist_ok=True, parents=True)

    model_path = models_dir / f'aml_model_{model_name}.pkl'
    metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'

    # Salvar modelo
    joblib.dump(model, model_path)

    # Preparar metadata serializável
    metadata = {
        'training_results': _make_serializable(training_results),
        'evaluation_results': _make_serializable(evaluation_results),
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name
    }

    # Salvar metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _extract_comparison_metrics(metadata: Dict, model_name: str) -> Dict[str, Any]:
    """Extrai métricas de comparação da metadata."""
    eval_results = metadata.get('evaluation_results', {})
    training_results = metadata.get('training_results', {})

    # Valores padrão
    default_metrics = {
        'model': model_name.upper(),
        'roc_auc_cv': 0.0,
        'roc_auc_test': 0.0,
        'pr_auc': 0.0,
        'recall_cv': 0.0,
        'recall_test': 0.0,
        'precision_test': 0.0,
        'f1_test': 0.0,
        'optimal_threshold': 0.5,
        'fraud_rate': 0.0,
        'compliant': False
    }

    try:
        # Extrair de evaluation_results
        roc_auc_test = eval_results.get('roc_auc', 0.0)
        pr_auc = eval_results.get('pr_auc', 0.0)
        optimal_threshold = eval_results.get('optimal_threshold', 0.5)

        # Extrair threshold analysis
        threshold_analysis = eval_results.get('threshold_analysis', [])
        if isinstance(threshold_analysis, list) and threshold_analysis:
            # Encontrar métricas do threshold ótimo
            optimal_data = None
            for threshold_data in threshold_analysis:
                if isinstance(threshold_data, dict) and threshold_data.get('threshold') == optimal_threshold:
                    optimal_data = threshold_data
                    break

            if optimal_data:
                recall_test = optimal_data.get('recall', 0.0)
                precision_test = optimal_data.get('precision', 0.0)
                f1_test = optimal_data.get('f1', 0.0)
                fraud_rate = optimal_data.get('predicted_fraud_rate', 0.0)
            else:
                recall_test = precision_test = f1_test = fraud_rate = 0.0
        else:
            recall_test = precision_test = f1_test = fraud_rate = 0.0

        # Extrair de training_results
        cv_results = training_results.get('cv_results', {})
        roc_auc_cv = cv_results.get('roc_auc', {}).get('mean', 0.0)
        recall_cv = cv_results.get('recall', {}).get('mean', 0.0)

        # Compliance
        regulatory_compliance = eval_results.get('regulatory_compliance', {})
        compliant = regulatory_compliance.get('overall_compliant', False)

        return {
            'model': model_name.upper(),
            'roc_auc_cv': roc_auc_cv,
            'roc_auc_test': roc_auc_test,
            'pr_auc': pr_auc,
            'recall_cv': recall_cv,
            'recall_test': recall_test,
            'precision_test': precision_test,
            'f1_test': f1_test,
            'optimal_threshold': optimal_threshold,
            'fraud_rate': fraud_rate,
            'compliant': compliant
        }

    except Exception as e:
        return default_metrics


def _make_serializable(obj: Any) -> Any:
    """Converte objeto para formato serializável em JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        # Para outros objetos, tentar converter para string
        try:
            return str(obj)
        except:
            return None