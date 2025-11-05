"""
Otimizador Optuna para modelos AML.
Busca automática de hiperparâmetros com validação temporal.
"""
import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from typing import Callable, Dict, Optional
import logging

from src.models.pipeline_factory import PipelineFactory


# Configurar logging do Optuna para reduzir verbosidade
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Otimizador de hiperparâmetros com Optuna e validação temporal.
    
    Features:
    - Cross-validation temporal (TimeSeriesSplit) para evitar data leakage
    - Pruning com Hyperband para eficiência computacional
    - Suporte para múltiplos modelos (XGBoost, LightGBM)
    - Callbacks customizados para logging limpo
    """
    
    def __init__(self, n_trials: int = 30, cv_splits: int = 5, random_seed: int = 42):
        """
        Inicializa otimizador.
        
        Args:
            n_trials: Número de trials para otimização
            cv_splits: Número de splits para validação temporal
            random_seed: Seed para reprodutibilidade
        """
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_seed = random_seed
        self.study = None
    
    def optimize(
        self,
        X_train,
        y_train,
        model_name: str,
        param_space_fn: Callable,
        scale_pos_weight: Optional[float] = None
    ) -> Dict:
        """
        Executa otimização de hiperparâmetros.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            model_name: Nome do modelo ('xgb' ou 'lgbm')
            param_space_fn: Função que define espaço de busca
            scale_pos_weight: Peso para classe positiva (para desbalanceamento)
            
        Returns:
            Dict com 'best_params' e 'best_value'
        """
        self.study = optuna.create_study(
            direction='maximize',
            pruner=HyperbandPruner()
        )
        self.study.set_user_attr('model_name', model_name)
        
        def objective(trial):
            # Obter parâmetros do espaço de busca
            params = param_space_fn(trial, model_name, self.random_seed, scale_pos_weight)
            
            # Validação cruzada temporal
            scores = self._cross_validate(X_train, y_train, model_name, params)
            
            return np.mean(scores)
        
        # Callback para logging limpo
        def print_callback(study, trial):
            model_upper = model_name.upper()
            print(f"\033[34m{model_upper}\033[0m | Trial {trial.number} | "
                  f"Score {trial.value:.4f} | Params {trial.params}")
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[print_callback],
            show_progress_bar=False
        )
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
    
    def _cross_validate(self, X_train, y_train, model_name: str, params: dict):
        """
        Valida modelo com TimeSeriesSplit.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            model_name: Nome do modelo
            params: Hiperparâmetros
            
        Returns:
            Lista de scores (PR-AUC) por fold
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Criar e treinar pipeline
            pipeline = PipelineFactory.create(model_name, params)
            pipeline.fit(X_train_fold, y_train_fold)
            
            # Avaliar com PR-AUC (ideal para desbalanceamento)
            y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            pr_auc = average_precision_score(y_val_fold, y_pred_proba)
            scores.append(pr_auc)
        
        return scores
    
    def get_study(self):
        """Retorna o study Optuna para visualizações."""
        return self.study


def get_param_space(trial, model_name: str, random_seed: int, scale_pos_weight: Optional[float]):
    """
    Define espaço de busca de hiperparâmetros.
    
    Args:
        trial: Trial do Optuna
        model_name: Nome do modelo ('xgb' ou 'lgbm')
        random_seed: Seed para reprodutibilidade
        scale_pos_weight: Peso para classe positiva
        
    Returns:
        Dict com hiperparâmetros sugeridos
    """
    if model_name == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1500, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'random_state': random_seed,
            'eval_metric': 'aucpr',
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
    
    elif model_name == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1500, step=100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'random_state': random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
    
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado")
    
    return params


# Função de compatibilidade para código legado
def objective(trial, model_name: str):
    """
    Wrapper de compatibilidade. Use OptunaOptimizer para novo código.
    """
    raise NotImplementedError(
        "Use OptunaOptimizer.optimize() ao invés desta função standalone"
    )
