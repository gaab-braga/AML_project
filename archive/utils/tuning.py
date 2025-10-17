"""
Hyperparameter Tuning and Model Selection Module.

This module consolidates four major components for production ML tuning:

1. NESTED CROSS-VALIDATION (NestedCrossValidator class)
   - Unbiased performance estimation with nested CV structure
   - Outer Loop (5 folds) - Estimates true generalization performance
   - Inner Loop (3 folds) - Hyperparameter tuning with Optuna
   - Prevents overfitting to validation set
   - Reference: Cawley & Talbot (2010)

2. STAGED TUNING STRATEGY (run_staged_tuning)
   - Two-stage approach: Coarse exploration → Fine refinement
   - Gating between stages based on improvement detection
   - Early abort for unpromising configurations
   - Production-optimized for time/performance trade-off

3. OPTUNA OBJECTIVES (create_*_objective functions)
   - LightGBM, XGBoost, CatBoost objective builders
   - Cross-validation integrated into objective
   - Automatic parameter space sampling
   - Pruning support via MedianPruner

4. CANDIDATE GATING (apply_gating_criteria)
   - Filters baseline models for tuning investment
   - Configurable thresholds (retention, stability, absolute minimum)
   - Statistical significance testing
   - JSON persistence for audit trail

Key Features:
- Unbiased performance estimation
- Parallel execution support
- Statistical significance testing
- Comprehensive reporting
- Progress tracking with visual feedback
- Compatible with Optuna, GridSearch, RandomSearch

Author: Data Science Team
Date: October 2025
"""

__all__ = [
    # Nested CV
    'NestedCVResult',
    'NestedCrossValidator',
    'compare_models_nested_cv',
    
    # Staged Tuning
    'run_staged_tuning',
    'print_progress',
    '_progress_states',  # Global state for progress tracking
    
    # Optuna Objectives
    'create_lgbm_objective',
    'create_xgb_objective',
    'create_catboost_objective',
    
    # Candidate Gating
    'apply_gating',
    'apply_gating_criteria',
    'save_candidate_results',
    'load_candidate_results',
    'format_gating_summary',
    'get_tuning_priority',
]

# Standard library
import json
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

# Data science stack
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer, roc_auc_score

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Statistical testing
from scipy import stats
from scipy.stats import wilcoxon

# ML libraries (imported dynamically in objectives)
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

# Internal imports (relative)
try:
    from modeling import FraudMetrics, get_cv_strategy
except ImportError:
    from .modeling import FraudMetrics, get_cv_strategy

# IPython (optional, for notebook progress bars)
try:
    from IPython.display import display, HTML, clear_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    display = None
    HTML = None
    clear_output = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class NestedCVResult:
    """Container for nested CV results."""
    outer_scores: List[float]
    inner_best_params: List[Dict]
    outer_fold_details: List[Dict]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    best_params_frequency: Dict
    execution_time: float
    metadata: Dict


class NestedCrossValidator:
    """
    Nested Cross-Validation for unbiased model selection and evaluation.
    
    Parameters
    ----------
    outer_cv : int or CV splitter, default=5
        Number of folds for outer loop (performance estimation)
    inner_cv : int or CV splitter, default=3
        Number of folds for inner loop (hyperparameter tuning)
    metric : str or callable, default='roc_auc'
        Scoring metric to optimize
    n_trials : int, default=20
        Number of Optuna trials per inner fold
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed)
        
    Examples
    --------
    >>> from nested_cv import NestedCrossValidator
    >>> from lightgbm import LGBMClassifier
    >>> 
    >>> def objective(trial):
    ...     return {
    ...         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
    ...         'max_depth': trial.suggest_int('max_depth', 3, 10)
    ...     }
    >>> 
    >>> ncv = NestedCrossValidator(outer_cv=5, inner_cv=3)
    >>> result = ncv.fit(X, y, LGBMClassifier, objective)
    >>> print(f"Mean Score: {result.mean_score:.4f} ± {result.std_score:.4f}")
    """
    
    def __init__(
        self,
        outer_cv: int = 5,
        inner_cv: int = 3,
        metric: str = 'roc_auc',
        n_trials: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.metric = metric
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_class: type,
        param_space: Callable,
        fixed_params: Optional[Dict] = None,
        metric_fn: Optional[Callable] = None
    ) -> NestedCVResult:
        """
        Run nested cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_class : class
            Model class (e.g., LGBMClassifier)
        param_space : callable
            Function that takes optuna.Trial and returns param dict
        fixed_params : dict, optional
            Fixed parameters not subject to tuning
        metric_fn : callable, optional
            Custom metric function with signature: metric_fn(y_true, y_pred) -> float
            
        Returns
        -------
        NestedCVResult
            Results object with scores, parameters, and statistics
        """
        start_time = datetime.now()
        
        if fixed_params is None:
            fixed_params = {}
        
        # Setup CV splitters
        if isinstance(self.outer_cv, int):
            outer_splitter = StratifiedKFold(
                n_splits=self.outer_cv,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            outer_splitter = self.outer_cv
        
        if isinstance(self.inner_cv, int):
            inner_splitter = StratifiedKFold(
                n_splits=self.inner_cv,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            inner_splitter = self.inner_cv
        
        # Storage for results
        outer_scores = []
        inner_best_params = []
        outer_fold_details = []
        
        logger.info(f"Starting Nested CV: {self.outer_cv} outer × {self.inner_cv} inner folds")
        logger.info(f"Trials per inner fold: {self.n_trials}")
        
        # Outer loop - Performance estimation
        for outer_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y), 1):
            if self.verbose >= 1:
                logger.info(f"\n{'='*60}")
                logger.info(f"Outer Fold {outer_idx}/{self.outer_cv}")
                logger.info(f"{'='*60}")
            
            X_train_outer = X.iloc[train_idx]
            y_train_outer = y.iloc[train_idx]
            X_test_outer = X.iloc[test_idx]
            y_test_outer = y.iloc[test_idx]
            
            # Inner loop - Hyperparameter tuning
            best_params = self._inner_optimization(
                X_train_outer,
                y_train_outer,
                model_class,
                param_space,
                fixed_params,
                inner_splitter,
                outer_idx
            )
            
            # Train final model with best params on full outer training set
            model = model_class(**{**fixed_params, **best_params})
            model.fit(X_train_outer, y_train_outer)
            
            # Evaluate on outer test set
            if metric_fn:
                y_pred = model.predict_proba(X_test_outer)[:, 1]
                score = metric_fn(y_test_outer, y_pred)
            else:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(self.metric)
                score = scorer(model, X_test_outer, y_test_outer)
            
            outer_scores.append(score)
            inner_best_params.append(best_params)
            
            # Store detailed results
            fold_detail = {
                'fold': outer_idx,
                'score': score,
                'best_params': best_params,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_fraud_rate': y_train_outer.mean(),
                'test_fraud_rate': y_test_outer.mean()
            }
            outer_fold_details.append(fold_detail)
            
            if self.verbose >= 1:
                logger.info(f"Outer Fold {outer_idx} Score: {score:.4f}")
                logger.info(f"Best params: {best_params}")
        
        # Calculate statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        # 95% confidence interval
        from scipy import stats
        ci = stats.t.interval(
            0.95,
            len(outer_scores) - 1,
            loc=mean_score,
            scale=stats.sem(outer_scores)
        )
        
        # Find most frequent parameter values
        best_params_frequency = self._analyze_param_frequency(inner_best_params)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create result object
        result = NestedCVResult(
            outer_scores=outer_scores,
            inner_best_params=inner_best_params,
            outer_fold_details=outer_fold_details,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=ci,
            best_params_frequency=best_params_frequency,
            execution_time=execution_time,
            metadata={
                'outer_cv': self.outer_cv,
                'inner_cv': self.inner_cv,
                'n_trials': self.n_trials,
                'metric': self.metric,
                'random_state': self.random_state,
                'model_class': model_class.__name__
            }
        )
        
        # Print summary
        if self.verbose >= 1:
            self._print_summary(result)
        
        return result
    
    def _inner_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_class: type,
        param_space: Callable,
        fixed_params: Dict,
        inner_splitter,
        outer_idx: int
    ) -> Dict:
        """
        Run hyperparameter optimization on inner folds.
        """
        if self.verbose >= 2:
            logger.info(f"  Inner optimization: {self.n_trials} trials × {self.inner_cv} folds")
        
        def objective(trial):
            # Get parameters from search space
            params = param_space(trial)
            
            # Cross-validate on inner folds
            inner_scores = []
            
            for inner_idx, (train_idx_inner, val_idx_inner) in enumerate(
                inner_splitter.split(X_train, y_train)
            ):
                X_train_inner = X_train.iloc[train_idx_inner]
                y_train_inner = y_train.iloc[train_idx_inner]
                X_val_inner = X_train.iloc[val_idx_inner]
                y_val_inner = y_train.iloc[val_idx_inner]
                
                # Train model
                model = model_class(**{**fixed_params, **params})
                model.fit(X_train_inner, y_train_inner)
                
                # Evaluate
                scorer = get_scorer(self.metric)
                score = scorer(model, X_val_inner, y_val_inner)
                inner_scores.append(score)
            
            return np.mean(inner_scores)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state + outer_idx)
        )
        
        # Suppress Optuna logs if verbose < 2
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.verbose >= 2 else optuna.logging.WARNING
        )
        
        # Run optimization
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        
        if self.verbose >= 2:
            logger.info(f"  Best inner score: {study.best_value:.4f}")
        
        return study.best_params
    
    def _analyze_param_frequency(self, params_list: List[Dict]) -> Dict:
        """
        Analyze frequency of parameter values across folds.
        """
        param_freq = {}
        
        for params in params_list:
            for key, value in params.items():
                if key not in param_freq:
                    param_freq[key] = {}
                
                # Convert to string for counting
                value_str = str(value)
                if value_str not in param_freq[key]:
                    param_freq[key][value_str] = 0
                
                param_freq[key][value_str] += 1
        
        # Find most common values
        most_common = {}
        for key, counts in param_freq.items():
            most_common[key] = max(counts.items(), key=lambda x: x[1])
        
        return most_common
    
    def _print_summary(self, result: NestedCVResult):
        """
        Print summary of nested CV results.
        """
        logger.info(f"\n{'='*60}")
        logger.info("NESTED CROSS-VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {result.metadata['model_class']}")
        logger.info(f"Metric: {result.metadata['metric']}")
        logger.info(f"Outer CV: {result.metadata['outer_cv']} folds")
        logger.info(f"Inner CV: {result.metadata['inner_cv']} folds")
        logger.info(f"Trials per fold: {result.metadata['n_trials']}")
        logger.info(f"\n{'='*60}")
        logger.info("PERFORMANCE")
        logger.info(f"{'='*60}")
        logger.info(f"Mean Score: {result.mean_score:.4f} ± {result.std_score:.4f}")
        logger.info(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        logger.info(f"Score Range: [{min(result.outer_scores):.4f}, {max(result.outer_scores):.4f}]")
        logger.info(f"\nScores by Fold:")
        for i, score in enumerate(result.outer_scores, 1):
            logger.info(f"  Fold {i}: {score:.4f}")
        
        logger.info(f"\n{'='*60}")
        logger.info("MOST COMMON PARAMETERS")
        logger.info(f"{'='*60}")
        for param, (value, count) in result.best_params_frequency.items():
            logger.info(f"  {param}: {value} ({count}/{len(result.outer_scores)} folds)")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Execution Time: {result.execution_time:.1f}s")
        logger.info(f"{'='*60}\n")


def compare_models_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models_config: Dict[str, Dict],
    outer_cv: int = 5,
    inner_cv: int = 3,
    n_trials: int = 20,
    metric: str = 'roc_auc',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare multiple models using nested cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    models_config : dict
        Dictionary with model configurations:
        {
            'model_name': {
                'model_class': ModelClass,
                'param_space': callable,
                'fixed_params': dict
            }
        }
    outer_cv, inner_cv : int
        Number of CV folds
    n_trials : int
        Optuna trials per inner fold
    metric : str
        Scoring metric
    random_state : int
        Random seed
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with statistics
        
    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> from xgboost import XGBClassifier
    >>> 
    >>> models_config = {
    ...     'LightGBM': {
    ...         'model_class': LGBMClassifier,
    ...         'param_space': lambda trial: {...},
    ...         'fixed_params': {'random_state': 42}
    ...     },
    ...     'XGBoost': {
    ...         'model_class': XGBClassifier,
    ...         'param_space': lambda trial: {...},
    ...         'fixed_params': {'random_state': 42}
    ...     }
    ... }
    >>> 
    >>> comparison = compare_models_nested_cv(X, y, models_config)
    """
    results = {}
    
    ncv = NestedCrossValidator(
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        metric=metric,
        n_trials=n_trials,
        random_state=random_state,
        verbose=1
    )
    
    for model_name, config in models_config.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'#'*60}")
        
        result = ncv.fit(
            X, y,
            model_class=config['model_class'],
            param_space=config['param_space'],
            fixed_params=config.get('fixed_params', {})
        )
        
        results[model_name] = result
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Mean_Score': result.mean_score,
            'Std_Score': result.std_score,
            'CI_Lower': result.confidence_interval[0],
            'CI_Upper': result.confidence_interval[1],
            'Min_Score': min(result.outer_scores),
            'Max_Score': max(result.outer_scores),
            'CV_Coefficient': result.std_score / result.mean_score if result.mean_score != 0 else np.inf,
            'Execution_Time_s': result.execution_time
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_Score', ascending=False)
    
    # Statistical significance testing
    comparison_df = _add_statistical_tests(comparison_df, results)
    
    return comparison_df


def _add_statistical_tests(comparison_df: pd.DataFrame, results: Dict) -> pd.DataFrame:
    """
    Add pairwise statistical significance tests.
    """
    from scipy.stats import wilcoxon
    
    model_names = list(results.keys())
    if len(model_names) < 2:
        return comparison_df
    
    # Wilcoxon signed-rank test (paired samples)
    best_model = comparison_df.iloc[0]['Model']
    
    significance = []
    for model_name in comparison_df['Model']:
        if model_name == best_model:
            significance.append('Best')
        else:
            # Compare with best model
            scores_best = results[best_model].outer_scores
            scores_current = results[model_name].outer_scores
            
            if len(scores_best) == len(scores_current):
                _, pvalue = wilcoxon(scores_best, scores_current)
                if pvalue < 0.01:
                    significance.append('*** (p<0.01)')
                elif pvalue < 0.05:
                    significance.append('** (p<0.05)')
                elif pvalue < 0.10:
                    significance.append('* (p<0.10)')
                else:
                    significance.append('n.s.')
            else:
                significance.append('N/A')
    
    comparison_df['Significance_vs_Best'] = significance
    
    return comparison_df


if __name__ == "__main__":
    print("Nested Cross-Validation Module")
    print("=" * 60)
    print("\nUsage Example:")
    print("""
# from nested_cv import NestedCrossValidator  # Module not implemented yet
from lightgbm import LGBMClassifier

def lgbm_param_space(trial):
    return {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

# ncv = NestedCrossValidator(outer_cv=5, inner_cv=3, n_trials=20)
# result = ncv.fit(X, y, LGBMClassifier, lgbm_param_space, 
#                  fixed_params={'random_state': 42, 'class_weight': 'balanced'})

# print(f"Score: {result.mean_score:.4f} ± {result.std_score:.4f}")
    """)


# ================================================================================
# STAGED TUNING STRATEGY
# ================================================================================
# Two-stage tuning with gating and early abort for production efficiency.
# ================================================================================


# Global state to track progress bars per model
_progress_states = {}

def print_progress(current: int, total: int, model_name: str, phase: str = "Coarse",
                   best_score: Optional[float] = None, best_params: Optional[Dict] = None,
                   start_time: Optional[float] = None, trial_info: Optional[Dict] = None):
    """
    Mostra progresso inteligente do tuning focado em early stopping e aprendizado mínimo.

    Em vez de barra até 100%, mostra:
    - Tendência de melhoria (↗ melhorando, → estabilizando, ↘ piorando)
    - Tempo decorrido e trials realizados
    - Status de pruning/early stopping
    - Melhor score encontrado até o momento

    Args:
        current: Trial atual
        total: Total de trials planejados
        model_name: Nome do modelo (LightGBM, XGBoost, CatBoost)
        phase: Fase do tuning ("Coarse" ou "Fine")
        best_score: Melhor score até o momento
        best_params: Parâmetros do melhor trial (opcional)
        start_time: Timestamp do início do tuning (opcional)
        trial_info: Informações adicionais do trial (opcional)

    Example:
        >>> print_progress(5, 10, "LightGBM", "Coarse", 0.3941, {'lr': 0.05, 'depth': 6})
        LightGBM [Coarse] ↗ 0.3941 | 5/10 trials | 2.3s
          ↳ lr=0.05, depth=6, leaves=31
    """
    # Use global HAS_IPYTHON flag set at module level
    use_html = HAS_IPYTHON

    # Initialize state for this model if needed
    if model_name not in _progress_states:
        _progress_states[model_name] = {
            'last_output': None,
            'completed': False,
            'start_time': start_time or datetime.now().timestamp(),
            'last_best_score': None,
            'trend': '→',  # Default: estabilizando
            'trial_history': [],
            'last_time_update': datetime.now().timestamp(),
            'display_seconds': 0
        }

    state = _progress_states[model_name]

    # Calculate elapsed time (independent counter)
    current_time = datetime.now().timestamp()
    elapsed_time = current_time - state['start_time']

    # Update display seconds independently (every 0.1 seconds)
    if current_time - state['last_time_update'] >= 0.1:
        state['display_seconds'] = int(elapsed_time)
        state['last_time_update'] = current_time

    # Format time display with independent seconds counter
    if elapsed_time < 60:
        # Show seconds as integer counter
        time_str = f"{state['display_seconds']}s"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        time_str = f"{hours}h {minutes}m"

    # Determine trend based on score improvement
    trend = '→'  # Default: estabilizando
    if best_score is not None:
        if state['last_best_score'] is None:
            trend = '↗'  # First score
        elif best_score > state['last_best_score']:
            trend = '↗'  # Melhorando
        elif best_score < state['last_best_score']:
            trend = '↘'  # Piorando
        # else: mantém → (estabilizando)

        state['last_best_score'] = best_score

    # Get trial information
    trial_status = ""
    if trial_info:
        if trial_info.get('pruned', False):
            trial_status = " 🪓"  # Pruning symbol
        elif trial_info.get('failed', False):
            trial_status = " ❌"  # Failed symbol

    # Calculate trials per minute rate
    trials_per_minute = current / max(elapsed_time / 60, 0.01)  # Avoid division by zero

    if use_html:
        # HTML com paleta amarela suave e discreta
        # Cores dark-mode palette (amber theme)
        trend_color = {
            '↗': '#10b981',  # green for improving
            '→': '#f59e0b',  # amber for stable
            '↘': '#ef4444'   # red for worsening
        }.get(trend, '#f59e0b')

        score_info = f' | <span style="background-color: #2d2416; color: #f59e0b; padding: 2px 6px; border-radius: 4px; font-weight: 500; border-left: 2px solid #f59e0b;">{best_score:.4f}</span>' if best_score else ""

        # Enhanced time display with independent seconds counter
        if elapsed_time < 60:
            time_info = f' | <span style="color: #10b981; font-size: 0.9em; font-weight: 500; font-family: monospace;">{state["display_seconds"]}s</span>'
        else:
            time_info = f' | <span style="color: #6b7280; font-size: 0.9em;">{time_str}</span>'
        
        trials_info = f' | <span style="color: #9ca3af; font-size: 0.9em;">{current}/{total} trials</span>'
        rate_info = f' | <span style="color: #9ca3af; font-size: 0.8em;">{trials_per_minute:.1f} trials/min</span>'

        # Trial status indicator
        status_indicator = ""
        if trial_status:
            if "🪓" in trial_status:
                status_indicator = f' <span style="color: #ef4444; font-size: 0.8em;">{trial_status.strip()}</span>'
            elif "❌" in trial_status:
                status_indicator = f' <span style="color: #dc2626; font-size: 0.8em;">{trial_status.strip()}</span>'

        # Format parameters for display (compact, top 3-4 most important)
        params_html = ""
        if best_params:
            # Select most relevant params based on model type
            param_priority = {
                'lightgbm': ['learning_rate', 'max_depth', 'num_leaves', 'min_child_samples'],
                'xgboost': ['learning_rate', 'max_depth', 'min_child_weight', 'subsample'],
                'catboost': ['learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']
            }

            priority_keys = param_priority.get(model_name.lower(), list(best_params.keys())[:4])
            selected_params = {k: best_params[k] for k in priority_keys if k in best_params}

            # Format params compactly
            params_str = ", ".join([
                f"{k.replace('_', '')}={v:.3f}" if isinstance(v, float) else f"{k.replace('_', '')}={v}"
                for k, v in list(selected_params.items())[:4]
            ])

            if params_str:
                params_html = f'<div style="margin-left: 20px; font-size: 0.85em; color: #9ca3af; font-family: monospace;">↳ {params_str}</div>'

        html_output = f'<div style="margin-bottom: 8px;"><strong>{model_name}</strong> [{phase}] <span style="color: {trend_color}; font-weight: bold; font-size: 1.1em;">{trend}</span>{score_info}{trials_info}{rate_info}{time_info}{status_indicator}{params_html}</div>'

        # Update state
        state['last_output'] = html_output

        # Build complete output with all model progress bars
        all_outputs = []
        for m_name in _progress_states:
            if _progress_states[m_name]['last_output']:
                all_outputs.append(_progress_states[m_name]['last_output'])

        # Display all progress bars together
        clear_output(wait=True)
        display(HTML(''.join(all_outputs)))

        # Mark as completed if done
        if current == total:
            state['completed'] = True
    else:
        # Fallback: output simples sem cores
        score_info = f" {best_score:.4f}" if best_score else ""
        trials_info = f" | {current}/{total} trials"
        rate_info = f" | {trials_per_minute:.1f} trials/min"

        params_str = ""
        if best_params:
            # Show top 3 params in terminal
            params_list = list(best_params.items())[:3]
            params_str = " | " + ", ".join([
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params_list
            ])

        output = f"{model_name} [{phase}] {trend}{score_info}{trials_info}{rate_info} | {time_str}{trial_status}{params_str}"

        # Only print if this is a new line (not overwriting)
        if state['last_output'] != output:
            print(output)
            state['last_output'] = output

        if current == total:
            state['completed'] = True


def print_progress_legacy(current: int, total: int, model_name: str, phase: str = "Coarse", 
                        best_score: Optional[float] = None, best_params: Optional[Dict] = None,
                        start_time: Optional[float] = None, trial_info: Optional[Dict] = None):
    """
    LEGACY: Mostra progresso de tuning com barra até 100% (comportamento antigo).
    
    Esta função mantém o comportamento original com barra de progresso até 100%,
    mas não é mais recomendada para cenários de early stopping.
    
    Args:
        current: Trial atual
        total: Total de trials
        model_name: Nome do modelo (LightGBM, XGBoost, CatBoost)
        phase: Fase do tuning ("Coarse" ou "Fine")
        best_score: Melhor score até o momento
        best_params: Parâmetros do melhor trial (opcional)
        start_time: Timestamp do início do tuning (opcional)
        trial_info: Informações adicionais do trial (opcional)
        
    Example:
        >>> print_progress_legacy(5, 10, "LightGBM", "Coarse", 0.3941, {'lr': 0.05, 'depth': 6})
        LightGBM [Coarse] ███████████████░░░░░░░░░░░░░░░ 50% (5/10) | Best: 0.3941 | 2.3s
          ↳ lr=0.05, depth=6, leaves=31
    """
    # Use global HAS_IPYTHON flag set at module level
    use_html = HAS_IPYTHON
    
    # Initialize state for this model if needed
    if model_name not in _progress_states:
        _progress_states[model_name] = {
            'last_output': None,
            'completed': False,
            'start_time': start_time or datetime.now().timestamp()
        }
    
    state = _progress_states[model_name]
    
    # Calculate progress
    percentage = (current / total) * 100
    bar_length = 30
    filled = int(bar_length * current / total)
    
    # Calculate elapsed time
    current_time = datetime.now().timestamp()
    elapsed_time = current_time - state['start_time']
    
    # Format time display
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.1f}s"
    elif elapsed_time < 3600:
        time_str = f"{elapsed_time/60:.1f}m"
    else:
        time_str = f"{elapsed_time/3600:.1f}h"
    
    # Calculate trials per minute rate
    trials_per_minute = current / max(elapsed_time / 60, 0.01)  # Avoid division by zero

    # Get trial information
    trial_status = ""
    if trial_info:
        if trial_info.get('pruned', False):
            trial_status = " 🪓"  # Pruning symbol
        elif trial_info.get('failed', False):
            trial_status = " ❌"  # Failed symbol

    if use_html:
        # HTML com paleta amarela suave e discreta
        # Cores dark-mode palette (amber theme)
        bar_filled = f'<span style="color: #f59e0b;">{"█" * filled}</span>'
        bar_empty = f'<span style="color: #3c3c3c;">{"░" * (bar_length - filled)}</span>' 
        bar = bar_filled + bar_empty
        
        # Enhanced score and time info
        score_info = f' | <span style="background-color: #2d2416; color: #f59e0b; padding: 2px 6px; border-radius: 4px; font-weight: 500; border-left: 2px solid #f59e0b;">Best: {best_score:.4f}</span>' if best_score else ""
        time_info = f' | <span style="color: #6b7280; font-size: 0.9em;">{time_str}</span>'
        
        # Trial status indicator
        status_indicator = ""
        if trial_status:
            if "🪓" in trial_status:
                status_indicator = f' <span style="color: #ef4444; font-size: 0.8em;">{trial_status.strip()}</span>'
            elif "❌" in trial_status:
                status_indicator = f' <span style="color: #dc2626; font-size: 0.8em;">{trial_status.strip()}</span>'
        
        # Format parameters for display (compact, top 3-4 most important)
        params_html = ""
        if best_params:
            # Select most relevant params based on model type
            param_priority = {
                'lightgbm': ['learning_rate', 'max_depth', 'num_leaves', 'min_child_samples'],
                'xgboost': ['learning_rate', 'max_depth', 'min_child_weight', 'subsample'],
                'catboost': ['learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']
            }
            
            priority_keys = param_priority.get(model_name.lower(), list(best_params.keys())[:4])
            selected_params = {k: best_params[k] for k in priority_keys if k in best_params}
            
            # Format params compactly
            params_str = ", ".join([
                f"{k.replace('_', '')}={v:.3f}" if isinstance(v, float) else f"{k.replace('_', '')}={v}"
                for k, v in list(selected_params.items())[:4]
            ])
            
            if params_str:
                params_html = f'<div style="margin-left: 20px; font-size: 0.85em; color: #9ca3af; font-family: monospace;">↳ {params_str}</div>'
        
        html_output = f'<div style="margin-bottom: 8px;"><strong>{model_name}</strong> [{phase}] {bar} <span style="color: #10b981;">{percentage:.0f}%</span> ({current}/{total}){score_info}{time_info}{status_indicator}{params_html}</div>'
        
        # Update state
        state['last_output'] = html_output
        
        # Build complete output with all model progress bars
        all_outputs = []
        for m_name in _progress_states:
            if _progress_states[m_name]['last_output']:
                all_outputs.append(_progress_states[m_name]['last_output'])
        
        # Display all progress bars together
        clear_output(wait=True)
        display(HTML(''.join(all_outputs)))
        
        # Mark as completed if done
        if current == total:
            state['completed'] = True
    else:
        # Fallback: output simples sem cores
        bar = '█' * filled + '░' * (bar_length - filled)
        score_info = f" | Best: {best_score:.4f}" if best_score else ""
        time_info = f" | {time_str}"
        
        params_str = ""
        if best_params:
            # Show top 3 params in terminal
            params_list = list(best_params.items())[:3]
            params_str = " | " + ", ".join([
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params_list
            ])
        
        output = f"{model_name} [{phase}] {bar} {percentage:.0f}% ({current}/{total}){score_info}{time_info}{trial_status}{params_str}"
        
        # Only print if this is a new line (not overwriting)
        if state['last_output'] != output:
            print(output)
            state['last_output'] = output
        
        if current == total:
            state['completed'] = True


def create_lgbm_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy,
    param_space: Dict[str, tuple],
    metric_fn: Callable,
    base_params: Optional[Dict] = None
) -> Callable:
    """
    Create Optuna objective for LightGBM.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv_strategy: CV splitter
        param_space: Parameter search space
        metric_fn: Scoring function
        base_params: Fixed base parameters
        
    Returns:
        Objective function for Optuna
    """
    from lightgbm import LGBMClassifier
    
    def objective(trial):
        try:
            # Sample hyperparameters
            params = {
                'num_leaves': trial.suggest_int('num_leaves', *param_space.get('num_leaves', (10, 300))),
                'learning_rate': trial.suggest_float('learning_rate', *param_space.get('learning_rate', (0.01, 0.3)), log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', *param_space.get('feature_fraction', (0.4, 1.0))),
                'bagging_fraction': trial.suggest_float('bagging_fraction', *param_space.get('bagging_fraction', (0.4, 1.0))),
                'min_child_samples': trial.suggest_int('min_child_samples', *param_space.get('min_child_samples', (5, 100))),
                'reg_alpha': trial.suggest_float('reg_alpha', *param_space.get('reg_alpha', (0, 10))),
                'reg_lambda': trial.suggest_float('reg_lambda', *param_space.get('reg_lambda', (0, 10))),
                'verbosity': -1,
                'force_col_wise': True
            }
            
            # Merge with base params
            if base_params:
                params = {**base_params, **params}
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_tr = y_train.iloc[train_idx]
                y_val = y_train.iloc[val_idx]
                
                model = LGBMClassifier(**params, n_estimators=500)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)]
                )
                
                y_proba = model.predict_proba(X_val)[:, 1]
                score = metric_fn(y_val.values, y_proba)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    return objective


def create_catboost_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy,
    param_space: Dict[str, tuple],
    metric_fn: Callable,
    base_params: Optional[Dict] = None
) -> Callable:
    """Create Optuna objective for CatBoost."""
    from catboost import CatBoostClassifier
    
    def objective(trial):
        try:
            params = {
                'depth': trial.suggest_int('depth', *param_space.get('depth', (4, 10))),
                'learning_rate': trial.suggest_float('learning_rate', *param_space.get('learning_rate', (0.01, 0.3)), log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *param_space.get('l2_leaf_reg', (1, 10))),
                'random_strength': trial.suggest_float('random_strength', *param_space.get('random_strength', (0, 10))),
                'bagging_temperature': trial.suggest_float('bagging_temperature', *param_space.get('bagging_temperature', (0, 1))),
                'border_count': trial.suggest_int('border_count', *param_space.get('border_count', (32, 255))),
                'verbose': False
            }
            
            if base_params:
                params = {**base_params, **params}
            
            cv_scores = []
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_tr = y_train.iloc[train_idx]
                y_val = y_train.iloc[val_idx]
                
                model = CatBoostClassifier(**params, iterations=500, early_stopping_rounds=50)
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
                
                y_proba = model.predict_proba(X_val)[:, 1]
                score = metric_fn(y_val.values, y_proba)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        except Exception:
            return 0.0
    
    return objective


def create_xgb_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy,
    param_space: Dict[str, tuple],
    metric_fn: Callable,
    base_params: Optional[Dict] = None
) -> Callable:
    """Create Optuna objective for XGBoost."""
    from xgboost import XGBClassifier
    
    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', *param_space.get('max_depth', (3, 10))),
                'learning_rate': trial.suggest_float('learning_rate', *param_space.get('learning_rate', (0.01, 0.3)), log=True),
                'subsample': trial.suggest_float('subsample', *param_space.get('subsample', (0.5, 1.0))),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *param_space.get('colsample_bytree', (0.5, 1.0))),
                'min_child_weight': trial.suggest_int('min_child_weight', *param_space.get('min_child_weight', (1, 10))),
                'reg_alpha': trial.suggest_float('reg_alpha', *param_space.get('reg_alpha', (0, 10))),
                'reg_lambda': trial.suggest_float('reg_lambda', *param_space.get('reg_lambda', (0, 10))),
                'verbosity': 0,
                'use_label_encoder': False
            }
            
            if base_params:
                params = {**base_params, **params}
            
            cv_scores = []
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_tr = y_train.iloc[train_idx]
                y_val = y_train.iloc[val_idx]
                
                model = XGBClassifier(**params, n_estimators=500, early_stopping_rounds=50)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                y_proba = model.predict_proba(X_val)[:, 1]
                score = metric_fn(y_val.values, y_proba)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        except Exception:
            return 0.0
    
    return objective


def run_staged_tuning(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    metric: str = 'pr_auc',
    cv_strategy = None,
    param_space: Optional[Dict] = None,
    base_params: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run staged hyperparameter tuning with gating.
    
    Two-stage approach:
    1. Coarse exploration (10-15 trials)
    2. Fine refinement (20-30 trials) if improvement detected
    
    Args:
        model_name: 'LightGBM', 'XGBoost', or 'RandomForest'
        X_train: Training features
        y_train: Training labels
        config: Tuning configuration
        metric: Optimization metric
        cv_strategy: CV splitter (optional)
        param_space: Custom parameter space
        base_params: Fixed parameters
        progress_callback: Optional callback(current, total, model, phase, best_score, best_params, start_time, trial_info)
        
    Returns:
        Dictionary with best parameters and tuning history
    """
    from utils.modeling import FraudMetrics, get_cv_strategy
    
    # Setup
    if cv_strategy is None:
        cv_folds = config.get('cv_folds', 3)
        cv_strategy = get_cv_strategy('stratified', n_splits=cv_folds, random_state=config.get('random_state', 42))
    
    # Metric function
    if metric == 'pr_auc':
        metric_fn = FraudMetrics.pr_auc_score
    else:
        from sklearn.metrics import roc_auc_score
        metric_fn = roc_auc_score
    
    # Create objective
    if model_name.lower() == 'lightgbm':
        objective = create_lgbm_objective(
            X_train, y_train, cv_strategy, 
            param_space or {}, metric_fn, base_params
        )
    elif model_name.lower() == 'xgboost':
        objective = create_xgb_objective(
            X_train, y_train, cv_strategy,
            param_space or {}, metric_fn, base_params
        )
    elif model_name.lower() == 'catboost':
        objective = create_catboost_objective(
            X_train, y_train, cv_strategy,
            param_space or {}, metric_fn, base_params
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported for staged tuning")
    
    # Stage 1: Coarse exploration
    n_trials_coarse = config.get('n_trials_coarse', 10)
    logger.info(f"🔍 Stage 1: Coarse exploration ({n_trials_coarse} trials)")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config.get('random_state', 42)),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Start time for progress tracking
    tuning_start_time = datetime.now().timestamp()
    
    # Callback para progresso
    if progress_callback:
        def optuna_callback(study, trial):
            # Get trial information
            trial_info = {}
            # Check if trial was pruned or failed (compatible with different Optuna versions)
            try:
                if hasattr(trial, 'state'):
                    # Try different ways to check trial state
                    if hasattr(optuna, 'TrialState'):
                        if trial.state == optuna.TrialState.PRUNED:
                            trial_info['pruned'] = True
                        elif trial.state == optuna.TrialState.FAIL:
                            trial_info['failed'] = True
                    else:
                        # Fallback: check trial value and other attributes
                        if trial.value is None:
                            trial_info['failed'] = True
            except (AttributeError, Exception):
                # If we can't determine state, continue without trial_info
                pass
            
            progress_callback(
                trial.number + 1, 
                n_trials_coarse, 
                model_name, 
                "Coarse", 
                study.best_value,
                study.best_params,
                start_time=tuning_start_time,
                trial_info=trial_info
            )
        
        study.optimize(objective, n_trials=n_trials_coarse, show_progress_bar=False, callbacks=[optuna_callback])
    else:
        study.optimize(objective, n_trials=n_trials_coarse, show_progress_bar=False)
    
    coarse_best = study.best_value
    logger.info(f"  Coarse best: {coarse_best:.4f}")
    
    # Stage 2: Fine refinement (if improvement detected)
    min_improvement = config.get('min_improvement_for_fine', 0.003)
    n_trials_fine = config.get('n_trials_fine', 20)
    
    # Simple baseline: mean of first few trials
    baseline = np.mean([t.value for t in study.trials[:3] if t.value is not None])
    improvement = coarse_best - baseline
    
    if improvement > min_improvement:
        logger.info(f"✓ Improvement detected ({improvement:.4f}), starting fine tuning...")
        logger.info(f"🎯 Stage 2: Fine refinement ({n_trials_fine} trials)")
        
        # Callback para progresso fine
        if progress_callback:
            def optuna_callback_fine(study, trial):
                # Get trial information
                trial_info = {}
                # Check if trial was pruned or failed (compatible with different Optuna versions)
                try:
                    if hasattr(trial, 'state'):
                        # Try different ways to check trial state
                        if hasattr(optuna, 'TrialState'):
                            if trial.state == optuna.TrialState.PRUNED:
                                trial_info['pruned'] = True
                            elif trial.state == optuna.TrialState.FAIL:
                                trial_info['failed'] = True
                        else:
                            # Fallback: check trial value and other attributes
                            if trial.value is None:
                                trial_info['failed'] = True
                except (AttributeError, Exception):
                    # If we can't determine state, continue without trial_info
                    pass
                
                progress_callback(
                    trial.number - n_trials_coarse + 1, 
                    n_trials_fine, 
                    model_name, 
                    "Fine", 
                    study.best_value,
                    study.best_params,
                    start_time=tuning_start_time,
                    trial_info=trial_info
                )
            
            study.optimize(objective, n_trials=n_trials_fine, show_progress_bar=False, callbacks=[optuna_callback_fine])
        else:
            study.optimize(objective, n_trials=n_trials_fine, show_progress_bar=False)
        
        final_best = study.best_value
        logger.info(f"  Fine best: {final_best:.4f}")
    else:
        logger.info(f"⚠️ Minimal improvement ({improvement:.4f}), skipping fine tuning")
        final_best = coarse_best
    
    # Results
    results = {
        'best_params': study.best_params,
        'best_score': final_best,
        'n_trials': len(study.trials),
        'stages': {
            'coarse': {
                'n_trials': n_trials_coarse,
                'best_score': coarse_best
            }
        },
        'trial_history': [
            {'trial': t.number, 'score': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ]
    }
    
    if improvement > min_improvement:
        results['stages']['fine'] = {
            'n_trials': n_trials_fine,
            'best_score': final_best,
            'improvement': final_best - coarse_best
        }
    
    logger.info(f"✅ Tuning complete: {model_name} → {final_best:.4f}")
    
    return results


def apply_gating(
    baseline_results: Dict[str, Dict],
    config: Dict[str, Any]
) -> list:
    """
    Apply gating criteria to select models for tuning.
    
    Args:
        baseline_results: Results from baseline evaluation
        config: Gating configuration
        
    Returns:
        List of model names that pass gating
    """
    pr_auc_retention = config.get('pr_auc_retention', 0.80)
    max_cv_stability = config.get('max_cv_stability', 0.20)
    max_candidates = config.get('max_candidates', 3)
    min_pr_auc_absolute = config.get('min_pr_auc_absolute', 0.004)
    
    logger.info("Applying gating criteria...")
    
    # Find best PR-AUC
    best_pr_auc = max(
        r.get('pr_auc_mean', 0) 
        for r in baseline_results.values() 
        if 'error' not in r
    )
    
    threshold = best_pr_auc * pr_auc_retention
    
    candidates = []
    for model_name, results in baseline_results.items():
        if 'error' in results:
            continue
        
        pr_auc_mean = results.get('pr_auc_mean', 0)
        pr_auc_std = results.get('pr_auc_std', 0)
        
        # Check criteria
        if pr_auc_mean < min_pr_auc_absolute:
            logger.info(f"  ❌ {model_name}: Below absolute threshold ({pr_auc_mean:.4f})")
            continue
        
        if pr_auc_mean < threshold:
            logger.info(f"  ❌ {model_name}: Below retention threshold ({pr_auc_mean:.4f} < {threshold:.4f})")
            continue
        
        cv_stability = pr_auc_std / pr_auc_mean if pr_auc_mean > 0 else 1.0
        if cv_stability > max_cv_stability:
            logger.info(f"  ❌ {model_name}: Unstable CV (CV={cv_stability:.2f})")
            continue
        
        candidates.append({
            'model': model_name,
            'score': pr_auc_mean,
            'stability': cv_stability
        })
        logger.info(f"  ✓ {model_name}: Passed gating (PR-AUC={pr_auc_mean:.4f})")
    
    # Sort and limit
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:max_candidates]
    
    selected = [c['model'] for c in candidates]
    logger.info(f"✅ {len(selected)} models selected for tuning: {selected}")
    
    return selected


# ================================================================================
# CANDIDATE GATING UTILITIES
# ================================================================================
# Logic for filtering and selecting models for tuning based on 
# baseline performance, stability, and configurable thresholds.
# ================================================================================


def apply_gating_criteria(df_models: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply gating criteria to select candidates for tuning.
    
    Args:
        df_models: DataFrame with baseline model results
        config: Configuration with gating parameters
        
    Returns:
        Dictionary with candidates, rejected models, and criteria applied
    """
    # Get gating config with defaults
    gating_config = config.get('modeling', {}).get('gating', {})
    
    pr_auc_retention = gating_config.get('pr_auc_retention', 0.90)  # Keep models >= 90% of best
    max_cv_stability = gating_config.get('max_cv_stability', 0.15)  # Max coef. variation 15%
    max_candidates = gating_config.get('max_candidates', 3)  # Limit tuning to top 3
    min_pr_auc_absolute = gating_config.get('min_pr_auc_absolute', 0.01)  # Absolute floor
    exclude_dummy = gating_config.get('exclude_dummy', True)  # Skip DummyClassifier from tuning
    
    # Filter out invalid models
    valid_models = df_models.dropna(subset=['PR_AUC']).copy()
    
    if exclude_dummy:
        valid_models = valid_models[valid_models['Model'] != 'DummyClassifier'].copy()
    
    if valid_models.empty:
        return {
            'candidates': pd.DataFrame(),
            'rejected': [],
            'criteria': {},
            'summary': "No valid models found"
        }
    
    # Find best performance
    best_pr_auc = valid_models['PR_AUC'].max()
    pr_auc_threshold = best_pr_auc * pr_auc_retention
    
    # Apply gating criteria
    candidates = []
    rejected = []
    
    for _, row in valid_models.iterrows():
        model_name = row['Model']
        variant = row['Variant']
        pr_auc = row['PR_AUC']
        cv_stability = row.get('CV_Stability', np.nan)
        
        # Check criteria
        reasons = []
        
        # PR_AUC retention check
        if pr_auc < pr_auc_threshold:
            reasons.append(f"PR_AUC {pr_auc:.4f} < {pr_auc_retention:.1%} * best ({pr_auc_threshold:.4f})")
        
        # Absolute minimum check
        if pr_auc < min_pr_auc_absolute:
            reasons.append(f"PR_AUC {pr_auc:.4f} < absolute minimum ({min_pr_auc_absolute})")
        
        # Stability check (if available)
        if pd.notna(cv_stability) and cv_stability > max_cv_stability:
            reasons.append(f"CV_Stability {cv_stability:.3f} > maximum ({max_cv_stability})")
        
        if reasons:
            rejected.append({
                'model': model_name,
                'variant': variant,
                'pr_auc': float(pr_auc),
                'cv_stability': float(cv_stability) if pd.notna(cv_stability) else None,
                'reasons': reasons
            })
        else:
            candidates.append(row)
    
    # Convert to DataFrame and limit count
    candidates_df = pd.DataFrame(candidates)
    if len(candidates_df) > max_candidates:
        # Sort by PR_AUC desc, CV_Stability asc (if available), then take top N
        sort_cols = ['PR_AUC']
        sort_ascending = [False]
        
        if 'CV_Stability' in candidates_df.columns:
            sort_cols.append('CV_Stability')
            sort_ascending.append(True)
        
        candidates_df = candidates_df.sort_values(sort_cols, ascending=sort_ascending)
        
        # Move excess to rejected
        excess = candidates_df.iloc[max_candidates:].copy()
        for _, row in excess.iterrows():
            rejected.append({
                'model': row['Model'],
                'variant': row['Variant'],
                'pr_auc': float(row['PR_AUC']),
                'cv_stability': float(row.get('CV_Stability', np.nan)) if pd.notna(row.get('CV_Stability')) else None,
                'reasons': [f"Exceeded max_candidates limit ({max_candidates})"]
            })
        
        candidates_df = candidates_df.iloc[:max_candidates].copy()
    
    # Summary
    summary = f"Selected {len(candidates_df)}/{len(valid_models)} candidates for tuning"
    
    criteria_applied = {
        'pr_auc_retention': pr_auc_retention,
        'pr_auc_threshold': float(pr_auc_threshold),
        'best_pr_auc': float(best_pr_auc),
        'max_cv_stability': max_cv_stability,
        'max_candidates': max_candidates,
        'min_pr_auc_absolute': min_pr_auc_absolute,
        'exclude_dummy': exclude_dummy
    }
    
    return {
        'candidates': candidates_df,
        'rejected': rejected,
        'criteria': criteria_applied,
        'summary': summary
    }


def save_candidate_results(gating_result: Dict[str, Any], artifacts_dir: Path) -> Path:
    """
    Save gating results to JSON file.
    
    Args:
        gating_result: Result from apply_gating_criteria
        artifacts_dir: Directory to save file
        
    Returns:
        Path to saved file
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)
    
    # Prepare data for JSON serialization
    candidates_list = []
    if not gating_result['candidates'].empty:
        for _, row in gating_result['candidates'].iterrows():
            candidate = {
                'model': row['Model'],
                'variant': row['Variant'],
                'pr_auc': float(row['PR_AUC']),
                'roc_auc': float(row.get('ROC_AUC', np.nan)),
                'f1': float(row.get('F1', np.nan)),
                'cv_pr_auc_mean': float(row.get('CV_PR_AUC_Mean', np.nan)),
                'cv_pr_auc_std': float(row.get('CV_PR_AUC_Std', np.nan)),
                'cv_stability': float(row.get('CV_Stability', np.nan)),
                'train_time_sec': float(row.get('Train_Time_Sec', np.nan)),
                'selected_for_tuning': True
            }
            # Clean NaNs
            for k, v in candidate.items():
                if isinstance(v, float) and np.isnan(v):
                    candidate[k] = None
            candidates_list.append(candidate)
    
    output_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'summary': gating_result['summary'],
        'criteria_applied': gating_result['criteria'],
        'candidates_count': len(candidates_list),
        'rejected_count': len(gating_result['rejected']),
        'candidates': candidates_list,
        'rejected': gating_result['rejected']
    }
    
    json_path = artifacts_dir / 'baseline_candidates.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return json_path


def load_candidate_results(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Load gating results from JSON file.
    
    Args:
        artifacts_dir: Directory containing the file
        
    Returns:
        Dictionary with candidate data or None if not found
    """
    json_path = Path(artifacts_dir) / 'baseline_candidates.json'
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def format_gating_summary(gating_result: Dict[str, Any]) -> str:
    """
    Format gating results as readable summary.
    
    Args:
        gating_result: Result from apply_gating_criteria
        
    Returns:
        Formatted summary string
    """
    lines = [
        "🚪 CANDIDATE GATING RESULTS:",
        f"   {gating_result['summary']}",
        ""
    ]
    
    criteria = gating_result['criteria']
    lines.extend([
        "📋 Criteria Applied:",
        f"   PR_AUC retention: >= {criteria['pr_auc_retention']:.1%} of best ({criteria['pr_auc_threshold']:.4f})",
        f"   Max CV stability: <= {criteria['max_cv_stability']:.1%}",
        f"   Max candidates: {criteria['max_candidates']}",
        f"   Absolute minimum: >= {criteria['min_pr_auc_absolute']:.3f}",
        ""
    ])
    
    if not gating_result['candidates'].empty:
        lines.append("✅ SELECTED FOR TUNING:")
        for _, row in gating_result['candidates'].iterrows():
            stability_str = f" (CV_Stab: {row.get('CV_Stability', 0):.3f})" if pd.notna(row.get('CV_Stability')) else ""
            lines.append(f"   🎯 {row['Model']} {row['Variant']}: PR_AUC={row['PR_AUC']:.4f}{stability_str}")
    
    if gating_result['rejected']:
        lines.extend(["", "❌ REJECTED:"])
        for rejected in gating_result['rejected']:
            reasons = "; ".join(rejected['reasons'])
            lines.append(f"   ⚠ {rejected['model']} {rejected['variant']}: {reasons}")
    
    return "\n".join(lines)


def get_tuning_priority(candidates_data: Dict[str, Any]) -> List[str]:
    """
    Get ordered list of model names for tuning based on priority.
    
    Args:
        candidates_data: Loaded candidate results
        
    Returns:
        List of model names in priority order
    """
    if not candidates_data or not candidates_data.get('candidates'):
        return []
    
    # Already sorted by gating process, just extract names
    priority_list = []
    for candidate in candidates_data['candidates']:
        model_key = f"{candidate['model']}"  # Could extend to include variant if needed
        if model_key not in priority_list:  # Avoid duplicates
            priority_list.append(model_key)
    
    return priority_list