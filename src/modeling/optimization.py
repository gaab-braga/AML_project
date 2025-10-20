import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
        'random_state': 42,
        'eval_metric': 'auc',
        'verbosity': 0,
        'tree_method': 'hist',
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)
        scores.append(pr_auc)

        trial.report(pr_auc, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

def objective_lgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
        'random_state': 42,
        'verbosity': -1,
        'metric': 'auc',
        'boosting_type': 'gbdt'
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)
        scores.append(pr_auc)

        trial.report(pr_auc, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

def objective_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)
        scores.append(pr_auc)

        trial.report(pr_auc, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

def run_hyperparameter_optimization(X, y, n_trials=1):
    OPTUNA_CONFIG = {
        'n_trials': n_trials,
        'timeout': 600,
        'n_jobs': 1,
        'direction': 'maximize',
        'study_name': 'aml_hyperopt_2025',
        'load_if_exists': True
    }

    optuna_results = {}

    for model_name, objective_func in [('xgboost', objective_xgb), ('lightgbm', objective_lgb), ('random_forest', objective_rf)]:
        study = optuna.create_study(
            study_name=f'aml_{model_name}_2025',
            direction=OPTUNA_CONFIG['direction'],
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
            load_if_exists=OPTUNA_CONFIG['load_if_exists']
        )

        study.optimize(lambda trial: objective_func(trial, X, y), n_trials=OPTUNA_CONFIG['n_trials'], timeout=OPTUNA_CONFIG['timeout'], n_jobs=OPTUNA_CONFIG['n_jobs'])

        best_params = study.best_params
        best_score = study.best_value

        optuna_results[model_name] = {'best_params': best_params, 'best_score': best_score, 'study': study}

    best_model_optuna = max(optuna_results.items(), key=lambda x: x[1]['best_score'])
    return best_model_optuna, optuna_results