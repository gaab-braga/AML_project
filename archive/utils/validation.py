"""
Model Validation and Monitoring Module.

This MEGA-MODULE consolidates 6 major validation components for production ML:

1. ADVERSARIAL VALIDATION (AdversarialValidator)
   - Detects data drift between train and test distributions
   - Trains classifier to distinguish train vs test samples
   - If AUC > 0.5, there's significant distribution shift
   - Identifies top drift-causing features
   
   Interpretation:
   - AUC ≈ 0.5: No drift (model can't distinguish)
   - AUC > 0.6: Moderate drift (caution!)
   - AUC > 0.7: Severe drift (DO NOT deploy)

2. WALK-FORWARD BACKTESTING (WalkForwardBacktester)
   - Temporal validation with proper time-series splits
   - Simulates production deployment over time
   - Detects performance degradation
   - Prevents data leakage with purge periods

3. VALIDATION SUMMARY GENERATOR (create_validation_summary)
   - Comprehensive model validation reports
   - Performance metrics with confidence intervals
   - Calibration assessment (ECE, Brier score)
   - Expected Value analysis
   - Data integrity checks

4. DRIFT MONITORING (population_stability_index)
   - PSI calculation for feature/score drift
   - Classification metrics tracking
   - Feature-level shift detection
   - Score distribution monitoring

5. PERFORMANCE ALERTING (PerformanceMonitor)
   - Automated threshold-based alerts
   - Email/Slack/Webhook notifications
   - Cooldown periods to prevent spam
   - Alert history and resolution tracking

6. TEMPORAL VALIDATION (validate_temporal_split_formal)
   - Formal validation of temporal splits
   - Prevents temporal leakage
   - Entity overlap detection
   - Gap analysis between train/test

Author: Data Science Team
Date: October 2025
Phase: 3.4 - Monitoring and Governance
"""

__all__ = [
    # Adversarial Validation
    'AdversarialValidationResult',
    'AdversarialValidator',
    'detect_drift_by_feature',
    'detect_drift_all_features',
    
    # Walk-Forward Backtesting
    'BacktestConfig',
    'WalkForwardBacktester',
    'save_backtest_results',
    'run_backtest_for_pipeline',
    
    # Validation Summary
    'create_validation_summary',
    'save_validation_summary',
    
    # Drift Monitoring
    'population_stability_index',
    'classification_metrics',
    'score_shift_report',
    'feature_shift_table',
    
    # Performance Alerting
    'AlertThreshold',
    'AlertEvent',
    'PerformanceMonitor',
    'create_performance_monitoring_system',
    'simulate_performance_monitoring_demo',
    
    # Temporal Validation
    'validate_temporal_split_formal',
    'save_temporal_validation_report',
    'run_temporal_validation_for_pipeline',
]

# Standard library
import json
import logging
import warnings
import smtplib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Data science stack
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Statistical testing
from scipy.stats import ks_2samp

# ML libraries
import lightgbm as lgb

# Internal imports (relative)
try:
    from metrics import bootstrap_metric, compute_calibration_metrics
    from production import check_data_integrity
except ImportError:
    from .metrics import bootstrap_metric, compute_calibration_metrics
    from .production import check_data_integrity

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# COMPONENT 1: ADVERSARIAL VALIDATION
# ================================================================================
# Detects data drift between train and test distributions
# ================================================================================


@dataclass
class AdversarialValidationResult:
    """Resultado da validação adversarial."""
    auc: float
    drift_severity: str  # 'No Drift', 'Moderate', 'Severe'
    top_drift_features: List[Tuple[str, float]]
    feature_importance_full: pd.DataFrame
    predictions: np.ndarray
    y_true: np.ndarray
    
    def __str__(self) -> str:
        return f"""
Adversarial Validation Result
==============================
AUC: {self.auc:.4f}
Drift Severity: {self.drift_severity}

Top 10 Features Causing Drift:
{chr(10).join(f"  {i+1}. {feat}: {imp:.4f}" for i, (feat, imp) in enumerate(self.top_drift_features[:10]))}
"""


class AdversarialValidator:
    """
    Validador adversarial para detectar data drift.
    
    Parâmetros
    ----------
    model_type : str, default='lightgbm'
        Tipo de modelo ('lightgbm', 'random_forest')
    n_folds : int, default=5
        Número de folds para cross-validation
    random_state : int, default=42
        Seed para reprodutibilidade
        
    Exemplo
    -------
    >>> validator = AdversarialValidator()
    >>> result = validator.validate(X_train, X_test)
    >>> print(f"AUC: {result.auc:.4f} - {result.drift_severity}")
    >>> validator.plot_results(result)
    """
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.model = None
        
        logger.info(f"🔍 AdversarialValidator inicializado: {model_type}")
    
    def validate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> AdversarialValidationResult:
        """
        Executa validação adversarial.
        
        Parâmetros
        ----------
        X_train : pd.DataFrame
            Dados de treino
        X_test : pd.DataFrame
            Dados de teste
        feature_names : List[str], optional
            Nomes das features (se None, usa colunas)
            
        Returns
        -------
        result : AdversarialValidationResult
            Resultado da validação
        """
        logger.info(f"🔍 Iniciando adversarial validation...")
        logger.info(f"   Train: {X_train.shape[0]:,} samples")
        logger.info(f"   Test: {X_test.shape[0]:,} samples")
        
        # Preparar dados
        X_combined, y_combined = self._prepare_data(X_train, X_test)
        
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Cross-validation
        predictions = np.zeros(len(y_combined))
        feature_importance = np.zeros(X_combined.shape[1])
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_fold_train = X_combined[train_idx]
            y_fold_train = y_combined[train_idx]
            X_fold_val = X_combined[val_idx]
            
            # Treinar modelo
            model = self._train_model(X_fold_train, y_fold_train)
            
            # Predições
            predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
            
            # Importância (acumulada)
            feature_importance += self._get_feature_importance(model)
            
            logger.info(f"   Fold {fold+1}/{self.n_folds} concluído")
        
        # Média de importância
        feature_importance /= self.n_folds
        
        # Calcular AUC
        auc = roc_auc_score(y_combined, predictions)
        
        # Classificar severidade
        drift_severity = self._classify_drift_severity(auc)
        
        # Top features
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        top_features = list(zip(
            feature_importance_df['feature'].head(20).values,
            feature_importance_df['importance'].head(20).values
        ))
        
        logger.info(f"✅ Adversarial Validation concluída:")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Drift Severity: {drift_severity}")
        
        return AdversarialValidationResult(
            auc=auc,
            drift_severity=drift_severity,
            top_drift_features=top_features,
            feature_importance_full=feature_importance_df,
            predictions=predictions,
            y_true=y_combined
        )
    
    def _prepare_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados: train=0, test=1."""
        # Garantir mesmo formato
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        
        # Combinar
        X_combined = np.vstack([X_train, X_test])
        
        # Labels: train=0, test=1
        y_combined = np.hstack([
            np.zeros(len(X_train)),
            np.ones(len(X_test))
        ])
        
        return X_combined, y_combined
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Treina modelo classificador."""
        if self.model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Modelo '{self.model_type}' não suportado")
        
        model.fit(X, y)
        return model
    
    def _get_feature_importance(self, model) -> np.ndarray:
        """Extrai importância de features."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            raise ValueError("Modelo não possui feature_importances_")
    
    def _classify_drift_severity(self, auc: float) -> str:
        """Classifica severidade do drift."""
        if auc < 0.55:
            return 'No Drift'
        elif auc < 0.65:
            return 'Mild Drift'
        elif auc < 0.75:
            return 'Moderate Drift'
        else:
            return 'Severe Drift'
    
    def plot_results(
        self,
        result: AdversarialValidationResult,
        save_path: Optional[str] = None
    ):
        """
        Visualiza resultados da validação adversarial.
        
        Parâmetros
        ----------
        result : AdversarialValidationResult
            Resultado da validação
        save_path : str, optional
            Caminho para salvar figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROC Curve
        ax = axes[0, 0]
        fpr, tpr, _ = roc_curve(result.y_true, result.predictions)
        ax.plot(fpr, tpr, label=f'AUC = {result.auc:.4f}', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Train vs Test Classification', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Adicionar indicador de drift
        drift_color = {
            'No Drift': 'green',
            'Mild Drift': 'yellow',
            'Moderate Drift': 'orange',
            'Severe Drift': 'red'
        }
        ax.text(
            0.6, 0.2,
            f'Drift: {result.drift_severity}',
            fontsize=14,
            fontweight='bold',
            color=drift_color.get(result.drift_severity, 'black'),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # 2. Feature Importance (Top 15)
        ax = axes[0, 1]
        top_15 = result.feature_importance_full.head(15).copy()
        ax.barh(range(len(top_15)), top_15['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels(top_15['feature'].values)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 15 Features Causing Drift', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Predictions Distribution
        ax = axes[1, 0]
        train_preds = result.predictions[result.y_true == 0]
        test_preds = result.predictions[result.y_true == 1]
        
        ax.hist(train_preds, bins=50, alpha=0.6, label='Train', color='blue', density=True)
        ax.hist(test_preds, bins=50, alpha=0.6, label='Test', color='red', density=True)
        ax.set_xlabel('Predicted Probability (Test=1)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Summary Text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
Adversarial Validation Summary
================================

AUC Score: {result.auc:.4f}
Drift Severity: {result.drift_severity}

Interpretation:
---------------
"""
        
        if result.auc < 0.55:
            summary_text += """
✅ No significant drift detected
   Train and test distributions are similar
   Model should generalize well
"""
        elif result.auc < 0.65:
            summary_text += """
⚠️ Mild drift detected
   Some distribution differences exist
   Monitor model performance closely
"""
        elif result.auc < 0.75:
            summary_text += """
⚠️ Moderate drift detected
   Significant distribution differences
   Consider retraining or domain adaptation
"""
        else:
            summary_text += """
🚨 SEVERE DRIFT DETECTED
   Train and test are very different
   DO NOT deploy model without investigation
   Retrain with more representative data
"""
        
        summary_text += f"""

Top 5 Drift-Causing Features:
------------------------------
"""
        for i, (feat, imp) in enumerate(result.top_drift_features[:5], 1):
            summary_text += f"{i}. {feat}: {imp:.4f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Plot salvo: {save_path}")
        
        plt.show()


def detect_drift_by_feature(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_col: str
) -> Dict[str, float]:
    """
    Detecta drift em uma feature específica usando KS statistic.
    
    Parâmetros
    ----------
    X_train : pd.DataFrame
        Dados de treino
    X_test : pd.DataFrame
        Dados de teste
    feature_col : str
        Nome da feature
        
    Returns
    -------
    stats : Dict[str, float]
        Estatísticas de drift (KS, p-value)
    """
    from scipy.stats import ks_2samp
    
    train_values = X_train[feature_col].dropna()
    test_values = X_test[feature_col].dropna()
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(train_values, test_values)
    
    return {
        'feature': feature_col,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'drift_detected': p_value < 0.05
    }


def detect_drift_all_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Detecta drift em todas as features.
    
    Parâmetros
    ----------
    X_train : pd.DataFrame
        Dados de treino
    X_test : pd.DataFrame
        Dados de teste
    top_n : int, default=20
        Top N features com maior drift
        
    Returns
    -------
    drift_df : pd.DataFrame
        DataFrame com estatísticas de drift por feature
    """
    logger.info("🔍 Detectando drift em todas as features...")
    
    results = []
    
    for col in X_train.columns:
        # Somente features numéricas
        if pd.api.types.is_numeric_dtype(X_train[col]):
            try:
                stats = detect_drift_by_feature(X_train, X_test, col)
                results.append(stats)
            except Exception as e:
                logger.warning(f"   Erro em '{col}': {e}")
    
    drift_df = pd.DataFrame(results)
    drift_df = drift_df.sort_values('ks_statistic', ascending=False)
    
    n_drift = drift_df['drift_detected'].sum()
    logger.info(f"✅ {n_drift}/{len(drift_df)} features com drift significativo (p < 0.05)")
    
    return drift_df.head(top_n)


# Exemplo de uso
if __name__ == "__main__":
    print("="*80)
    print("TESTE: Adversarial Validation")
    print("="*80)
    
    # Simular dados
    np.random.seed(42)
    
    # Train: distribuição original
    n_train = 5000
    X_train = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.exponential(2, n_train),
        'feature_3': np.random.uniform(-5, 5, n_train),
        'feature_4': np.random.normal(10, 2, n_train),
        'feature_5': np.random.poisson(3, n_train)
    })
    
    # Test: COM DRIFT em feature_2 e feature_4
    n_test = 2000
    X_test = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_test),  # SEM drift
        'feature_2': np.random.exponential(4, n_test),  # COM DRIFT (escala diferente)
        'feature_3': np.random.uniform(-5, 5, n_test),  # SEM drift
        'feature_4': np.random.normal(15, 2, n_test),  # COM DRIFT (média diferente)
        'feature_5': np.random.poisson(3, n_test)  # SEM drift
    })
    
    print("\n1. Adversarial Validation (modelo classificador)")
    print("-" * 80)
    
    validator = AdversarialValidator(model_type='lightgbm')
    result = validator.validate(X_train, X_test)
    
    print(result)
    
    # Plot
    validator.plot_results(result, save_path='adversarial_validation_test.png')
    
    print("\n2. Drift por Feature (KS Test)")
    print("-" * 80)
    
    drift_df = detect_drift_all_features(X_train, X_test)
    print(drift_df.to_string(index=False))
    
    print("\n✅ Teste concluído!")
    print("   Esperado: feature_2 e feature_4 com drift significativo")
    print("   Esperado: AUC > 0.6 (drift moderado)")
"""
Backtesting Walk-Forward Module

Comprehensive backtesting with walk-forward validation for time series
to validate model stability and performance over time.
"""

# import json  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from datetime import datetime, timedelta  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, List, Tuple, Optional, Union  # Already imported at top
# from dataclasses import dataclass  # Already imported at top
# import warnings  # Already imported at top
# from sklearn.model_selection import TimeSeriesSplit  # Already imported at top
# from sklearn.metrics import precision_recall_curve, auc  # Already imported at top
# from sklearn.calibration import CalibratedClassifierCV  # Already imported at top
# from sklearn.preprocessing import StandardScaler  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #2

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class BacktestConfig:
    """Configuration for backtesting setup."""
    initial_train_months: int = 12
    step_months: int = 1
    test_months: int = 3
    max_windows: int = 12
    min_train_samples: int = 1000
    min_test_samples: int = 100
    purge_days: int = 0  # Gap between train and test to prevent leakage
    date_column: str = 'Timestamp'
    target_column: str = 'Is_Laundering'


class WalkForwardBacktester:
    """Walk-forward backtesting with proper temporal validation."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = []
        self.summary_stats = {}
        
    def create_temporal_windows(
        self, 
        data: pd.DataFrame,
        date_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create walk-forward temporal windows.
        
        Args:
            data: DataFrame with temporal data
            date_column: Name of date column (uses config default if None)
            
        Returns:
            List of window configurations with train/test indices
        """
        date_col = date_column or self.config.date_column
        
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        # Ensure dates are parsed
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(date_col).reset_index(drop=True)
        
        # Get date range
        min_date = data[date_col].min()
        max_date = data[date_col].max()
        
        print(f"📅 Creating walk-forward windows: {min_date.date()} to {max_date.date()}")
        
        windows = []
        current_start = min_date
        window_id = 1
        
        while window_id <= self.config.max_windows:
            # Define train period
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=self.config.initial_train_months)
            
            # Define test period (with purge gap)
            test_start = train_end + pd.Timedelta(days=self.config.purge_days)
            test_end = test_start + pd.DateOffset(months=self.config.test_months)
            
            # Check if we have enough data for test period
            if test_end > max_date:
                print(f"   Stopping at window {window_id-1}: insufficient data for test period")
                break
            
            # Get indices for train and test
            train_mask = (data[date_col] >= train_start) & (data[date_col] < train_end)
            test_mask = (data[date_col] >= test_start) & (data[date_col] < test_end)
            
            train_indices = data[train_mask].index.tolist()
            test_indices = data[test_mask].index.tolist()
            
            # Validate sample sizes
            if len(train_indices) < self.config.min_train_samples:
                print(f"   Stopping at window {window_id}: insufficient train samples ({len(train_indices)})")
                break
                
            if len(test_indices) < self.config.min_test_samples:
                print(f"   Stopping at window {window_id}: insufficient test samples ({len(test_indices)})")
                break
            
            window = {
                'window_id': window_id,
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat(),
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_samples': len(train_indices),
                'test_samples': len(test_indices),
                'purge_days': self.config.purge_days
            }
            
            windows.append(window)
            
            print(f"   Window {window_id}: Train {len(train_indices)} samples, Test {len(test_indices)} samples")
            
            # Move to next window
            current_start += pd.DateOffset(months=self.config.step_months)
            window_id += 1
        
        print(f"✅ Created {len(windows)} temporal windows for backtesting")
        return windows
    
    def run_backtest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_pipeline,
        windows: Optional[List[Dict]] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtesting.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_pipeline: Trained model pipeline or callable
            windows: Pre-computed windows (will create if None)
            feature_columns: Specific features to use (uses all if None)
            
        Returns:
            Comprehensive backtesting results
        """
        print("🚀 Starting walk-forward backtesting...")
        
        # Create windows if not provided
        if windows is None:
            windows = self.create_temporal_windows(X)
        
        if not windows:
            raise ValueError("No valid temporal windows could be created")
        
        # Prepare features
        if feature_columns is not None:
            X_features = X[feature_columns].copy()
        else:
            # Use only numeric columns (exclude date/categorical columns)
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X_features = X[numeric_cols].copy()
        
        print(f"📊 Using {len(X_features.columns)} features for backtesting")
        
        # Initialize results storage
        window_results = []
        all_predictions = []
        performance_over_time = []
        
        for window in windows:
            window_id = window['window_id']
            print(f"\n🔄 Processing Window {window_id}/{len(windows)}...")
            
            try:
                result = self._evaluate_window(
                    X_features, y, model_pipeline, window
                )
                window_results.append(result)
                
                # Store predictions for later analysis
                if 'predictions' in result:
                    all_predictions.extend(result['predictions'])
                
                # Track performance over time
                performance_over_time.append({
                    'window_id': window_id,
                    'test_start': window['test_start'],
                    'pr_auc': result['metrics']['pr_auc'],
                    'precision_at_10': result['metrics'].get('precision_at_10', np.nan),
                    'recall_at_10': result['metrics'].get('recall_at_10', np.nan),
                    'calibration_error': result['metrics'].get('calibration_error', np.nan)
                })
                
                print(f"   ✅ Window {window_id}: PR-AUC = {result['metrics']['pr_auc']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Window {window_id} failed: {str(e)}")
                continue
        
        # Compute summary statistics
        self.summary_stats = self._compute_summary_statistics(window_results, performance_over_time)
        
        # Compile full results
        backtest_results = {
            'backtest_timestamp': datetime.utcnow().isoformat(),
            'config': {
                'initial_train_months': self.config.initial_train_months,
                'step_months': self.config.step_months,
                'test_months': self.config.test_months,
                'max_windows': self.config.max_windows,
                'purge_days': self.config.purge_days
            },
            'windows_processed': len(window_results),
            'windows_failed': len(windows) - len(window_results),
            'window_results': window_results,
            'performance_over_time': performance_over_time,
            'summary_statistics': self.summary_stats,
            'stability_assessment': self._assess_stability(performance_over_time)
        }
        
        print(f"\n✅ Backtesting completed: {len(window_results)}/{len(windows)} windows successful")
        return backtest_results
    
    def _evaluate_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_pipeline,
        window: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate model performance on a single temporal window."""
        
        # Extract train/test data
        train_idx = window['train_indices']
        test_idx = window['test_indices']
        
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train medians for test
        
        # Train model for this window
        if hasattr(model_pipeline, 'fit'):
            # Fresh model training
            model = model_pipeline
            model.fit(X_train, y_train)
        else:
            # Assume it's a callable that returns a trained model
            model = model_pipeline(X_train, y_train)
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] > 1:
                y_scores = y_proba[:, 1]  # Positive class probability
            else:
                y_scores = y_proba.ravel()
        else:
            y_scores = model.decision_function(X_test)
            # Normalize scores to [0,1] range
            y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        
        # Compute metrics
        metrics = self._compute_window_metrics(y_test, y_scores)
        
        # Store predictions for analysis
        predictions = [
            {
                'true_label': int(true),
                'predicted_score': float(score),
                'window_id': window['window_id']
            }
            for true, score in zip(y_test, y_scores)
        ]
        
        return {
            'window_id': window['window_id'],
            'window_config': {
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end']
            },
            'sample_sizes': {
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'test_positives': int(y_test.sum()),
                'test_negatives': int((y_test == 0).sum())
            },
            'metrics': metrics,
            'predictions': predictions
        }
    
    def _compute_window_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics for a single window."""
        
        metrics = {}
        
        try:
            # PR-AUC (primary metric)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(recall, precision)
            
            # Precision/Recall at top percentiles
            top_10_threshold = np.percentile(y_scores, 90)
            top_5_threshold = np.percentile(y_scores, 95)
            
            top_10_predictions = (y_scores >= top_10_threshold).astype(int)
            top_5_predictions = (y_scores >= top_5_threshold).astype(int)
            
            if top_10_predictions.sum() > 0:
                metrics['precision_at_10'] = (y_true[top_10_predictions == 1]).mean()
                metrics['recall_at_10'] = (top_10_predictions[y_true == 1]).mean()
            else:
                metrics['precision_at_10'] = 0.0
                metrics['recall_at_10'] = 0.0
            
            if top_5_predictions.sum() > 0:
                metrics['precision_at_5'] = (y_true[top_5_predictions == 1]).mean()
                metrics['recall_at_5'] = (top_5_predictions[y_true == 1]).mean()
            else:
                metrics['precision_at_5'] = 0.0
                metrics['recall_at_5'] = 0.0
            
            # Calibration metrics (simplified)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_scores[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['calibration_error'] = ece
            
            # Distribution metrics
            metrics['base_rate'] = y_true.mean()
            metrics['score_mean'] = y_scores.mean()
            metrics['score_std'] = y_scores.std()
            
        except Exception as e:
            print(f"      Warning: Error computing metrics: {str(e)}")
            metrics = {'pr_auc': 0.0, 'error': str(e)}
        
        return metrics
    
    def _compute_summary_statistics(
        self, 
        window_results: List[Dict], 
        performance_over_time: List[Dict]
    ) -> Dict[str, Any]:
        """Compute summary statistics across all windows."""
        
        if not window_results:
            return {'error': 'No successful windows to analyze'}
        
        # Extract PR-AUC values
        pr_aucs = [r['metrics']['pr_auc'] for r in window_results if 'pr_auc' in r['metrics']]
        
        if not pr_aucs:
            return {'error': 'No valid PR-AUC values found'}
        
        pr_aucs = np.array(pr_aucs)
        
        summary = {
            'n_windows': len(window_results),
            'pr_auc_statistics': {
                'mean': float(pr_aucs.mean()),
                'std': float(pr_aucs.std()),
                'min': float(pr_aucs.min()),
                'max': float(pr_aucs.max()),
                'median': float(np.median(pr_aucs)),
                'q25': float(np.percentile(pr_aucs, 25)),
                'q75': float(np.percentile(pr_aucs, 75))
            }
        }
        
        # Performance degradation analysis
        if len(pr_aucs) > 1:
            # Linear trend
            time_indices = np.arange(len(pr_aucs))
            trend_coef = np.corrcoef(time_indices, pr_aucs)[0, 1]
            summary['performance_trend'] = {
                'correlation_with_time': float(trend_coef),
                'interpretation': 'declining' if trend_coef < -0.3 else 'stable' if abs(trend_coef) < 0.3 else 'improving'
            }
            
            # Consecutive window degradation
            degradations = []
            for i in range(1, len(pr_aucs)):
                degradations.append(pr_aucs[i] - pr_aucs[i-1])
            
            consecutive_declines = 0
            max_consecutive_declines = 0
            for deg in degradations:
                if deg < 0:
                    consecutive_declines += 1
                    max_consecutive_declines = max(max_consecutive_declines, consecutive_declines)
                else:
                    consecutive_declines = 0
            
            summary['stability_metrics'] = {
                'max_consecutive_declines': max_consecutive_declines,
                'mean_window_change': float(np.mean(degradations)),
                'volatility': float(np.std(degradations))
            }
        
        return summary
    
    def _assess_stability(self, performance_over_time: List[Dict]) -> Dict[str, Any]:
        """Assess model stability over time."""
        
        if len(performance_over_time) < 3:
            return {'status': 'insufficient_data', 'recommendation': 'Need at least 3 windows for stability assessment'}
        
        pr_aucs = [p['pr_auc'] for p in performance_over_time if not np.isnan(p['pr_auc'])]
        
        if len(pr_aucs) < 3:
            return {'status': 'insufficient_valid_data', 'recommendation': 'Too many windows with invalid PR-AUC values'}
        
        pr_aucs = np.array(pr_aucs)
        
        # Stability criteria
        mean_performance = pr_aucs.mean()
        std_performance = pr_aucs.std()
        cv = std_performance / mean_performance if mean_performance > 0 else np.inf
        
        # Performance thresholds
        min_acceptable_performance = 0.1  # Minimum PR-AUC
        max_acceptable_cv = 0.3  # Maximum coefficient of variation
        
        stability_assessment = {
            'mean_performance': float(mean_performance),
            'coefficient_of_variation': float(cv),
            'performance_range': float(pr_aucs.max() - pr_aucs.min()),
            'windows_below_threshold': int(np.sum(pr_aucs < min_acceptable_performance))
        }
        
        # Overall assessment
        if mean_performance < min_acceptable_performance:
            status = 'poor_performance'
            recommendation = f'Mean PR-AUC ({mean_performance:.3f}) below acceptable threshold ({min_acceptable_performance})'
        elif cv > max_acceptable_cv:
            status = 'unstable'
            recommendation = f'High performance variability (CV={cv:.3f}) indicates model instability'
        elif stability_assessment['windows_below_threshold'] > len(pr_aucs) * 0.2:
            status = 'inconsistent'
            recommendation = 'More than 20% of windows have poor performance'
        else:
            status = 'stable'
            recommendation = 'Model shows stable performance across time windows'
        
        stability_assessment.update({
            'status': status,
            'recommendation': recommendation,
            'production_ready': status == 'stable'
        })
        
        return stability_assessment


def save_backtest_results(results: Dict[str, Any], output_path: Path) -> Path:
    """Save backtesting results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return output_path


def run_backtest_for_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """
    Run comprehensive backtesting for the pipeline.
    
    Args:
        data_dir: Directory containing prepared datasets
        artifacts_dir: Directory to save backtesting results
        config: Backtesting configuration (uses defaults if None)
        
    Returns:
        Backtesting results dictionary
    """
    if config is None:
        config = BacktestConfig()
    
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    
    print("🔍 Loading data for backtesting...")
    
    # Load the temporal or engineered dataset
    temporal_features_path = data_dir / 'X_train_temporal.csv'
    temporal_target_path = data_dir / 'y_train_temporal.csv'
    
    if temporal_features_path.exists() and temporal_target_path.exists():
        X = pd.read_csv(temporal_features_path)
        y = pd.read_csv(temporal_target_path)[config.target_column]
        print("   Using temporal dataset for backtesting")
    else:
        # Fallback to engineered data
        X = pd.read_csv(data_dir / 'X_train_engineered.csv')
        y = pd.read_csv(data_dir / 'y_train_engineered.csv')[config.target_column]
        print("   Using engineered dataset for backtesting")
    
    print(f"   Loaded {len(X)} samples with {len(X.columns)} features")
    
    # Initialize backtester
    backtester = WalkForwardBacktester(config)
    
    # Create a simple model pipeline for backtesting
    # In practice, this would use your actual trained model
    def simple_model_pipeline(X_train, y_train):
        # RF and Pipeline already imported at module level
        
        # Simple RF model for backtesting
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        model.fit(X_train, y_train)
        return model
    
    # Run backtesting
    backtest_results = backtester.run_backtest(
        X=X,
        y=y,
        model_pipeline=simple_model_pipeline
    )
    
    # Save results
    results_path = save_backtest_results(
        backtest_results,
        artifacts_dir / 'backtest_results.json'
    )
    
    print(f"💾 Backtest results saved: {results_path}")
    
    # Print summary
    summary = backtest_results['summary_statistics']
    stability = backtest_results['stability_assessment']
    
    print(f"\n📊 Backtesting Summary:")
    print(f"   Windows processed: {backtest_results['windows_processed']}")
    print(f"   Mean PR-AUC: {summary['pr_auc_statistics']['mean']:.4f} ± {summary['pr_auc_statistics']['std']:.4f}")
    print(f"   Stability: {stability['status']} - {stability['recommendation']}")
    
    return backtest_results

"""
Model Validation Summary Generator

Creates comprehensive validation reports consolidating:
- PR_AUC + Confidence Intervals
- Calibration metrics (ECE, Brier)
- Expected Value analysis
- Drift monitoring baseline
- Data integrity checks
- Temporal validation results
"""

# import json  # Already imported at top
# import numpy as np  # Already imported at top
# import pandas as pd  # Already imported at top
# from datetime import datetime  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, Optional, List  # Already imported at top
# from metrics import bootstrap_metric, compute_calibration_metrics  # Already imported at top
# from governance import check_data_integrity, validate_temporal_split  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #3


def create_validation_summary(
    model_metadata: Dict[str, Any],
    y_true: np.ndarray,
    y_score: np.ndarray,
    config: Dict[str, Any],
    artifacts_dir: Path,
    data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create comprehensive model validation summary.
    
    Args:
        model_metadata: Best model metadata from selection
        y_true: True test labels
        y_score: Model scores on test set
        config: Configuration dictionary
        artifacts_dir: Directory containing artifacts
        data_dir: Directory containing datasets (for integrity check)
        
    Returns:
        Comprehensive validation summary dictionary
    """
    print("🔍 Generating comprehensive validation summary...")
    
    summary = {
        'generated_at': datetime.utcnow().isoformat(),
        'model_info': {
            'name': model_metadata.get('model_name', 'unknown'),
            'variant': model_metadata.get('variant', 'unknown'),
            'source': model_metadata.get('source', 'unknown'),
            'selection_stage': model_metadata.get('selection_stage', 'unknown')
        }
    }
    
    # 1. Performance with Confidence Intervals
    print("   Computing performance metrics with bootstrap CI...")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        # Primary metrics
        pr_auc = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        
        # Bootstrap confidence intervals
        pr_auc_bootstrap = bootstrap_metric(
            y_true, y_score, average_precision_score, 
            n_bootstrap=1000, random_state=42
        )
        
        roc_auc_bootstrap = bootstrap_metric(
            y_true, y_score, roc_auc_score, 
            n_bootstrap=1000, random_state=42
        )
        
        summary['performance'] = {
            'pr_auc': {
                'value': float(pr_auc),
                'ci_lower': pr_auc_bootstrap['ci_lower'],
                'ci_upper': pr_auc_bootstrap['ci_upper'],
                'bootstrap_std': pr_auc_bootstrap['std']
            },
            'roc_auc': {
                'value': float(roc_auc),
                'ci_lower': roc_auc_bootstrap['ci_lower'],
                'ci_upper': roc_auc_bootstrap['ci_upper'],
                'bootstrap_std': roc_auc_bootstrap['std']
            },
            'base_rate': float(np.mean(y_true)),
            'n_samples': len(y_true),
            'n_positives': int(np.sum(y_true))
        }
        
    except Exception as e:
        print(f"   Warning: Performance metrics failed: {e}")
        summary['performance'] = {'error': str(e)}
    
    # 2. Calibration Assessment
    print("   Assessing model calibration...")
    try:
        cal_metrics = compute_calibration_metrics(y_true, y_score)
        summary['calibration'] = {
            'ece': cal_metrics['ece'],
            'brier_score': cal_metrics['brier_score'],
            'calibration_quality': _assess_calibration_quality(cal_metrics['ece']),
            'reliability_bins': len(cal_metrics['reliability_data'])
        }
    except Exception as e:
        print(f"   Warning: Calibration metrics failed: {e}")
        summary['calibration'] = {'error': str(e)}
    
    # 3. Expected Value Analysis
    print("   Computing Expected Value analysis...")
    try:
        ev_config = config.get('scoring', {}).get('expected_value', {})
        if ev_config:
            ev_analysis = _compute_ev_analysis(y_true, y_score, ev_config)
            summary['expected_value'] = ev_analysis
        else:
            summary['expected_value'] = {'status': 'not_configured'}
    except Exception as e:
        print(f"   Warning: Expected Value analysis failed: {e}")
        summary['expected_value'] = {'error': str(e)}
    
    # 4. Operational Metrics
    print("   Computing operational metrics...")
    try:
        k_values = config.get('scoring', {}).get('k_values', [50, 100, 200, 500])
        operational = _compute_operational_metrics(y_true, y_score, k_values)
        summary['operational'] = operational
    except Exception as e:
        print(f"   Warning: Operational metrics failed: {e}")
        summary['operational'] = {'error': str(e)}
    
    # 5. Stability Assessment
    print("   Assessing model stability...")
    try:
        stability = _assess_stability(model_metadata)
        summary['stability'] = stability
    except Exception as e:
        print(f"   Warning: Stability assessment failed: {e}")
        summary['stability'] = {'error': str(e)}
    
    # 6. Data Integrity
    if data_dir:
        print("   Checking data integrity...")
        try:
            integrity = check_data_integrity(data_dir)
            summary['data_integrity'] = {
                'status': 'verified',
                'hashes': integrity,
                'files_checked': len(integrity),
                'missing_files': sum(1 for v in integrity.values() if v == 'FILE_NOT_FOUND'),
                'error_files': sum(1 for v in integrity.values() if v.startswith('ERROR'))
            }
        except Exception as e:
            print(f"   Warning: Data integrity check failed: {e}")
            summary['data_integrity'] = {'error': str(e)}
    
    # 7. Artifact Completeness
    print("   Checking artifact completeness...")
    try:
        completeness = _check_artifact_completeness(artifacts_dir)
        summary['artifact_completeness'] = completeness
    except Exception as e:
        print(f"   Warning: Artifact completeness check failed: {e}")
        summary['artifact_completeness'] = {'error': str(e)}
    
    # 8. Model Quality Assessment
    print("   Generating overall quality assessment...")
    quality = _assess_overall_quality(summary, config)
    summary['quality_assessment'] = quality
    
    print("✅ Validation summary completed")
    return summary


def _assess_calibration_quality(ece: float) -> str:
    """Assess calibration quality based on ECE."""
    if ece < 0.05:
        return "excellent"
    elif ece < 0.10:
        return "good"
    elif ece < 0.15:
        return "fair"
    else:
        return "poor"


def _compute_ev_analysis(y_true: np.ndarray, y_score: np.ndarray, ev_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Expected Value analysis."""
    v_tp = ev_config.get('v_tp', 10.0)
    c_fp = ev_config.get('c_fp', 2.0)
    c_fn = ev_config.get('c_fn', 50.0)
    c_review = ev_config.get('c_review', 0.5)
    
    # Analyze EV at different capacity levels
    capacity_scenarios = ev_config.get('capacity_scenarios', [50, 100, 200, 500])
    ev_results = []
    
    for capacity in capacity_scenarios:
        # Find threshold that flags approximately 'capacity' cases
        sorted_indices = np.argsort(y_score)[::-1]
        threshold_idx = min(capacity - 1, len(y_score) - 1)
        threshold = y_score[sorted_indices[threshold_idx]]
        
        # Compute confusion matrix
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        flagged = np.sum(y_pred)
        
        # Compute EV
        ev = tp * v_tp - fp * c_fp - fn * c_fn - flagged * c_review
        
        ev_results.append({
            'capacity': capacity,
            'threshold': float(threshold),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'flagged': int(flagged),
            'expected_value': float(ev),
            'precision': float(tp / flagged if flagged > 0 else 0),
            'recall': float(tp / np.sum(y_true) if np.sum(y_true) > 0 else 0)
        })
    
    return {
        'parameters': ev_config,
        'capacity_analysis': ev_results,
        'optimal_capacity': max(ev_results, key=lambda x: x['expected_value'])['capacity'] if ev_results else None
    }


def _compute_operational_metrics(y_true: np.ndarray, y_score: np.ndarray, k_values: List[int]) -> Dict[str, Any]:
    """Compute operational metrics at different K values."""
    sorted_indices = np.argsort(y_score)[::-1]
    y_sorted = y_true[sorted_indices]
    total_positives = np.sum(y_true)
    
    metrics_at_k = []
    for k in k_values:
        k = min(k, len(y_true))
        if k <= 0:
            continue
            
        top_k = y_sorted[:k]
        tp_k = np.sum(top_k)
        
        precision_k = tp_k / k if k > 0 else 0
        recall_k = tp_k / total_positives if total_positives > 0 else 0
        lift_k = precision_k / np.mean(y_true) if np.mean(y_true) > 0 else 0
        
        metrics_at_k.append({
            'k': k,
            'precision': float(precision_k),
            'recall': float(recall_k),
            'lift': float(lift_k),
            'true_positives': int(tp_k)
        })
    
    return {
        'metrics_at_k': metrics_at_k,
        'workload_efficiency': _compute_workload_efficiency(metrics_at_k)
    }


def _compute_workload_efficiency(metrics_at_k: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute workload efficiency metrics."""
    if not metrics_at_k:
        return {}
    
    # Find K with best precision/recall trade-off
    efficiency_scores = []
    for metric in metrics_at_k:
        # Harmonic mean of precision and recall (F1-like for @K)
        p, r = metric['precision'], metric['recall']
        f1_k = 2 * p * r / (p + r) if (p + r) > 0 else 0
        efficiency_scores.append(f1_k)
    
    best_idx = np.argmax(efficiency_scores)
    
    return {
        'most_efficient_k': metrics_at_k[best_idx]['k'],
        'efficiency_score': efficiency_scores[best_idx],
        'precision_at_efficient_k': metrics_at_k[best_idx]['precision'],
        'recall_at_efficient_k': metrics_at_k[best_idx]['recall']
    }


def _assess_stability(model_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assess model stability from metadata."""
    stability_metrics = model_metadata.get('stability_metrics', {})
    
    cv_mean = stability_metrics.get('cv_pr_auc_mean', np.nan)
    cv_std = stability_metrics.get('cv_pr_auc_std', np.nan)
    cv_coef = stability_metrics.get('cv_stability_coef', np.nan)
    
    # Assess stability
    if not np.isnan(cv_coef):
        if cv_coef < 0.05:
            stability_assessment = "very_stable"
        elif cv_coef < 0.10:
            stability_assessment = "stable"
        elif cv_coef < 0.20:
            stability_assessment = "moderately_stable"
        else:
            stability_assessment = "unstable"
    else:
        stability_assessment = "unknown"
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_coefficient_variation': cv_coef,
        'assessment': stability_assessment,
        'stability_notes': _get_stability_notes(stability_assessment)
    }


def _get_stability_notes(assessment: str) -> List[str]:
    """Get stability assessment notes."""
    notes_map = {
        "very_stable": ["Excellent cross-validation stability", "Low performance variance"],
        "stable": ["Good cross-validation stability", "Acceptable performance variance"],
        "moderately_stable": ["Some performance variance", "Monitor stability in production"],
        "unstable": ["High performance variance", "Consider feature engineering or regularization"],
        "unknown": ["Stability assessment unavailable", "Recommend stability analysis"]
    }
    return notes_map.get(assessment, [])


def _check_artifact_completeness(artifacts_dir: Path) -> Dict[str, Any]:
    """Check completeness of pipeline artifacts."""
    critical_artifacts = [
        'best_model_meta.json',
        'baseline_models.csv',
        'baseline_metrics_at_k.csv',
        'thresholds.json',
        'model_card.json'
    ]
    
    optional_artifacts = [
        'ensemble_metadata.json',
        'risk_scores_ensemble.csv',
        'tuning_results.json',
        'validation_report.json',
        'monitor_summary.json'
    ]
    
    critical_status = {}
    optional_status = {}
    
    for artifact in critical_artifacts:
        path = artifacts_dir / artifact
        critical_status[artifact] = {
            'exists': path.exists(),
            'size_bytes': path.stat().st_size if path.exists() else 0
        }
    
    for artifact in optional_artifacts:
        path = artifacts_dir / artifact
        optional_status[artifact] = {
            'exists': path.exists(),
            'size_bytes': path.stat().st_size if path.exists() else 0
        }
    
    critical_missing = [k for k, v in critical_status.items() if not v['exists']]
    critical_complete = len(critical_missing) == 0
    
    return {
        'critical_artifacts': critical_status,
        'optional_artifacts': optional_status,
        'critical_complete': critical_complete,
        'critical_missing_count': len(critical_missing),
        'critical_missing': critical_missing,
        'completeness_score': (len(critical_artifacts) - len(critical_missing)) / len(critical_artifacts)
    }


def _assess_overall_quality(summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall model quality and readiness."""
    issues = []
    warnings = []
    recommendations = []
    
    # Performance checks
    perf = summary.get('performance', {})
    if 'value' in perf.get('pr_auc', {}):
        pr_auc = perf['pr_auc']['value']
        if pr_auc < 0.01:
            issues.append("Very low PR_AUC (<0.01) - model may not be useful")
        elif pr_auc < 0.05:
            warnings.append("Low PR_AUC (<0.05) - consider feature engineering")
        
        # Check CI width
        ci_width = perf['pr_auc'].get('ci_upper', 0) - perf['pr_auc'].get('ci_lower', 0)
        if ci_width > 0.1:
            warnings.append("Wide confidence interval - performance may be unstable")
    
    # Calibration checks
    cal = summary.get('calibration', {})
    if 'ece' in cal:
        if cal['ece'] > 0.15:
            warnings.append("Poor calibration (ECE > 0.15) - probabilities may be unreliable")
        elif cal['ece'] > 0.10:
            recommendations.append("Consider recalibration to improve probability estimates")
    
    # Stability checks
    stability = summary.get('stability', {})
    if stability.get('assessment') == 'unstable':
        issues.append("Model shows high cross-validation instability")
    elif stability.get('assessment') == 'moderately_stable':
        warnings.append("Model shows some instability - monitor in production")
    
    # Artifact completeness
    completeness = summary.get('artifact_completeness', {})
    if not completeness.get('critical_complete', True):
        issues.append(f"Missing critical artifacts: {completeness.get('critical_missing', [])}")
    
    # Overall assessment
    if issues:
        overall_status = "not_ready"
    elif warnings:
        overall_status = "ready_with_caution"
    else:
        overall_status = "ready"
    
    return {
        'overall_status': overall_status,
        'issues': issues,
        'warnings': warnings,
        'recommendations': recommendations,
        'quality_score': _compute_quality_score(summary),
        'production_readiness': overall_status in ['ready', 'ready_with_caution']
    }


def _compute_quality_score(summary: Dict[str, Any]) -> float:
    """Compute overall quality score (0-1)."""
    score = 0.0
    max_score = 0.0
    
    # Performance component (30%)
    perf = summary.get('performance', {})
    if 'value' in perf.get('pr_auc', {}):
        pr_auc = perf['pr_auc']['value']
        # Normalize PR_AUC (assuming >0.1 is excellent for rare class)
        perf_score = min(pr_auc / 0.1, 1.0) * 0.3
        score += perf_score
    max_score += 0.3
    
    # Calibration component (20%)
    cal = summary.get('calibration', {})
    if 'ece' in cal:
        # Lower ECE is better
        cal_score = max(0, (0.15 - cal['ece']) / 0.15) * 0.2
        score += cal_score
    max_score += 0.2
    
    # Stability component (25%)
    stability = summary.get('stability', {})
    stability_map = {
        'very_stable': 1.0,
        'stable': 0.8,
        'moderately_stable': 0.6,
        'unstable': 0.3,
        'unknown': 0.5
    }
    if stability.get('assessment') in stability_map:
        stab_score = stability_map[stability['assessment']] * 0.25
        score += stab_score
    max_score += 0.25
    
    # Completeness component (25%)
    completeness = summary.get('artifact_completeness', {})
    if 'completeness_score' in completeness:
        comp_score = completeness['completeness_score'] * 0.25
        score += comp_score
    max_score += 0.25
    
    return score / max_score if max_score > 0 else 0.0


def save_validation_summary(summary: Dict[str, Any], output_path: Path) -> Path:
    """Save validation summary to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    return output_path

# import numpy as np  # Already imported at top
# import pandas as pd  # Already imported at top
# from typing import Tuple, Dict  # Already imported at top
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #4


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    exp = expected.dropna()
    act = actual.dropna()
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = exp.quantile(quantiles).values
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf
    exp_counts = np.histogram(exp, bins=cut_points)[0]
    act_counts = np.histogram(act, bins=cut_points)[0]
    exp_perc = exp_counts / exp_counts.sum()
    act_perc = act_counts / act_counts.sum()
    psi = 0.0
    for e, a in zip(exp_perc, act_perc):
        if e == 0 or a == 0:
            continue
        psi += (a - e) * np.log(a / e)
    return float(psi)


def classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true))>1 else np.nan,
        'pr_auc': average_precision_score(y_true, y_proba) if len(np.unique(y_true))>1 else np.nan,
    }


def score_shift_report(ref_scores: pd.Series, new_scores: pd.Series) -> Dict[str, float]:
    return {
        'psi_scores': population_stability_index(ref_scores, new_scores),
        'mean_ref': float(ref_scores.mean()),
        'mean_new': float(new_scores.mean()),
        'std_ref': float(ref_scores.std()),
        'std_new': float(new_scores.std()),
    }


def feature_shift_table(ref_df: pd.DataFrame, new_df: pd.DataFrame, sample_max: int = 20000) -> pd.DataFrame:
    common = [c for c in ref_df.columns if c in new_df.columns]
    rows = []
    if len(ref_df) > sample_max:
        ref_df = ref_df.sample(sample_max, random_state=42)
    if len(new_df) > sample_max:
        new_df = new_df.sample(sample_max, random_state=42)
    for c in common:
        if pd.api.types.is_numeric_dtype(ref_df[c]):
            psi = population_stability_index(ref_df[c], new_df[c])
            rows.append({'feature': c, 'psi': psi})
    return pd.DataFrame(rows).sort_values('psi', ascending=False)

"""
Automated Performance Alerting System

Comprehensive monitoring and alerting for ML pipeline performance degradation.
Tracks key metrics and sends notifications when thresholds are breached.
"""

# import json  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from datetime import datetime, timedelta  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, List, Optional, Union, Callable  # Already imported at top
# from dataclasses import dataclass, asdict  # Already imported at top
# import logging  # Already imported at top
# import smtplib  # Already imported at top
# from email.mime.text import MIMEText  # Already imported at top
# from email.mime.multipart import MIMEMultipart  # Already imported at top
# import warnings  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #5


@dataclass
class AlertThreshold:
    """Definition of a performance threshold for alerting."""
    metric_name: str
    threshold_value: float
    comparison_type: str  # 'below', 'above', 'outside_range'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    cooldown_hours: int = 24  # Minimum time between same alerts
    consecutive_breaches_required: int = 1  # Breaches needed to trigger
    
    def __post_init__(self):
        valid_comparisons = ['below', 'above', 'outside_range']
        valid_severities = ['low', 'medium', 'high', 'critical']
        
        if self.comparison_type not in valid_comparisons:
            raise ValueError(f"comparison_type must be one of {valid_comparisons}")
        if self.severity not in valid_severities:
            raise ValueError(f"severity must be one of {valid_severities}")


@dataclass
class AlertEvent:
    """Single alert event record."""
    alert_id: str
    timestamp: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[str] = None


class PerformanceMonitor:
    """Core performance monitoring and alerting system."""
    
    def __init__(self, config: Dict[str, Any], artifacts_dir: Path):
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Alert configuration
        self.alert_thresholds = self._load_alert_thresholds()
        self.alert_history = self._load_alert_history()
        self.performance_history = self._load_performance_history()
        
        # Notification configuration
        self.notification_config = config.get('notifications', {})
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated logger for alerts."""
        logger = logging.getLogger('performance_alerts')
        logger.setLevel(logging.INFO)
        
        # File handler for alert log
        alert_log_path = self.artifacts_dir / 'performance_alerts.log'
        file_handler = logging.FileHandler(alert_log_path)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_alert_thresholds(self) -> List[AlertThreshold]:
        """Load alert threshold configurations."""
        thresholds_config = self.config.get('performance_alerts', {}).get('thresholds', [])
        
        if not thresholds_config:
            # Default thresholds for AML pipeline
            thresholds_config = [
                {
                    'metric_name': 'pr_auc',
                    'threshold_value': 0.05,
                    'comparison_type': 'below',
                    'severity': 'critical',
                    'description': 'PR-AUC fell below minimum acceptable level',
                    'consecutive_breaches_required': 2
                },
                {
                    'metric_name': 'precision_at_10',
                    'threshold_value': 0.1,
                    'comparison_type': 'below',
                    'severity': 'high',
                    'description': 'Precision@10% significantly degraded',
                    'consecutive_breaches_required': 2
                },
                {
                    'metric_name': 'prediction_drift_psi',
                    'threshold_value': 0.2,
                    'comparison_type': 'above',
                    'severity': 'medium',
                    'description': 'High prediction drift detected',
                    'consecutive_breaches_required': 1
                },
                {
                    'metric_name': 'feature_drift_psi',
                    'threshold_value': 0.3,
                    'comparison_type': 'above',
                    'severity': 'medium',
                    'description': 'Significant feature drift detected',
                    'consecutive_breaches_required': 1
                },
                {
                    'metric_name': 'null_percentage',
                    'threshold_value': 15.0,
                    'comparison_type': 'above',
                    'severity': 'high',
                    'description': 'High percentage of missing values',
                    'consecutive_breaches_required': 1
                },
                {
                    'metric_name': 'prediction_volume',
                    'threshold_value': 0.5,
                    'comparison_type': 'below',
                    'severity': 'medium',
                    'description': 'Low prediction volume - potential data pipeline issue',
                    'consecutive_breaches_required': 1
                }
            ]
        
        thresholds = []
        for threshold_config in thresholds_config:
            try:
                threshold = AlertThreshold(**threshold_config)
                thresholds.append(threshold)
            except Exception as e:
                self.logger.warning(f"Failed to create threshold from config {threshold_config}: {e}")
        
        return thresholds
    
    def _load_alert_history(self) -> List[AlertEvent]:
        """Load alert history from file."""
        history_path = self.artifacts_dir / 'alert_history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                return [AlertEvent(**alert_data) for alert_data in history_data]
            except Exception as e:
                self.logger.warning(f"Failed to load alert history: {e}")
        
        return []
    
    def _load_performance_history(self) -> List[Dict[str, Any]]:
        """Load performance metrics history."""
        history_path = self.artifacts_dir / 'performance_history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load performance history: {e}")
        
        return []
    
    def _save_alert_history(self) -> None:
        """Save alert history to file."""
        history_path = self.artifacts_dir / 'alert_history.json'
        try:
            history_data = [asdict(alert) for alert in self.alert_history]
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save alert history: {e}")
    
    def _save_performance_history(self) -> None:
        """Save performance history to file."""
        history_path = self.artifacts_dir / 'performance_history.json'
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")
    
    def record_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record new performance metrics."""
        timestamp = datetime.utcnow().isoformat()
        
        performance_record = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 1000 records to prevent unbounded growth
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self._save_performance_history()
        
        # Check for threshold breaches
        self._check_thresholds(metrics, timestamp)
    
    def _check_thresholds(self, current_metrics: Dict[str, Any], timestamp: str) -> List[AlertEvent]:
        """Check current metrics against thresholds and generate alerts."""
        new_alerts = []
        
        for threshold in self.alert_thresholds:
            metric_name = threshold.metric_name
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Check if threshold is breached
            is_breached = self._is_threshold_breached(current_value, threshold)
            
            if is_breached:
                # Check consecutive breaches if required
                if threshold.consecutive_breaches_required > 1:
                    consecutive_breaches = self._count_consecutive_breaches(
                        metric_name, threshold
                    )
                    if consecutive_breaches < threshold.consecutive_breaches_required:
                        continue
                
                # Check cooldown period
                if self._is_in_cooldown(threshold):
                    continue
                
                # Generate alert
                alert = self._create_alert(current_value, threshold, timestamp)
                new_alerts.append(alert)
                self.alert_history.append(alert)
                
                # Log alert
                self.logger.warning(f"ALERT: {alert.message}")
                
                # Send notification
                self._send_notification(alert)
        
        if new_alerts:
            self._save_alert_history()
        
        return new_alerts
    
    def _is_threshold_breached(self, value: float, threshold: AlertThreshold) -> bool:
        """Check if a value breaches the threshold."""
        if threshold.comparison_type == 'below':
            return value < threshold.threshold_value
        elif threshold.comparison_type == 'above':
            return value > threshold.threshold_value
        elif threshold.comparison_type == 'outside_range':
            # Assume threshold_value is center, and we check ±20% range
            range_size = threshold.threshold_value * 0.2
            return (value < threshold.threshold_value - range_size or 
                    value > threshold.threshold_value + range_size)
        
        return False
    
    def _count_consecutive_breaches(self, metric_name: str, threshold: AlertThreshold) -> int:
        """Count consecutive breaches for a metric."""
        if len(self.performance_history) < threshold.consecutive_breaches_required:
            return 0
        
        consecutive_count = 0
        
        # Check recent history in reverse order
        for record in reversed(self.performance_history[-threshold.consecutive_breaches_required:]):
            metrics = record['metrics']
            if metric_name in metrics:
                value = metrics[metric_name]
                if self._is_threshold_breached(value, threshold):
                    consecutive_count += 1
                else:
                    break
        
        return consecutive_count
    
    def _is_in_cooldown(self, threshold: AlertThreshold) -> bool:
        """Check if threshold is in cooldown period."""
        if threshold.cooldown_hours <= 0:
            return False
        
        cooldown_cutoff = datetime.utcnow() - timedelta(hours=threshold.cooldown_hours)
        
        # Check if there's a recent alert for this threshold
        for alert in reversed(self.alert_history):
            if alert.metric_name == threshold.metric_name:
                alert_time = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
                if alert_time > cooldown_cutoff:
                    return True
                break  # Only check most recent alert
        
        return False
    
    def _create_alert(self, current_value: float, threshold: AlertThreshold, timestamp: str) -> AlertEvent:
        """Create an alert event."""
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        message = (
            f"{threshold.description}. "
            f"Current value: {current_value:.4f}, "
            f"Threshold: {threshold.threshold_value:.4f} "
            f"({threshold.comparison_type})"
        )
        
        return AlertEvent(
            alert_id=alert_id,
            timestamp=timestamp,
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            message=message
        )
    
    def _send_notification(self, alert: AlertEvent) -> None:
        """Send notification for an alert."""
        try:
            notification_methods = self.notification_config.get('methods', ['log'])
            
            if 'email' in notification_methods:
                self._send_email_notification(alert)
            
            if 'slack' in notification_methods:
                self._send_slack_notification(alert)
            
            if 'webhook' in notification_methods:
                self._send_webhook_notification(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to send notification for alert {alert.alert_id}: {e}")
    
    def _send_email_notification(self, alert: AlertEvent) -> None:
        """Send email notification."""
        email_config = self.notification_config.get('email', {})
        
        if not email_config.get('enabled', False):
            return
        
        try:
            # Email configuration
            smtp_server = email_config.get('smtp_server', 'localhost')
            smtp_port = email_config.get('smtp_port', 587)
            username = email_config.get('username')
            password = email_config.get('password')
            from_email = email_config.get('from_email', username)
            to_emails = email_config.get('to_emails', [])
            
            if not to_emails:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.upper()}] AML Pipeline Alert: {alert.metric_name}"
            
            body = f"""
            AML Pipeline Performance Alert
            
            Severity: {alert.severity.upper()}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value:.4f}
            Threshold: {alert.threshold_value:.4f}
            Timestamp: {alert.timestamp}
            
            Description:
            {alert.message}
            
            Alert ID: {alert.alert_id}
            
            Please investigate and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, alert: AlertEvent) -> None:
        """Send Slack notification."""
        # Placeholder for Slack integration
        # Would use slack_sdk or webhook
        self.logger.info(f"Slack notification would be sent for alert {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: AlertEvent) -> None:
        """Send webhook notification."""
        # Placeholder for webhook integration
        self.logger.info(f"Webhook notification would be sent for alert {alert.alert_id}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts in the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00')) > cutoff_time
        ]
        
        # Group by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Group by metric
        metric_counts = {}
        for alert in recent_alerts:
            metric_counts[alert.metric_name] = metric_counts.get(alert.metric_name, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'time_window_hours': hours,
            'severity_breakdown': severity_counts,
            'metric_breakdown': metric_counts,
            'recent_alerts': [asdict(alert) for alert in recent_alerts[-10:]]  # Last 10
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.utcnow().isoformat()
                self._save_alert_history()
                self.logger.info(f"Alert {alert_id} marked as resolved")
                return True
        
        return False
    
    def get_performance_trends(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance trends for a specific metric."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        recent_records = [
            record for record in self.performance_history
            if datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]
        
        values = []
        timestamps = []
        
        for record in recent_records:
            if metric_name in record['metrics']:
                values.append(record['metrics'][metric_name])
                timestamps.append(record['timestamp'])
        
        if not values:
            return {'error': f'No data found for metric {metric_name}'}
        
        values = np.array(values)
        
        return {
            'metric_name': metric_name,
            'time_window_days': days,
            'data_points': len(values),
            'current_value': float(values[-1]) if len(values) > 0 else None,
            'mean_value': float(values.mean()),
            'std_value': float(values.std()),
            'min_value': float(values.min()),
            'max_value': float(values.max()),
            'trend_slope': self._calculate_trend_slope(values),
            'timestamps': timestamps,
            'values': values.tolist()
        }
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


def create_performance_monitoring_system(
    config: Dict[str, Any],
    artifacts_dir: Path
) -> PerformanceMonitor:
    """Create and configure performance monitoring system."""
    
    print("🔔 Setting up performance monitoring system...")
    
    monitor = PerformanceMonitor(config, artifacts_dir)
    
    print(f"   ✅ Loaded {len(monitor.alert_thresholds)} alert thresholds")
    print(f"   ✅ Alert history: {len(monitor.alert_history)} previous alerts")
    print(f"   ✅ Performance history: {len(monitor.performance_history)} data points")
    
    return monitor


def simulate_performance_monitoring_demo(artifacts_dir: Path) -> Dict[str, Any]:
    """Demonstrate performance monitoring with simulated data."""
    
    print("🎭 Running performance monitoring demonstration...")
    
    # Create mock configuration
    config = {
        'performance_alerts': {
            'thresholds': [
                {
                    'metric_name': 'pr_auc',
                    'threshold_value': 0.1,
                    'comparison_type': 'below',
                    'severity': 'critical',
                    'description': 'PR-AUC critically low',
                    'consecutive_breaches_required': 1
                }
            ]
        },
        'notifications': {
            'methods': ['log'],
            'email': {'enabled': False}
        }
    }
    
    # Create monitor
    monitor = create_performance_monitoring_system(config, artifacts_dir)
    
    # Simulate performance metrics over time
    demo_results = {
        'simulation_timestamp': datetime.utcnow().isoformat(),
        'metrics_recorded': [],
        'alerts_generated': []
    }
    
    # Good performance initially
    good_metrics = {
        'pr_auc': 0.25,
        'precision_at_10': 0.40,
        'prediction_drift_psi': 0.05,
        'null_percentage': 2.0
    }
    
    monitor.record_performance_metrics(good_metrics)
    demo_results['metrics_recorded'].append(('Good performance', good_metrics))
    
    # Degraded performance - should trigger alert
    bad_metrics = {
        'pr_auc': 0.05,  # Below threshold
        'precision_at_10': 0.15,
        'prediction_drift_psi': 0.25,  # Above threshold
        'null_percentage': 18.0  # Above threshold
    }
    
    monitor.record_performance_metrics(bad_metrics)
    demo_results['metrics_recorded'].append(('Degraded performance', bad_metrics))
    
    # Get alert summary
    alert_summary = monitor.get_alert_summary(hours=1)
    demo_results['alert_summary'] = alert_summary
    
    # Get performance trends
    if 'pr_auc' in good_metrics:
        trends = monitor.get_performance_trends('pr_auc', days=1)
        demo_results['performance_trends'] = trends
    
    print(f"   ✅ Recorded {len(demo_results['metrics_recorded'])} metric sets")
    print(f"   ✅ Generated {alert_summary['total_alerts']} alerts")
    
    return demo_results

"""
Temporal Validation Module

Formal validation of temporal splits to prevent data leakage
and ensure compliance with time-series modeling requirements.
"""

# import json  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from datetime import datetime  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, Optional, List, Tuple  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #6


def validate_temporal_split_formal(
    train_files: Dict[str, str],
    test_files: Dict[str, str],
    date_column: str = 'Timestamp',
    entity_column: Optional[str] = None,
    min_gap_days: int = 0
) -> Dict[str, Any]:
    """
    Formal validation of temporal split with comprehensive checks.
    
    Args:
        train_files: Dict mapping dataset names to train file paths
        test_files: Dict mapping dataset names to test file paths  
        date_column: Name of the date/timestamp column
        entity_column: Name of entity identifier column (optional)
        min_gap_days: Minimum gap required between train and test periods
        
    Returns:
        Comprehensive validation report
    """
    validation_report = {
        'validation_timestamp': datetime.utcnow().isoformat(),
        'parameters': {
            'date_column': date_column,
            'entity_column': entity_column,
            'min_gap_days': min_gap_days
        },
        'datasets_checked': list(train_files.keys()),
        'validation_results': {},
        'overall_status': 'PASS',
        'critical_issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    print("🕐 Running formal temporal validation...")
    
    for dataset_name in train_files.keys():
        if dataset_name not in test_files:
            validation_report['critical_issues'].append(
                f"Missing test file for dataset: {dataset_name}"
            )
            continue
            
        train_path = train_files[dataset_name]
        test_path = test_files[dataset_name]
        
        print(f"   Validating {dataset_name}...")
        
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Validate individual dataset
            dataset_result = _validate_dataset_pair(
                train_df, test_df, dataset_name, date_column, entity_column, min_gap_days
            )
            
            validation_report['validation_results'][dataset_name] = dataset_result
            
            # Aggregate issues
            if not dataset_result['temporal_valid']:
                validation_report['critical_issues'].extend(dataset_result['issues'])
                validation_report['overall_status'] = 'FAIL'
            
            validation_report['warnings'].extend(dataset_result['warnings'])
            
        except Exception as e:
            error_msg = f"Failed to validate {dataset_name}: {str(e)}"
            validation_report['critical_issues'].append(error_msg)
            validation_report['overall_status'] = 'FAIL'
            
            validation_report['validation_results'][dataset_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    # Generate recommendations
    validation_report['recommendations'] = _generate_recommendations(validation_report)
    
    print(f"   ✅ Temporal validation completed: {validation_report['overall_status']}")
    
    return validation_report


def _validate_dataset_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    date_column: str,
    entity_column: Optional[str],
    min_gap_days: int
) -> Dict[str, Any]:
    """Validate a single dataset pair."""
    
    result = {
        'dataset': dataset_name,
        'temporal_valid': True,
        'entity_valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if date column exists
    if date_column not in train_df.columns:
        result['issues'].append(f"Date column '{date_column}' not found in train data")
        result['temporal_valid'] = False
        return result
        
    if date_column not in test_df.columns:
        result['issues'].append(f"Date column '{date_column}' not found in test data")
        result['temporal_valid'] = False
        return result
    
    try:
        # Parse dates
        train_dates = pd.to_datetime(train_df[date_column])
        test_dates = pd.to_datetime(test_df[date_column])
        
        # Basic statistics
        result['statistics']['train_date_range'] = {
            'min': train_dates.min().isoformat(),
            'max': train_dates.max().isoformat(),
            'count': len(train_dates),
            'unique_dates': train_dates.nunique()
        }
        
        result['statistics']['test_date_range'] = {
            'min': test_dates.min().isoformat(),
            'max': test_dates.max().isoformat(),
            'count': len(test_dates),
            'unique_dates': test_dates.nunique()
        }
        
        # Critical check: temporal ordering
        train_max = train_dates.max()
        test_min = test_dates.min()
        
        if train_max >= test_min:
            gap_days = (test_min - train_max).days
            result['issues'].append(
                f"Temporal leakage detected: train_max ({train_max.date()}) >= test_min ({test_min.date()}), gap: {gap_days} days"
            )
            result['temporal_valid'] = False
        else:
            gap_days = (test_min - train_max).days
            result['statistics']['temporal_gap_days'] = gap_days
            
            if gap_days < min_gap_days:
                result['warnings'].append(
                    f"Temporal gap ({gap_days} days) is less than required minimum ({min_gap_days} days)"
                )
        
        # Entity overlap check
        if entity_column and entity_column in train_df.columns and entity_column in test_df.columns:
            train_entities = set(train_df[entity_column].dropna())
            test_entities = set(test_df[entity_column].dropna())
            overlapping_entities = train_entities.intersection(test_entities)
            
            result['statistics']['entity_overlap'] = {
                'train_unique_entities': len(train_entities),
                'test_unique_entities': len(test_entities),
                'overlapping_entities': len(overlapping_entities),
                'overlap_percentage': len(overlapping_entities) / len(train_entities) * 100 if train_entities else 0
            }
            
            if overlapping_entities:
                result['warnings'].append(
                    f"Entity overlap detected: {len(overlapping_entities)} entities appear in both train and test"
                )
                
                # Critical if overlap is significant
                overlap_pct = len(overlapping_entities) / len(train_entities) * 100 if train_entities else 0
                if overlap_pct > 10:  # More than 10% overlap is concerning
                    result['issues'].append(
                        f"High entity overlap: {overlap_pct:.1f}% of training entities also appear in test"
                    )
                    result['entity_valid'] = False
        
        # Data quality checks
        train_nulls = train_dates.isnull().sum()
        test_nulls = test_dates.isnull().sum()
        
        if train_nulls > 0:
            result['warnings'].append(f"Train data has {train_nulls} null dates")
            
        if test_nulls > 0:
            result['warnings'].append(f"Test data has {test_nulls} null dates")
        
        # Temporal distribution check
        train_span = (train_dates.max() - train_dates.min()).days
        test_span = (test_dates.max() - test_dates.min()).days
        
        result['statistics']['temporal_spans'] = {
            'train_span_days': train_span,
            'test_span_days': test_span
        }
        
        if test_span > train_span * 2:
            result['warnings'].append(
                f"Test period span ({test_span} days) is much longer than train span ({train_span} days)"
            )
            
    except Exception as e:
        result['issues'].append(f"Date parsing error: {str(e)}")
        result['temporal_valid'] = False
    
    return result


def _generate_recommendations(validation_report: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on validation results."""
    recommendations = []
    
    if validation_report['overall_status'] == 'FAIL':
        recommendations.append("❌ CRITICAL: Pipeline should not proceed to production with temporal validation failures")
        recommendations.append("🔧 Fix temporal leakage issues before continuing")
    
    # Check for entity overlap across datasets
    high_overlap_datasets = []
    for dataset, result in validation_report['validation_results'].items():
        if isinstance(result, dict) and 'statistics' in result:
            entity_stats = result['statistics'].get('entity_overlap', {})
            overlap_pct = entity_stats.get('overlap_percentage', 0)
            if overlap_pct > 5:
                high_overlap_datasets.append(f"{dataset} ({overlap_pct:.1f}%)")
    
    if high_overlap_datasets:
        recommendations.append(f"⚠️ Consider entity-based splitting for: {', '.join(high_overlap_datasets)}")
    
    # Gap recommendations
    gaps = []
    for dataset, result in validation_report['validation_results'].items():
        if isinstance(result, dict) and 'statistics' in result:
            gap = result['statistics'].get('temporal_gap_days')
            if gap is not None:
                gaps.append(gap)
    
    if gaps:
        min_gap = min(gaps)
        if min_gap < 7:
            recommendations.append("📅 Consider increasing temporal gap to at least 7 days for model stability")
        elif min_gap < 30:
            recommendations.append("📅 Consider 30+ day gap for production deployment to account for delayed labels")
    
    if not recommendations:
        recommendations.append("✅ Temporal validation passed - ready for production deployment")
    
    return recommendations


def save_temporal_validation_report(report: Dict[str, Any], output_path: Path) -> Path:
    """Save temporal validation report to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return output_path


def run_temporal_validation_for_pipeline(data_dir: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Run temporal validation for standard pipeline datasets.
    
    Args:
        data_dir: Directory containing train/test datasets
        artifacts_dir: Directory to save validation report
        
    Returns:
        Validation report dictionary
    """
    data_dir = Path(data_dir)
    
    # Define standard dataset pairs
    dataset_pairs = {
        'engineered': {
            'train': data_dir / 'X_train_engineered.csv',
            'test': data_dir / 'X_test_engineered.csv'
        }
    }
    
    # Check for temporal datasets
    temporal_train = data_dir / 'X_train_temporal.csv'
    temporal_test = data_dir / 'X_test_temporal.csv'
    if temporal_train.exists() and temporal_test.exists():
        dataset_pairs['temporal'] = {
            'train': temporal_train,
            'test': temporal_test
        }
    
    # Check for scaled datasets
    scaled_train = data_dir / 'X_train_scaled.csv'
    scaled_test = data_dir / 'X_test_scaled.csv'
    if scaled_train.exists() and scaled_test.exists():
        dataset_pairs['scaled'] = {
            'train': scaled_train,
            'test': scaled_test
        }
    
    # Prepare file mappings
    train_files = {name: str(files['train']) for name, files in dataset_pairs.items()}
    test_files = {name: str(files['test']) for name, files in dataset_pairs.items()}
    
    # Run validation
    validation_report = validate_temporal_split_formal(
        train_files=train_files,
        test_files=test_files,
        date_column='Timestamp',  # Adjust based on your data
        entity_column='Account',  # Adjust based on your data
        min_gap_days=1
    )
    
    # Save report
    report_path = save_temporal_validation_report(
        validation_report, 
        artifacts_dir / 'temporal_validation.json'
    )
    
    print(f"💾 Temporal validation report saved: {report_path}")
    
    return validation_report

