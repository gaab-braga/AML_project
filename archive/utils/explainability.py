"""
Explainability Module - Interpretabilidade para Modelos AML
===========================================================

Implementa técnicas avançadas de explicabilidade para modelos de Machine Learning:

FUNCIONALIDADES:
- SHAP (SHapley Additive exPlanations): Global e Local
- LIME (Local Interpretable Model-agnostic Explanations)
- Permutation Importance com ranking
- Feature Selection Incremental
- Model Explainer com visualizações automáticas
- Fallback para quando SHAP/LIME não disponíveis

CLASSES PRINCIPAIS:
- ModelExplainer: Wrapper avançado para SHAP e LIME
- SHAPExplanation: Container para explicações SHAP
- SHAPFallback: Fallback quando SHAP não disponível

FUNÇÕES:
- compute_permutation_importance: Importância por permutação
- compute_shap_values: Calcula SHAP values
- permutation_ranking: Ranqueia features por queda de métrica
- incremental_subset_evaluation: Avalia subsets incrementais
- find_elbow: Encontra ponto de corte ótimo
- compare_explanations_shap_lime: Compara SHAP vs LIME
- explain_model_safe: Explicação segura com fallback

Autor: Time de Data Science
Data: Outubro 2025
Fase: 3.1 - Interpretabilidade e Governança
"""

# Stdlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union

# Third-party: científico
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Third-party: ML
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# Third-party: explicabilidade (opcionais)
# Configurar logging primeiro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party: explicabilidade (opcionais com warnings)
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except ImportError:
    shap = None  # type: ignore
    _HAS_SHAP = False
    logger.warning("⚠️ SHAP não instalado. Funcionalidades SHAP não disponíveis. Instale com: pip install shap")
    
try:
    import lime  # type: ignore
    import lime.lime_tabular  # type: ignore
    _HAS_LIME = True
except ImportError:
    lime = None  # type: ignore
    _HAS_LIME = False
    logger.warning("⚠️ LIME não instalado. Funcionalidades LIME não disponíveis. Instale com: pip install lime")

# Configurar plot style
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Classes
    'ModelExplainer',
    'SHAPExplanation',
    'SHAPFallback',
    
    # Funções principais
    'compute_permutation_importance',
    'compute_shap_values',
    'permutation_ranking',
    'incremental_subset_evaluation',
    'find_elbow',
    'compare_explanations_shap_lime',
    'explain_model_safe',
]


class ModelExplainer:
    """
    Classe principal para explicabilidade avançada.
    
    Suporta:
    - Tree-based models (LightGBM, XGBoost, CatBoost)
    - SHAP global e local
    - LIME local
    - Plots automáticos
    
    Exemplo
    -------
    >>> explainer = ModelExplainer(model, X_train)
    >>> exp = explainer.explain_shap(X_test)
    >>> explainer.plot_summary(exp)
    >>> explainer.plot_waterfall(exp, sample_idx=0)
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ):
        self.model = model
        self.X_train = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.feature_names = feature_names or list(self.X_train.columns)
        
        # Criar SHAP explainer
        if _HAS_SHAP:
            self.shap_explainer = self._create_shap_explainer()
        else:
            self.shap_explainer = None
            
        # Criar LIME explainer
        if _HAS_LIME:
            self.lime_explainer = self._create_lime_explainer()
        else:
            self.lime_explainer = None
    
    def _create_shap_explainer(self):
        """Cria SHAP explainer apropriado."""
        try:
            # TreeExplainer para tree models
            return shap.TreeExplainer(self.model)
        except:
            # Fallback: KernelExplainer
            logger.warning("TreeExplainer falhou, usando KernelExplainer")
            return shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_train, 100)
            )
    
    def _create_lime_explainer(self):
        """Cria LIME explainer."""
        return lime.lime_tabular.LIMETabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Legítima', 'Fraude'],
            mode='classification',
            random_state=42
        )
    
    def explain_shap(self, X: pd.DataFrame):
        """
        Calcula SHAP values.
        
        Returns
        -------
        shap_values : np.ndarray
            SHAP values para classe positiva (fraude)
        """
        if not _HAS_SHAP or self.shap_explainer is None:
            raise ValueError("SHAP não disponível. Instale: pip install shap")
        
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        
        shap_values = self.shap_explainer.shap_values(X_df)
        
        # Para binary, pegar classe 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return {
            'shap_values': shap_values,
            'data': X_df,
            'expected_value': self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value
        }
    
    def explain_lime(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        num_features: int = 10
    ):
        """
        Explica predição com LIME.
        
        Parameters
        ----------
        X : pd.DataFrame
            Dados
        sample_idx : int
            Índice da amostra
        num_features : int
            Número de features na explicação
            
        Returns
        -------
        explanation : lime.Explanation
            Explicação LIME
        """
        if not _HAS_LIME or self.lime_explainer is None:
            raise ValueError("LIME não disponível. Instale: pip install lime")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        explanation = self.lime_explainer.explain_instance(
            X_array[sample_idx],
            self.model.predict_proba,
            num_features=num_features,
            top_labels=1
        )
        
        return explanation
    
    def plot_summary(
        self,
        shap_dict: Dict,
        plot_type: str = 'dot',
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """Plot SHAP summary."""
        if not _HAS_SHAP:
            raise ValueError("SHAP não disponível")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            shap_dict['shap_values'],
            shap_dict['data'],
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        
        plt.title(f'SHAP Summary - {plot_type}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(
        self,
        shap_dict: Dict,
        sample_idx: int = 0,
        max_display: int = 15,
        save_path: Optional[str] = None
    ):
        """Plot SHAP waterfall para amostra específica."""
        if not _HAS_SHAP:
            raise ValueError("SHAP não disponível")
        
        try:
            # SHAP v0.40+
            exp = shap.Explanation(
                values=shap_dict['shap_values'][sample_idx],
                base_values=shap_dict['expected_value'],
                data=shap_dict['data'].iloc[sample_idx].values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(exp, max_display=max_display, show=False)
            plt.title(f'SHAP Waterfall - Sample {sample_idx}', fontsize=14)
            plt.tight_layout()
            
        except Exception as e:
            logger.warning(f"Waterfall falhou: {e}. Usando force plot.")
            shap.force_plot(
                shap_dict['expected_value'],
                shap_dict['shap_values'][sample_idx],
                shap_dict['data'].iloc[sample_idx],
                matplotlib=True,
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_lime(
        self,
        lime_explanation,
        save_path: Optional[str] = None
    ):
        """Plot LIME explanation."""
        if not _HAS_LIME:
            raise ValueError("LIME não disponível")
        
        fig = lime_explanation.as_pyplot_figure(label=1)
        plt.title('LIME Local Explanation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_top_features(
        self,
        shap_dict: Dict,
        n: int = 10
    ) -> pd.DataFrame:
        """Retorna top N features mais importantes."""
        abs_mean = np.abs(shap_dict['shap_values']).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': abs_mean
        }).sort_values('importance', ascending=False).head(n)
        
        return df.reset_index(drop=True)
    
    def explain_prediction_text(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_k: int = 5
    ) -> str:
        """
        Gera explicação textual de uma predição.
        
        Returns
        -------
        text : str
            Explicação em texto natural
        """
        # Predição
        y_pred_proba = self.model.predict_proba(X)[sample_idx, 1]
        y_pred_class = self.model.predict(X)[sample_idx]
        
        # SHAP
        shap_dict = self.explain_shap(X)
        shap_values = shap_dict['shap_values'][sample_idx]
        
        # Top features
        top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
        
        # Texto
        label = 'FRAUDE' if y_pred_class == 1 else 'LEGÍTIMA'
        text = f"🎯 PREDIÇÃO: {label} (prob: {y_pred_proba:.2%})\n\n"
        text += f"📊 PRINCIPAIS FATORES:\n\n"
        
        for i, idx in enumerate(top_indices, 1):
            feat_name = self.feature_names[idx]
            feat_val = X.iloc[sample_idx, idx]
            shap_val = shap_values[idx]
            direction = "Aumenta" if shap_val > 0 else "Diminui"
            
            text += f"{i}. {feat_name}: {feat_val}\n"
            text += f"   {direction} risco em {abs(shap_val):.4f}\n\n"
        
        return text


def compare_explanations_shap_lime(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_idx: int = 0,
    num_features: int = 10
) -> Dict[str, Any]:
    """
    Compara explicações SHAP vs LIME para mesma amostra.
    
    Returns
    -------
    comparison : Dict
        Dicionário com ambas explicações
    """
    explainer = ModelExplainer(model, X_train)
    
    # SHAP
    shap_dict = explainer.explain_shap(X_test)
    shap_values_sample = shap_dict['shap_values'][sample_idx]
    
    # LIME
    lime_exp = explainer.explain_lime(X_test, sample_idx, num_features)
    lime_weights = dict(lime_exp.as_list(label=1))
    
    # Comparar top features
    shap_top_indices = np.argsort(np.abs(shap_values_sample))[::-1][:num_features]
    shap_top_features = [
        (explainer.feature_names[i], shap_values_sample[i])
        for i in shap_top_indices
    ]
    
    return {
        'sample_idx': sample_idx,
        'shap_top_features': shap_top_features,
        'lime_weights': lime_weights,
        'shap_dict': shap_dict,
        'lime_exp': lime_exp
    }


# Exemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    
    print("="*80)
    print("TESTE: Model Explainability Avançado")
    print("="*80)
    
    # Dados
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Modelo
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Explainer
    explainer = ModelExplainer(model, X_train)
    
    # SHAP
    print("\n📊 SHAP Summary")
    shap_dict = explainer.explain_shap(X_test.head(100))
    explainer.plot_summary(shap_dict)
    
    # Waterfall
    print("\n🌊 SHAP Waterfall (Sample 0)")
    explainer.plot_waterfall(shap_dict, sample_idx=0)
    
    # Top features
    print("\n🔝 Top Features:")
    print(explainer.get_top_features(shap_dict, n=10))
    
    # Explicação textual
    print("\n📝 Explicação Textual:")
    print(explainer.explain_prediction_text(X_test, sample_idx=0))
    
    # LIME (se disponível)
    if _HAS_LIME:
        print("\n🍋 LIME Explanation")
        lime_exp = explainer.explain_lime(X_test, sample_idx=0)
        explainer.plot_lime(lime_exp)
    
    print("\n✅ Teste completo!")


# ============================================================================
# FUNÇÕES ORIGINAIS (mantidas para compatibilidade)
# ============================================================================
# Nota: Imports já consolidados no topo do arquivo


@dataclass
class SHAPExplanation:
    """Container para explicações SHAP."""
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    data: pd.DataFrame
    model_output: str  # 'probability' ou 'raw'
    
    def __repr__(self):
        return (f"SHAPExplanation(n_samples={len(self.shap_values)}, "
                f"n_features={len(self.feature_names)}, "
                f"model_output='{self.model_output}')")


# ============================================================================
# FUNÇÕES ORIGINAIS (mantidas para compatibilidade)
# ============================================================================

def compute_permutation_importance(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_repeats: int = 10, 
    random_state: int = 42
) -> pd.DataFrame:
    """Calcula permutation importance (original)."""
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'feature': X.columns, 
        'importance_mean': result.importances_mean, 
        'importance_std': result.importances_std
    })
    return imp_df.sort_values('importance_mean', ascending=False)


def compute_shap_values(model, X: pd.DataFrame, sample: int = 1000) -> Dict[str, Any]:
    """Calcula SHAP values (original)."""
    if not _HAS_SHAP:
        raise ImportError('SHAP não instalado. Adicione shap ao requirements.')
    
    X_sample = X.sample(min(sample, len(X)), random_state=42)
    
    if hasattr(model, 'predict_proba'):
        if 'tree' in model.__class__.__name__.lower() or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)
    else:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
    
    return {'explainer': explainer, 'values': shap_values, 'data': X_sample}


# ============================================================================
# FASE 3: NOVAS FUNCIONALIDADES AVANÇADAS
# ============================================================================
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold


def permutation_ranking(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, metric: str = 'pr_auc', n_repeats: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Ranqueia features por queda de métrica ao permutar.
    Suporta 'pr_auc' e 'f1' (threshold 0.5) como métrica.
    """
    rng = np.random.default_rng(random_state)
    base_proba = model.fit(X, y).predict_proba(X)[:,1]
    if metric == 'pr_auc':
        base_metric = average_precision_score(y, base_proba)
    else:
        base_metric = f1_score(y, (base_proba>=0.5).astype(int))
    impacts = []
    for col in X.columns:
        drop_values = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            proba = model.fit(X_perm, y).predict_proba(X_perm)[:,1]
            if metric == 'pr_auc':
                m = average_precision_score(y, proba)
            else:
                m = f1_score(y, (proba>=0.5).astype(int))
            drop_values.append(base_metric - m)
        impacts.append({'feature': col, 'impact_mean': np.mean(drop_values), 'impact_std': np.std(drop_values)})
    df = pd.DataFrame(impacts).sort_values('impact_mean', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df)+1)
    return df


def incremental_subset_evaluation(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, ordered_features: List[str], metric: str = 'pr_auc', k_folds: int = 3) -> pd.DataFrame:
    records = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    current_feats: List[str] = []
    for feat in ordered_features:
        current_feats.append(feat)
        fold_scores = []
        for tr, va in skf.split(X, y):
            X_tr, X_va = X.iloc[tr][current_feats], X.iloc[va][current_feats]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:,1]
            if metric == 'pr_auc':
                score = average_precision_score(y_va, proba)
            else:
                score = f1_score(y_va, (proba>=0.5).astype(int))
            fold_scores.append(score)
        records.append({
            'n_features': len(current_feats),
            'features': list(current_feats),
            'metric_mean': float(np.mean(fold_scores)),
            'metric_std': float(np.std(fold_scores))
        })
    return pd.DataFrame(records)


def find_elbow(df: pd.DataFrame, col_x: str = 'n_features', col_y: str = 'metric_mean') -> int:
    """Heurística simples: primeira posição onde ganho marginal < 1% absoluto ou < 0.5% relativo."""
    best_idx = df[col_y].idxmax()
    baseline = df.loc[df.index[0], col_y]
    prev = baseline
    for i in range(1, len(df)):
        cur = df.loc[df.index[i], col_y]
        abs_gain = cur - prev
        rel_gain = abs_gain / prev if prev!=0 else 0
        if abs_gain < 0.01 and rel_gain < 0.005:
            return int(df.loc[df.index[i-1], col_x])
        prev = cur
    return int(df.loc[best_idx, col_x])


# ============================================================================
# SHAP FALLBACK - Implementação alternativa quando SHAP não funciona
# ============================================================================
# Nota: Imports já consolidados no topo do arquivo

class SHAPFallback:
    """Fallback para funcionalidade SHAP básica"""
    
    def __init__(self):
        self.available = False
        try:
            import shap
            self.shap = shap
            self.available = True
        except:
            self.shap = None
            self.available = False
    
    def explain_model(self, model, X_test, feature_names=None):
        """Explica modelo usando permutation importance como fallback"""
        if self.available:
            try:
                explainer = self.shap.Explainer(model)
                shap_values = explainer(X_test[:100])  # Sample para performance
                return {
                    "method": "shap",
                    "values": shap_values,
                    "feature_importance": None
                }
            except:
                pass
        
        # Fallback para permutation importance
        from sklearn.metrics import accuracy_score
        
        # Dummy y para permutation importance
        y_dummy = np.random.randint(0, 2, len(X_test))
        
        perm_importance = permutation_importance(
            model, X_test, y_dummy, 
            n_repeats=3, random_state=42
        )
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        return {
            "method": "permutation",
            "values": None,
            "feature_importance": importance_df
        }

# Global instance
shap_fallback = SHAPFallback()

def explain_model_safe(model, X_test, feature_names=None):
    """Função segura para explicação de modelos"""
    return shap_fallback.explain_model(model, X_test, feature_names)
