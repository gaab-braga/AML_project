"""
Feature Engineering and Preprocessing Utilities for AML Detection

Este mega-módulo consolidado contém 7 sub-módulos:
1. IV Analysis & Feature Selection
2. Custom Encoders (Frequency, Safe Label, Pipeline)
3. Feature Engineering Pipeline (modular functions)
4. Feature Engineering Utilities (time, encoding, balancing)
5. Feature Manifest (versioning & lineage tracking)
6. Graph Features (NetworkX integration)
7. Dataset Governance (hashing & version control)

IMPORTANTE: Este arquivo foi consolidado de múltiplos módulos.
Imports duplicados foram removidos e marcados com comentários.
"""

import pandas as pd
import numpy as np
import hashlib
import warnings
import json
import pickle
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict

# Third-party imports
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

warnings.filterwarnings('ignore')

# Internal imports (relative)
try:
    from .metrics import ThresholdConfig, evaluate_thresholds as _core_threshold_evaluate
except ImportError:
    ThresholdConfig = None
    _core_threshold_evaluate = None

__all__ = [
    # IV Analysis (3)
    'calculate_iv',
    'calculate_iv_safe',
    'analise_iv',
    
    # EDA Reports (1)
    'gera_relatorios_aed',
    
    # Visualization (1)
    'plot_elbow_curve',
    
    # Custom Encoders (3 classes)
    'FrequencyEncoder',
    'SafeLabelEncoder',
    'FeatureEngineeringPipeline',
    
    # Feature Selection Pipeline (6)
    'run_iv_feature_selection',
    'create_frequency_features',
    'prepare_final_dataset',
    'create_simple_graph_features',
    'apply_label_encoding',
    'apply_dataset_balancing',
    
    # Core Feature Selection (3)
    'run_core_feature_selection',
    'save_feature_selection_artifacts',
    'compare_scaling_methods_simple',
    
    # Complete Pipeline (2)
    'run_complete_feature_engineering_pipeline',
    'save_all_datasets',
    
    # Feature Engineering Utilities (10)
    'extract_time_features',
    'frequency_encode',
    'reduce_cardinality',
    'target_encode_safe',
    'compute_iv_and_select',
    'balance_dataset',
    'build_feature_matrix',
    'evaluate_thresholds',
    'generate_lift_table',
    'compare_models',
    
    # Scaling (4)
    'apply_feature_scaling',
    'load_scaler',
    'transform_with_saved_scaler',
    'compare_scaling_methods',
    
    # Advanced Temporal Features (8)
    'create_rolling_features',
    'create_velocity_features',
    'create_lag_features',
    'create_time_since_features',
    'create_seasonality_features',
    'create_deviation_features',
    'create_all_temporal_features',
    'create_advanced_features',
    
    # Feature Manifest (3 classes + 1 function)
    'FeatureDefinition',
    'DatasetManifest',
    'FeatureManifestManager',
    'create_feature_manifest_for_pipeline',
    
    # Graph Features (3)
    'build_transaction_graph',
    'graph_centrality_features',
    'attach_graph_features',
    
    # Dataset Governance (8)
    'hash_dataset',
    'get_git_commit',
    'get_git_branch',
    'get_git_status',
    'create_dataset_manifest',
    'save_manifest',
    'verify_dataset_integrity',
    'enhance_metadata_with_governance',
    
    # Feature Ranking (3)
    'permutation_ranking',
    'incremental_subset_evaluation',
    'find_elbow',
    
    # Configuration
    'DEFAULT_CONFIG',
]


# ============================================================================
# SUB-MODULE 1: IV ANALYSIS & FEATURE SELECTION
# ============================================================================

def calculate_iv(df, target_col, bins=10, max_iv=10.0, min_samples=10):
    """Cálculo de Information Value (IV) baseado em WOE com proteções contra overfitting."""
    iv_list = []
    # Total de eventos (1) e não-eventos (0)
    total_eventos = (df[target_col] == 1).sum()
    total_naoeventos = (df[target_col] == 0).sum()
    
    # Verificar se há casos suficientes
    if total_eventos < 5 or total_naoeventos < 5:
        print(f"Aviso: Poucos casos para cálculo confiável de IV (eventos={total_eventos}, não-eventos={total_naoeventos})")

    for col in df.columns:
        if col == target_col:
            continue
        try:
            x = df[col].copy()
            
            # Criar bins se for numérica contínua
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
                x_binned = pd.qcut(x, bins, duplicates='drop')
            else:
                # Para variáveis categóricas ou com poucos valores únicos
                # Converter para string primeiro, depois tratar NaN
                x_str = x.astype(str)
                x_binned = x_str.replace('nan', '**NA**')
            
            # Tabela de contingência: contagem de 0 e 1 por categoria/faixa
            df_cut = pd.crosstab(x_binned, df[target_col])
            
            # Se faltar alguma coluna (0 ou 1), pular variável
            if 0 not in df_cut.columns or 1 not in df_cut.columns:
                continue
            
            # Filtrar categorias com poucos samples para evitar overfitting
            df_cut = df_cut[(df_cut[0] + df_cut[1]) >= min_samples]
            
            if len(df_cut) == 0:
                continue
            
            iv = 0.0
            # Calcular WOE e acumular IV
            for idx, row in df_cut.iterrows():
                count_nao, count_ev = row.get(0, 0), row.get(1, 0)
                # Suavização Laplace para evitar zero/infinito
                ev = count_ev + 0.5
                ne = count_nao + 0.5
                pct_ev = ev / (total_eventos + 1)
                pct_ne = ne / (total_naoeventos + 1)
                
                # Calcular WOE com proteção contra valores extremos
                if pct_ev > 0 and pct_ne > 0:
                    woe = np.log(pct_ev / pct_ne)
                    # Limitar WOE para evitar valores extremos
                    woe = np.clip(woe, -5, 5)
                    iv += (pct_ev - pct_ne) * woe
            
            # Limitar IV máximo para evitar overfitting
            iv = min(iv, max_iv)
            iv_list.append({'variável': col, 'IV': iv})
            
        except Exception as e:
            print(f"Erro ao calcular IV para {col}: {e}")
            continue
    
    return pd.DataFrame(iv_list).sort_values(by='IV', ascending=False)


# Manter compatibilidade com nome antigo
def calculate_iv_safe(df, target_col):
    """Wrapper para manter compatibilidade."""
    result = calculate_iv(df, target_col)
    # Renomear colunas para manter compatibilidade
    result = result.rename(columns={'variável': 'Variable'})
    return result


def gera_relatorios_aed(df, target_feat, 
                        html_pp='base_aed_pp.html', 
                        html_sv='base_aed_sv.html'):
    """
    Função desabilitada devido a conflitos de dependências.
    Para usar, instale pandas-profiling e sweetviz compatíveis.
    """
    print("⚠️ Função desabilitada devido a conflitos de dependências")
    print("Para usar relatórios automáticos, instale:")
    print("- ydata-profiling (substituto do pandas-profiling)")
    print("- sweetviz")
    return None, None
   
    

class analise_iv:
        
    # função private
    def __get_tab_bivariada(self, var_escolhida):
     
        # Cria a contagem de Target_1 e Target_0
        df_aux = self.df.copy() 
        df_aux['target2'] = self.df[self.target]
        df2 = df_aux.pivot_table(values='target2',
                                 index=var_escolhida,
                                 columns=self.target,
                                 aggfunc='count')
        
        df2 = df2.rename(columns={0:'#Target_0',
                                  1:'#Target_1'})
        df2.fillna(0, inplace=True)

        # Cria as demais colunas da tabela bivariada
        df2['Total'] = (df2['#Target_0'] + df2['#Target_1'])
        df2['%Freq'] = (df2['Total'] / (df2['Total'].sum()) * 100).round(decimals=2)
        df2['%Target_1'] = (df2['#Target_1'] / (df2['#Target_1'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = (df2['#Target_0'] / (df2['#Target_0'].sum()) * 100).round(decimals=2)
        df2['%Target_0'] = df2['%Target_0'].apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['%Taxa_de_Target_1'] = (df2['#Target_1'] / df2['Total'] * 100).round(decimals=2)
        df2['Odds'] = (df2['%Target_1'] / df2['%Target_0']).round(decimals=2)
        df2['Odds'] = df2.Odds.apply(lambda x: 0.01 if x == 0 else x) #corrige problema do log indeterminado
        df2['LN(Odds)'] = np.log(df2['Odds']).round(decimals=2)
        df2['IV'] = (((df2['%Target_1'] / 100 - df2['%Target_0'] / 100) * df2['LN(Odds)'])).round(decimals=2)
        df2['IV'] = np.where(df2['Odds'] == 0.01, 0 , df2['IV']) 

        df2 = df2.reset_index()
        df2['Variavel'] = var_escolhida
        df2 = df2.rename(columns={var_escolhida: 'Var_Range'})
        df2 = df2[['Variavel','Var_Range', '#Target_1','#Target_0', 'Total', '%Freq', '%Target_1', '%Target_0',
       '%Taxa_de_Target_1', 'Odds', 'LN(Odds)', 'IV']]
        
        # Guarda uma cópia da tabela no histórico
        self.df_tabs_iv = pd.concat([self.df_tabs_iv, df2], axis = 0)
        
        return df2
        
    def get_bivariada(self, var_escolhida='all_vars'):
        
        if var_escolhida == 'all_vars':
                       
            #vars = self.df.drop(self.target,axis = 1).columns
            vars = self.get_lista_iv().index
            for var in vars:
                tabela = self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]
                print('==> "{}" tem IV de {}'.format(var,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var]['IV'].sum().round(decimals=2)))
                # printa a tabela no Jupyter
                display(tabela)
            
            return
        
        else:
            print('==> "{}" tem IV de {}'.format(var_escolhida,self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]['IV'].sum().round(decimals=2)))
            return self.df_tabs_iv[self.df_tabs_iv['Variavel'] == var_escolhida]
                   
            
    def get_lista_iv(self):
        
    
        # agrupa a lista de IV's em ordem descrescente
        lista = (self.df_tabs_iv.groupby('Variavel').agg({'IV':'sum'})).sort_values(by=['IV'],ascending=False)
            
        return lista
    
    

    def __init__(self, df, target, nbins=10):

        self.df = df.copy()
        self.target = target

        #lista de variaveis numericas
        df_num = self.df.loc[:,((self.df.dtypes == 'int32') | 
                                (self.df.dtypes == 'int64') | 
                                (self.df.dtypes == 'float64')
                               )
                            ]

        vars = df_num.drop(target,axis = 1).columns

        for var in vars:
            nome_var = 'fx_' + var 
            df_num[nome_var] = pd.qcut(df_num[var], 
                                       q=nbins, 
                                       precision=2,
                                       duplicates='drop')
            df_num = df_num.drop(var, axis = 1)
            df_num = df_num.rename(columns={nome_var: var})

        #lista de variaveis qualitativas
        df_str = self.df.loc[:,((self.df.dtypes == 'object') | 
                                (self.df.dtypes == 'category') |
                                (self.df.dtypes == 'bool'))]


        self.df = pd.concat([df_num,df_str],axis = 1)


         # inicializa tab historica
        self.df_tabs_iv = pd.DataFrame()

        vars = self.df.drop(self.target,axis = 1).columns
        for var in vars:
            self.__get_tab_bivariada(var);

        # remove tabs de iv duplicadas
        self.df_tabs_iv = self.df_tabs_iv.drop_duplicates(subset=['Variavel','Var_Range'], keep='last')


def plot_elbow_curve(curve_df, elbow_n, core_metric, metric_name='PR_AUC', figsize=(10, 6)):
    """
    Plota a curva de elbow para seleção de features.
    
    Parameters:
    -----------
    curve_df : pd.DataFrame
        DataFrame com colunas 'n_features' e 'metric_mean'
    elbow_n : int
        Número de features no ponto de elbow
    core_metric : float
        Valor da métrica no ponto de elbow
    metric_name : str
        Nome da métrica para labels (default: 'PR_AUC')
    figsize : tuple
        Tamanho da figura (default: (10, 6))
    
    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    plt.plot(curve_df['n_features'], curve_df['metric_mean'], 'b-o', linewidth=2, markersize=4)
    plt.axvline(x=elbow_n, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Elbow Point (n={elbow_n})')
    plt.axhline(y=core_metric, color='red', linestyle=':', alpha=0.5)
    
    # Destacar o ponto de elbow
    plt.plot(elbow_n, core_metric, 'ro', markersize=8, label=f'{metric_name}={core_metric:.3f}')
    
    plt.xlabel('Número de Features')
    plt.ylabel(f'{metric_name} Score')
    plt.title('Curva de Performance vs Número de Features (Método Elbow)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adicionar texto com informações
    plt.text(0.02, 0.98, f'Ponto ótimo: {elbow_n} features\nPerformance: {core_metric:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Calcular e mostrar ganho marginal
    max_metric = curve_df['metric_mean'].max()
    marginal_gain = max_metric - core_metric
    print(f'Ganho marginal após {elbow_n} features: {marginal_gain:.4f}')
        
"""
Custom Encoders for ML Pipeline - Anti-Money Laundering
Prevents data leakage by fitting only on training data
"""
# import numpy as np  # Already imported at top
# import pandas as pd  # Already imported at top
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoder that prevents data leakage."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.freq_maps_ = {}
    
    def fit(self, X, y=None):
        """Fit frequency maps on training data only."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self.columns if self.columns else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols:
            if col in X.columns:
                self.freq_maps_[col] = X[col].value_counts().to_dict()
        return self
    
    def transform(self, X):
        """Transform using fitted frequency maps."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, freq_map in self.freq_maps_.items():
            if col in X.columns:
                # Unseen categories get frequency 0
                X[f'{col}_freq'] = X[col].map(freq_map).fillna(0).astype(int)
        return X


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder that handles unseen categories gracefully."""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.label_maps_ = {}
    
    def fit(self, X, y=None):
        """Fit label maps on training data only."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        cols = self.columns if self.columns else X.select_dtypes(include=['object', 'category']).columns
        
        for col in cols:
            if col in X.columns:
                # Create label map: {category: label}
                unique_vals = sorted(X[col].dropna().unique())
                self.label_maps_[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self
    
    def transform(self, X):
        """Transform using fitted label maps."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, label_map in self.label_maps_.items():
            if col in X.columns:
                # Unseen categories get label -1 (or map to first category)
                X[col] = X[col].map(label_map).fillna(-1).astype(int)
        
        # Ensure all columns are numeric (critical for LightGBM)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
        
        return X


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Complete feature engineering pipeline without leakage."""
    
    def __init__(self, categorical_cols=None, high_card_cols=None):
        self.categorical_cols = categorical_cols or []
        self.high_card_cols = high_card_cols or []
        self.freq_encoder = FrequencyEncoder(columns=high_card_cols)
        self.label_encoder = SafeLabelEncoder(columns=categorical_cols)
    
    def fit(self, X, y=None):
        """Fit all encoders on training data."""
        # Fit frequency encoder
        if self.high_card_cols:
            self.freq_encoder.fit(X, y)
        
        # Fit label encoder
        if self.categorical_cols:
            self.label_encoder.fit(X, y)
        
        return self
    
    def transform(self, X):
        """Apply all transformations."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Apply frequency encoding
        if self.high_card_cols:
            X = self.freq_encoder.transform(X)
        
        # Apply label encoding
        if self.categorical_cols:
            X = self.label_encoder.transform(X)
        
        return X

# -*- coding: utf-8 -*-
"""
Feature Engineering Pipeline - Modular Functions
==============================================

Funções modularizadas para feature engineering otimizado:
- Feature selection com Information Value
- Frequency encoding para alta cardinalidade  
- Balanceamento de datasets
- Feature scaling e normalização
- Graph features simples
- Core feature selection pipeline

Elimina redundâncias do notebook 03_feature_engineering.ipynb
"""

# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import json
import pickle

warnings.filterwarnings('ignore')

# ============================
# FEATURE SELECTION PIPELINE
# ============================

def run_iv_feature_selection(df, target_col='Is Laundering', min_iv=0.15, max_iv=5.0, min_samples=20):
    """
    Pipeline completo de seleção por Information Value
    
    Returns:
        dict: {
            'selected_features': list,
            'iv_results': DataFrame,
            'iv_filtered': DataFrame
        }
    """
    # Calcular IV com proteções (função já definida neste módulo)
    iv_results = calculate_iv(df, target_col, max_iv=max_iv, min_samples=min_samples)
    
    # Filtrar por IV mínimo e excluir alta cardinalidade
    exclude_cols = ['Timestamp', 'Account', 'Dest Account', 'Account.1']
    
    iv_filtered = iv_results[
        (~iv_results['variável'].isin(exclude_cols)) & 
        (iv_results['IV'] >= min_iv)
    ].copy()
    
    selected_features = iv_filtered['variável'].tolist()
    
    print(f"✅ Features selecionadas por IV: {len(selected_features)} (IV ≥ {min_iv})")
    
    return {
        'selected_features': selected_features,
        'iv_results': iv_results,
        'iv_filtered': iv_filtered
    }

def create_frequency_features(df, high_card_cols=None, min_freq=5, suffix='_freq'):
    """
    Aplicar frequency encoding para variáveis de alta cardinalidade
    
    Returns:
        tuple: (df_enhanced, freq_mappings)
    """
    if high_card_cols is None:
        high_card_cols = ['From Bank', 'To Bank', 'Account', 'Dest Account']
    
    # Importar função de feature engineering
    import sys
    sys.path.append('..')
    from utils import feature_engineering as fe
    
    df_enhanced, freq_mappings = fe.frequency_encode(
        df, 
        cols=high_card_cols, 
        min_freq=min_freq,
        suffix=suffix
    )
    
    engineered_features = [f"{col}{suffix}" for col in high_card_cols 
                          if f"{col}{suffix}" in df_enhanced.columns]
    
    print(f"✅ Frequency encoding: {len(engineered_features)} features criadas")
    
    return df_enhanced, freq_mappings, engineered_features

def prepare_final_dataset(df_enhanced, iv_selected_features, engineered_features, target_col='Is Laundering'):
    """
    Combinar features selecionadas e preparar dataset final
    
    Returns:
        tuple: (X_final, y_final, final_features)
    """
    # Combinar features
    final_features = iv_selected_features + engineered_features
    
    # Dataset final
    X_final = df_enhanced[final_features]
    y_final = df_enhanced[target_col]
    
    print(f"✅ Dataset final: {X_final.shape} | "
          f"Features: {len(iv_selected_features)} IV + {len(engineered_features)} engineered")
    print(f"Desbalanceamento: {y_final.value_counts()[0]/y_final.value_counts()[1]:.0f}:1")
    
    return X_final, y_final, final_features

# ============================
# GRAPH FEATURES
# ============================

def create_simple_graph_features(df):
    """
    Criar features básicas de conectividade sem NetworkX
    
    Returns:
        DataFrame: Dataset com graph features adicionadas
    """
    df_graph = df.copy()
    
    # 1. Features de grau (número de conexões únicas)
    if 'From Bank' in df.columns and 'To Bank' in df.columns:
        # Grau de saída e entrada
        from_counts = df.groupby('From Bank').size()
        to_counts = df.groupby('To Bank').size()
        
        df_graph['from_bank_degree'] = df['From Bank'].map(from_counts).fillna(0)
        df_graph['to_bank_degree'] = df['To Bank'].map(to_counts).fillna(0)
        
        # Self-loops
        df_graph['is_self_loop'] = (df['From Bank'] == df['To Bank']).astype(int)
        
        # Diversidade de conexões
        unique_from = df.groupby('From Bank')['To Bank'].nunique()
        unique_to = df.groupby('To Bank')['From Bank'].nunique()
        
        df_graph['from_bank_diversity'] = df['From Bank'].map(unique_from).fillna(0)
        df_graph['to_bank_diversity'] = df['To Bank'].map(unique_to).fillna(0)
        
        print(f"✅ Graph features criadas: 5 novas features de conectividade")
    
    # 2. Binning de frequências
    freq_cols = [col for col in df.columns if '_freq' in col]
    for col in freq_cols:
        if col in df.columns:
            df_graph[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    
    new_features = set(df_graph.columns) - set(df.columns)
    return df_graph, list(new_features)

# ============================
# ENCODING E BALANCEAMENTO
# ============================

def apply_label_encoding(X, categorical_cols=None):
    """
    Aplicar LabelEncoder nas variáveis categóricas
    
    Returns:
        tuple: (X_encoded, label_encoders)
    """
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    print(f"✅ Label encoding: {len(categorical_cols)} colunas categóricas convertidas")
    
    return X_encoded, label_encoders

def apply_dataset_balancing(X_train, y_train, strategy='smote_under', balance_params=None):
    """
    Aplicar balanceamento usando SMOTE + Undersampling
    
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    import sys
    sys.path.append('..')
    from utils import feature_engineering as fe
    
    if balance_params is None:
        balance_params = {
            'over_strategy': {1: 600},
            'under_strategy': {0: 1800, 1: 600}
        }
    
    X_train_balanced, y_train_balanced = fe.balance_dataset(
        X_train, 
        y_train,
        strategy=strategy,
        params=balance_params
    )
    
    print(f"✅ Balanceamento aplicado: {X_train_balanced.shape} "
          f"({y_train_balanced.sum()} positivos)")
    
    return X_train_balanced, y_train_balanced

# ============================
# CORE FEATURE SELECTION
# ============================

def run_core_feature_selection(X_train, y_train, config=None):
    """
    Pipeline completo de core feature selection
    
    Returns:
        dict: {
            'ranking_df': DataFrame,
            'curve_df': DataFrame, 
            'core_features': list,
            'elbow_n': int,
            'core_metric': float
        }
    """
    # Funções já definidas neste módulo (linhas 3421, 3449, 3475)
    
    # Configurações padrão
    if config is None:
        config = {
            'metric': 'pr_auc',
            'n_repeats': 5,
            'k_folds': 5,
            'random_state': 42,
            'model_params': {
                'n_estimators': 150,
                'max_depth': 3,
                'random_state': 42
            }
        }
    
    # Modelo base
    base_model = GradientBoostingClassifier(**config['model_params'])
    
    # 1. Permutation importance ranking
    ranking_df = permutation_ranking(
        base_model,
        X_train,
        y_train,
        metric=config['metric'],
        n_repeats=config['n_repeats'],
        random_state=config['random_state']
    )
    
    print(f"✅ Ranking completo: {len(ranking_df)} features analisadas")
    
    # 2. Curva incremental
    ordered_features = ranking_df['feature'].tolist()
    curve_df = incremental_subset_evaluation(
        base_model,
        X_train,
        y_train,
        ordered_features,
        metric=config['metric'],
        k_folds=config['k_folds']
    )
    
    # 3. Encontrar elbow
    elbow_n = find_elbow(curve_df)
    core_features = curve_df.loc[curve_df['n_features'] == elbow_n, 'features'].iloc[0]
    core_metric = curve_df.loc[curve_df['n_features'] == elbow_n, 'metric_mean'].iloc[0]
    
    print(f"✅ Elbow encontrado: {elbow_n} features | "
          f"{config['metric'].upper()} = {core_metric:.4f}")
    
    return {
        'ranking_df': ranking_df,
        'curve_df': curve_df,
        'core_features': core_features,
        'elbow_n': elbow_n,
        'core_metric': core_metric,
        'config': config
    }

def save_feature_selection_artifacts(results, artifacts_dir):
    """
    Salvar todos os artefatos de feature selection
    
    Args:
        results: dict retornado por run_core_feature_selection
        artifacts_dir: Path para diretório de artefatos
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)
    
    # Salvar arquivos
    ranking_path = artifacts_dir / 'feature_ranking_permutation.csv'
    curve_path = artifacts_dir / 'feature_core_curve.csv'
    core_txt_path = artifacts_dir / 'core_features.txt'
    metadata_path = artifacts_dir / 'core_features_metadata.json'
    
    results['ranking_df'].to_csv(ranking_path, index=False)
    results['curve_df'].to_csv(curve_path, index=False)
    
    with open(core_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results['core_features']))
    
    # Metadata
    metadata = {
        'created_at_utc': pd.Timestamp.utcnow().isoformat(),
        'metric': results['config']['metric'],
        'n_repeats': results['config']['n_repeats'],
        'k_folds': results['config']['k_folds'],
        'model': 'GradientBoostingClassifier',
        'model_params': results['config']['model_params'],
        'elbow_n_features': int(results['elbow_n']),
        'core_metric_mean': float(results['core_metric']),
        'full_metric_max': float(results['curve_df']['metric_mean'].max()),
        'feature_ranking_rows': int(len(results['ranking_df'])),
        'curve_rows': int(len(results['curve_df']))
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("✅ Artefatos salvos:")
    for path in [ranking_path, curve_path, core_txt_path, metadata_path]:
        print(f"   • {path.name}")
    
    return metadata

# ============================
# SCALING PIPELINE
# ============================

def compare_scaling_methods_simple(X_train, y_train, X_test, y_test, sample_size=1000):
    """
    Comparação rápida de métodos de scaling
    
    Returns:
        dict: {'logistic': DataFrame, 'rf': DataFrame}
    """
    # compare_scaling_methods já definida neste módulo (linha 1638)
    
    # Amostra para comparação rápida
    if len(X_train) > sample_size:
        idx_sample = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[idx_sample]
        y_train_sample = y_train.iloc[idx_sample]
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    # Comparações
    comparison_logistic = compare_scaling_methods(
        X_train_sample, y_train_sample, X_test, y_test,
        methods=['none', 'standard', 'robust', 'minmax'],
        model_type='logistic'
    )
    
    comparison_rf = compare_scaling_methods(
        X_train_sample, y_train_sample, X_test, y_test,
        methods=['none', 'standard', 'robust', 'minmax'],
        model_type='rf'
    )
    
    print("✅ Comparação de scaling concluída")
    
    return {
        'logistic': comparison_logistic,
        'rf': comparison_rf
    }

# ============================
# COMPLETE PIPELINE
# ============================

def run_complete_feature_engineering_pipeline(df_original, config=None, save_artifacts=True):
    """
    Pipeline completo de feature engineering - substituindo múltiplas células
    
    Args:
        df_original: DataFrame original
        config: dict com configurações
        save_artifacts: bool para salvar artefatos
    
    Returns:
        dict: Todos os resultados e datasets processados
    """
    print("🚀 INICIANDO PIPELINE COMPLETO DE FEATURE ENGINEERING")
    print("=" * 60)
    
    if config is None:
        config = {
            'sample_size': 50000,
            'min_iv': 0.15,
            'balance_params': {
                'over_strategy': {1: 600},
                'under_strategy': {0: 1800, 1: 600}
            },
            'scaling_method': 'standard',
            'random_state': 42
        }
    
    results = {}
    
    # 1. Amostragem estratificada
    print("1️⃣ Aplicando amostragem estratificada...")
    if len(df_original) > config['sample_size']:
        df_sample, _ = train_test_split(
            df_original, 
            train_size=config['sample_size']/len(df_original),
            stratify=df_original['Is Laundering'],
            random_state=config['random_state']
        )
        df = df_sample
    else:
        df = df_original
    
    print(f"   Dataset: {df.shape[0]:,} linhas")
    
    # 2. Transformações básicas
    print("2️⃣ Aplicando transformações básicas...")
    df['To Bank'] = df['To Bank'].astype('object')
    df['From Bank'] = df['From Bank'].astype('object')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 3. Features temporais
    print("3️⃣ Extraindo features temporais...")
    import sys
    sys.path.append('..')
    from utils import feature_engineering as fe
    
    df_fe = fe.extract_time_features(df, 'Timestamp', drop_original=True)
    print(f"   Features temporais: {df_fe.shape}")
    
    # 4. Seleção por IV
    print("4️⃣ Seleção por Information Value...")
    iv_results = run_iv_feature_selection(df_fe, min_iv=config['min_iv'])
    results.update(iv_results)
    
    # 5. Frequency encoding
    print("5️⃣ Aplicando frequency encoding...")
    df_enhanced, freq_mappings, engineered_features = create_frequency_features(df_fe)
    results['freq_mappings'] = freq_mappings
    results['engineered_features'] = engineered_features
    
    # 6. Dataset final
    print("6️⃣ Preparando dataset final...")
    X_final, y_final, final_features = prepare_final_dataset(
        df_enhanced, iv_results['selected_features'], engineered_features
    )
    results.update({
        'X_final': X_final,
        'y_final': y_final, 
        'final_features': final_features
    })
    
    # 7. Label encoding
    print("7️⃣ Aplicando label encoding...")
    X_encoded, label_encoders = apply_label_encoding(X_final)
    results['label_encoders'] = label_encoders
    
    # 8. Train/test split
    print("8️⃣ Dividindo treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_final, 
        test_size=0.3, 
        random_state=config['random_state'], 
        stratify=y_final
    )
    
    # 9. Balanceamento
    print("9️⃣ Aplicando balanceamento...")
    X_train_balanced, y_train_balanced = apply_dataset_balancing(
        X_train, y_train, balance_params=config['balance_params']
    )
    
    results.update({
        'X_train': X_train_balanced,
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_test': y_test
    })
    
    # 10. Graph features (opcional)
    print("🔟 Criando graph features...")
    X_train_graph, graph_features = create_simple_graph_features(X_train_balanced)
    X_test_graph, _ = create_simple_graph_features(X_test)
    
    results.update({
        'X_train_graph': X_train_graph,
        'X_test_graph': X_test_graph,
        'graph_features': graph_features
    })
    
    print("\n✅ PIPELINE COMPLETO FINALIZADO!")
    print(f"   📊 Datasets finais: Treino {X_train_balanced.shape} | Teste {X_test.shape}")
    print(f"   🎯 Features finais: {len(final_features)} + {len(graph_features)} graph")
    print(f"   ⚖️ Balanceamento: {y_train_balanced.sum()} positivos")
    
    return results

# ============================
# SAVE/LOAD UTILITIES
# ============================

def save_all_datasets(results, data_dir):
    """
    Salvar todos os datasets gerados
    """
    data_dir = Path(data_dir)
    
    datasets_to_save = [
        ('X_train_engineered.csv', results['X_train']),
        ('X_test_engineered.csv', results['X_test']),
        ('y_train_engineered.csv', results['y_train']),
        ('y_test_engineered.csv', results['y_test']),
        ('X_train_graph.csv', results['X_train_graph']),
        ('X_test_graph.csv', results['X_test_graph'])
    ]
    
    for filename, data in datasets_to_save:
        if data is not None:
            data.to_csv(data_dir / filename, index=False)
    
    # Salvar mappings e features
    with open(data_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(results['label_encoders'], f)
    with open(data_dir / 'freq_mappings.pkl', 'wb') as f:
        pickle.dump(results['freq_mappings'], f)
    with open(data_dir / 'final_features.txt', 'w') as f:
        for feature in results['final_features']:
            f.write(f"{feature}\n")
    
    print("✅ Todos os datasets salvos em:", data_dir)

"""
Feature Engineering Utilities for Money Laundering Detection
============================================================

Este módulo contém funções para engenharia de features, seleção de variáveis
e preparação de dados para modelos de machine learning.
"""

# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Imports já declarados no topo do arquivo
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline as ImbPipeline
# import warnings
# warnings.filterwarnings('ignore')

# Import interno já movido para o topo (linha ~58)
# try:
#     from .metrics import ThresholdConfig, evaluate_thresholds as _core_threshold_evaluate
# except ImportError:
#     ThresholdConfig = None
#     _core_threshold_evaluate = None


def extract_time_features(df, timestamp_col='Timestamp', drop_original=True):
    """
    Extrai features temporais de uma coluna timestamp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com coluna timestamp
    timestamp_col : str
        Nome da coluna timestamp
    drop_original : bool
        Se deve remover a coluna original
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com features temporais adicionadas
    """
    df = df.copy()
    
    # Converter para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extrair features temporais
    df['Year'] = df[timestamp_col].dt.year
    df['Month'] = df[timestamp_col].dt.month
    df['Day'] = df[timestamp_col].dt.day
    df['Hour'] = df[timestamp_col].dt.hour
    df['DayOfWeek'] = df[timestamp_col].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Quarter'] = df[timestamp_col].dt.quarter
    
    # Features de período do dia
    df['PeriodOfDay'] = pd.cut(df['Hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    if drop_original:
        df = df.drop(columns=[timestamp_col])
    
    print(f"✅ Extracted time features from {timestamp_col}")
    return df


def frequency_encode(df, cols, min_freq=10, suffix='_freq'):
    """
    Aplica frequency encoding para variáveis categóricas de alta cardinalidade.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com variáveis categóricas
    cols : list
        Lista de colunas para encoding
    min_freq : int
        Frequência mínima para manter categoria original
    suffix : str
        Sufixo para novas colunas
        
    Returns:
    --------
    pandas.DataFrame, dict
        DataFrame transformado e dicionário de mapeamentos
    """
    df = df.copy()
    freq_maps = {}
    
    for col in cols:
        if col not in df.columns:
            print(f"⚠️ Column {col} not found, skipping...")
            continue
            
        # Calcular frequências
        freq_map = df[col].value_counts().to_dict()
        
        # Criar nova coluna com frequências
        df[col + suffix] = df[col].map(freq_map)
        
        # Opcional: agrupar categorias raras
        rare_categories = [k for k, v in freq_map.items() if v < min_freq]
        if rare_categories:
            df[col + '_grouped'] = df[col].apply(
                lambda x: '__RARE__' if x in rare_categories else x
            )
        
        freq_maps[col] = freq_map
        print(f"✅ Frequency encoded {col}: {len(freq_map)} categories")
    
    return df, freq_maps


def reduce_cardinality(df, col, top_n=50, other_label='__OTHER__'):
    """
    Reduz cardinalidade mantendo apenas top N categorias mais frequentes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame original
    col : str
        Nome da coluna
    top_n : int
        Número de categorias a manter
    other_label : str
        Label para categorias agrupadas
        
    Returns:
    --------
    pandas.DataFrame, dict
        DataFrame transformado e mapeamento
    """
    df = df.copy()
    
    if col not in df.columns:
        print(f"⚠️ Column {col} not found")
        return df, {}
    
    # Obter top N categorias
    top_categories = df[col].value_counts().head(top_n).index.tolist()
    
    # Criar mapeamento
    mapping = {cat: cat for cat in top_categories}
    for cat in df[col].unique():
        if cat not in top_categories:
            mapping[cat] = other_label
    
    # Aplicar mapeamento
    df[col + '_reduced'] = df[col].map(mapping)
    
    print(f"✅ Reduced {col}: {len(df[col].unique())} → {len(df[col + '_reduced'].unique())} categories")
    
    return df, mapping


def target_encode_safe(df, col, target, min_samples=50, smoothing=1.0, cv_folds=5):
    """
    Target encoding com regularização para evitar overfitting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com dados
    col : str
        Coluna a ser encodada
    target : str
        Variável target
    min_samples : int
        Mínimo de amostras para usar encoding específico
    smoothing : float
        Parâmetro de suavização
    cv_folds : int
        Número de folds para cross-validation
        
    Returns:
    --------
    pandas.DataFrame, dict
        DataFrame com nova coluna e mapeamento
    """
    df = df.copy()
    
    # Calcular mean global do target
    global_mean = df[target].mean()
    
    # Calcular mean por categoria
    category_means = df.groupby(col)[target].agg(['mean', 'count']).to_dict('index')
    
    # Aplicar regularização
    encoding_map = {}
    for category, stats in category_means.items():
        cat_mean = stats['mean']
        cat_count = stats['count']
        
        if cat_count < min_samples:
            # Usar mean global para categorias raras
            encoded_value = global_mean
        else:
            # Aplicar suavização: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
            encoded_value = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
        
        encoding_map[category] = encoded_value
    
    # Aplicar encoding
    df[col + '_target_enc'] = df[col].map(encoding_map).fillna(global_mean)
    
    print(f"✅ Target encoded {col}: regularization applied")
    
    return df, encoding_map


def compute_iv_and_select(df, target, min_iv=0.02, exclude_cols=None):
    """
    Calcula IV e seleciona variáveis acima do threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com features
    target : str
        Nome da variável target
    min_iv : float
        IV mínimo para seleção
    exclude_cols : list
        Colunas a excluir da análise
        
    Returns:
    --------
    pandas.DataFrame, list
        DataFrame com IV results e lista de features selecionadas
    """
    # calculate_iv_safe já definida neste módulo
    if exclude_cols is None:
        exclude_cols = []
    
    # Calcular IV para todas as variáveis
    iv_results = calculate_iv_safe(df, target)
    
    # Aplicar filtros
    selected_features = iv_results[iv_results['IV'] >= min_iv]['Variable'].tolist()
    
    # Remover colunas excluídas
    selected_features = [col for col in selected_features if col not in exclude_cols]
    
    print(f"📊 IV Analysis Results:")
    print(f"   Features with IV >= {min_iv}: {len(selected_features)}")
    print(f"   Top 5 features: {selected_features[:5]}")
    
    return iv_results, selected_features


def balance_dataset(X, y, strategy='smote_under', params=None):
    """
    Balanceia dataset usando diferentes estratégias.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Features
    y : pandas.Series or numpy.array
        Target
    strategy : str
        Estratégia de balanceamento: 'smote', 'under', 'smote_under'
    params : dict
        Parâmetros específicos da estratégia
        
    Returns:
    --------
    X_balanced, y_balanced
        Dados balanceados
    """
    if params is None:
        params = {}
    
    print(f"🔄 Applying {strategy} balancing...")
    print(f"   Original distribution: {np.bincount(y)}")
    
    if strategy == 'smote':
        sampler = SMOTE(random_state=42, **params)
        X_bal, y_bal = sampler.fit_resample(X, y)
        
    elif strategy == 'under':
        sampler = RandomUnderSampler(random_state=42, **params)
        X_bal, y_bal = sampler.fit_resample(X, y)
        
    elif strategy == 'smote_under':
        # Pipeline: SMOTE + UnderSampling
        over_strategy = params.get('over_strategy', {1: 1000})
        under_strategy = params.get('under_strategy', {0: 3000, 1: 1000})
        
        pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy=over_strategy, random_state=42)),
            ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42))
        ])
        
        X_bal, y_bal = pipeline.fit_resample(X, y)
    
    else:
        raise ValueError(f"Strategy {strategy} not supported")
    
    print(f"   New distribution: {np.bincount(y_bal)}")
    print(f"✅ Balancing completed!")
    
    return X_bal, y_bal


def build_feature_matrix(df_raw, config):
    """
    Função principal para construir matriz de features.
    
    Parameters:
    -----------
    df_raw : pandas.DataFrame
        DataFrame original
    config : dict
        Configuração do pipeline de features
        
    Returns:
    --------
    pandas.DataFrame, dict
        DataFrame processado e metadados
    """
    print("🔧 Building feature matrix...")
    df = df_raw.copy()
    metadata = {'steps': [], 'mappings': {}}
    
    # 1. Extrair features temporais
    if config.get('extract_time', False):
        timestamp_col = config.get('timestamp_col', 'Timestamp')
        if timestamp_col in df.columns:
            df = extract_time_features(df, timestamp_col)
            metadata['steps'].append('time_extraction')
    
    # 2. Frequency encoding para alta cardinalidade
    freq_cols = config.get('frequency_encode_cols', [])
    if freq_cols:
        df, freq_maps = frequency_encode(df, freq_cols, 
                                       min_freq=config.get('min_freq', 10))
        metadata['mappings']['frequency'] = freq_maps
        metadata['steps'].append('frequency_encoding')
    
    # 3. Redução de cardinalidade
    reduce_cols = config.get('reduce_cardinality_cols', [])
    for col in reduce_cols:
        if col in df.columns:
            df, mapping = reduce_cardinality(df, col, 
                                           top_n=config.get('top_n', 50))
            metadata['mappings'][f'{col}_reduction'] = mapping
    
    # 4. Target encoding (opcional)
    target_encode_cols = config.get('target_encode_cols', [])
    target_col = config.get('target_col')
    if target_encode_cols and target_col:
        for col in target_encode_cols:
            if col in df.columns:
                df, mapping = target_encode_safe(df, col, target_col)
                metadata['mappings'][f'{col}_target'] = mapping
    
    # 5. Limpeza final
    drop_cols = config.get('drop_cols', [])
    if drop_cols:
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=existing_drop_cols)
        metadata['steps'].append(f'dropped_{len(existing_drop_cols)}_cols')
    
    print(f"✅ Feature matrix built: {df.shape}")
    print(f"   Steps applied: {', '.join(metadata['steps'])}")
    
    return df, metadata


def evaluate_thresholds(y_true, y_proba, metric='f1', step=0.01, verbose=True):
    """Wrapper around threshold_optimizer to keep backward compatibility."""
    if ThresholdConfig is None or _core_threshold_evaluate is None:
        raise ImportError(
            "threshold_optimizer module is required for evaluate_thresholds."
        )

    cfg = ThresholdConfig(
        min_threshold=step,
        max_threshold=0.99,
        step=step,
        optimize_metric=metric
    )

    result = _core_threshold_evaluate(y_true, y_proba, cfg)
    results_df = pd.DataFrame(result['curve'])

    best_metrics = result['best_metrics']
    best_threshold = result['best_threshold']
    best_score = best_metrics.get(metric)

    if verbose and best_score is not None:
        print(f"🎯 Best threshold for {metric}: {best_threshold:.3f}")
        print(f"   Best {metric}: {best_score:.4f}")
        print(f"   Precision: {best_metrics.get('precision', np.nan):.4f}")
        print(f"   Recall: {best_metrics.get('recall', np.nan):.4f}")
        print(f"   Accuracy: {best_metrics.get('accuracy', np.nan):.4f}")

    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'results_df': results_df,
        'best_metrics': best_metrics,
    }


def generate_lift_table(y_true, y_proba, q=10):
    """
    Gera tabela de lift/gains para análise de performance.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_proba : array-like
        Probabilidades preditas
    q : int
        Número de quantis
        
    Returns:
    --------
    pandas.DataFrame
        Tabela de lift
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba
    })
    
    # Ordenar por probabilidade decrescente
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    
    # Criar quantis
    df['quantile'] = pd.qcut(df.index, q=q, labels=False) + 1
    
    # Calcular métricas por quantil
    lift_table = df.groupby('quantile').agg({
        'y_true': ['count', 'sum'],
        'y_proba': 'mean'
    }).round(4)
    
    lift_table.columns = ['total', 'positives', 'avg_proba']
    lift_table['positive_rate'] = lift_table['positives'] / lift_table['total']
    
    # Calcular lift
    overall_rate = df['y_true'].mean()
    lift_table['lift'] = lift_table['positive_rate'] / overall_rate
    
    # Calcular gains cumulativos
    lift_table['cum_positives'] = lift_table['positives'].cumsum()
    lift_table['cum_total'] = lift_table['total'].cumsum()
    lift_table['cum_positive_rate'] = lift_table['cum_positives'] / lift_table['cum_total']
    lift_table['gains'] = lift_table['cum_positives'] / df['y_true'].sum()
    
    return lift_table.reset_index()


def compare_models(models_dict, X_train, y_train, X_test, y_test, model_names=None):
    """
    Compara performance de múltiplos modelos.
    
    Parameters:
    -----------
    models_dict : dict
        Dicionário {nome: modelo_treinado}
    X_train, y_train, X_test, y_test : array-like
        Dados de treino e teste
    model_names : list
        Nomes personalizados para os modelos
        
    Returns:
    --------
    pandas.DataFrame
        Tabela comparativa de performance
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    
    results = []
    
    for name, model in models_dict.items():
        try:
            # Predições
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            metrics = {
                'Model': name,
                'Train_Accuracy': accuracy_score(y_train, y_pred_train),
                'Test_Accuracy': accuracy_score(y_test, y_pred_test),
                'Test_Precision': precision_score(y_test, y_pred_test, zero_division=0),
                'Test_Recall': recall_score(y_test, y_pred_test, zero_division=0),
                'Test_F1': f1_score(y_test, y_pred_test, zero_division=0),
                'Test_ROC_AUC': roc_auc_score(y_test, y_proba_test)
            }
            
            results.append(metrics)
            
        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)
    
    # Adicionar ranking
    comparison_df['F1_Rank'] = comparison_df['Test_F1'].rank(ascending=False)
    comparison_df['AUC_Rank'] = comparison_df['Test_ROC_AUC'].rank(ascending=False)
    
    return comparison_df.sort_values('Test_F1', ascending=False)


# Exemplo de configuração padrão
DEFAULT_CONFIG = {
    'extract_time': True,
    'timestamp_col': 'Timestamp',
    'frequency_encode_cols': ['From Bank', 'To Bank', 'Account', 'Dest Account'],
    'reduce_cardinality_cols': ['From Bank', 'To Bank'],
    'target_encode_cols': [],
    'top_n': 50,
    'min_freq': 10,
    'drop_cols': ['Timestamp'],
    'target_col': 'Is Laundering'
}

def apply_feature_scaling(X_train, X_test=None, method='standard', exclude_cols=None, 
                          save_scaler=True, scaler_path='artifacts/scaler.pkl'):
    """
    Apply feature scaling/normalization to datasets.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features to fit scaler and transform
    X_test : pandas.DataFrame, optional
        Test features to transform (if provided)
    method : str
        Scaling method: 'standard', 'robust', 'minmax', 'maxabs', 'none'
    exclude_cols : list, optional
        Columns to exclude from scaling (e.g., binary flags, already normalized features)
    save_scaler : bool
        Whether to save fitted scaler for production use
    scaler_path : str
        Path to save the scaler object
        
    Returns:
    --------
    X_train_scaled, X_test_scaled (or None), scaler
        Scaled datasets and fitted scaler object
    """
    import pickle
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
    
    print(f"Applying {method} scaling...")
    
    if method == 'none':
        print("   Scaling disabled - returning original data")
        return X_train, X_test, None
    
    # Select columns to scale
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    cols_to_keep = [col for col in X_train.columns if col not in cols_to_scale]
    
    print(f"   Scaling {len(cols_to_scale)} numeric columns")
    print(f"   Excluding {len(cols_to_keep)} columns from scaling")
    
    # Initialize scaler
    scaler_map = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler()
    }
    
    if method not in scaler_map:
        raise ValueError(f"Unsupported scaling method: {method}. Choose from {list(scaler_map.keys())}")
    
    scaler = scaler_map[method]
    
    # Fit scaler on training data
    if cols_to_scale:
        scaler.fit(X_train[cols_to_scale])
        
        # Transform training data
        X_train_scaled_array = scaler.transform(X_train[cols_to_scale])
        X_train_scaled_df = pd.DataFrame(
            X_train_scaled_array, 
            columns=cols_to_scale,
            index=X_train.index
        )
        
        # Combine scaled and unscaled columns
        X_train_result = pd.concat([
            X_train_scaled_df,
            X_train[cols_to_keep]
        ], axis=1).reindex(columns=X_train.columns)
        
        # Transform test data if provided
        X_test_result = None
        if X_test is not None:
            X_test_scaled_array = scaler.transform(X_test[cols_to_scale])
            X_test_scaled_df = pd.DataFrame(
                X_test_scaled_array,
                columns=cols_to_scale, 
                index=X_test.index
            )
            
            X_test_result = pd.concat([
                X_test_scaled_df,
                X_test[cols_to_keep]
            ], axis=1).reindex(columns=X_test.columns)
        
        # Save scaler
        if save_scaler:
            scaler_path = Path(scaler_path)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            
            scaler_metadata = {
                'scaler': scaler,
                'method': method,
                'cols_scaled': cols_to_scale,
                'cols_excluded': exclude_cols,
                'feature_names': X_train.columns.tolist(),
                'n_features': len(cols_to_scale)
            }
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_metadata, f)
            
            print(f"   Scaler saved to {scaler_path}")
        
        print(f"SUCCESS: Feature scaling completed using {method} method")
        
        return X_train_result, X_test_result, scaler
    
    else:
        print("   No numeric columns found to scale")
        return X_train, X_test, None


def load_scaler(scaler_path='artifacts/scaler.pkl'):
    """
    Load a previously saved scaler for inference.
    
    Parameters:
    -----------
    scaler_path : str
        Path to saved scaler file
        
    Returns:
    --------
    scaler_metadata : dict
        Dictionary containing scaler and metadata
    """
    import pickle
    from pathlib import Path
    
    scaler_path = Path(scaler_path)
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler_metadata = pickle.load(f)
    
    print(f"SUCCESS: Scaler loaded: {scaler_metadata['method']} method")
    print(f"   Features scaled: {len(scaler_metadata['cols_scaled'])}")
    
    return scaler_metadata


def transform_with_saved_scaler(X, scaler_path='artifacts/scaler.pkl'):
    """
    Transform new data using a previously saved scaler.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features to transform
    scaler_path : str
        Path to saved scaler file
        
    Returns:
    --------
    pandas.DataFrame
        Transformed features
    """
    scaler_metadata = load_scaler(scaler_path)
    scaler = scaler_metadata['scaler']
    cols_to_scale = scaler_metadata['cols_scaled']
    
    # Validate feature compatibility
    missing_cols = [col for col in cols_to_scale if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for scaling: {missing_cols}")
    
    # Transform
    if cols_to_scale:
        X_scaled_array = scaler.transform(X[cols_to_scale])
        X_scaled_df = pd.DataFrame(
            X_scaled_array,
            columns=cols_to_scale,
            index=X.index
        )
        
        # Combine scaled and unscaled columns
        cols_to_keep = [col for col in X.columns if col not in cols_to_scale]
        X_result = pd.concat([
            X_scaled_df,
            X[cols_to_keep]
        ], axis=1).reindex(columns=X.columns)
        
        return X_result
    else:
        return X


def compare_scaling_methods(X_train, y_train, X_test, y_test, 
                           methods=['none', 'standard', 'robust', 'minmax'],
                           model_type='logistic', exclude_cols=None):
    """
    Compare different scaling methods on model performance.
    
    Parameters:
    -----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data
    X_test, y_test : pandas.DataFrame, pandas.Series  
        Test data
    methods : list
        List of scaling methods to compare
    model_type : str
        Type of model to use for comparison: 'logistic', 'rf'
    exclude_cols : list, optional
        Columns to exclude from scaling
        
    Returns:
    --------
    pandas.DataFrame
        Comparison results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    print(f"Comparing scaling methods using {model_type} model...")
    
    # Model selection
    if model_type == 'logistic':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'rf':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    results = []
    
    for method in methods:
        print(f"   Testing {method} scaling...")
        
        # Apply scaling
        X_tr_scaled, X_te_scaled, scaler = apply_feature_scaling(
            X_train, X_test, method=method, exclude_cols=exclude_cols, 
            save_scaler=False
        )
        
        # Train and evaluate model
        model = base_model.fit(X_tr_scaled, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_te_scaled)[:, 1]
        else:
            y_pred_scores = model.decision_function(X_te_scaled)
            y_pred_proba = (y_pred_scores - y_pred_scores.min()) / (y_pred_scores.max() - y_pred_scores.min())
        
        # Compute metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        results.append({
            'scaling_method': method,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'model_type': model_type
        })
    
    results_df = pd.DataFrame(results)
    print(f"SUCCESS: Scaling comparison completed!")
    
    return results_df


# ============================================================================
# FASE 2: Features Temporais Avançadas
# ============================================================================


def create_rolling_features(
    df: pd.DataFrame,
    group_by_col: str,
    value_col: str,
    windows: list = None,
    agg_funcs: list = None,
    timestamp_col: str = 'Timestamp'
) -> pd.DataFrame:
    """
    Cria features de rolling windows (médias móveis, somas, etc).
    
    Útil para capturar padrões temporais e comportamento histórico.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str
        Coluna para agrupar (ex: 'Account', 'From Bank')
    value_col : str
        Coluna com valores (ex: 'Amount Paid', 'Amount Received')
    windows : list, default=[7, 30, 90]
        Janelas de tempo em dias
    agg_funcs : list, default=['mean', 'sum', 'std', 'count']
        Funções de agregação
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de rolling windows
        
    Exemplo
    -------
    >>> df = create_rolling_features(
    ...     df, 
    ...     group_by_col='Account', 
    ...     value_col='Amount Paid',
    ...     windows=[7, 30, 90]
    ... )
    >>> # Cria: Amount_Paid_7d_mean, Amount_Paid_30d_mean, etc.
    """
    df = df.copy()
    
    if windows is None:
        windows = [7, 30, 90]
    if agg_funcs is None:
        agg_funcs = ['mean', 'sum', 'std', 'count']
    
    # Converter timestamp se necessário
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Ordenar por timestamp
    df = df.sort_values([group_by_col, timestamp_col])
    
    print(f"🔄 Criando rolling features para {value_col}...")
    
    for window in windows:
        for agg_func in agg_funcs:
            col_name = f'{value_col}_rolling_{window}d_{agg_func}'
            
            # Rolling window com groupby
            df[col_name] = (
                df.groupby(group_by_col)[value_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).agg(agg_func))
            )
            
            print(f"   ✅ {col_name}")
    
    return df


def create_velocity_features(
    df: pd.DataFrame,
    group_by_col: str,
    timestamp_col: str = 'Timestamp',
    time_windows: list = None
) -> pd.DataFrame:
    """
    Cria features de velocidade (transações por unidade de tempo).
    
    Captura frequência de transações - útil para detectar comportamento anômalo.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str
        Coluna para agrupar (ex: 'Account', 'From Bank')
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
    time_windows : list, default=['1H', '24H', '7D']
        Janelas de tempo (formato pandas: H=hora, D=dia)
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de velocidade
        
    Exemplo
    -------
    >>> df = create_velocity_features(df, group_by_col='Account')
    >>> # Cria: tx_velocity_1H, tx_velocity_24H, tx_velocity_7D
    """
    df = df.copy()
    
    if time_windows is None:
        time_windows = ['1H', '24H', '7D']
    
    # Converter timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df = df.sort_values([group_by_col, timestamp_col])
    
    print(f"⚡ Criando velocity features para {group_by_col}...")
    
    for window in time_windows:
        col_name = f'tx_velocity_{window}'
        
        # Contar transações na janela
        df[col_name] = (
            df.groupby(group_by_col)[timestamp_col]
            .transform(lambda x: x.rolling(window, on=x).count())
        )
        
        print(f"   ✅ {col_name}")
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    group_by_col: str,
    value_cols: list,
    lags: list = None,
    timestamp_col: str = 'Timestamp'
) -> pd.DataFrame:
    """
    Cria features de lag (valores históricos atrasados).
    
    Captura valores de transações anteriores - útil para modelar sequências.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str
        Coluna para agrupar (ex: 'Account')
    value_cols : list
        Colunas para criar lags (ex: ['Amount Paid', 'Amount Received'])
    lags : list, default=[1, 2, 3, 5, 10]
        Número de períodos para atrasar
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de lag
        
    Exemplo
    -------
    >>> df = create_lag_features(
    ...     df, 
    ...     group_by_col='Account',
    ...     value_cols=['Amount Paid'],
    ...     lags=[1, 2, 3]
    ... )
    >>> # Cria: Amount_Paid_lag1, Amount_Paid_lag2, Amount_Paid_lag3
    """
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    # Converter timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df = df.sort_values([group_by_col, timestamp_col])
    
    print(f"📊 Criando lag features...")
    
    for col in value_cols:
        if col not in df.columns:
            print(f"   ⚠️  Coluna {col} não encontrada, pulando...")
            continue
            
        for lag in lags:
            lag_col_name = f'{col}_lag{lag}'
            
            df[lag_col_name] = (
                df.groupby(group_by_col)[col]
                .shift(lag)
            )
            
            print(f"   ✅ {lag_col_name}")
    
    return df


def create_time_since_features(
    df: pd.DataFrame,
    group_by_col: str,
    timestamp_col: str = 'Timestamp',
    unit: str = 'hours'
) -> pd.DataFrame:
    """
    Cria features de tempo desde última transação.
    
    Captura tempo decorrido - útil para detectar transações incomuns.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str
        Coluna para agrupar (ex: 'Account')
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
    unit : str, default='hours'
        Unidade de tempo: 'seconds', 'minutes', 'hours', 'days'
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de time since
        
    Exemplo
    -------
    >>> df = create_time_since_features(df, group_by_col='Account')
    >>> # Cria: hours_since_last_tx
    """
    df = df.copy()
    
    # Converter timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df = df.sort_values([group_by_col, timestamp_col])
    
    print(f"⏱️  Criando time since features...")
    
    # Tempo desde última transação
    df[f'{unit}_since_last_tx'] = (
        df.groupby(group_by_col)[timestamp_col]
        .diff()
        .dt.total_seconds()
    )
    
    # Converter para unidade desejada
    conversions = {
        'seconds': 1,
        'minutes': 60,
        'hours': 3600,
        'days': 86400
    }
    
    if unit in conversions:
        df[f'{unit}_since_last_tx'] = df[f'{unit}_since_last_tx'] / conversions[unit]
    
    # Preencher NaN (primeira transação) com mediana
    median_value = df[f'{unit}_since_last_tx'].median()
    df[f'{unit}_since_last_tx'].fillna(median_value, inplace=True)
    
    print(f"   ✅ {unit}_since_last_tx")
    
    return df


def create_seasonality_features(
    df: pd.DataFrame,
    timestamp_col: str = 'Timestamp'
) -> pd.DataFrame:
    """
    Cria features de sazonalidade (dia da semana, hora, etc).
    
    Captura padrões cíclicos - útil para detectar transações fora do padrão.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de sazonalidade
        
    Exemplo
    -------
    >>> df = create_seasonality_features(df)
    >>> # Cria: day_of_week, hour_of_day, is_weekend, is_business_hours, etc.
    """
    df = df.copy()
    
    # Converter timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    print(f"📅 Criando seasonality features...")
    
    # Dia da semana (0=Segunda, 6=Domingo)
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    
    # Hora do dia (0-23)
    df['hour_of_day'] = df[timestamp_col].dt.hour
    
    # Fim de semana
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Horário comercial (9h-18h, Seg-Sex)
    df['is_business_hours'] = (
        (df['hour_of_day'] >= 9) & 
        (df['hour_of_day'] < 18) & 
        (df['day_of_week'] < 5)
    ).astype(int)
    
    # Período do dia
    df['period_of_day'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # Mês
    df['month'] = df[timestamp_col].dt.month
    
    # Trimestre
    df['quarter'] = df[timestamp_col].dt.quarter
    
    # Dia do mês
    df['day_of_month'] = df[timestamp_col].dt.day
    
    # Features cíclicas (sin/cos para capturar periodicidade)
    # Hora do dia
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # Dia da semana
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Mês
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"   ✅ day_of_week, hour_of_day, is_weekend")
    print(f"   ✅ is_business_hours, period_of_day")
    print(f"   ✅ month, quarter, day_of_month")
    print(f"   ✅ Cyclic features (sin/cos)")
    
    return df


def create_deviation_features(
    df: pd.DataFrame,
    group_by_col: str,
    value_cols: list,
    timestamp_col: str = 'Timestamp',
    window: int = 30
) -> pd.DataFrame:
    """
    Cria features de desvio do comportamento normal.
    
    Captura quanto cada transação se desvia do histórico - útil para anomalias.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str
        Coluna para agrupar (ex: 'Account')
    value_cols : list
        Colunas para calcular desvio (ex: ['Amount Paid'])
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
    window : int, default=30
        Janela de tempo para calcular média/std histórica
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com features de desvio
        
    Exemplo
    -------
    >>> df = create_deviation_features(
    ...     df,
    ...     group_by_col='Account',
    ...     value_cols=['Amount Paid']
    ... )
    >>> # Cria: Amount_Paid_deviation, Amount_Paid_zscore
    """
    df = df.copy()
    
    # Converter timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df = df.sort_values([group_by_col, timestamp_col])
    
    print(f"📈 Criando deviation features (window={window}d)...")
    
    for col in value_cols:
        if col not in df.columns:
            print(f"   ⚠️  Coluna {col} não encontrada, pulando...")
            continue
        
        # Média e std históricos
        rolling_mean = (
            df.groupby(group_by_col)[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        
        rolling_std = (
            df.groupby(group_by_col)[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
        
        # Desvio absoluto
        df[f'{col}_deviation'] = df[col] - rolling_mean
        
        # Z-score (desvio padronizado)
        df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Desvio percentual
        df[f'{col}_deviation_pct'] = (
            (df[col] - rolling_mean) / (rolling_mean + 1e-8) * 100
        )
        
        print(f"   ✅ {col}_deviation, {col}_zscore, {col}_deviation_pct")
    
    return df


def create_all_temporal_features(
    df: pd.DataFrame,
    group_by_col: str = 'Account',
    value_cols: list = None,
    timestamp_col: str = 'Timestamp',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cria todas as features temporais avançadas de uma vez.
    
    Wrapper conveniente que aplica todas as funções de feature engineering temporal.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados
    group_by_col : str, default='Account'
        Coluna para agrupar
    value_cols : list, optional
        Colunas com valores (se None, usa ['Amount Paid', 'Amount Received'])
    timestamp_col : str, default='Timestamp'
        Coluna com timestamp
    verbose : bool, default=True
        Se True, imprime progresso
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame com todas as features temporais
        
    Exemplo
    -------
    >>> df_enriched = create_all_temporal_features(
    ...     df,
    ...     group_by_col='Account',
    ...     value_cols=['Amount Paid', 'Amount Received']
    ... )
    """
    if value_cols is None:
        value_cols = ['Amount Paid', 'Amount Received']
    
    if verbose:
        print("="*80)
        print("🚀 CRIANDO FEATURES TEMPORAIS AVANÇADAS")
        print("="*80)
        print(f"   Group by: {group_by_col}")
        print(f"   Value cols: {value_cols}")
        print(f"   Timestamp: {timestamp_col}")
        print()
    
    # 1. Seasonality
    df = create_seasonality_features(df, timestamp_col)
    
    # 2. Rolling windows
    for col in value_cols:
        if col in df.columns:
            df = create_rolling_features(
                df, group_by_col, col,
                windows=[7, 30, 90],
                agg_funcs=['mean', 'sum', 'std', 'count'],
                timestamp_col=timestamp_col
            )
    
    # 3. Velocity
    df = create_velocity_features(df, group_by_col, timestamp_col)
    
    # 4. Lags
    df = create_lag_features(
        df, group_by_col, value_cols,
        lags=[1, 2, 3, 5],
        timestamp_col=timestamp_col
    )
    
    # 5. Time since
    df = create_time_since_features(df, group_by_col, timestamp_col, unit='hours')
    
    # 6. Deviation
    df = create_deviation_features(
        df, group_by_col, value_cols,
        timestamp_col=timestamp_col,
        window=30
    )
    
    if verbose:
        n_new_features = len(df.columns) - len(value_cols) - 1  # Aproximado
        print()
        print("="*80)
        print(f"✅ FEATURES TEMPORAIS CRIADAS")
        print(f"   Total de features: {len(df.columns)}")
        print(f"   Features temporais: ~{n_new_features}")
        print("="*80)
    
    return df


def create_advanced_features(X, y, agg_stats=None, fit_mode=True):
    """
    Cria features avançadas de interação e agregação para notebook 07.
    DEVE SER APLICADO ANTES DO SMOTE!
    
    Parameters:
    -----------
    X : DataFrame
        Dataset com features base (DEVE incluir 'Timestamp')
    y : Series
        Target variable
    agg_stats : dict
        Estatísticas de agregação (apenas para test - evita leakage)
    fit_mode : bool
        Se True, calcula estatísticas no train. Se False, usa stats do train.
    
    Returns:
    --------
    X_featured : DataFrame
        Dataset com features adicionais (sem Timestamp)
    y : Series
        Target (retornado inalterado)
    agg_stats : dict
        Estatísticas calculadas (para aplicar no test)
    """
    X_feat = X.copy()
    
    # ============================================================================
    # 1. TEMPORAL FEATURES
    # ============================================================================
    print("🕒 Creating temporal features...")
    X_feat['Hour'] = X_feat['Timestamp'].dt.hour
    X_feat['DayOfWeek'] = X_feat['Timestamp'].dt.dayofweek
    X_feat['IsWeekend'] = (X_feat['DayOfWeek'] >= 5).astype(int)
    X_feat['IsBusinessHours'] = ((X_feat['Hour'] >= 9) & (X_feat['Hour'] <= 18)).astype(int)
    X_feat['IsNightTransaction'] = ((X_feat['Hour'] >= 0) & (X_feat['Hour'] <= 6)).astype(int)
    
    # ============================================================================
    # 2. INTERACTION FEATURES
    # ============================================================================
    print("💫 Creating interaction features...")
    X_feat['Amount_per_Hour'] = X_feat['Amount Paid'] / (X_feat['Hour'] + 1)
    X_feat['Amount_From_Bank'] = X_feat['Amount Paid'] * X_feat['From Bank']
    X_feat['Amount_To_Bank'] = X_feat['Amount Paid'] * X_feat['To Bank']
    X_feat['Amount_Account'] = X_feat['Amount Paid'] * X_feat['Account']
    
    # ============================================================================
    # 3. AGGREGATION FEATURES (FIT ON TRAIN ONLY)
    # ============================================================================
    if fit_mode:
        print("📊 Computing aggregation statistics (TRAIN only)...")
        agg_stats = {}
        
        # Agregar por Account
        agg_stats['account'] = X_feat.groupby('Account').agg({
            'Amount Paid': ['count', 'sum', 'mean', 'std'],
            'Timestamp': ['min', 'max']
        }).reset_index()
        agg_stats['account'].columns = ['Account', 'Account_Count', 'Account_Sum', 
                                         'Account_Mean', 'Account_Std', 'Account_MinTime', 'Account_MaxTime']
        
        # Agregar por From Bank
        agg_stats['from_bank'] = X_feat.groupby('From Bank').agg({
            'Amount Paid': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        agg_stats['from_bank'].columns = ['From Bank', 'FromBank_Count', 'FromBank_Sum', 
                                           'FromBank_Mean', 'FromBank_Std']
        
        # Agregar por To Bank
        agg_stats['to_bank'] = X_feat.groupby('To Bank').agg({
            'Amount Paid': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        agg_stats['to_bank'].columns = ['To Bank', 'ToBank_Count', 'ToBank_Sum', 
                                         'ToBank_Mean', 'ToBank_Std']
    else:
        print("📊 Applying aggregation statistics from TRAIN...")
    
    # Merge aggregations
    X_feat = X_feat.merge(agg_stats['account'], on='Account', how='left')
    X_feat = X_feat.merge(agg_stats['from_bank'], on='From Bank', how='left')
    X_feat = X_feat.merge(agg_stats['to_bank'], on='To Bank', how='left')
    
    # Fill NaN for unseen categories (test only)
    agg_cols = [col for col in X_feat.columns if col.startswith(('Account_', 'FromBank_', 'ToBank_'))]
    X_feat[agg_cols] = X_feat[agg_cols].fillna(0)
    
    # ============================================================================
    # 4. DERIVED FEATURES
    # ============================================================================
    print("🧮 Creating derived features...")
    X_feat['Amount_Account_Ratio'] = X_feat['Amount Paid'] / (X_feat['Account_Sum'] + 1)
    
    # Velocity: transações por dia (handle NaN timestamps from unseen accounts in test)
    if pd.api.types.is_datetime64_any_dtype(X_feat['Account_MinTime']):
        X_feat['Account_Days'] = (X_feat['Account_MaxTime'] - X_feat['Account_MinTime']).dt.total_seconds() / 86400
    else:
        X_feat['Account_Days'] = 0
    X_feat['Account_Velocity'] = X_feat['Account_Count'] / (X_feat['Account_Days'] + 1)
    
    X_feat['Amount_vs_Account_Mean'] = (X_feat['Amount Paid'] - X_feat['Account_Mean']) / (X_feat['Account_Std'] + 1)
    
    # ============================================================================
    # 5. CLEANUP
    # ============================================================================
    X_feat = X_feat.drop(columns=['Timestamp', 'Account_MinTime', 'Account_MaxTime', 'Account_Days'], errors='ignore')
    
    print(f"[OK] Created {len(X_feat.columns) - len(X.columns) + 1} new features")
    print(f"   Total features: {len(X_feat.columns)}")
    
    return X_feat, y, agg_stats


print("SUCCESS: Feature Engineering module loaded successfully!")

"""
Feature Manifest Module

Comprehensive feature versioning and lineage tracking for model governance
and reproducibility in production ML systems.
"""

# import json  # Already imported at top
# import pandas as pd  # Already imported at top
# import numpy as np  # Already imported at top
# from datetime import datetime  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, List, Tuple, Optional, Union  # Already imported at top
# import hashlib  # Already imported at top
# import pickle  # Already imported at top
# from dataclasses import dataclass, asdict  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #4


@dataclass
class FeatureDefinition:
    """Definition of a single feature with metadata."""
    name: str
    dtype: str
    source_stage: str  # e.g., 'engineered', 'scaled', 'temporal'
    creation_method: str  # e.g., 'frequency_encoding', 'interaction', 'scaling'
    dependencies: List[str]  # Source features this depends on
    importance_score: Optional[float] = None
    stability_score: Optional[float] = None
    business_meaning: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None


@dataclass 
class DatasetManifest:
    """Manifest for a complete dataset with all features."""
    dataset_name: str
    creation_timestamp: str
    feature_count: int
    sample_count: int
    feature_definitions: List[FeatureDefinition]
    data_hash: str
    source_files: List[str]
    processing_pipeline: List[str]
    quality_metrics: Dict[str, Any]


class FeatureManifestManager:
    """Manages feature versioning and lineage tracking."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.manifest_path = self.artifacts_dir / 'feature_manifest.json'
        self.history_path = self.artifacts_dir / 'feature_manifest_history.json'
        
        # Load existing manifest if available
        self.current_manifest = self._load_current_manifest()
        self.history = self._load_history()
        
    def _load_current_manifest(self) -> Optional[Dict[str, Any]]:
        """Load current feature manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load feature manifest history."""
        if self.history_path.exists():
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataset for change detection."""
        # Use a subset for performance on large datasets
        sample_size = min(1000, len(df))
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df
            
        # Create hash from column names, dtypes, and sample data
        columns_hash = hashlib.md5(str(sorted(df.columns.tolist())).encode()).hexdigest()
        dtypes_hash = hashlib.md5(str(df.dtypes.to_dict()).encode()).hexdigest()
        
        # Sample data hash
        data_str = sample_df.to_string()
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        # Combine hashes
        combined = f"{columns_hash}_{dtypes_hash}_{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def create_feature_definitions(
        self,
        df: pd.DataFrame,
        source_stage: str,
        feature_importance: Optional[Dict[str, float]] = None,
        feature_stability: Optional[Dict[str, float]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> List[FeatureDefinition]:
        """Create feature definitions from DataFrame."""
        
        feature_definitions = []
        
        for col in df.columns:
            # Determine creation method based on column name patterns
            creation_method = self._infer_creation_method(col, processing_metadata)
            
            # Determine dependencies
            dependencies = self._infer_dependencies(col, processing_metadata)
            
            # Get importance and stability scores
            importance = feature_importance.get(col) if feature_importance else None
            stability = feature_stability.get(col) if feature_stability else None
            
            # Infer business meaning
            business_meaning = self._infer_business_meaning(col)
            
            # Create validation rules
            validation_rules = self._create_validation_rules(df[col])
            
            feature_def = FeatureDefinition(
                name=col,
                dtype=str(df[col].dtype),
                source_stage=source_stage,
                creation_method=creation_method,
                dependencies=dependencies,
                importance_score=importance,
                stability_score=stability,
                business_meaning=business_meaning,
                validation_rules=validation_rules
            )
            
            feature_definitions.append(feature_def)
        
        return feature_definitions
    
    def _infer_creation_method(self, column_name: str, metadata: Optional[Dict] = None) -> str:
        """Infer how a feature was created based on naming patterns."""
        col_lower = column_name.lower()
        
        # Check metadata first
        if metadata and 'feature_creation_methods' in metadata:
            if column_name in metadata['feature_creation_methods']:
                return metadata['feature_creation_methods'][column_name]
        
        # Pattern-based inference
        if '_encoded' in col_lower or '_freq' in col_lower:
            return 'frequency_encoding'
        elif '_scaled' in col_lower or '_norm' in col_lower:
            return 'scaling'
        elif '_interaction' in col_lower or '_product' in col_lower:
            return 'interaction'
        elif '_rolling' in col_lower or '_lag' in col_lower:
            return 'temporal_feature'
        elif '_ratio' in col_lower or '_pct' in col_lower:
            return 'ratio_calculation'
        elif 'anomaly' in col_lower or 'outlier' in col_lower:
            return 'anomaly_detection'
        elif 'graph' in col_lower or 'network' in col_lower:
            return 'graph_feature'
        elif col_lower.startswith('is_') or col_lower.endswith('_flag'):
            return 'binary_encoding'
        else:
            return 'direct_feature'
    
    def _infer_dependencies(self, column_name: str, metadata: Optional[Dict] = None) -> List[str]:
        """Infer feature dependencies."""
        # Check metadata first
        if metadata and 'feature_dependencies' in metadata:
            if column_name in metadata['feature_dependencies']:
                return metadata['feature_dependencies'][column_name]
        
        # Pattern-based inference for derived features
        dependencies = []
        col_lower = column_name.lower()
        
        # Interaction features
        if '_interaction' in col_lower:
            # Try to extract base feature names
            parts = column_name.split('_interaction')[0].split('_')
            if len(parts) >= 2:
                dependencies = [f"{parts[0]}_{parts[1]}", f"{parts[2]}_{parts[3]}"] if len(parts) >= 4 else parts
        
        # Ratio features  
        elif '_ratio' in col_lower or '_pct' in col_lower:
            base_name = column_name.split('_ratio')[0].split('_pct')[0]
            dependencies = [f"{base_name}_numerator", f"{base_name}_denominator"]
        
        # Encoded features
        elif '_encoded' in col_lower or '_freq' in col_lower:
            base_name = column_name.replace('_encoded', '').replace('_freq', '')
            dependencies = [base_name]
        
        # Scaled features
        elif '_scaled' in col_lower:
            base_name = column_name.replace('_scaled', '')
            dependencies = [base_name]
        
        return dependencies
    
    def _infer_business_meaning(self, column_name: str) -> str:
        """Infer business meaning from column name."""
        col_lower = column_name.lower()
        
        # AML/Financial domain patterns
        if 'amount' in col_lower or 'value' in col_lower:
            return 'Transaction amount or financial value'
        elif 'freq' in col_lower or 'count' in col_lower:
            return 'Transaction frequency or count metric'
        elif 'time' in col_lower or 'date' in col_lower:
            return 'Temporal information'
        elif 'account' in col_lower:
            return 'Account identifier or metadata'
        elif 'bank' in col_lower:
            return 'Bank or financial institution information'
        elif 'ratio' in col_lower or 'pct' in col_lower:
            return 'Calculated ratio or percentage metric'
        elif 'anomaly' in col_lower or 'outlier' in col_lower:
            return 'Anomaly detection score or flag'
        elif 'graph' in col_lower or 'network' in col_lower:
            return 'Graph network analysis feature'
        elif col_lower.startswith('is_') or col_lower.endswith('_flag'):
            return 'Binary indicator or flag'
        else:
            return 'General feature - business meaning to be defined'
    
    def _create_validation_rules(self, series: pd.Series) -> Dict[str, Any]:
        """Create validation rules for a feature."""
        rules = {}
        
        # Data type validation
        rules['expected_dtype'] = str(series.dtype)
        
        # Missing value patterns
        null_pct = series.isnull().sum() / len(series) * 100
        rules['max_null_percentage'] = min(null_pct * 1.5, 50.0)  # Allow 50% more nulls than current
        
        # Numeric validation
        if pd.api.types.is_numeric_dtype(series):
            # Range validation
            q01 = series.quantile(0.01)
            q99 = series.quantile(0.99)
            
            rules['expected_range'] = {
                'min_value': float(q01 - abs(q01) * 0.1),  # 10% buffer
                'max_value': float(q99 + abs(q99) * 0.1)
            }
            
            # Distribution validation
            rules['distribution_checks'] = {
                'mean_range': [float(series.mean() * 0.8), float(series.mean() * 1.2)],
                'std_range': [float(series.std() * 0.5), float(series.std() * 2.0)]
            }
        
        # Categorical validation
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            unique_values = series.unique()
            if len(unique_values) <= 50:  # Store values for small cardinality
                rules['expected_values'] = [str(v) for v in unique_values if pd.notna(v)]
            else:
                rules['max_unique_values'] = len(unique_values) * 2  # Allow double current cardinality
        
        return rules
    
    def create_dataset_manifest(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_stage: str,
        source_files: List[str],
        processing_pipeline: List[str],
        feature_importance: Optional[Dict[str, float]] = None,
        feature_stability: Optional[Dict[str, float]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> DatasetManifest:
        """Create complete dataset manifest."""
        
        print(f"📋 Creating feature manifest for {dataset_name}...")
        
        # Create feature definitions
        feature_definitions = self.create_feature_definitions(
            df, source_stage, feature_importance, feature_stability, processing_metadata
        )
        
        # Compute data hash
        data_hash = self._compute_data_hash(df)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(df)
        
        # Create manifest
        manifest = DatasetManifest(
            dataset_name=dataset_name,
            creation_timestamp=datetime.utcnow().isoformat(),
            feature_count=len(df.columns),
            sample_count=len(df),
            feature_definitions=feature_definitions,
            data_hash=data_hash,
            source_files=source_files,
            processing_pipeline=processing_pipeline,
            quality_metrics=quality_metrics
        )
        
        print(f"   ✅ Created manifest with {len(feature_definitions)} features")
        return manifest
    
    def _compute_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute data quality metrics."""
        
        metrics = {
            'missing_value_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'numeric_features': int(df.select_dtypes(include=[np.number]).shape[1]),
            'categorical_features': int(df.select_dtypes(include=['object', 'category']).shape[1]),
            'datetime_features': int(df.select_dtypes(include=['datetime64']).shape[1])
        }
        
        # Feature-level quality
        feature_quality = {}
        for col in df.columns:
            feature_quality[col] = {
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique()),
                'cardinality_ratio': float(df[col].nunique() / len(df))
            }
        
        metrics['feature_quality'] = feature_quality
        
        return metrics
    
    def save_manifest(self, manifest: DatasetManifest) -> Path:
        """Save dataset manifest and update history."""
        
        # Convert to dictionary
        manifest_dict = asdict(manifest)
        
        # Update current manifest
        self.current_manifest = manifest_dict
        
        # Save current manifest
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_dict, f, indent=2, ensure_ascii=False)
        
        # Update history
        history_entry = {
            'timestamp': manifest.creation_timestamp,
            'dataset_name': manifest.dataset_name,
            'feature_count': manifest.feature_count,
            'sample_count': manifest.sample_count,
            'data_hash': manifest.data_hash,
            'quality_summary': {
                'missing_value_percentage': manifest.quality_metrics['missing_value_percentage'],
                'duplicate_percentage': manifest.quality_metrics['duplicate_percentage'],
                'numeric_features': manifest.quality_metrics['numeric_features']
            }
        }
        
        self.history.append(history_entry)
        
        # Save history
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Feature manifest saved: {self.manifest_path}")
        return self.manifest_path
    
    def validate_against_manifest(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        tolerance_factor: float = 1.2
    ) -> Dict[str, Any]:
        """Validate new dataset against existing manifest."""
        
        if not self.current_manifest:
            return {
                'status': 'no_baseline',
                'message': 'No existing manifest found for validation'
            }
        
        print(f"✅ Validating {dataset_name} against feature manifest...")
        
        validation_results = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'dataset_name': dataset_name,
            'baseline_manifest': self.current_manifest['dataset_name'],
            'validation_status': 'PASS',
            'issues': [],
            'warnings': [],
            'feature_validation': {}
        }
        
        baseline_features = {fd['name']: fd for fd in self.current_manifest['feature_definitions']}
        
        # Check feature presence
        current_features = set(df.columns)
        baseline_feature_names = set(baseline_features.keys())
        
        missing_features = baseline_feature_names - current_features
        new_features = current_features - baseline_feature_names
        
        if missing_features:
            validation_results['issues'].append(f"Missing features: {list(missing_features)}")
            validation_results['validation_status'] = 'FAIL'
        
        if new_features:
            validation_results['warnings'].append(f"New features detected: {list(new_features)}")
        
        # Validate individual features
        for feature_name in current_features.intersection(baseline_feature_names):
            feature_validation = self._validate_feature(
                df[feature_name], 
                baseline_features[feature_name],
                tolerance_factor
            )
            
            validation_results['feature_validation'][feature_name] = feature_validation
            
            if feature_validation['status'] == 'FAIL':
                validation_results['issues'].extend(feature_validation['issues'])
                validation_results['validation_status'] = 'FAIL'
            elif feature_validation['status'] == 'WARNING':
                validation_results['warnings'].extend(feature_validation['warnings'])
        
        # Overall data quality validation
        current_quality = self._compute_quality_metrics(df)
        baseline_quality = self.current_manifest['quality_metrics']
        
        # Check missing value percentage
        current_missing_pct = current_quality['missing_value_percentage']
        baseline_missing_pct = baseline_quality['missing_value_percentage']
        
        if current_missing_pct > baseline_missing_pct * tolerance_factor:
            validation_results['warnings'].append(
                f"Missing value percentage increased: {current_missing_pct:.2f}% vs baseline {baseline_missing_pct:.2f}%"
            )
        
        # Check feature count
        if len(df.columns) != self.current_manifest['feature_count']:
            validation_results['warnings'].append(
                f"Feature count changed: {len(df.columns)} vs baseline {self.current_manifest['feature_count']}"
            )
        
        print(f"   Validation status: {validation_results['validation_status']}")
        if validation_results['issues']:
            print(f"   Issues found: {len(validation_results['issues'])}")
        if validation_results['warnings']:
            print(f"   Warnings: {len(validation_results['warnings'])}")
        
        return validation_results
    
    def _validate_feature(
        self,
        series: pd.Series,
        baseline_definition: Dict[str, Any],
        tolerance_factor: float
    ) -> Dict[str, Any]:
        """Validate individual feature against baseline."""
        
        validation = {
            'feature_name': series.name,
            'status': 'PASS',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Data type validation
        current_dtype = str(series.dtype)
        expected_dtype = baseline_definition['expected_dtype']
        
        if current_dtype != expected_dtype:
            validation['issues'].append(f"Data type mismatch: {current_dtype} vs expected {expected_dtype}")
            validation['status'] = 'FAIL'
        
        # Validation rules check
        if 'validation_rules' in baseline_definition and baseline_definition['validation_rules']:
            rules = baseline_definition['validation_rules']
            
            # Missing value check
            current_null_pct = series.isnull().sum() / len(series) * 100
            max_null_pct = rules.get('max_null_percentage', 50)
            
            if current_null_pct > max_null_pct:
                validation['issues'].append(
                    f"Null percentage exceeded: {current_null_pct:.2f}% > {max_null_pct:.2f}%"
                )
                validation['status'] = 'FAIL'
            
            # Numeric range validation
            if 'expected_range' in rules and pd.api.types.is_numeric_dtype(series):
                range_rules = rules['expected_range']
                min_val, max_val = series.min(), series.max()
                
                if min_val < range_rules['min_value']:
                    validation['warnings'].append(f"Min value below expected: {min_val} < {range_rules['min_value']}")
                    if validation['status'] == 'PASS':
                        validation['status'] = 'WARNING'
                
                if max_val > range_rules['max_value']:
                    validation['warnings'].append(f"Max value above expected: {max_val} > {range_rules['max_value']}")
                    if validation['status'] == 'PASS':
                        validation['status'] = 'WARNING'
            
            # Categorical validation
            if 'expected_values' in rules:
                current_values = set(str(v) for v in series.unique() if pd.notna(v))
                expected_values = set(rules['expected_values'])
                
                unexpected_values = current_values - expected_values
                if unexpected_values:
                    validation['warnings'].append(f"Unexpected categorical values: {list(unexpected_values)}")
                    if validation['status'] == 'PASS':
                        validation['status'] = 'WARNING'
        
        return validation
    
    def generate_feature_lineage_report(self) -> Dict[str, Any]:
        """Generate comprehensive feature lineage report."""
        
        if not self.current_manifest:
            return {'error': 'No current manifest available'}
        
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'dataset_info': {
                'name': self.current_manifest['dataset_name'],
                'feature_count': self.current_manifest['feature_count'],
                'sample_count': self.current_manifest['sample_count']
            },
            'feature_lineage': {},
            'processing_pipeline': self.current_manifest.get('processing_pipeline', []),
            'quality_summary': self.current_manifest.get('quality_metrics', {}),
            'feature_categories': {}
        }
        
        # Analyze feature lineage and categories
        creation_methods = {}
        source_stages = {}
        dependency_graph = {}
        
        for feature_def in self.current_manifest['feature_definitions']:
            feature_name = feature_def['name']
            
            # Categorize by creation method
            method = feature_def['creation_method']
            if method not in creation_methods:
                creation_methods[method] = []
            creation_methods[method].append(feature_name)
            
            # Categorize by source stage
            stage = feature_def['source_stage']
            if stage not in source_stages:
                source_stages[stage] = []
            source_stages[stage].append(feature_name)
            
            # Build dependency graph
            dependencies = feature_def.get('dependencies', [])
            dependency_graph[feature_name] = {
                'dependencies': dependencies,
                'creation_method': method,
                'business_meaning': feature_def.get('business_meaning', ''),
                'importance_score': feature_def.get('importance_score'),
                'stability_score': feature_def.get('stability_score')
            }
        
        report['feature_categories'] = {
            'by_creation_method': creation_methods,
            'by_source_stage': source_stages
        }
        
        report['feature_lineage'] = dependency_graph
        
        # Feature statistics
        importance_scores = [
            fd.get('importance_score') 
            for fd in self.current_manifest['feature_definitions'] 
            if fd.get('importance_score') is not None
        ]
        
        if importance_scores:
            report['importance_statistics'] = {
                'mean_importance': float(np.mean(importance_scores)),
                'top_10_features': sorted(
                    [(fd['name'], fd.get('importance_score', 0)) 
                     for fd in self.current_manifest['feature_definitions'] 
                     if fd.get('importance_score') is not None],
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
        
        return report


def create_feature_manifest_for_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create feature manifest for all pipeline datasets.
    
    Args:
        data_dir: Directory containing datasets
        artifacts_dir: Directory to save manifest
        config: Configuration dictionary
        
    Returns:
        Manifest creation results
    """
    if config is None:
        config = {'enabled': True}
    
    if not config.get('enabled', True):
        return {'status': 'disabled', 'message': 'Feature manifest creation disabled in config'}
    
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    
    print("📋 Creating comprehensive feature manifest...")
    
    # Initialize manifest manager
    manager = FeatureManifestManager(artifacts_dir)
    
    # Find available datasets
    dataset_configs = [
        {
            'name': 'engineered_features',
            'path': data_dir / 'X_train_engineered.csv',
            'stage': 'engineered',
            'pipeline': ['data_prep', 'feature_engineering']
        },
        {
            'name': 'scaled_features', 
            'path': data_dir / 'X_train_scaled.csv',
            'stage': 'scaled',
            'pipeline': ['data_prep', 'feature_engineering', 'scaling']
        },
        {
            'name': 'temporal_features',
            'path': data_dir / 'X_train_temporal.csv', 
            'stage': 'temporal',
            'pipeline': ['data_prep', 'feature_engineering', 'temporal_split']
        },
        {
            'name': 'graph_features',
            'path': data_dir / 'X_train_graph.csv',
            'stage': 'graph',
            'pipeline': ['data_prep', 'feature_engineering', 'graph_analysis']
        }
    ]
    
    results = {
        'creation_timestamp': datetime.utcnow().isoformat(),
        'datasets_processed': [],
        'datasets_skipped': [],
        'manifest_files': []
    }
    
    # Load feature importance if available
    feature_importance = None
    importance_path = artifacts_dir / 'permutation_importance.csv'
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
            feature_importance = dict(zip(importance_df['feature'], importance_df['importance']))
    
    # Process each dataset
    primary_manifest = None
    for dataset_config in dataset_configs:
        dataset_path = dataset_config['path']
        
        if not dataset_path.exists():
            results['datasets_skipped'].append({
                'name': dataset_config['name'],
                'reason': 'file_not_found',
                'path': str(dataset_path)
            })
            continue
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Create manifest
            manifest = manager.create_dataset_manifest(
                df=df,
                dataset_name=dataset_config['name'],
                source_stage=dataset_config['stage'],
                source_files=[str(dataset_path)],
                processing_pipeline=dataset_config['pipeline'],
                feature_importance=feature_importance
            )
            
            # Save manifest (this updates the current manifest)
            manifest_path = manager.save_manifest(manifest)
            
            results['datasets_processed'].append({
                'name': dataset_config['name'],
                'feature_count': manifest.feature_count,
                'sample_count': manifest.sample_count,
                'data_hash': manifest.data_hash
            })
            
            # Use the most comprehensive dataset as primary
            if dataset_config['name'] == 'scaled_features' or primary_manifest is None:
                primary_manifest = manifest
                
        except Exception as e:
            results['datasets_skipped'].append({
                'name': dataset_config['name'],
                'reason': 'processing_error',
                'error': str(e),
                'path': str(dataset_path)
            })
    
    # Generate lineage report
    if primary_manifest:
        lineage_report = manager.generate_feature_lineage_report()
        lineage_path = artifacts_dir / 'feature_lineage_report.json'
        
        with open(lineage_path, 'w', encoding='utf-8') as f:
            json.dump(lineage_report, f, indent=2, ensure_ascii=False)
        
        results['lineage_report'] = str(lineage_path)
        
        print(f"💾 Feature lineage report saved: {lineage_path}")
    
    results['manifest_files'] = [str(manager.manifest_path), str(manager.history_path)]
    
    print(f"✅ Feature manifest creation completed:")
    print(f"   Processed: {len(results['datasets_processed'])} datasets")
    print(f"   Skipped: {len(results['datasets_skipped'])} datasets")
    
    return results

# import pandas as pd  # Already imported at top
# import networkx as nx  # Already imported at top
# from typing import List, Dict  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #5

def build_transaction_graph(df: pd.DataFrame, src_col: str, dst_col: str, amount_col: str = None):  # type: ignore
    """Build transaction graph from DataFrame (requires NetworkX)."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph features. Install with: pip install networkx")
    
    assert nx is not None  # For type checker
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s = row[src_col]
        d = row[dst_col]
        w = float(row[amount_col]) if amount_col and pd.notnull(row[amount_col]) else 1.0
        if G.has_edge(s, d):
            G[s][d]['weight'] += w
            G[s][d]['count'] += 1
        else:
            G.add_edge(s, d, weight=w, count=1)
    return G

def graph_centrality_features(G) -> pd.DataFrame:  # type: ignore
    """Compute graph centrality features (requires NetworkX)."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph features. Install with: pip install networkx")
    
    assert nx is not None  # For type checker
    
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    deg = dict(G.degree())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    between = nx.betweenness_centrality(G, k=min(500, len(G))) if len(G) > 50 else nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, max_iter=200) if G.number_of_nodes() < 10000 else {}
    df_feat = pd.DataFrame({
        'node': list(G.nodes()),
        'g_degree': [deg.get(n,0) for n in G.nodes()],
        'g_in_degree': [in_deg.get(n,0) for n in G.nodes()],
        'g_out_degree': [out_deg.get(n,0) for n in G.nodes()],
        'g_betweenness': [between.get(n,0) for n in G.nodes()],
        'g_pagerank': [pagerank.get(n,0) for n in G.nodes()],
    })
    return df_feat

def attach_graph_features(df: pd.DataFrame, node_col: str, features: pd.DataFrame) -> pd.DataFrame:
    return df.merge(features, left_on=node_col, right_on='node', how='left')

"""
Utilitários para hash e governança de datasets.

Garante reprodutibilidade através de:
- Hash SHA-256 de datasets
- Captura de git commit atual
- Metadados de versionamento
"""

import hashlib
# import subprocess  # Already imported at top
# import json  # Already imported at top
# import pandas as pd  # Already imported at top
# from pathlib import Path  # Already imported at top
# from typing import Dict, Any, Optional  # Already imported at top
# from datetime import datetime  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #6


def hash_dataset(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calcula hash SHA-256 de um arquivo de dataset.
    
    Args:
        file_path: Caminho para o arquivo
        chunk_size: Tamanho do chunk para leitura (para arquivos grandes)
        
    Returns:
        Hash SHA-256 em hexadecimal
        
    Raises:
        FileNotFoundError: Se arquivo não existe
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Ler em chunks para arquivos grandes
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_git_commit() -> Optional[str]:
    """
    Obtém o hash do commit atual do git.
    
    Returns:
        Hash do commit atual ou None se não estiver em repo git
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """
    Obtém o nome da branch atual do git.
    
    Returns:
        Nome da branch atual ou None se não estiver em repo git
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_status() -> Dict[str, Any]:
    """
    Obtém status detalhado do git repository.
    
    Returns:
        Dicionário com informações do git
    """
    git_info = {
        'commit_hash': get_git_commit(),
        'branch': get_git_branch(),
        'has_uncommitted_changes': False,
        'repository_clean': True
    }
    
    try:
        # Verificar se há mudanças não commitadas
        result = subprocess.run(
            ['git', 'status', '--porcelain'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if result.stdout.strip():
            git_info['has_uncommitted_changes'] = True
            git_info['repository_clean'] = False
            git_info['uncommitted_files'] = result.stdout.strip().split('\n')
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return git_info


def create_dataset_manifest(dataset_path: str, additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Cria manifesto completo de um dataset com hash e git info.
    
    Args:
        dataset_path: Caminho para o dataset
        additional_metadata: Metadados adicionais opcionais
        
    Returns:
        Dicionário com manifesto completo
    """
    dataset_path = Path(dataset_path)
    
    manifest = {
        'dataset_info': {
            'file_path': str(dataset_path),
            'file_name': dataset_path.name,
            'file_size_bytes': dataset_path.stat().st_size if dataset_path.exists() else None,
            'creation_timestamp': datetime.now().isoformat(),
            'sha256_hash': hash_dataset(dataset_path) if dataset_path.exists() else None
        },
        'git_info': get_git_status(),
        'environment_info': {
            'python_version': None,  # Será preenchido quando importar sys
            'working_directory': str(Path.cwd())
        }
    }
    
    # Adicionar informações do pandas se for CSV
    if dataset_path.suffix.lower() == '.csv' and dataset_path.exists():
        try:
            df_sample = pd.read_csv(dataset_path, nrows=5)  # Só para metadados
            manifest['dataset_info'].update({
                'columns': list(df_sample.columns),
                'estimated_rows': None,  # Seria muito lento contar todas
                'dtypes': df_sample.dtypes.to_dict()
            })
        except Exception:
            pass
    
    # Incluir metadados adicionais
    if additional_metadata:
        manifest['additional_metadata'] = additional_metadata
    
    return manifest


def save_manifest(manifest: Dict[str, Any], output_path: str) -> None:
    """
    Salva manifesto em arquivo JSON.
    
    Args:
        manifest: Dicionário do manifesto
        output_path: Caminho para salvar o arquivo JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def verify_dataset_integrity(dataset_path: str, expected_hash: str) -> bool:
    """
    Verifica se o hash de um dataset corresponde ao esperado.
    
    Args:
        dataset_path: Caminho para o dataset
        expected_hash: Hash esperado (SHA-256)
        
    Returns:
        True se hashes coincidirem, False caso contrário
    """
    try:
        current_hash = hash_dataset(dataset_path)
        return current_hash == expected_hash
    except FileNotFoundError:
        return False


def enhance_metadata_with_governance(existing_metadata: Dict[str, Any], 
                                   dataset_path: str) -> Dict[str, Any]:
    """
    Enriquece metadados existentes com informações de governança.
    
    Args:
        existing_metadata: Metadados já existentes
        dataset_path: Caminho do dataset principal
        
    Returns:
        Metadados enriquecidos com hash e git info
    """
    enhanced = existing_metadata.copy()
    
    # Adicionar hash do dataset
    enhanced['source_data_hash'] = hash_dataset(dataset_path)
    
    # Adicionar informações do git
    git_info = get_git_status()
    enhanced['git_commit'] = git_info['commit_hash']
    enhanced['git_branch'] = git_info['branch']
    enhanced['repository_clean'] = git_info['repository_clean']
    
    # Timestamp de quando foi enriquecido
    enhanced['governance_timestamp'] = datetime.now().isoformat()
    
    return enhanced


# Exemplo de uso
if __name__ == "__main__":
    # Teste com arquivo fictício
    test_file = Path("data/df_Money_Laundering.csv")
    
    if test_file.exists():
        print("=== TESTE DE GOVERNANÇA DE DATASET ===")
        
        # Hash do dataset
        dataset_hash = hash_dataset(test_file)
        print(f"Hash do dataset: {dataset_hash[:16]}...")
        
        # Informações do git
        git_info = get_git_status()
        print(f"Git commit: {git_info['commit_hash'][:8] if git_info['commit_hash'] else 'N/A'}")
        print(f"Git branch: {git_info['branch'] or 'N/A'}")
        print(f"Repository clean: {git_info['repository_clean']}")
        
        # Manifesto completo
        manifest = create_dataset_manifest(test_file, {
            "description": "Dataset principal de detecção de lavagem de dinheiro",
            "source": "Simulado para desenvolvimento"
        })
        
        print(f"\nManifesto criado com {len(manifest)} seções principais")
        
        # Salvar manifesto
        save_manifest(manifest, "artifacts/dataset_manifest.json")
        print("Manifesto salvo em artifacts/dataset_manifest.json")
        
    else:
        print(f"Arquivo de teste não encontrado: {test_file}")
        print("Execute este script no diretório raiz do projeto.")


# import numpy as np  # Already imported at top
# import pandas as pd  # Already imported at top
# from typing import List, Dict, Tuple  # Already imported at top
# from sklearn.base import BaseEstimator  # Already imported at top
# from sklearn.metrics import average_precision_score, f1_score  # Already imported at top
# from sklearn.model_selection import StratifiedKFold  # Already imported at top
# DUPLICATE IMPORTS REMOVED - Originated from concatenated file #7


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
