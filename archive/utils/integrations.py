"""
Configuration loader for Money Laundering Detection Pipeline
Centralizes all parameters from config.yaml
"""

__all__ = [
    # Config management
    'get_default_config',
    'load_config',
    'get_paths',
    'get_model_params',
    'get_k_values',
    'get_psi_interpretation',
    'setup_paths',
    'get_ensemble_config',
    # Dependency checking
    'check_dependencies_smart',
    'fix_dependencies_auto',
    'create_dependency_report',
    # Module loading
    'load_internal_modules',
    'get_pipeline_functions',
    # Legacy utilities
    'log_message',
    'gera_relatorios_aed',
    'analise_iv',
    'ks_stat',
    'calcula_desempenho',
    'matriz_confusao',
    'calcula_desempenho_reg',
    'dispersao_modelo',
    'cria_grafico_var_qualitativa',
    'tabela_bivariada',
    # Product summary
    'main',  # product summary main function
]

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    
from pathlib import Path
from typing import Dict, Any, Optional

def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration as fallback when YAML is not available.
    """
    return {
        'random_state': 42,
        'paths': {
            'data': 'data',
            'artifacts': 'artifacts',
            'models': 'models',
            'utils': 'utils'
        },
        'scoring': {
            'k_values': [50, 100, 200, 500],
            'risk_levels': {
                'HIGH': 0.95,
                'MEDIUM': 0.85,
                'LOW': 0.70
            }
        },
        'ensemble': {
            'weights': {
                'supervised': 0.7,
                'anomaly': 0.3,
                'graph': 0.0
            },
            'blend_method': 'rank_mean'
        },
        'monitoring': {
            'psi_thresholds': {
                'stable': 0.1,
                'small_change': 0.2,
                'significant_change': 0.3
            }
        },
        'modeling': {
            'primary_metric': 'pr_auc'
        }
    }

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in project root.
        
    Returns:
        Dictionary with all configuration parameters
    """
    if not HAS_YAML:
        print("⚠️  PyYAML não disponível. Usando configuração padrão.")
        return get_default_config()
        
    if config_path is None:
        # Try to find config.yaml from current location
        current = Path.cwd()
        
        # Check current directory and parent directories
        for path in [current] + list(current.parents):
            config_file = path / "config.yaml"
            if config_file.exists():
                config_path = str(config_file)
                break
        
        if config_path is None:
            print("⚠️  config.yaml não encontrado. Usando configuração padrão.")
            return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Erro ao carregar config.yaml: {e}. Usando configuração padrão.")
        return get_default_config()

def get_paths(config: Dict[str, Any], base_path: Optional[Path] = None) -> Dict[str, Path]:
    """
    Convert path strings from config to Path objects.
    
    Args:
        config: Configuration dictionary
        base_path: Base path to resolve relative paths. If None, uses current directory.
        
    Returns:
        Dictionary of Path objects
    """
    if base_path is None:
        base_path = Path.cwd()
        
    paths = {}
    for key, path_str in config['paths'].items():
        paths[key] = base_path / path_str
        
    return paths

def get_model_params(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get model parameters for specific algorithm.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model (e.g., 'GradientBoosting')
        
    Returns:
        Dictionary of model parameters
    """
    baseline_models = config['modeling']['baseline_models']
    
    if model_name in baseline_models:
        params = baseline_models[model_name].copy()
        
        # Add random_state if not present
        if 'random_state' not in params and model_name in ['GradientBoosting', 'RandomForest', 'LightGBM']:
            params['random_state'] = config['random_state']
            
        return params
    
    return {}

def get_k_values(config: Dict[str, Any], n_samples: int) -> list:
    """
    Get adaptive K values for @K metrics based on dataset size.
    
    Args:
        config: Configuration dictionary
        n_samples: Number of samples in test set
        
    Returns:
        List of K values adapted to dataset size
    """
    base_k_values = config['scoring']['k_values']
    
    # Adapt K values to dataset size
    adapted_k = []
    for k in base_k_values:
        adapted = min(k, n_samples // 2)  # Don't exceed half of dataset
        if adapted > 0:
            adapted_k.append(adapted)
    
    return sorted(set(adapted_k))  # Remove duplicates and sort

def get_psi_interpretation(psi_value: float, config: Dict[str, Any]) -> str:
    """
    Get PSI drift interpretation based on configured thresholds.
    
    Args:
        psi_value: PSI value to interpret
        config: Configuration dictionary
        
    Returns:
        String interpretation of drift level
    """
    thresholds = config['monitoring']['psi_thresholds']
    
    if psi_value < thresholds['stable']:
        return 'Estável'
    elif psi_value < thresholds['small_change']:
        return 'Mudança pequena'
    elif psi_value < thresholds['significant_change']:
        return 'Mudança moderada'
    else:
        return 'Mudança significativa'

# Example usage functions
def setup_paths(config: Dict[str, Any], create_dirs: bool = True) -> Dict[str, Path]:
    """
    Setup project directory structure.
    
    Args:
        config: Configuration dictionary
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        Dictionary of Path objects
    """
    paths = get_paths(config)
    
    if create_dirs:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    return paths

def get_ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ensemble configuration with computed parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with ensemble parameters
    """
    ensemble_config = config['ensemble'].copy()
    
    # Add computed parameters
    ensemble_config['primary_metric'] = config['modeling']['primary_metric']
    ensemble_config['random_state'] = config['random_state']
    
    return ensemble_config

"""
🔧 DEPENDENCY CHECKER INTELIGENTE
Verifica e corrige dependências automaticamente com fallbacks
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def check_dependencies_smart(verbose: bool = True) -> Dict[str, bool]:
    """
    Checker inteligente de dependências com fallbacks automáticos
    
    Returns:
        Dict com status de cada dependência
    """
    
    # Dependências críticas (falha se ausente)
    critical_deps = {
        'pandas': '2.0.3',
        'numpy': '1.24.3', 
        'sklearn': '1.3.0',
        'matplotlib': '3.7.2',
        'pathlib': None,  # Built-in
        'json': None,     # Built-in
        'yaml': '6.0.1'
    }
    
    # Dependências opcionais com fallbacks
    optional_deps = {
        'plotly': {'version': '5.15.0', 'fallback': 'matplotlib', 'description': 'Visualizações interativas'},
        'shap': {'version': '0.42.1', 'fallback': 'permutation_importance', 'description': 'Interpretabilidade avançada'},
        'networkx': {'version': '3.1', 'fallback': 'basic_graphs', 'description': 'Análise de grafos'},
        'streamlit': {'version': '1.25.0', 'fallback': 'jupyter_widgets', 'description': 'Dashboard web'},
        'lightgbm': {'version': '4.0.0', 'fallback': 'sklearn.ensemble', 'description': 'LightGBM models'},
        'optuna': {'version': '3.3.0', 'fallback': 'sklearn.model_selection', 'description': 'Hyperparameter tuning'}
    }
    
    results = {'critical': {}, 'optional': {}, 'status': 'ok', 'errors': [], 'warnings': []}
    
    if verbose:
        print("🔍 VERIFICANDO DEPENDÊNCIAS...")
        print("=" * 50)
    
    # Check critical dependencies
    for dep_name, required_version in critical_deps.items():
        try:
            if dep_name == 'sklearn':
                import sklearn
                module = sklearn
                dep_display = 'scikit-learn'
            else:
                module = importlib.import_module(dep_name)
                dep_display = dep_name
            
            version = getattr(module, '__version__', 'unknown')
            
            if required_version and version != 'unknown':
                version_ok = _check_version_compatibility(version, required_version)
                if not version_ok:
                    error_msg = f"❌ {dep_display}: versão {version} (requerida: {required_version})"
                    results['errors'].append(error_msg)
                    if verbose:
                        print(error_msg)
                else:
                    results['critical'][dep_name] = True
                    if verbose:
                        print(f"✅ {dep_display}: {version}")
            else:
                results['critical'][dep_name] = True
                if verbose:
                    print(f"✅ {dep_display}: {version}")
                    
        except ImportError as e:
            error_msg = f"❌ CRÍTICO: {dep_name} não encontrado"
            results['errors'].append(error_msg)
            results['critical'][dep_name] = False
            results['status'] = 'critical_error'
            if verbose:
                print(error_msg)
    
    # Check optional dependencies
    for dep_name, dep_info in optional_deps.items():
        try:
            module = importlib.import_module(dep_name)
            version = getattr(module, '__version__', 'unknown')
            
            required_version = dep_info['version']
            if required_version and version != 'unknown':
                version_ok = _check_version_compatibility(version, required_version)
                if not version_ok:
                    warning_msg = f"⚠️ {dep_name}: versão {version} (recomendada: {required_version})"
                    results['warnings'].append(warning_msg)
                    if verbose:
                        print(warning_msg)
                else:
                    results['optional'][dep_name] = True
                    if verbose:
                        print(f"✅ {dep_name}: {version}")
            else:
                results['optional'][dep_name] = True
                if verbose:
                    print(f"✅ {dep_name}: {version}")
                    
        except ImportError:
            fallback = dep_info['fallback']
            description = dep_info['description']
            warning_msg = f"⚠️ {dep_name} ausente ({description}) - fallback: {fallback}"
            results['warnings'].append(warning_msg)
            results['optional'][dep_name] = False
            if verbose:
                print(warning_msg)
    
    # Summary
    if verbose:
        print("=" * 50)
        if results['status'] == 'critical_error':
            print("🚨 ERRO CRÍTICO: Dependências essenciais ausentes!")
            print("💡 Execute: pip install -r requirements_fixed.txt")
        elif results['warnings']:
            print(f"⚠️ {len(results['warnings'])} aviso(s) - funcionalidade reduzida")
            print("💡 Execute: pip install -r requirements_fixed.txt")
        else:
            print("🎉 TODAS AS DEPENDÊNCIAS OK!")
    
    return results

def _check_version_compatibility(current: str, required: str) -> bool:
    """Verifica compatibilidade de versões (major.minor level)"""
    try:
        current_parts = [int(x) for x in current.split('.')[:2]]
        required_parts = [int(x) for x in required.split('.')[:2]]
        
        # Check major.minor compatibility
        return current_parts[0] == required_parts[0] and current_parts[1] >= required_parts[1]
    except:
        return True  # Se não conseguir parsear, assume OK

def fix_dependencies_auto(requirements_file: str = "requirements_fixed.txt") -> bool:
    """
    Tenta corrigir dependências automaticamente
    """
    print("🔧 TENTANDO CORRIGIR DEPENDÊNCIAS...")
    
    requirements_path = Path(requirements_file)
    if not requirements_path.exists():
        print(f"❌ Arquivo {requirements_file} não encontrado")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install from requirements
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], 
                              check=True, capture_output=True, text=True)
        
        print("✅ Dependências instaladas com sucesso!")
        print("🔄 Reinicie o kernel para aplicar as mudanças")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na instalação: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def create_dependency_report(output_file: str = "artifacts/dependency_report.json") -> None:
    """Cria relatório detalhado de dependências"""
    import json
    from datetime import datetime
    
    results = check_dependencies_smart(verbose=False)
    
    # Add system info
    results['system_info'] = {
        'python_version': sys.version,
        'platform': sys.platform,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add installed packages
    try:
        import pkg_resources
        installed = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
        results['installed_packages'] = installed
    except:
        results['installed_packages'] = {}
    
    # Save report
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Relatório salvo em: {output_file}")

def main():
    """Função principal para execução standalone"""
    print("🚀 DEPENDENCY CHECKER INTELIGENTE")
    print("=" * 50)
    
    # Check dependencies
    results = check_dependencies_smart(verbose=True)
    
    # Generate report
    create_dependency_report()
    
    # Auto-fix if requested
    if results['status'] == 'critical_error':
        print("\n🤔 Tentar corrigir automaticamente? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes', 's', 'sim']:
                fix_dependencies_auto()
        except:
            pass
    
    return results

"""
Module Loading Utilities

Helper functions to load internal modules consistently across notebooks.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict
import logging

# Set up logger
logger = logging.getLogger(__name__)


def load_internal_modules(base_dir: Path = None) -> Dict[str, Any]:
    """
    Load all essential internal modules for model evaluation pipeline.
    
    Args:
        base_dir: Base directory (defaults to ../utils from current location)
        
    Returns:
        Dictionary with loaded modules
    """
    if base_dir is None:
        base_dir = Path("utils")
    
    modules = {}
    
    # List of essential modules
    module_specs = [
        ('config_loader', 'config_loader.py'),
        ('model_selection', 'model_selection.py'),
        ('candidate_gating', 'candidate_gating.py'),
        ('metrics_utils', 'metrics_utils.py'),
        ('ranking_metrics', 'ranking_metrics.py'),
        ('expected_value', 'expected_value.py'),
        ('governance', 'governance.py'),
        ('temporal_split', 'temporal_split.py'),
        ('advanced_metrics', 'advanced_metrics.py'),
        ('ensemble', 'ensemble.py'),
        ('threshold_optimizer', 'threshold_optimizer.py')
    ]
    
    for module_name, filename in module_specs:
        module_path = base_dir / filename
        if module_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[module_name] = module
                logger.info(f"✅ {module_name} loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {module_name}: {e}")
        else:
            logger.warning(f"⚠️ {module_path} not found")
    
    return modules


def get_pipeline_functions(modules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract commonly used functions from loaded modules.
    
    Args:
        modules: Dictionary of loaded modules
        
    Returns:
        Dictionary of functions ready to use
    """
    functions = {}
    
    # Config functions
    if 'config_loader' in modules:
        functions['load_config'] = modules['config_loader'].load_config
    
    # Model selection functions
    if 'model_selection' in modules:
        ms = modules['model_selection']
        functions['select_best_model'] = ms.select_best_model
        functions['format_selection_summary'] = ms.format_selection_summary
    
    # Candidate gating functions
    if 'candidate_gating' in modules:
        cg = modules['candidate_gating']
        functions['apply_gating_criteria'] = cg.apply_gating_criteria
        functions['save_candidate_results'] = cg.save_candidate_results
        functions['format_gating_summary'] = cg.format_gating_summary
        functions['get_tuning_priority'] = cg.get_tuning_priority
    
    # Metrics functions (excluding deprecated @K functions)
    if 'metrics_utils' in modules:
        mu = modules['metrics_utils']
        functions['evaluate_model'] = mu.evaluate_model
        functions['save_curves'] = mu.save_curves
        functions['convert_for_json'] = mu.convert_for_json
        functions['compute_cv_metrics'] = mu.compute_cv_metrics
    
    # Ranking metrics functions (consolidated - PRIMARY SOURCE)
    if 'ranking_metrics' in modules:
        rm = modules['ranking_metrics']
        # Core ranking functions
        functions['compute_at_k'] = rm.compute_at_k
        functions['format_at_k'] = rm.format_at_k
        # Advanced ranking analysis
        functions['generate_lift_table'] = rm.generate_lift_table
        functions['compute_cumulative_gains'] = rm.compute_cumulative_gains
        functions['compute_top_decile_lift'] = rm.compute_top_decile_lift
        functions['compute_efficiency_curve'] = rm.compute_efficiency_curve
        functions['compare_ranking_performance'] = rm.compare_ranking_performance
        functions['get_ranking_summary'] = rm.get_ranking_summary
        # Visualization and export
        functions['save_ranking_curves'] = rm.save_ranking_curves
    
    # Expected Value functions
    if 'expected_value' in modules:
        ev = modules['expected_value']
        functions['compute_expected_value'] = ev.compute_expected_value
        functions['find_optimal_threshold_ev'] = ev.find_optimal_threshold_ev
        functions['get_aml_cost_params_template'] = ev.get_aml_cost_params_template
        functions['format_ev_summary'] = ev.format_ev_summary
    
    # Governance functions
    if 'governance' in modules:
        gov = modules['governance']
        functions['compute_data_hash'] = gov.compute_data_hash
        functions['validate_temporal_split'] = gov.validate_temporal_split
        functions['create_model_card'] = gov.create_model_card
        functions['save_model_card'] = gov.save_model_card
        functions['check_data_integrity'] = gov.check_data_integrity
        functions['compute_file_hash'] = gov.compute_file_hash
        functions['hash_artifacts'] = gov.hash_artifacts
        functions['build_lineage_record'] = gov.build_lineage_record
        functions['update_lineage_registry'] = gov.update_lineage_registry
    
    # Temporal split functions
    if 'temporal_split' in modules:
        ts = modules['temporal_split']
        functions['create_temporal_split'] = ts.create_temporal_split
        functions['validate_split_quality'] = ts.validate_split_quality
        functions['prepare_temporal_datasets'] = ts.prepare_temporal_datasets
        functions['format_temporal_split_report'] = ts.format_temporal_split_report
    
    # Advanced metrics functions
    if 'advanced_metrics' in modules:
        am = modules['advanced_metrics']
        functions['compute_calibration_metrics'] = am.compute_calibration_metrics
        functions['compute_ndcg_at_k'] = am.compute_ndcg_at_k
        functions['compute_aml_specialized_metrics'] = am.compute_aml_specialized_metrics
        functions['compute_comprehensive_metrics'] = am.compute_comprehensive_metrics
        functions['format_advanced_metrics_summary'] = am.format_advanced_metrics_summary

    # Ensemble functions
    if 'ensemble' in modules:
        en = modules['ensemble']
        functions['blend_scores'] = en.blend_scores
        functions['build_ensemble_outputs'] = en.build_ensemble_outputs

    # Threshold optimization functions
    if 'threshold_optimizer' in modules:
        to_mod = modules['threshold_optimizer']
        functions['threshold_config_cls'] = to_mod.ThresholdConfig
        functions['evaluate_thresholds_curve'] = to_mod.evaluate_thresholds
        functions['save_threshold_artifact'] = to_mod.save_threshold_artifact
    
    return functions


import pandas as pd
import numpy as np
from math import sqrt
from IPython.display import display
# import pandas_profiling  # Comentado devido a incompatibilidade com numpy mais recente
import datetime

# import sweetviz as sv  # Comentado por enquanto para evitar conflitos
import matplotlib
import matplotlib.pyplot as plt

# import tensorflow as tf  # Comentado devido a incompatibilidade com numpy mais recente

# Métricas de Desempenho
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import plot_confusion_matrix  # Removido nas versões recentes do sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')

def log_message(message):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S - "), message)


def gera_relatorios_aed(df, target_feat=None, 
                        html_pp='base_aed_pp.html', 
                        html_sv='base_aed_sv.html'):
    '''
    '''
    # Gera relatório usando Pandas Profiling
    perfil_pp = df.profile_report()
    perfil_pp.to_file(output_file=html_pp)
    
    # Gera relatório usando SweetViz (comentado devido a incompatibilidades)
    # perfil_sv = sv.analyze(df, target_feat=target_feat)
    # perfil_sv.show_html(html_sv)
    perfil_sv = None  # Placeholder quando SweetViz não está disponível
    
    return perfil_pp, perfil_sv
    

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
        
        
# Função para cálculo do KS
def ks_stat(y, y_pred):
    return ks_2samp(y_pred[y==1], y_pred[y!=1]).statistic

# Função para cálculo do desempenho de modelos
def calcula_desempenho(modelo, x_train, y_train, x_test, y_test):
    
    # Verifica se é um modelo TensorFlow/Keras (comentado devido a incompatibilidades)
    # if isinstance(modelo, tf.keras.Model):
    #     ypred_train = (modelo.predict(x_train) > 0.5).astype("int32")
    #     ypred_proba_train = modelo.predict(x_train)[:,0]
    #     
    #     ypred_test = (modelo.predict(x_test) > 0.5).astype("int32")
    #     ypred_proba_test = modelo.predict(x_test)[:,0]
    # else:
    
    # Assumindo modelos sklearn por padrão
    try:
        # Cálculo dos valores preditos
        ypred_train = modelo.predict(x_train)
        ypred_proba_train = modelo.predict_proba(x_train)[:,1]

        ypred_test = modelo.predict(x_test)
        ypred_proba_test = modelo.predict_proba(x_test)[:,1]
            
    except Exception as e:
        print(f'Modelo não suportado: {e}')
        return None

    # Métricas de Desempenho
    acc_train = accuracy_score(y_train, ypred_train)
    acc_test = accuracy_score(y_test, ypred_test)
    
    roc_train = roc_auc_score(y_train, ypred_proba_train)
    roc_test = roc_auc_score(y_test, ypred_proba_test)
    
    ks_train = ks_stat(y_train, ypred_proba_train)
    ks_test = ks_stat(y_test, ypred_proba_test)
    
    prec_train = precision_score(y_train, ypred_train, zero_division=0)
    prec_test = precision_score(y_test, ypred_test, zero_division=0)
    
    recl_train = recall_score(y_train, ypred_train)
    recl_test = recall_score(y_test, ypred_test)
    
    f1_train = f1_score(y_train, ypred_train)
    f1_test = f1_score(y_test, ypred_test)

    df_desemp = pd.DataFrame({'Treino':[acc_train, roc_train, ks_train, 
                                        prec_train, recl_train, f1_train],
                              'Teste':[acc_test, roc_test, ks_test,
                                       prec_test, recl_test, f1_test]},
                            index=['Acurácia','AUROC','KS',
                                   'Precision','Recall','F1'])
    
    df_desemp['Variação'] = round(df_desemp['Teste'] / df_desemp['Treino'] - 1, 2)
    
    return df_desemp


def matriz_confusao(modelo, X_train, y_train, X_test, y_test):
    
    # Usar ConfusionMatrixDisplay (nova API do sklearn)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matriz de confusão para treino
    ConfusionMatrixDisplay.from_estimator(modelo, X_train, y_train, ax=axes[0])
    axes[0].set_title('Treino')
    
    # Matriz de confusão para teste
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, ax=axes[1])
    axes[1].set_title('Teste')
    
    plt.tight_layout()
    plt.show()
    
    

# Função para cálculo do desempenho de modelos
def calcula_desempenho_reg(modelo, x_train, y_train, x_test, y_test, y_scaler, historico=None):

    # Calcula valores preditos pelo modelo
    if historico is not None:
        rmse_hist = historico.history[list(historico.history.keys())[1]]
        val_rmse_hist = historico.history[list(historico.history.keys())[3]]
        
    else:
        rmse_hist = modelo.history_[list(modelo.history_.keys())[1]]
        val_rmse_hist = modelo.history_[list(modelo.history_.keys())[3]]

    y_train_pred = y_scaler.inverse_transform(modelo.predict(x_train))
    y_test_pred  = y_scaler.inverse_transform(modelo.predict(x_test))
    
    # Métricas de Desempenho
    r2_train = r2_score(y_train, y_train_pred)
    r2_test =  r2_score(y_test, y_test_pred)

    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = sqrt(mean_squared_error(y_test,  y_test_pred))
    
    df_desemp = pd.DataFrame({'Treino':[r2_train, rmse_train],
                              'Teste':[r2_test, rmse_test]
                             },
                            index=['R²','RMSE'])
    
    df_desemp['Treino'] = df_desemp['Treino'].round(2)
    df_desemp['Teste'] = df_desemp['Teste'].round(2)
    df_desemp['Variação'] = round(df_desemp['Teste'] / df_desemp['Treino'] - 1, 2)
    
    
    # Gráfico com a evolução do treinamento
    matplotlib.use('module://ipykernel.pylab.backend_inline')

    fig, ax = plt.subplots(figsize=(8, 8))    
    plt.plot(rmse_hist)
    plt.plot(val_rmse_hist)
    plt.title('RMSE do Modelo')
    plt.ylabel('RMSE')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()
    
    
    # Dispersão dos valores Observados vs. Preditos
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)
    fig.suptitle('Valores Observados vs. Predições', fontsize = 16)
    min_val = min([np.min(y_train.min()),y_train_pred.min()])*0.5
    max_val = max([np.max(y_train.max()),y_train_pred.max()])*1.1

    axs[0].plot(y_train, y_train_pred, 'ro');
    axs[0].plot([y_train.min()[0], y_train_pred.max()], [y_train.min()[0], y_train_pred.max()], 'k--', lw=1)
    axs[0].set_xlim([min_val, max_val])
    axs[0].set_ylim([min_val, max_val])

    axs[1].plot(y_test, y_test_pred, 'ro');
    axs[1].plot([y_test.min()[0], y_test_pred.max()], [y_test.min()[0], y_test_pred.max()], 'k--', lw=1)
    axs[1].set_xlim([min_val, max_val])
    axs[1].set_ylim([min_val, max_val])

    for ax in axs.flat:
        ax.set(xlabel='Valores Observados', ylabel='Valore Preditos')
    plt.show()
    
    return df_desemp


def dispersao_modelo(y_obs, y_pred):
    matplotlib.use('module://ipykernel.pylab.backend_inline')

    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    min_val = min([np.min(y_obs.min()),y_pred.min()])*0.5
    max_val = max([np.max(y_obs.max()),y_pred.max()])*1.1
    
    plt.plot(y_obs, y_pred, 'ro')
    plt.xlabel('Observados', fontsize = 10)
    plt.ylabel('Preditos', fontsize = 10)    
    plt.title('Predições vs. Observados', fontsize = 10)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    plt.show()
    
    
def cria_grafico_var_qualitativa(tab):

    # Aumenta o tamanho do gráfico (largura 8 e altura 4)
    fig = plt.figure(figsize=(8,4))

    # Cria um gráfico de barras usando o indice da tabela como rótulos do eixo X
    cor = np.random.rand(3)
    plt.bar(tab.index,tab['Freq_Relativa'],width = 0.7, tick_label=tab.index,color=cor,alpha=0.6)

    plt.ylim(0,tab['Freq_Relativa'].max()+0.2)
    plt.title("Frequência Relativa de {}".format(list(tab.columns)[0]))

    # cria um conjunto de pares de rótulos e frequencias relativas
    for x,y in zip(tab.index,tab['Freq_Relativa']):

        # formata o rotulo do percentual
        rotulo = "{:.4f}".format(y)

        # coloca o rotulo na posição (x,y), alinhado ao centro e com distância 0,5 do ponto (x,y)
        plt.annotate(rotulo,(x,y),ha='center',textcoords="offset points",xytext=(0,5))
        
        
def tabela_bivariada(data, var):
    
    df = pd.DataFrame(data[var].value_counts()).sort_values(by=var,ascending=False)
    total = df[var].sum()
    df['Freq_Relativa'] = (df[var]/total).round(decimals=4)
    df['Freq_Acumulada'] = df['Freq_Relativa'].cumsum().round(decimals=4)
    return df

"""Quick product summary utility.

Uso:
    python -m utils.product_summary

Objetivo:
    Fornecer uma visão consolidada dos artefatos finais gerados pelo pipeline,
    facilitando entendimento do "produto" entregue (modelo oficial, thresholds,
    ensemble, governança, monitoramento e explicabilidade).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")

KEY_FILES = [
    "best_model_meta.json",
    "model_card.json",
    "baseline_candidates.json",
    "baseline_models.csv",
    "baseline_metrics_at_k.csv",
    "tuning_results.json",
    "thresholds.json",
    "threshold_analysis.csv",
    "risk_scores_ensemble.csv",
    "ensemble_metadata.json",
    "monitor_feature_shift.csv",
    "monitor_score_shift.csv",
    "monitor_summary.json",
    "validation_report.json",
    "permutation_importance.csv",
    "shap_importance.csv",
    "pipeline_results.json",
    "lineage_registry.json"
]

SEPARATOR = "=" * 78


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not _exists(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_best_model(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Model Name      : {meta.get('model_name')}")
    parts.append(f"Variant         : {meta.get('variant')}")
    parts.append(f"Source Stage    : {meta.get('source')}")
    if meta.get('primary_metric') and meta.get('primary_value') is not None:
        parts.append(f"Primary Metric  : {meta.get('primary_metric')}={meta.get('primary_value'):.4f}")
    if meta.get('decision_action'):
        parts.append(f"Decision Action : {meta.get('decision_action')}")
    if meta.get('improvement_over_baseline') is not None:
        parts.append(f"Δ vs Baseline   : {meta.get('improvement_over_baseline'):.4f}")
    if meta.get('model_file_path'):
        parts.append(f"Model File      : {meta.get('model_file_path')}")
    else:
        # Reconstroi caminho provável
        source = meta.get('source')
        variant = meta.get('variant')
        if source == 'tuning':
            guess = MODELS_DIR / 'best_model_tuned.pkl'
        elif variant == 'core':
            guess = MODELS_DIR / 'best_baseline_core.pkl'
        else:
            guess = MODELS_DIR / 'best_baseline.pkl'
        parts.append(f"(Possível arquivo) : {guess}")
    return "\n".join(parts)


def summarize_thresholds(payload: Dict[str, Any]) -> str:
    best = payload.get('best_threshold')
    metrics = payload.get('best_metrics', {})
    optimize_metric = payload.get('threshold_config', {}).get('optimize_metric')
    flagged = metrics.get('flagged_cases')
    lines = [
        f"Best Threshold : {best}",
        f"Optimize Metric: {optimize_metric}",
    ]
    if optimize_metric in metrics:
        lines.append(f"{optimize_metric}={metrics.get(optimize_metric):.4f}")
    if flagged is not None:
        lines.append(f"Flagged Cases  : {flagged}")
    return "\n".join(lines)


def summarize_ensemble(meta: Dict[str, Any]) -> str:
    comps = ", ".join(meta.get('components_used', []))
    bands = meta.get('band_distribution', {})
    band_str = ", ".join(f"{k}:{v}" for k, v in bands.items()) if bands else "-"
    return (
        f"Components  : {comps}\n"
        f"Blend Method: {meta.get('blend_method')}\n"
        f"Risk Bands  : {band_str}"
    )


def file_summary() -> List[str]:
    rows: List[str] = []
    for fname in KEY_FILES:
        path = ARTIFACTS_DIR / fname
        status = "OK" if _exists(path) else "MISSING"
        size = path.stat().st_size if path.exists() else 0
        rows.append(f"{fname:<30} | {status:<8} | {size:>7} bytes")
    return rows


def main() -> None:
    print(SEPARATOR)
    print("PIPELINE PRODUCT SUMMARY")
    print(SEPARATOR)

    print("\n[1] Arquivos-Chave")
    for line in file_summary():
        print(line)

    # Best model
    meta = _read_json(ARTIFACTS_DIR / 'best_model_meta.json')
    if meta:
        print("\n[2] Modelo Oficial Selecionado")
        print(summarize_best_model(meta))
    else:
        print("\n[2] Modelo Oficial Selecionado\n(best_model_meta.json ausente – execute estágio baselines/tuning)")

    # Thresholds
    thresholds = _read_json(ARTIFACTS_DIR / 'thresholds.json')
    if thresholds:
        print("\n[3] Threshold Otimizado")
        print(summarize_thresholds(thresholds))
    else:
        print("\n[3] Threshold Otimizado\n( thresholds.json ausente – execute estágio thresholds )")

    # Ensemble
    ensemble_meta = _read_json(ARTIFACTS_DIR / 'ensemble_metadata.json')
    if ensemble_meta:
        print("\n[4] Ensemble / Risco")
        print(summarize_ensemble(ensemble_meta))
    else:
        print("\n[4] Ensemble / Risco\n( ensemble_metadata.json ausente – execute estágio ensemble )")

    # Governance / Validation
    validation = _read_json(ARTIFACTS_DIR / 'validation_report.json')
    if validation:
        art_status = validation.get('artifacts_status', {})
        completeness = art_status.get('completeness')
        print("\n[5] Validação & Governança")
        print(f"Artifacts completeness: {completeness}")
        missing = art_status.get('missing') or []
        if missing:
            print(f"Faltando: {', '.join(missing)}")
    else:
        print("\n[5] Validação & Governança\n(validation_report.json ausente – execute estágio validation)")

    # Explainability quick existence
    perm_ok = _exists(ARTIFACTS_DIR / 'permutation_importance.csv')
    shap_ok = _exists(ARTIFACTS_DIR / 'shap_importance.csv')
    print("\n[6] Explainability")
    print(f"Permutation Importance: {'OK' if perm_ok else 'MISSING'}")
    print(f"SHAP Importance       : {'OK' if shap_ok else 'MISSING'}")

    print("\n[7] Próximos Passos Sugeridos")
    suggestions: List[str] = []
    if not meta:
        suggestions.append("Executar: baselines (e tuning se aplicável) para gerar best_model_meta.json")
    if meta and not thresholds:
        suggestions.append("Executar thresholds para definir corte operacional")
    if thresholds and not ensemble_meta:
        suggestions.append("Gerar ensemble para consolidar risco final")
    if ensemble_meta and not validation:
        suggestions.append("Rodar validation para consolidar auditoria")
    if not suggestions:
        suggestions.append("Pipeline completo – avaliar deploy / servir modelo / monitoração contínua")
    for s in suggestions:
        print(f"- {s}")

    print("\n" + SEPARATOR)
    print("Fim do resumo.")


if __name__ == "__main__":  # pragma: no cover
    main()
