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

if __name__ == "__main__":
    main()