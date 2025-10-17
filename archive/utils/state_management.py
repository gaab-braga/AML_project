# ============================================================================
# NOTEBOOK STATE MANAGEMENT SYSTEM
# ============================================================================
"""
Sistema para garantir execução reprodutível e consistente entre notebooks.

Este módulo resolve problemas de:
- Execução não sequencial (execution counts não sequenciais)
- Variáveis inconsistentes entre execuções
- Dependências não satisfeitas entre notebooks
- Resultados não reprodutíveis

Uso:
    from utils.state_management import NotebookStateManager
    state_mgr = NotebookStateManager(notebook_name="01_data_preparation")
    state_mgr.validate_dependencies()
    state_mgr.check_data_consistency()
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

class NotebookConfig:
    """Configuração centralizada para todos os notebooks"""

    # Caminhos base
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks" / "05_model_dev"

    # Estado global
    STATE_FILE = ARTIFACTS_DIR / "notebook_state.json"
    CACHE_DIR = ARTIFACTS_DIR / "cache"

    # Configurações de cache
    CACHE_EXPIRY_HOURS = 24
    FORCE_REFRESH = False

    # Versões esperadas
    EXPECTED_VERSIONS = {
        "data_version": "v2.1",
        "model_version": "v1.0",
        "feature_engineering_version": "v1.2"
    }

# ============================================================================
# STATE MANAGER CLASS
# ============================================================================

class NotebookStateManager:
    """
    Gerenciador de estado para garantir execução reprodutível entre notebooks.

    Funcionalidades:
    - Validação de dependências entre notebooks
    - Verificação de consistência de dados
    - Cache inteligente com timestamps
    - Reset de estado quando necessário
    """

    def __init__(self, notebook_name: str, config: Optional[NotebookConfig] = None):
        """
        Inicializar gerenciador de estado.

        Args:
            notebook_name: Nome do notebook (ex: "01_data_preparation")
            config: Configuração customizada (opcional)
        """
        self.notebook_name = notebook_name
        self.config = config or NotebookConfig()
        self.state_data = self._load_state()
        self.notebook_state = self.state_data.get('notebooks', {}).get(notebook_name, {})

        # Criar diretórios se não existirem
        self.config.CACHE_DIR.mkdir(exist_ok=True)
        self.config.ARTIFACTS_DIR.mkdir(exist_ok=True)

        print(f"📋 NotebookStateManager inicializado para: {notebook_name}")

    def _load_state(self) -> Dict[str, Any]:
        """Carregar estado global dos notebooks"""
        if self.config.STATE_FILE.exists():
            try:
                with open(self.config.STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Erro ao carregar estado: {e}")
                return self._create_default_state()
        else:
            return self._create_default_state()

    def _create_default_state(self) -> Dict[str, Any]:
        """Criar estado padrão"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "notebooks": {},
            "global_variables": {},
            "cache_info": {}
        }

    def _save_state(self):
        """Salvar estado global"""
        try:
            with open(self.config.STATE_FILE, 'w') as f:
                json.dump(self.state_data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Erro ao salvar estado: {e}")

    # ============================================================================
    # DEPENDENCY VALIDATION
    # ============================================================================

    def validate_dependencies(self, required_notebooks: Optional[List[str]] = None) -> bool:
        """
        Validar que notebooks dependentes foram executados corretamente.

        Args:
            required_notebooks: Lista de notebooks que devem ter sido executados antes

        Returns:
            True se todas as dependências estão satisfeitas
        """
        print(f"\n🔍 VALIDANDO DEPENDÊNCIAS PARA: {self.notebook_name}")
        print("=" * 60)

        if required_notebooks is None:
            required_notebooks = self._get_default_dependencies()

        all_valid = True

        for dep_notebook in required_notebooks:
            dep_state = self.state_data.get('notebooks', {}).get(dep_notebook, {})

            if not dep_state:
                print(f"❌ {dep_notebook}: NÃO EXECUTADO")
                all_valid = False
                continue

            # Verificar se foi executado recentemente (últimas 24h)
            last_execution = dep_state.get('last_execution')
            if last_execution:
                last_exec_dt = datetime.fromisoformat(last_execution)
                if datetime.now() - last_exec_dt > timedelta(hours=24):
                    print(f"⚠️  {dep_notebook}: EXECUTADO HÁ MAIS DE 24h ({last_execution})")
                    print("   💡 Considere re-executar para garantir consistência")

            # Verificar variáveis críticas
            critical_vars = self._get_critical_variables(dep_notebook)
            missing_vars = []

            for var in critical_vars:
                if var not in dep_state.get('variables', {}):
                    missing_vars.append(var)

            if missing_vars:
                print(f"❌ {dep_notebook}: VARIÁVEIS FALTANDO: {missing_vars}")
                all_valid = False
            else:
                print(f"✅ {dep_notebook}: DEPENDÊNCIA SATISFEITA")

        if all_valid:
            print("\n✅ TODAS AS DEPENDÊNCIAS VALIDADAS")
        else:
            print("\n❌ DEPENDÊNCIAS NÃO SATISFEITAS")
            print("💡 Execute os notebooks dependentes primeiro")

        return all_valid

    def _get_default_dependencies(self) -> List[str]:
        """Obter dependências padrão baseadas no nome do notebook"""
        dependencies = {
            "02_baseline_models": ["01_data_preparation"],
            "03_hyperparameter_tuning": ["01_data_preparation", "02_baseline_models"],
            "04_model_evaluation": ["01_data_preparation", "02_baseline_models", "03_hyperparameter_tuning"],
            "05_final_results": ["01_data_preparation", "02_baseline_models", "03_hyperparameter_tuning", "04_model_evaluation"]
        }
        return dependencies.get(self.notebook_name, [])

    def _get_critical_variables(self, notebook_name: str) -> List[str]:
        """Obter variáveis críticas que devem existir após execução do notebook"""
        critical_vars = {
            "01_data_preparation": ["X_train_balanced", "y_train_balanced", "X_test_featured", "y_test"],
            "02_baseline_models": ["baseline_results", "baseline_models"],
            "03_hyperparameter_tuning": ["tuning_results"],
            "04_model_evaluation": ["best_model", "test_metrics"],
            "05_final_results": ["final_model"]
        }
        return critical_vars.get(notebook_name, [])

    # ============================================================================
    # DATA CONSISTENCY CHECKS
    # ============================================================================

    def check_data_consistency(self) -> bool:
        """
        Verificar consistência dos dados carregados.

        Returns:
            True se dados são consistentes
        """
        print(f"\n🔍 VERIFICANDO CONSISTÊNCIA DE DADOS")
        print("=" * 60)

        issues = []

        # Verificar se arquivos de dados existem
        data_files = [
            self.config.DATA_DIR / "X_train_engineered.csv",
            self.config.DATA_DIR / "X_test_engineered.csv",
            self.config.DATA_DIR / "y_train_processed.parquet",
            self.config.DATA_DIR / "y_test_processed.parquet"
        ]

        for data_file in data_files:
            if not data_file.exists():
                issues.append(f"Arquivo não encontrado: {data_file.name}")

        # Verificar versões dos dados
        version_file = self.config.ARTIFACTS_DIR / "data_version.json"
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    version_info = json.load(f)

                current_version = version_info.get('version')
                expected_version = self.config.EXPECTED_VERSIONS['data_version']

                if current_version != expected_version:
                    issues.append(f"Versão dos dados incompatível: {current_version} vs {expected_version}")
            except Exception as e:
                issues.append(f"Erro ao verificar versão dos dados: {e}")

        # Verificar integridade dos dados (se variáveis existem)
        try:
            # Verificar se variáveis globais críticas existem
            critical_globals = ['CONFIG', 'ARTIFACTS_DIR', 'DATA_DIR']
            missing_globals = []

            for var in critical_globals:
                if var not in globals():
                    missing_globals.append(var)

            if missing_globals:
                issues.append(f"Variáveis globais faltando: {missing_globals}")

        except:
            issues.append("Erro ao verificar variáveis globais")

        if issues:
            print("❌ PROBLEMAS DE CONSISTÊNCIA ENCONTRADOS:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ DADOS CONSISTENTES")
            return True

    # ============================================================================
    # STATE MANAGEMENT
    # ============================================================================

    def update_execution_state(self, variables: Optional[Dict[str, Any]] = None):
        """
        Atualizar estado após execução bem-sucedida do notebook.

        Args:
            variables: Dicionário com variáveis importantes do notebook
        """
        # Atualizar estado do notebook
        notebook_state = {
            'last_execution': datetime.now().isoformat(),
            'execution_count': self.notebook_state.get('execution_count', 0) + 1,
            'variables': variables or {},
            'data_hash': self._calculate_data_hash()
        }

        # Atualizar estado global
        if 'notebooks' not in self.state_data:
            self.state_data['notebooks'] = {}

        self.state_data['notebooks'][self.notebook_name] = notebook_state
        self.notebook_state = notebook_state

        # Salvar estado
        self._save_state()

        print(f"✅ Estado atualizado para {self.notebook_name}")

    def _calculate_data_hash(self) -> str:
        """Calcular hash dos dados principais para verificar mudanças"""
        try:
            # Usar arquivos de dados principais para calcular hash
            data_files = [
                self.config.DATA_DIR / "X_train_engineered.csv",
                self.config.DATA_DIR / "y_train_processed.parquet"
            ]

            hasher = hashlib.md5()
            for data_file in data_files:
                if data_file.exists():
                    with open(data_file, 'rb') as f:
                        hasher.update(f.read())

            return hasher.hexdigest()
        except:
            return "unknown"

    # ============================================================================
    # CACHE MANAGEMENT
    # ============================================================================

    def get_cached_result(self, cache_key: str, max_age_hours: Optional[int] = None) -> Optional[Any]:
        """
        Obter resultado do cache se ainda válido.

        Args:
            cache_key: Chave única para o resultado em cache
            max_age_hours: Idade máxima do cache em horas (padrão: 24h)

        Returns:
            Resultado em cache ou None se inválido/expirado
        """
        if self.config.FORCE_REFRESH:
            return None

        cache_file = self.config.CACHE_DIR / f"{cache_key}.pkl"
        cache_meta_file = self.config.CACHE_DIR / f"{cache_key}_meta.json"

        if not cache_file.exists() or not cache_meta_file.exists():
            return None

        try:
            # Verificar metadata do cache
            with open(cache_meta_file, 'r') as f:
                meta = json.load(f)

            created_at = datetime.fromisoformat(meta['created_at'])
            max_age = max_age_hours or self.config.CACHE_EXPIRY_HOURS

            if datetime.now() - created_at > timedelta(hours=max_age):
                print(f"📋 Cache expirado para {cache_key}")
                return None

            # Carregar resultado do cache
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)

            print(f"✅ Cache válido encontrado para {cache_key}")
            return result

        except Exception as e:
            print(f"⚠️  Erro ao carregar cache {cache_key}: {e}")
            return None

    def save_to_cache(self, cache_key: str, result: Any, metadata: Optional[Dict] = None):
        """
        Salvar resultado no cache.

        Args:
            cache_key: Chave única para o resultado
            result: Resultado a ser armazenado
            metadata: Metadados adicionais
        """
        try:
            cache_file = self.config.CACHE_DIR / f"{cache_key}.pkl"
            cache_meta_file = self.config.CACHE_DIR / f"{cache_key}_meta.json"

            # Salvar resultado
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            # Salvar metadata
            meta = {
                'created_at': datetime.now().isoformat(),
                'notebook': self.notebook_name,
                'cache_key': cache_key
            }
            if metadata:
                meta.update(metadata)

            with open(cache_meta_file, 'w') as f:
                json.dump(meta, f, indent=2, default=str)

            print(f"💾 Resultado salvo no cache: {cache_key}")

        except Exception as e:
            print(f"⚠️  Erro ao salvar cache {cache_key}: {e}")

    # ============================================================================
    # RESET FUNCTIONALITY
    # ============================================================================

    def reset_notebook_state(self, confirm: bool = False):
        """
        Resetar estado do notebook atual.

        Args:
            confirm: Confirmação para evitar resets acidentais
        """
        if not confirm:
            print("⚠️  Use confirm=True para resetar o estado")
            return

        if self.notebook_name in self.state_data.get('notebooks', {}):
            del self.state_data['notebooks'][self.notebook_name]
            self._save_state()
            print(f"🔄 Estado resetado para {self.notebook_name}")
        else:
            print(f"ℹ️  Nenhum estado encontrado para {self.notebook_name}")

    def reset_all_state(self, confirm: bool = False):
        """
        Resetar todo o estado global (usar com cuidado!).

        Args:
            confirm: Confirmação dupla para evitar resets acidentais
        """
        if not confirm:
            print("⚠️  Use confirm=True para resetar TODO o estado")
            return

        confirm_text = input("Digite 'RESET_ALL' para confirmar reset total: ")
        if confirm_text != "RESET_ALL":
            print("❌ Reset cancelado")
            return

        self.state_data = self._create_default_state()
        self._save_state()
        print("🔄 TODO o estado foi resetado")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_notebook_state_manager(notebook_name: str) -> NotebookStateManager:
    """
    Factory function para criar gerenciador de estado.

    Args:
        notebook_name: Nome do notebook

    Returns:
        Instância configurada do NotebookStateManager
    """
    return NotebookStateManager(notebook_name)

def validate_notebook_execution_order():
    """
    Validar ordem de execução dos notebooks no diretório atual.
    Útil para verificar se notebooks foram executados na ordem correta.
    """
    print("🔍 VALIDANDO ORDEM DE EXECUÇÃO DOS NOTEBOOKS")
    print("=" * 60)

    config = NotebookConfig()
    state_file = config.STATE_FILE

    if not state_file.exists():
        print("❌ Nenhum estado encontrado - execute os notebooks primeiro")
        return False

    try:
        with open(state_file, 'r') as f:
            state_data = json.load(f)

        notebooks = state_data.get('notebooks', {})

        if not notebooks:
            print("❌ Nenhum notebook executado ainda")
            return False

        # Ordenar por tempo de execução
        sorted_notebooks = sorted(
            notebooks.items(),
            key=lambda x: x[1].get('last_execution', '2000-01-01')
        )

        print("📋 Ordem de execução detectada:")
        for i, (name, state) in enumerate(sorted_notebooks, 1):
            exec_time = state.get('last_execution', 'unknown')
            exec_count = state.get('execution_count', 0)
            print(f"   {i}. {name} (execuções: {exec_count}, última: {exec_time})")

        # Verificar ordem lógica
        expected_order = [
            "01_data_preparation",
            "02_baseline_models",
            "03_hyperparameter_tuning",
            "04_model_evaluation",
            "05_final_results"
        ]

        actual_order = [name for name, _ in sorted_notebooks]
        expected_order_filtered = [n for n in expected_order if n in actual_order]

        if actual_order == expected_order_filtered:
            print("\n✅ Ordem de execução CORRETA")
            return True
        else:
            print("\n⚠️  Ordem de execução pode estar INCORRETA")
            print(f"   Esperado: {expected_order_filtered}")
            print(f"   Atual:    {actual_order}")
            return False

    except Exception as e:
        print(f"❌ Erro ao validar ordem: {e}")
        return False

# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Alias para facilitar uso
StateManager = NotebookStateManager
create_state_manager = create_notebook_state_manager

if __name__ == "__main__":
    # Teste básico
    print("🧪 Testando NotebookStateManager...")

    mgr = NotebookStateManager("test_notebook")
    print("✅ Módulo funcionando corretamente")