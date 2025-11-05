"""
Configuração centralizada para notebooks AML.
Todos os parâmetros configuráveis em um único lugar.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NotebookConfig:
    """Configuração imutável para notebooks de modelagem AML."""
    
    # Diretórios
    data_dir: Path = field(default_factory=lambda: Path('..') / 'data' / 'processed')
    artifacts_dir: Path = field(default_factory=lambda: Path('..') / 'artifacts')
    
    # Arquivos
    features_file_pkl: str = 'features_with_patterns_sampled.pkl'
    
    # Parâmetros de treinamento
    random_seed: int = 42
    test_size: float = 0.2
    
    # Parâmetros de otimização
    optuna_trials: int = 30
    cv_splits: int = 5
    
    # Estado do modelo (será preenchido durante execução)
    best_model_name: Optional[str] = None
    best_model_params: dict = field(default_factory=dict)
    
    def to_dict(self):
        """Converte para dicionário para compatibilidade com código legado."""
        return {
            'data_dir': self.data_dir,
            'artifacts_dir': self.artifacts_dir,
            'features_file_pkl': self.features_file_pkl,
            'random_seed': self.random_seed,
            'test_size': self.test_size,
            'optuna_trials': self.optuna_trials,
            'best_model_name': self.best_model_name,
            'best_model_params': self.best_model_params
        }


# Instância global padrão
CONFIG = NotebookConfig()
