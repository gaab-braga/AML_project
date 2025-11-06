# 🚀 Roadmap de Refatoração: AML Project - Do Notebook à Produção

> **Objetivo:** Transformar um projeto de ciência de dados baseado em notebooks em um sistema profissional, modular e pronto para produção, seguindo princípios de Clean Code, MLOps e arquitetura de software.

---

## 📋 Sumário Executivo

### Situação Atual
- ✅ **Notebooks completos e funcionais** (7 notebooks de ponta a ponta)
- ✅ **Modelos treinados e avaliados** com métricas excelentes
- ✅ **Estrutura inicial de produção** (API, dashboard, scripts)
- ⚠️ **Código duplicado** entre notebooks e módulos `src/`
- ⚠️ **Referências legadas** em scripts e configurações
- ⚠️ **Complexidade desnecessária** em alguns módulos

### Objetivo Final
Um sistema de ML profissional com:
- **Modularidade:** Cada módulo tem responsabilidade única e clara
- **Reprodutibilidade:** Qualquer pessoa pode executar o pipeline completo
- **Manutenibilidade:** Código limpo, documentado e testado
- **Production-Ready:** API, monitoramento, logging e deployment

---

## 🎯 Princípios Orientadores

### 1. **KISS (Keep It Simple, Stupid)**
- Se pode ser feito em 10 linhas, não faça em 100
- Evite abstrações prematuras
- Prefira clareza sobre "elegância"

### 2. **DRY (Don't Repeat Yourself)**
- Função escrita uma vez, usada em múltiplos lugares
- Configurações centralizadas
- Zero duplicação entre notebooks e código de produção

### 3. **Single Responsibility Principle**
- Cada módulo/função faz **UMA COISA** e a faz bem
- Nomes descritivos que revelam a intenção
- Separação clara entre: dados, features, modelos, avaliação, deploy

### 4. **Human-Readable Code**
- Código auto-explicativo (nomes claros > comentários excessivos)
- Estrutura intuitiva de diretórios
- Documentação concisa e objetiva

---

## 📂 Análise da Estrutura Atual

### ✅ O que **MANTER**

#### 1. `data/` - Dados do Projeto
```
data/
├── raw/          # ✅ Dados originais, imutáveis
└── processed/    # ✅ Dados prontos para modelagem
```
**Status:** Estrutura correta. Manter.

#### 2. `models/` - Artefatos de Modelo
```
models/
├── *.pkl         # Modelos serializados
├── *.pt          # Modelos PyTorch
└── metadata.yaml # Metadados dos modelos
```
**Status:** Correto. Adicionar versionamento semântico.

#### 3. `artifacts/` - Resultados e Métricas
```
artifacts/
├── *_results.json
├── *_report.json
├── *.csv
└── shap_plots/
```
**Status:** Bom para rastreabilidade. Manter.

#### 4. `notebooks/` - Exploração e Relatórios
```
notebooks/
├── 01_Data_Ingestion_EDA.ipynb
├── 02_IBM_Benchmark.ipynb
├── 03_Model_Selection_Tuning.ipynb
├── 04_Ensemble_Modeling.ipynb
├── 05_Model_Interpretation.ipynb
├── 06_Robustness_Validation.ipynb
└── 07_Executive_Summary.ipynb
```
**Status:** Excelente documentação. Notebooks devem **importar** funções de `src/`, não reimplementá-las.

#### 5. `config/` - Configurações
```
config/
├── pipeline_config.yaml       # ✅ Pipeline ML
├── features.yaml              # ✅ Feature engineering
├── monitoring_config.yaml     # ✅ Monitoramento
├── dashboard_config.yaml      # ✅ Dashboard
└── security_config.yaml       # ✅ Segurança
```
**Status:** Bem organizado. Consolidar em um único `config.yaml` principal.

---

### ⚠️ O que **REFATORAR**

#### 1. `src/` - Código Fonte (CRÍTICO)

**Estrutura Atual:**
```
src/
├── data/
├── eda/
├── evaluation/
├── evaluation_module/    # ⚠️ Duplicação com evaluation/
├── features/
├── interfaces/
├── modeling/
├── models/               # ⚠️ Confuso com models/ na raiz
├── optimization/
├── orchestration/
├── reporting/
├── utils/
├── visualization/
└── monitoring_service.py
```

**Problemas Identificados:**
- **Duplicação:** `evaluation/` vs `evaluation_module/`
- **Confusão:** `src/models/` vs `models/` (raiz)
- **Over-engineering:** Módulos como `interfaces/`, `orchestration/` podem estar vazios ou subutilizados
- **Falta de clareza:** Muitos subdiretórios dificultam navegação

**Estrutura Proposta (LIMPA):**
```
src/
├── __init__.py
├── config.py              # Carregamento centralizado de configs
│
├── data/
│   ├── __init__.py
│   ├── loader.py          # Carregar dados brutos
│   └── preprocessing.py   # Limpeza e transformação
│
├── features/
│   ├── __init__.py
│   ├── engineering.py     # Criar features (temporal, network, etc)
│   └── selection.py       # Seleção de features
│
├── models/
│   ├── __init__.py
│   ├── train.py           # Treinamento de modelos
│   ├── predict.py         # Predição
│   └── evaluate.py        # Métricas e avaliação
│
├── explainability/
│   ├── __init__.py
│   └── shap_analysis.py   # SHAP e interpretabilidade
│
├── monitoring/
│   ├── __init__.py
│   └── service.py         # Monitoramento de produção
│
└── utils/
    ├── __init__.py
    ├── logger.py          # Logging centralizado
    └── helpers.py         # Funções auxiliares genéricas

entrypoints/
├── __init__.py
├── cli.py                 # Interface de linha de comando (Click/Typer)
├── api.py                 # API REST (FastAPI)
├── batch.py               # Processamento em batch
└── stream.py              # Processamento em stream (Kafka/Redis)
```

**Ações:**
1. **Deletar:** `evaluation_module/`, `src/models/`, `interfaces/`, `orchestration/`, `eda/`, `reporting/`
2. **Consolidar:** Mover código útil para os módulos principais
3. **Simplificar:** Reduzir de 15+ subdiretórios para 6 essenciais
4. **Criar:** Novo diretório `entrypoints/` para separar pontos de entrada do core business logic

---

#### 2. `entrypoints/` - Pontos de Entrada do Sistema (NOVO!)

**Por que criar este diretório?**

Em uma arquitetura limpa, **separamos a lógica de negócio (src/) dos pontos de entrada (entrypoints/)**. Isso permite:

- ✅ **Múltiplas interfaces** para o mesmo core: CLI, API REST, batch jobs, streaming
- ✅ **Testabilidade:** Testar lógica de negócio sem inicializar servidor/CLI
- ✅ **Flexibilidade:** Adicionar novos entrypoints sem modificar `src/`
- ✅ **Clareza:** Fica explícito onde o sistema pode ser invocado

**Estrutura Proposta:**
```
entrypoints/
├── __init__.py
│
├── cli.py                 # Interface de linha de comando
│   └── Comandos: train, predict, evaluate, serve
│
├── api.py                 # API REST (FastAPI)
│   └── Endpoints: /predict, /batch, /health
│
├── batch.py               # Processamento em lote (Airflow/Cron)
│   └── Processa grandes volumes de dados
│
└── stream.py              # Processamento em tempo real (Kafka/Redis)
    └── Consome eventos e faz predições
```

**Exemplo de Uso:**
```powershell
# Via CLI
python -m entrypoints.cli train --config config/pipeline_config.yaml
python -m entrypoints.cli predict --input data.csv --output predictions.csv

# Via API
python -m entrypoints.api
# ou
uvicorn entrypoints.api:app --reload

# Via Batch (scheduled)
python -m entrypoints.batch --date 2025-11-06

# Via Stream
python -m entrypoints.stream --kafka-topic transactions
```

**Ações:**
1. **Criar:** Novo diretório `entrypoints/` na raiz (mesmo nível que `src/`)
2. **Migrar:** Código de `api/production_api.py` → `entrypoints/api.py`
3. **Migrar:** Lógica de `scripts/train_pipeline.py` → `entrypoints/cli.py`
4. **Deletar:** Diretório `api/` antigo (substituído por `entrypoints/api.py`)

---

#### 3. `scripts/` - Scripts Utilitários (REDUZIDO)

**Estrutura Atual:**
```
scripts/
├── integrate_patterns.py
├── pattern_integration_demo.py
├── production_readiness.py
├── reproduce_all.py
├── retraining_pipeline.py
├── robustness_validation.py
├── scenario_analysis_demo.py
└── README_convert_to_multignn_csv.md
```

**Problemas:**
- Nomes genéricos (`demo`, `integrate`)
- Mistura de responsabilidades (alguns deveriam estar em `entrypoints/`)
- Alguns podem estar obsoletos

**Estrutura Proposta (MINIMALISTA):**
```
scripts/
├── setup_project.sh       # Setup inicial do projeto
├── export_to_multignn.py  # Conversão para benchmark IBM
└── generate_report.py     # Relatórios ad-hoc para stakeholders
```

**Ações:**
1. **Migrar:** `reproduce_all.py`, `retraining_pipeline.py` → `entrypoints/cli.py`
2. **Deletar:** Todos os arquivos `*_demo.py` (código deve estar em `src/`)
3. **Deletar:** `production_readiness.py` (lógica em `src/`, invocado via `entrypoints/`)
4. **Manter:** Apenas scripts auxiliares que NÃO fazem parte do fluxo principal

---

#### 4. `dashboard/` - Dashboard Streamlit

**Estrutura Atual:**
```
dashboard/
└── app.py
```

**Status:** Funcional. Garantir que importa de `src/` e não reimplementa lógica.

---

### 🗑️ O que **DELETAR**

#### Diretórios a Remover:
1. **`benchmark/`** - Se não for usado ativamente, mover para documentação externa
2. **`deploy/`** - Deployment deve ser via Docker/K8s (não apenas manifests soltos)
3. **`logs/`** - Logs devem ser gerados dinamicamente, não versionados
4. **`__pycache__/`** - Sempre no `.gitignore`

#### Arquivos Obsoletos:
- Scripts duplicados ou de teste
- Notebooks antigos (`*_old.ipynb`, `*_backup.ipynb`)
- Configs não utilizados

---

## 🛠️ Plano de Execução Detalhado

### **FASE 1: LIMPEZA E ORGANIZAÇÃO** (2-3 horas)

#### Passo 1.1: Backup e Preparação
```powershell
# Criar branch de refatoração
git checkout -b refactor/mlops-structure

# Backup completo
git commit -am "Checkpoint antes da refatoração"
```

#### Passo 1.2: Deletar Diretórios Desnecessários
```powershell
# Remover caches e logs
Remove-Item -Recurse -Force __pycache__, logs/, .pytest_cache/

# Remover duplicações em src/
Remove-Item -Recurse -Force src/evaluation_module/, src/interfaces/, src/orchestration/, src/eda/, src/reporting/
```

#### Passo 1.3: Reestruturar `src/`
```powershell
# Criar nova estrutura limpa
New-Item -ItemType Directory -Force src/explainability
New-Item -ItemType Directory -Force src/monitoring

# Mover arquivos para locais corretos
Move-Item src/monitoring_service.py src/monitoring/service.py
```

---

### **FASE 2: REFATORAÇÃO DO CÓDIGO** (8-12 horas)

#### Passo 2.1: Criar `src/config.py`

**Objetivo:** Centralizar carregamento de todas as configurações.

```python
# src/config.py
"""
Gerenciamento centralizado de configurações.
Carrega e valida arquivos YAML de config/.
"""
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Singleton para configurações do projeto."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load()
    
    def load(self, config_path: str = "config/pipeline_config.yaml"):
        """Carrega configuração principal."""
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Acessa valor de configuração via dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    @property
    def data_path(self) -> Path:
        return Path(self.get('data.path', 'data/'))
    
    @property
    def model_path(self) -> Path:
        return Path(self.get('output.model_path', 'models/'))

# Instância global
config = Config()
```

**Uso em qualquer módulo:**
```python
from src.config import config

data_dir = config.data_path
model_params = config.get('model.params')
```

---

#### Passo 2.2: Refatorar `src/data/`

**`src/data/loader.py`** - Carregamento de dados
```python
"""
Carregamento de dados brutos e processados.
Responsabilidade: I/O de dados.
"""
import pandas as pd
from pathlib import Path
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data(filename: str = "transactions.csv") -> pd.DataFrame:
    """
    Carrega dados brutos do diretório raw/.
    
    Args:
        filename: Nome do arquivo CSV
        
    Returns:
        DataFrame com dados brutos
    """
    filepath = config.data_path / "raw" / filename
    logger.info(f"Carregando dados de {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Carregados {len(df)} registros com {df.shape[1]} colunas")
    
    return df

def load_processed_data(split: str = "train") -> pd.DataFrame:
    """
    Carrega dados processados (train/test).
    
    Args:
        split: 'train' ou 'test'
        
    Returns:
        DataFrame processado
    """
    filepath = config.data_path / "processed" / f"{split}.pkl"
    logger.info(f"Carregando dados processados de {filepath}")
    
    return pd.read_pickle(filepath)

def save_processed_data(df: pd.DataFrame, split: str = "train"):
    """Salva dados processados."""
    filepath = config.data_path / "processed" / f"{split}.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_pickle(filepath)
    logger.info(f"Dados salvos em {filepath}")
```

**`src/data/preprocessing.py`** - Limpeza e transformação
```python
"""
Pré-processamento de dados: limpeza, imputação, split.
Responsabilidade: Transformações de dados brutos.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza básica: remover duplicatas, tratar nulos.
    
    Args:
        df: DataFrame bruto
        
    Returns:
        DataFrame limpo
    """
    logger.info("Iniciando limpeza de dados")
    
    # Remover duplicatas
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removidas {initial_rows - len(df)} duplicatas")
    
    # Tratar valores ausentes (estratégia simples)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('MISSING')
    
    logger.info("Limpeza concluída")
    return df

def split_train_test(df: pd.DataFrame, target_col: str = None) -> tuple:
    """
    Divide dados em treino e teste.
    
    Args:
        df: DataFrame completo
        target_col: Nome da coluna alvo (pega do config se None)
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_col is None:
        target_col = config.get('model.target_column', 'is_laundering')
    
    test_size = config.get('validation.test_size', 0.2)
    random_state = config.get('model.random_state', 42)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Split: {len(X_train)} treino, {len(X_test)} teste")
    return X_train, X_test, y_train, y_test
```

---

#### Passo 2.3: Refatorar `src/features/`

**`src/features/engineering.py`** - Criação de features
```python
"""
Feature Engineering: criação de features temporais, de rede, estatísticas.
Responsabilidade: Transformar dados limpos em features para ML.
"""
import pandas as pd
import numpy as np
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Cria features temporais (hora do dia, dia da semana, etc).
    
    Args:
        df: DataFrame com coluna de timestamp
        timestamp_col: Nome da coluna de timestamp
        
    Returns:
        DataFrame com features temporais adicionadas
    """
    logger.info("Criando features temporais")
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(22, 6).astype(int)
    
    logger.info("Features temporais criadas")
    return df

def create_aggregation_features(
    df: pd.DataFrame, 
    group_col: str, 
    agg_col: str,
    windows: list = None
) -> pd.DataFrame:
    """
    Cria features de agregação (soma, média, contagem em janelas).
    
    Args:
        df: DataFrame
        group_col: Coluna para agrupar (ex: 'account_id')
        agg_col: Coluna para agregar (ex: 'amount')
        windows: Lista de janelas temporais (dias)
        
    Returns:
        DataFrame com features agregadas
    """
    if windows is None:
        windows = config.get('features.windows', [7, 30])
    
    logger.info(f"Criando agregações para {group_col} sobre {agg_col}")
    
    df = df.copy()
    
    for window in windows:
        df[f'{agg_col}_sum_{window}d'] = (
            df.groupby(group_col)[agg_col]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        df[f'{agg_col}_mean_{window}d'] = (
            df.groupby(group_col)[agg_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    logger.info(f"Agregações criadas para janelas {windows}")
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    
    Args:
        df: DataFrame limpo
        
    Returns:
        DataFrame com todas as features
    """
    logger.info("Iniciando pipeline de feature engineering")
    
    df = create_temporal_features(df)
    df = create_aggregation_features(df, 'account_id', 'amount')
    
    logger.info("Feature engineering concluído")
    return df
```

---

#### Passo 2.4: Refatorar `src/models/`

**`src/models/train.py`** - Treinamento
```python
"""
Treinamento de modelos ML.
Responsabilidade: Treinar, validar e serializar modelos.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    'random_forest': RandomForestClassifier,
    'xgboost': XGBClassifier,
    'lightgbm': LGBMClassifier
}

def get_model(model_name: str = None):
    """
    Instancia modelo com hiperparâmetros do config.
    
    Args:
        model_name: Nome do modelo (default: pega do config)
        
    Returns:
        Instância do modelo configurado
    """
    if model_name is None:
        model_name = config.get('model.name', 'xgboost')
    
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Modelo '{model_name}' não encontrado no registro")
    
    params = config.get('model.params', {})
    logger.info(f"Instanciando {model_name} com parâmetros: {params}")
    
    return model_class(**params)

def train_model(X_train, y_train, model_name: str = None):
    """
    Treina modelo.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        model_name: Nome do modelo
        
    Returns:
        Modelo treinado
    """
    logger.info("Iniciando treinamento")
    
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    logger.info("Treinamento concluído")
    return model

def save_model(model, filename: str = None):
    """
    Serializa modelo treinado.
    
    Args:
        model: Modelo treinado
        filename: Nome do arquivo (default: pega do config)
    """
    if filename is None:
        filename = config.get('output.model_path', 'models/model.pkl')
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    logger.info(f"Modelo salvo em {filepath}")

def load_model(filename: str = None):
    """Carrega modelo serializado."""
    if filename is None:
        filename = config.get('output.model_path', 'models/model.pkl')
    
    logger.info(f"Carregando modelo de {filename}")
    return joblib.load(filename)
```

**`src/models/predict.py`** - Predição
```python
"""
Predição em produção.
Responsabilidade: Inferência em novos dados.
"""
import pandas as pd
from src.models.train import load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(X, model=None, return_proba: bool = True):
    """
    Faz predições em novos dados.
    
    Args:
        X: Features
        model: Modelo (carrega se None)
        return_proba: Se True, retorna probabilidades
        
    Returns:
        Array de predições ou probabilidades
    """
    if model is None:
        model = load_model()
    
    logger.info(f"Fazendo predições para {len(X)} amostras")
    
    if return_proba and hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(X)[:, 1]
    else:
        predictions = model.predict(X)
    
    return predictions

def predict_batch(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predição em batch com retorno estruturado.
    
    Args:
        df: DataFrame com features
        model: Modelo treinado
        
    Returns:
        DataFrame com predições e probabilidades
    """
    if model is None:
        model = load_model()
    
    predictions = predict(df, model, return_proba=False)
    probabilities = predict(df, model, return_proba=True)
    
    result = df.copy()
    result['prediction'] = predictions
    result['probability'] = probabilities
    
    return result
```

**`src/models/evaluate.py`** - Avaliação
```python
"""
Avaliação de modelos: métricas, curvas, relatórios.
Responsabilidade: Computar e reportar performance.
"""
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """
    Calcula métricas de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        y_proba: Probabilidades (opcional)
        
    Returns:
        Dicionário com métricas
    """
    logger.info("Calculando métricas de avaliação")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    logger.info(f"Métricas calculadas: {metrics}")
    return metrics

def save_evaluation_report(metrics: dict, filepath: str = "artifacts/evaluation.json"):
    """Salva relatório de avaliação."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Relatório salvo em {filepath}")

def print_evaluation_summary(metrics: dict):
    """Imprime resumo das métricas."""
    print("\n" + "="*50)
    print("RESUMO DA AVALIAÇÃO")
    print("="*50)
    print(f"Acurácia:  {metrics['accuracy']:.4f}")
    print(f"Precisão:  {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print("="*50 + "\n")
```

---

#### Passo 2.5: Criar `src/utils/logger.py`

```python
"""
Logging centralizado para todo o projeto.
Responsabilidade: Configurar e fornecer loggers consistentes.
"""
import logging
import sys
from pathlib import Path

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Cria logger configurado.
    
    Args:
        name: Nome do módulo (__name__)
        level: Nível de log
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger
```

---

### **FASE 3: ENTRYPOINTS - CLI** (3-4 horas)

#### Passo 3.1: Criar `entrypoints/cli.py` - Interface de Linha de Comando

Vamos usar **Typer** (ou Click) para criar uma CLI profissional e moderna.

```python
#!/usr/bin/env python3
"""
Interface de Linha de Comando (CLI) para o sistema AML.
Ponto de entrada principal para operações via terminal.

Comandos disponíveis:
- train: Treinar modelo
- predict: Fazer predições
- evaluate: Avaliar modelo
- serve: Iniciar servidor API
"""
import typer
from typing import Optional
from pathlib import Path
import pandas as pd

from src.data.loader import load_raw_data, save_processed_data, load_processed_data
from src.data.preprocessing import clean_data, split_train_test
from src.features.engineering import build_features
from src.models.train import train_model, save_model, load_model
from src.models.evaluate import evaluate_model, save_evaluation_report, print_evaluation_summary
from src.models.predict import predict_batch
from src.utils.logger import get_logger

# Criar aplicação CLI
app = typer.Typer(
    name="aml-cli",
    help="Sistema de Detecção de Lavagem de Dinheiro - Interface de Linha de Comando"
)

logger = get_logger(__name__)

@app.command()
def train(
    data_file: Optional[str] = typer.Option(None, "--data", "-d", help="Arquivo CSV de dados brutos"),
    config_file: Optional[str] = typer.Option("config/pipeline_config.yaml", "--config", "-c", help="Arquivo de configuração"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="Nome do modelo (xgboost, lightgbm, random_forest)"),
):
    """
    Treina um modelo de detecção de lavagem de dinheiro.
    
    Exemplo:
        python -m entrypoints.cli train --data data/raw/transactions.csv --model xgboost
    """
    typer.secho("\n🚀 INICIANDO TREINAMENTO DO MODELO\n", fg=typer.colors.GREEN, bold=True)
    
    try:
        # 1. Carregar dados
        typer.echo("📂 [1/7] Carregando dados brutos...")
        df = load_raw_data(data_file) if data_file else load_raw_data()
        typer.secho(f"   ✓ Carregados {len(df)} registros", fg=typer.colors.GREEN)
        
        # 2. Limpeza
        typer.echo("\n🧹 [2/7] Limpando dados...")
        df_clean = clean_data(df)
        typer.secho(f"   ✓ Dados limpos: {len(df_clean)} registros", fg=typer.colors.GREEN)
        
        # 3. Feature Engineering
        typer.echo("\n🔧 [3/7] Criando features...")
        df_features = build_features(df_clean)
        typer.secho(f"   ✓ Features criadas: {df_features.shape[1]} colunas", fg=typer.colors.GREEN)
        
        # 4. Split
        typer.echo("\n✂️  [4/7] Dividindo em treino/teste...")
        X_train, X_test, y_train, y_test = split_train_test(df_features)
        typer.secho(f"   ✓ Treino: {len(X_train)} | Teste: {len(X_test)}", fg=typer.colors.GREEN)
        
        # Salvar dados processados
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        save_processed_data(train_data, 'train')
        save_processed_data(test_data, 'test')
        
        # 5. Treinar
        typer.echo("\n🎯 [5/7] Treinando modelo...")
        model = train_model(X_train, y_train, model_name)
        typer.secho(f"   ✓ Modelo treinado com sucesso", fg=typer.colors.GREEN)
        
        # 6. Avaliar
        typer.echo("\n📊 [6/7] Avaliando modelo...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = evaluate_model(y_test, y_pred, y_proba)
        print_evaluation_summary(metrics)
        
        # 7. Salvar
        typer.echo("\n💾 [7/7] Salvando artefatos...")
        save_model(model)
        save_evaluation_report(metrics)
        typer.secho(f"   ✓ Modelo salvo", fg=typer.colors.GREEN)
        
        typer.secho("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!\n", fg=typer.colors.GREEN, bold=True)
        
    except Exception as e:
        typer.secho(f"\n❌ ERRO: {str(e)}\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

@app.command()
def predict(
    input_file: str = typer.Argument(..., help="Arquivo CSV com dados para predição"),
    output_file: str = typer.Option("predictions.csv", "--output", "-o", help="Arquivo de saída com predições"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Caminho do modelo treinado"),
):
    """
    Faz predições em novos dados.
    
    Exemplo:
        python -m entrypoints.cli predict data/new_transactions.csv -o results.csv
    """
    typer.secho("\n🔮 FAZENDO PREDIÇÕES\n", fg=typer.colors.BLUE, bold=True)
    
    try:
        # Carregar dados
        typer.echo(f"📂 Carregando dados de {input_file}...")
        df = pd.read_csv(input_file)
        typer.secho(f"   ✓ {len(df)} registros carregados", fg=typer.colors.GREEN)
        
        # Carregar modelo
        typer.echo("\n🤖 Carregando modelo...")
        model = load_model(model_path) if model_path else load_model()
        typer.secho(f"   ✓ Modelo carregado", fg=typer.colors.GREEN)
        
        # Feature engineering (você deve aplicar as mesmas transformações do treino)
        typer.echo("\n🔧 Processando features...")
        df_features = build_features(df)
        
        # Predição
        typer.echo("\n🎯 Fazendo predições...")
        results = predict_batch(df_features, model)
        
        # Salvar
        results.to_csv(output_file, index=False)
        typer.secho(f"\n✅ Predições salvas em {output_file}", fg=typer.colors.GREEN, bold=True)
        
        # Estatísticas
        suspicious_count = results['prediction'].sum()
        typer.echo(f"\n📈 Estatísticas:")
        typer.echo(f"   • Total de transações: {len(results)}")
        typer.echo(f"   • Transações suspeitas: {suspicious_count} ({suspicious_count/len(results)*100:.2f}%)")
        typer.echo(f"   • Risco médio: {results['probability'].mean():.4f}")
        
    except Exception as e:
        typer.secho(f"\n❌ ERRO: {str(e)}\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

@app.command()
def evaluate(
    test_data: Optional[str] = typer.Option(None, "--test-data", "-t", help="Arquivo de teste (default: usa data/processed/test.pkl)"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m", help="Caminho do modelo"),
):
    """
    Avalia um modelo treinado.
    
    Exemplo:
        python -m entrypoints.cli evaluate --model models/xgboost_model.pkl
    """
    typer.secho("\n📊 AVALIANDO MODELO\n", fg=typer.colors.YELLOW, bold=True)
    
    try:
        # Carregar dados de teste
        if test_data:
            df_test = pd.read_csv(test_data)
        else:
            df_test = load_processed_data('test')
        
        # Separar X e y
        from src.config import config
        target_col = config.get('model.target_column', 'is_laundering')
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]
        
        # Carregar modelo
        model = load_model(model_path) if model_path else load_model()
        
        # Predição
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Avaliar
        metrics = evaluate_model(y_test, y_pred, y_proba)
        print_evaluation_summary(metrics)
        
        # Salvar relatório
        save_evaluation_report(metrics, "artifacts/evaluation_report.json")
        typer.secho("\n✅ Relatório salvo em artifacts/evaluation_report.json", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"\n❌ ERRO: {str(e)}\n", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host do servidor"),
    port: int = typer.Option(8000, "--port", "-p", help="Porta do servidor"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload em desenvolvimento"),
):
    """
    Inicia o servidor API.
    
    Exemplo:
        python -m entrypoints.cli serve --port 8000 --reload
    """
    typer.secho(f"\n🚀 INICIANDO SERVIDOR API em {host}:{port}\n", fg=typer.colors.MAGENTA, bold=True)
    
    import uvicorn
    uvicorn.run(
        "entrypoints.api:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    app()
```

**Adicionar ao `requirements.txt`:**
```
typer[all]>=0.9.0  # CLI framework moderno
```

**Uso:**
```powershell
# Treinar modelo
python -m entrypoints.cli train --data data/raw/transactions.csv --model xgboost

# Fazer predições
python -m entrypoints.cli predict data/new_data.csv -o predictions.csv

# Avaliar modelo
python -m entrypoints.cli evaluate --model models/my_model.pkl

# Iniciar API
python -m entrypoints.cli serve --port 8000 --reload

# Ver ajuda
python -m entrypoints.cli --help
python -m entrypoints.cli train --help
```

---

### **FASE 4: ENTRYPOINTS - API DE PRODUÇÃO** (4-6 horas)

#### Passo 4.1: Criar `entrypoints/api.py` - API FastAPI

**`entrypoints/api.py`**
```python
"""
API de Produção para Sistema AML.
Framework: FastAPI
Features: Auto-documentação, validação, async, type hints
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

from src.models.predict import predict, load_model
from src.monitoring.service import AMLMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Inicializar app
app = FastAPI(
    title="AML Detection API",
    description="API para detecção de lavagem de dinheiro em tempo real",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo na inicialização
model = None
monitor = AMLMonitor()

@app.on_event("startup")
async def startup_event():
    """Carrega modelo ao iniciar."""
    global model
    logger.info("Carregando modelo...")
    model = load_model()
    logger.info("Modelo carregado com sucesso")

# Schemas Pydantic
class Transaction(BaseModel):
    """Schema de uma transação."""
    account_id: str
    amount: float = Field(..., gt=0, description="Valor da transação (deve ser positivo)")
    timestamp: str
    # Adicione outros campos conforme necessário
    
    class Config:
        schema_extra = {
            "example": {
                "account_id": "ACC123456",
                "amount": 15000.50,
                "timestamp": "2025-11-06T14:30:00"
            }
        }

class PredictionResponse(BaseModel):
    """Schema da resposta de predição."""
    transaction_id: Optional[str] = None
    is_suspicious: bool
    risk_score: float = Field(..., ge=0, le=1, description="Probabilidade de lavagem (0-1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")

# Endpoints
@app.get("/")
async def root():
    """Health check simples."""
    return {"status": "online", "service": "AML Detection API"}

@app.get("/health")
async def health_check():
    """
    Endpoint de health check completo.
    Verifica status do modelo e sistema.
    """
    health_report = monitor.get_health_report()
    
    if health_report['status'] == 'critical':
        raise HTTPException(status_code=503, detail="Sistema crítico")
    
    return health_report

@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """
    Faz predição para uma transação.
    
    Args:
        transaction: Dados da transação
        
    Returns:
        Resposta com flag de suspeita e score de risco
    """
    try:
        # Converter para DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Feature engineering (simplificado - você deve usar src.features aqui)
        # df = build_features(df)
        
        # Predição
        risk_score = float(predict(df, model, return_proba=True)[0])
        
        # Classificar nível de risco
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.85:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Monitorar
        monitor.log_prediction(risk_score)
        
        return PredictionResponse(
            transaction_id=transaction.account_id,
            is_suspicious=risk_score > 0.5,
            risk_score=risk_score,
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(transactions: List[Transaction]):
    """
    Predição em batch para múltiplas transações.
    
    Args:
        transactions: Lista de transações
        
    Returns:
        Lista de predições
    """
    results = []
    
    for txn in transactions:
        result = await predict_transaction(txn)
        results.append(result)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Passo 4.2: Criar `entrypoints/batch.py` - Processamento em Lote

```python
"""
Processamento em lote (batch) para grandes volumes.
Ideal para jobs agendados (Airflow, Cron, GitHub Actions).

Exemplo de uso:
    python -m entrypoints.batch --date 2025-11-06 --input data/transactions_20251106.csv
"""
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.data.preprocessing import clean_data
from src.features.engineering import build_features
from src.models.predict import predict_batch, load_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def process_batch(input_file: str, output_dir: str = "data/predictions", date: str = None):
    """
    Processa um lote de transações e salva predições.
    
    Args:
        input_file: Arquivo CSV com transações
        output_dir: Diretório de saída
        date: Data do lote (formato YYYY-MM-DD)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Processando batch para data: {date}")
    
    # Carregar dados
    logger.info(f"Carregando {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Carregados {len(df)} registros")
    
    # Preprocessar
    df_clean = clean_data(df)
    df_features = build_features(df_clean)
    
    # Carregar modelo
    model = load_model()
    
    # Predição
    logger.info("Fazendo predições em batch...")
    results = predict_batch(df_features, model)
    
    # Salvar resultados
    output_path = Path(output_dir) / f"predictions_{date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    logger.info(f"Predições salvas em {output_path}")
    
    # Estatísticas
    suspicious = results[results['prediction'] == 1]
    logger.info(f"Total: {len(results)} | Suspeitos: {len(suspicious)} ({len(suspicious)/len(results)*100:.2f}%)")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processamento em batch AML")
    parser.add_argument("--input", "-i", required=True, help="Arquivo de entrada")
    parser.add_argument("--output", "-o", default="data/predictions", help="Diretório de saída")
    parser.add_argument("--date", "-d", help="Data do batch (YYYY-MM-DD)")
    
    args = parser.parse_args()
    process_batch(args.input, args.output, args.date)
```

**Uso:**
```powershell
# Processar batch de hoje
python -m entrypoints.batch --input data/daily/transactions_today.csv

# Processar batch específico
python -m entrypoints.batch --input data/historical/2025-11-06.csv --date 2025-11-06 --output results/
```

**Integração com Airflow (exemplo):**
```python
# dags/aml_batch_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    'aml_batch_processing',
    start_date=datetime(2025, 11, 1),
    schedule_interval='@daily'
)

process_batch = BashOperator(
    task_id='process_batch',
    bash_command='python -m entrypoints.batch --input data/daily/{{ ds }}.csv --date {{ ds }}',
    dag=dag
)
```

---

**Executar API:**
```powershell
# Via CLI helper
python -m entrypoints.cli serve --reload

# Diretamente
uvicorn entrypoints.api:app --reload

# Produção
uvicorn entrypoints.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Acessar documentação automática:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

### **FASE 5: TESTES E QUALIDADE** (4-6 horas)

#### Passo 5.1: Estrutura de Testes

```
tests/
├── __init__.py
├── conftest.py              # Fixtures compartilhados
├── test_data_loader.py
├── test_preprocessing.py
├── test_features.py
├── test_models.py
└── test_api.py
```

#### Passo 5.2: Exemplo de Testes

**`tests/conftest.py`** - Fixtures
```python
"""
Fixtures compartilhados para testes.
"""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_dataframe():
    """DataFrame de exemplo para testes."""
    return pd.DataFrame({
        'account_id': ['A001', 'A002', 'A003'],
        'amount': [100.0, 250.0, 50.0],
        'timestamp': pd.date_range('2025-01-01', periods=3),
        'is_laundering': [0, 1, 0]
    })

@pytest.fixture
def sample_features():
    """Features de exemplo."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0],
        'feature_2': [10.0, 20.0, 30.0]
    })

@pytest.fixture
def sample_labels():
    """Labels de exemplo."""
    return pd.Series([0, 1, 0])
```

**`tests/test_data_loader.py`**
```python
"""
Testes para módulo de carregamento de dados.
"""
import pytest
from src.data.loader import load_raw_data, load_processed_data

def test_load_raw_data_returns_dataframe():
    """Testa se load_raw_data retorna um DataFrame."""
    # Mock ou use arquivo de teste
    # df = load_raw_data('test_data.csv')
    # assert isinstance(df, pd.DataFrame)
    pass  # Implementar com dados de teste

def test_load_raw_data_not_empty():
    """Testa se dados carregados não estão vazios."""
    pass  # Implementar
```

**`tests/test_preprocessing.py`**
```python
"""
Testes para pré-processamento.
"""
import pytest
import pandas as pd
from src.data.preprocessing import clean_data, split_train_test

def test_clean_data_removes_duplicates(sample_dataframe):
    """Testa se clean_data remove duplicatas."""
    df_with_dup = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]])
    df_clean = clean_data(df_with_dup)
    
    assert len(df_clean) == len(sample_dataframe)

def test_split_train_test_returns_four_elements(sample_dataframe):
    """Testa se split retorna 4 elementos."""
    result = split_train_test(sample_dataframe, 'is_laundering')
    
    assert len(result) == 4

def test_split_preserves_total_samples(sample_dataframe):
    """Testa se split preserva número total de amostras."""
    X_train, X_test, y_train, y_test = split_train_test(sample_dataframe, 'is_laundering')
    
    total = len(X_train) + len(X_test)
    assert total == len(sample_dataframe)
```

**Executar testes:**
```powershell
# Todos os testes
pytest

# Com coverage
pytest --cov=src --cov-report=html

# Testes específicos
pytest tests/test_preprocessing.py -v
```

---

### **FASE 6: CONTAINERIZAÇÃO E DEPLOY** (3-4 horas)

#### Passo 6.1: Criar `Dockerfile`

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código
COPY . .

# Expor porta
EXPOSE 8000

# Comando para iniciar API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Passo 6.2: Criar `docker-compose.yml`

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    container_name: aml-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - ENV=production
    restart: unless-stopped
  
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: aml-dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    restart: unless-stopped
```

**Executar:**
```powershell
# Build e start
docker-compose up --build

# Background
docker-compose up -d

# Parar
docker-compose down
```

---

### **FASE 7: DOCUMENTAÇÃO E FINALIZAÇÃO** (2-3 horas)

#### Passo 7.1: Atualizar `README.md`

```markdown
# AML Project - Sistema de Detecção de Lavagem de Dinheiro

Sistema completo de Machine Learning para detecção de lavagem de dinheiro, com API de produção, monitoramento e dashboard interativo.

## 🚀 Quick Start

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/gaab-braga/AML_project.git
cd AML_project

# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar Dados

Coloque seus dados brutos em `data/raw/transactions.csv`.

### 3. Treinar Modelo

```bash
python scripts/train_pipeline.py
```

### 4. Iniciar API

```bash
uvicorn api.main:app --reload
```

Acesse: `http://localhost:8000/docs`

### 5. Dashboard

```bash
streamlit run dashboard/app.py
```

Acesse: `http://localhost:8501`

## 📁 Estrutura do Projeto

```
AML_project/
├── entrypoints/      # 🎯 Pontos de entrada do sistema
│   ├── cli.py        # Interface de linha de comando (train, predict, evaluate)
│   ├── api.py        # API REST (FastAPI)
│   ├── batch.py      # Processamento em lote
│   └── stream.py     # Processamento em tempo real (opcional)
│
├── src/              # 🔧 Lógica de negócio (core)
│   ├── data/         # Carregamento e pré-processamento
│   ├── features/     # Feature engineering
│   ├── models/       # Treinamento, predição, avaliação
│   ├── explainability/ # SHAP e interpretabilidade
│   ├── monitoring/   # Monitoramento de produção
│   └── utils/        # Utilitários (logger, config, etc)
│
├── config/           # ⚙️ Configurações (YAML)
├── data/             # 📊 Dados (não versionados)
├── models/           # 🤖 Modelos treinados (não versionados)
├── artifacts/        # 📈 Métricas e relatórios
├── notebooks/        # 📓 Exploração e documentação
├── scripts/          # 🛠️ Scripts auxiliares (setup, export)
├── tests/            # ✅ Testes unitários
└── dashboard/        # 📱 Dashboard Streamlit (opcional)
```

## 🛠️ Desenvolvimento

### Executar Testes

```bash
pytest
pytest --cov=src --cov-report=html
```

### Linting e Formatação

```bash
# Formatar código
black src/ tests/

# Lint
flake8 src/
```

### Docker

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Logs
docker-compose logs -f
```

## 📊 Notebooks

1. `01_Data_Ingestion_EDA.ipynb` - Análise exploratória
2. `02_IBM_Benchmark.ipynb` - Benchmark com IBM MultiGNN
3. `03_Model_Selection_Tuning.ipynb` - Seleção e tuning
4. `04_Ensemble_Modeling.ipynb` - Ensemble e calibração
5. `05_Model_Interpretation.ipynb` - SHAP e explicabilidade
6. `06_Robustness_Validation.ipynb` - Validação de robustez
7. `07_Executive_Summary.ipynb` - Resumo executivo

## 🔧 Configuração

Edite `config/pipeline_config.yaml` para personalizar:
- Caminhos de dados
- Hiperparâmetros do modelo
- Thresholds de decisão
- Configurações de validação

## 📈 Performance

- **ROC-AUC:** 0.99
- **PR-AUC:** 0.39
- **Recall @ Top 1%:** 0.85
- **Latência API:** <50ms (p95)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie branch: `git checkout -b feature/nova-feature`
3. Commit: `git commit -am 'Add nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Pull Request

## 📝 Licença

MIT License

## 👤 Autor

Gabriel Braga - [@gaab-braga](https://github.com/gaab-braga)
```

---

## 📋 Checklist Final de Refatoração

### Estrutura de Diretórios
- [ ] `data/` organizado (raw/, processed/)
- [ ] `models/` limpo (apenas artefatos finais)
- [ ] `src/` modular e enxuto (6 submódulos principais)
- [ ] `entrypoints/` criado (cli.py, api.py, batch.py)
- [ ] `scripts/` minimalista (apenas auxiliares)
- [ ] `tests/` com cobertura básica
- [ ] `config/` consolidado

### Código
- [ ] Todo código de processamento em `src/data/`
- [ ] Feature engineering em `src/features/`
- [ ] Treino/predição/avaliação em `src/models/`
- [ ] Logging centralizado em `src/utils/logger.py`
- [ ] Configurações carregadas via `src/config.py`
- [ ] Zero duplicação entre notebooks e `src/`

### Notebooks
- [ ] Notebooks importam de `src/` (não reimplementam)
- [ ] Focados em análise e visualização
- [ ] Documentação clara com markdown

### Scripts
- [ ] `train_pipeline.py` funcional
- [ ] Scripts com argumentos de linha de comando
- [ ] Nomes descritivos e objetivos

### API
- [ ] Migrado para FastAPI
- [ ] Schemas Pydantic para validação
- [ ] Health checks funcionais
- [ ] Documentação automática (/docs)

### Qualidade
- [ ] Testes básicos implementados
- [ ] `.gitignore` completo
- [ ] `requirements.txt` atualizado
- [ ] README.md atualizado

### Deploy
- [ ] `Dockerfile` funcional
- [ ] `docker-compose.yml` configurado
- [ ] Variáveis de ambiente documentadas

---

## 🎯 Métricas de Sucesso

Após a refatoração, você deve conseguir:

1. **Executar pipeline completo com 1 comando:**
   ```powershell
   python -m entrypoints.cli train
   ```

2. **Fazer predições com 1 comando:**
   ```powershell
   python -m entrypoints.cli predict data.csv -o predictions.csv
   ```

3. **Iniciar API em produção com 1 comando:**
   ```powershell
   python -m entrypoints.cli serve --port 8000
   # ou
   docker-compose up -d
   ```

4. **Processar batch agendado:**
   ```powershell
   python -m entrypoints.batch --input daily_data.csv --date 2025-11-06
   ```

5. **Modificar hiperparâmetros sem tocar no código:**
   - Editar `config/pipeline_config.yaml`
   - Re-executar: `python -m entrypoints.cli train`

6. **Adicionar nova feature em 1 arquivo:**
   - Editar `src/features/engineering.py`
   - Função é automaticamente usada por TODOS os entrypoints

7. **Reproduzir resultados em qualquer máquina:**
   - Clone repo
   - `pip install -r requirements.txt`
   - `python -m entrypoints.cli train`

---

## 🚨 Anti-Patterns a Evitar

### ❌ NÃO FAÇA:
1. **Código duplicado** entre notebooks e `src/`
2. **Hardcoding** de caminhos, parâmetros, thresholds
3. **Notebooks gigantes** (>500 linhas de código)
4. **Funções com >50 linhas** sem refatoração
5. **Classes com múltiplas responsabilidades**
6. **Import relativos** confusos (`from ../../utils`)
7. **Logs com `print()`** ao invés de logging apropriado
8. **Secrets no código** (API keys, passwords)

### ✅ FAÇA:
1. **Função única por responsabilidade**
2. **Configurações em YAML**
3. **Notebooks importam de `src/`**
4. **Type hints** em funções
5. **Docstrings** em funções públicas
6. **Logging estruturado**
7. **Testes para lógica crítica**
8. **Versionamento semântico** de modelos

---

## 📅 Timeline Estimado

| Fase | Descrição | Tempo | Prioridade |
|------|-----------|-------|------------|
| 1 | Limpeza e organização | 2-3h | 🔴 ALTA |
| 2 | Refatoração de `src/` | 8-12h | 🔴 ALTA |
| 3 | Entrypoints - CLI | 3-4h | 🔴 ALTA |
| 4 | Entrypoints - API & Batch | 4-6h | 🟡 MÉDIA |
| 5 | Testes | 4-6h | 🟡 MÉDIA |
| 6 | Docker/Deploy | 3-4h | 🟢 BAIXA |
| 7 | Documentação | 2-3h | 🟡 MÉDIA |
| **TOTAL** | | **26-38h** | |

**Recomendação:** Execute as fases em ordem. Priorize Fases 1-3 antes de avançar.

---

## 🎓 Recursos Adicionais

### Leitura Recomendada
- [Clean Code (Robert C. Martin)](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Python Best Practices](https://realpython.com/tutorials/best-practices/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLOps Principles (Google)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Ferramentas Úteis
- **Formatação:** `black`, `isort`
- **Linting:** `flake8`, `pylint`
- **Type Checking:** `mypy`
- **Testing:** `pytest`, `pytest-cov`
- **API Testing:** `httpx`, `pytest-asyncio`

---

## 💡 Dicas Finais

1. **Comece pequeno:** Refatore um módulo por vez, teste, commit.
2. **Teste frequentemente:** Não acumule mudanças. Teste a cada refatoração.
3. **Git é seu amigo:** Commits pequenos e frequentes. Use branches.
4. **Documentação é código:** README e docstrings são tão importantes quanto o código.
5. **Performance vem depois:** Primeiro faça funcionar, depois otimize.
6. **Peça feedback:** Code review é essencial para qualidade.

---

## 🏁 Conclusão

Este roadmap transforma seu projeto de notebooks acadêmicos em um sistema profissional de ML. O resultado final será:

- ✅ **Limpo:** Estrutura intuitiva, código legível
- ✅ **Modular:** Componentes reutilizáveis e desacoplados
- ✅ **Testável:** Cobertura de testes, CI/CD ready
- ✅ **Deployável:** API em produção, containerizado
- ✅ **Manutenível:** Fácil de estender e debugar

**Lembre-se:** "Perfeição é atingida não quando não há mais nada a adicionar, mas quando não há mais nada a remover." - Antoine de Saint-Exupéry

Boa refatoração! 🚀

---

**Versão:** 1.0.0  
**Data:** Novembro 2025  
**Autor:** GitHub Copilot + Gabriel Braga  
**Status:** READY FOR IMPLEMENTATION
