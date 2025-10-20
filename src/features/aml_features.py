"""
Feature engineering utilities for AML project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame, windows: List[int] = [7, 30]) -> pd.DataFrame:
    """
    Create temporal aggregation features for each entity using pandas with memory-efficient approach.

    Args:
        df: Transaction DataFrame with 'source', 'timestamp', 'amount'
        windows: Rolling window sizes in days

    Returns:
        DataFrame with temporal features
    """
    logger.info("Creating temporal features with pandas...")
    start_total = time.time()
    print("[temporal] Starting temporal feature creation with pandas...")

    # Work with a copy to avoid modifying original
    df_temp = df.copy()

    # Ensure timestamp is datetime
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])

    # Create temporal features for each window
    for window in windows:
        start_win = time.time()
        print(f"[temporal] Processing window={window} days...")

        # Group by source and create rolling features
        grouped = df_temp.groupby('source')

        # Initialize new columns
        df_temp[f'source_amount_sum_{window}d'] = 0.0
        df_temp[f'source_amount_mean_{window}d'] = 0.0
        df_temp[f'source_amount_std_{window}d'] = 0.0
        df_temp[f'source_transaction_count_{window}d'] = 0

        # Process each group
        for name, group in grouped:
            if len(group) > 0:
                # Sort group by timestamp
                group_sorted = group.sort_values('timestamp').copy()

                # Create rolling window aggregations
                rolling = group_sorted.set_index('timestamp')['amount'].rolling(
                    window=f'{window}D',
                    min_periods=1
                )

                # Calculate aggregations
                sum_vals = rolling.sum()
                mean_vals = rolling.mean()
                std_vals = rolling.std()
                count_vals = rolling.count()

                # Update the main dataframe using loc with the original indices
                df_temp.loc[group_sorted.index, f'source_amount_sum_{window}d'] = sum_vals.values
                df_temp.loc[group_sorted.index, f'source_amount_mean_{window}d'] = mean_vals.values
                df_temp.loc[group_sorted.index, f'source_amount_std_{window}d'] = std_vals.values
                df_temp.loc[group_sorted.index, f'source_transaction_count_{window}d'] = count_vals.values

        elapsed_win = time.time() - start_win
        print(f"[temporal] Finished window={window}d in {elapsed_win:.2f}s")

    # Fill NaN values with 0
    temporal_cols = [col for col in df_temp.columns if any(s in col for s in ['source_amount_', 'source_transaction_count_'])]
    df_temp[temporal_cols] = df_temp[temporal_cols].fillna(0)

    # Add time-based features
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    df_temp['day_of_week'] = df_temp['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
    df_temp['is_business_hours'] = df_temp['timestamp'].dt.hour.between(9, 17).astype(int)
    df_temp['is_weekend'] = df_temp['timestamp'].dt.weekday.isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday

    total_elapsed = time.time() - start_total
    all_temporal_cols = temporal_cols + ['hour', 'day_of_week', 'is_business_hours', 'is_weekend']
    logger.info(f"Created {len(all_temporal_cols)} temporal features using pandas")
    print(f"[temporal] Done. Total time: {total_elapsed:.2f}s")

    return df_temp

def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create network-based features using degree centrality.

    Args:
        df: Transaction DataFrame with 'source', 'target'

    Returns:
        DataFrame with network features
    """
    logger.info("Creating network features...")

    try:
        import networkx as nx

        # Create directed graph
        G = nx.from_pandas_edgelist(
            df, 'source', 'target',
            create_using=nx.DiGraph()
        )

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)

        # Create features DataFrame
        network_features = []
        for node in G.nodes():
            network_features.append({
                'node': node,
                'degree_centrality': degree_centrality.get(node, 0),
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'degree': G.degree(node),
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node)
            })

        network_df = pd.DataFrame(network_features)

    except ImportError:
        logger.warning("NetworkX not available, creating basic degree features...")

        # Basic degree calculation without NetworkX
        source_degrees = df.groupby('source').size().rename('source_degree')
        target_degrees = df.groupby('target').size().rename('target_degree')

        # Combine all unique nodes
        all_nodes = set(df['source'].unique()) | set(df['target'].unique())

        network_features = []
        for node in all_nodes:
            network_features.append({
                'node': node,
                'degree': source_degrees.get(node, 0) + target_degrees.get(node, 0),
                'in_degree': target_degrees.get(node, 0),  # Incoming connections
                'out_degree': source_degrees.get(node, 0),  # Outgoing connections
                'degree_centrality': (source_degrees.get(node, 0) + target_degrees.get(node, 0)) / len(all_nodes),
                'in_degree_centrality': target_degrees.get(node, 0) / len(all_nodes),
                'out_degree_centrality': source_degrees.get(node, 0) / len(all_nodes)
            })

        network_df = pd.DataFrame(network_features)

    logger.info(f"Created network features for {len(network_df)} nodes")

    return network_df

def encode_categorical_features(df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features without data leakage.

    Args:
        df: DataFrame with categorical columns
        target_col: Target column name

    Returns:
        Tuple of (encoded DataFrame, encoding mappings)
    """
    logger.info("Encoding categorical features...")

    df_encoded = df.copy()
    encoders = {}

    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]

    for col in categorical_cols:
        # Frequency encoding (safe for temporal data)
        freq_encoding = df_encoded[col].value_counts(normalize=True)
        df_encoded[col] = df_encoded[col].map(freq_encoding)

        encoders[col] = freq_encoding.to_dict()

        # Fill NaN with global mean
        df_encoded[col] = df_encoded[col].fillna(freq_encoding.mean())

    logger.info(f"Encoded {len(encoders)} categorical columns")

    return df_encoded, encoders

def create_aml_feature_pipeline(config: Optional[Dict] = None) -> 'AMLFeaturePipeline':
    """
    Create AML feature engineering pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Configured pipeline instance
    """
    if config is None:
        config = {
            'scaler_type': 'robust',
            'temporal_windows': [7, 30],
            'network_features': True,
            'categorical_encoding': 'frequency'
        }

    pipeline = AMLFeaturePipeline(config)
    logger.info("Created AML feature pipeline")

    return pipeline

class AMLFeaturePipeline:
    """Modular feature engineering pipeline for AML."""

    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform features."""
        logger.info("Fitting and transforming features...")

        X_transformed = X.copy()

        # Handle categorical encoding
        if self.config.get('categorical_encoding'):
            X_transformed, self.encoders = encode_categorical_features(X_transformed)

        # Robust scaling
        if self.config.get('scaler_type') == 'robust':
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'is_fraud']

            for col in numeric_cols:
                median = X_transformed[col].median()
                q75, q25 = np.percentile(X_transformed[col].dropna(), [75, 25])
                iqr = q75 - q25

                if iqr == 0:
                    iqr = 1

                X_transformed[col] = (X_transformed[col] - median) / iqr
                self.scalers[col] = {'median': median, 'iqr': iqr}

        self.feature_names = X_transformed.columns.tolist()

        logger.info(f"Pipeline fitted with {len(self.feature_names)} features")

        return X_transformed.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted pipeline."""
        X_transformed = X.copy()

        # Apply categorical encoding
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                freq_map = pd.Series(encoder)
                X_transformed[col] = X_transformed[col].map(freq_map)
                X_transformed[col] = X_transformed[col].fillna(freq_map.mean())

        # Apply scaling
        for col, params in self.scalers.items():
            if col in X_transformed.columns:
                X_transformed[col] = (X_transformed[col] - params['median']) / params['iqr']

        return X_transformed.values

    def save(self, path: str):
        """Save pipeline to disk."""
        import joblib
        joblib.dump({
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'AMLFeaturePipeline':
        """Load pipeline from disk."""
        import joblib
        data = joblib.load(path)

        pipeline = cls(data['config'])
        pipeline.scalers = data['scalers']
        pipeline.encoders = data['encoders']
        pipeline.feature_names = data['feature_names']

        logger.info(f"Pipeline loaded from {path}")
        return pipeline

# ==========================================
# DASHBOARD & STREAMING UTILITIES
# ==========================================

class SSELiveCallback:
    """Callback para enviar métricas de treinamento em tempo real via SSE."""
    
    def __init__(self, model_name, server_url='http://localhost:5000', update_interval=5):
        self.model_name = model_name
        self.server_url = server_url
        self.update_interval = update_interval
        self.iterations = []
        self.train_auc = []
        self.val_auc = []
        self.oob_scores = []
        self.start_time = time.time()
        
    def send_to_dashboard(self, event_type, data):
        """Envia dados para o dashboard via HTTP POST."""
        try:
            response = requests.post(
                f"{self.server_url}/send_training_data",
                json={'type': event_type, 'data': data},
                timeout=1.0
            )
            if response.status_code == 200:
                logger.info(f"📡 Enviado {event_type} para dashboard")
            else:
                logger.warning(f"⚠️ Falha ao enviar para dashboard: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao conectar dashboard: {e}")
    
    def add_metrics(self, iteration, train_auc=None, val_auc=None, oob_score=None):
        """Adiciona métricas e envia para dashboard se necessário."""
        self.iterations.append(iteration)
        if train_auc is not None:
            self.train_auc.append(train_auc)
        if val_auc is not None:
            self.val_auc.append(val_auc)
        if oob_score is not None:
            self.oob_scores.append(oob_score)
        
        if iteration % self.update_interval == 0 or iteration == 1:
            self.send_update()
    
    def send_start(self, message="Training started"):
        """Envia evento de início."""
        self.send_to_dashboard('start', {'message': message})
    
    def send_update(self):
        """Envia atualização de métricas."""
        data = {
            'iteration': self.iterations[-1] if self.iterations else 0,
            'roc_auc': self.val_auc[-1] if self.val_auc else None,
            'accuracy': self.train_auc[-1] if self.train_auc else None,  # Usando train_auc como proxy
            'loss': 1 - (self.val_auc[-1] if self.val_auc else 0) if self.val_auc else None
        }
        self.send_to_dashboard('update', data)
    
    def send_complete(self, final_metrics=None):
        """Envia evento de conclusão."""
        data = {'message': 'Training completed'}
        if final_metrics:
            data.update(final_metrics)
        self.send_to_dashboard('complete', data)

def ensure_server_running(server_url='http://localhost:5000', script_path='flask_sse.py', python_path=None):
    """
    Garante que o servidor SSE está rodando.

    Args:
        server_url: URL do servidor
        script_path: Caminho para o script do servidor
        python_path: Caminho para o executável Python (opcional)

    Returns:
        True se servidor está rodando ou foi iniciado
    """
    import requests
    import time
    import subprocess
    import sys
    from pathlib import Path

    try:
        # Verificar se servidor já está rodando
        response = requests.get(f"{server_url}/status", timeout=2)
        if response.status_code == 200:
            print("✅ Servidor SSE já está rodando")
            return True
    except:
        pass

    # Iniciar servidor em background
    print("🚀 Iniciando servidor SSE...")
    try:
        # Usar subprocess para iniciar servidor em background
        if python_path is None:
            python_path = sys.executable

        script_full_path = Path(script_path).resolve()
        if not script_full_path.exists():
            # Tentar encontrar no diretório pai
            script_full_path = Path('..') / script_path
            if not script_full_path.exists():
                raise FileNotFoundError(f"Script {script_path} não encontrado")

        process = subprocess.Popen([
            python_path, str(script_full_path)
        ], cwd=script_full_path.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Aguardar servidor iniciar
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get(f"{server_url}/status", timeout=2)
                if response.status_code == 200:
                    print("✅ Servidor SSE iniciado com sucesso")
                    return True
            except:
                continue

        print("❌ Falha ao iniciar servidor SSE")
        return False
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        return False

def wait_for_client_connection(server_url='http://localhost:5000', timeout=10):
    """
    Aguarda um cliente SSE conectar.

    Args:
        server_url: URL do servidor
        timeout: Tempo máximo de espera em segundos

    Returns:
        True se cliente conectou, False caso contrário
    """
    import requests
    import time

    print("⏳ Aguardando conexão do navegador...")
    for i in range(timeout):
        try:
            response = requests.get(f"{server_url}/status", timeout=2)
            data = response.json()
            if data.get('clients_connected', 0) > 0:
                print("✅ Cliente conectado! Pronto para streaming.")
                return True
        except:
            pass
        time.sleep(1)
    print("⚠️ Nenhum cliente conectado, mas continuando...")
    return False

def start_live_dashboard(script_path='flask_sse.py'):
    """
    Função helper simplificada para iniciar dashboard live.

    Use esta função no notebook para setup rápido do dashboard.

    Args:
        script_path: Caminho para flask_sse.py (relativo ao projeto)

    Returns:
        True se dashboard foi configurado com sucesso

    Example:
        >>> from src.features import start_live_dashboard
        >>> start_live_dashboard()  # Setup completo em uma linha
    """
    return setup_live_dashboard(script_path=script_path)

def setup_live_dashboard(server_url='http://localhost:5000', script_path='flask_sse.py', python_path=None):
    """
    Configura o dashboard live automaticamente para monitoramento de treinamento AML.

    Esta função:
    1. Inicia o servidor Flask SSE se não estiver rodando
    2. Abre o dashboard no navegador padrão
    3. Aguarda conexão do cliente para confirmar que está pronto

    Args:
        server_url: URL do servidor SSE (padrão: http://localhost:5000)
        script_path: Caminho relativo para o script flask_sse.py
        python_path: Caminho para o executável Python (opcional)

    Returns:
        True se setup foi bem-sucedido e cliente conectou

    Example:
        >>> from src.features import setup_live_dashboard
        >>> setup_live_dashboard()  # Inicia dashboard automaticamente
        ✅ Servidor SSE já está rodando
        🌐 Dashboard aberto no navegador - aguarde alguns segundos...
        ⏳ Aguardando conexão do navegador...
        ✅ Cliente conectado! Pronto para streaming.
        True
    """
    import webbrowser

    # Garantir que o servidor está rodando
    if not ensure_server_running(server_url, script_path, python_path):
        print("❌ Não foi possível iniciar o servidor SSE. Abortando.")
        return False

    # Abrir dashboard no navegador
    try:
        webbrowser.open(server_url)
        print("🌐 Dashboard aberto no navegador - aguarde alguns segundos...")
    except Exception as e:
        print(f"⚠️ Não foi possível abrir navegador automaticamente: {e}")
        print(f"🌐 Por favor, abra manualmente: {server_url}")

    # Aguardar conexão do navegador
    return wait_for_client_connection(server_url)

def parse_laundering_patterns(file_path: str) -> List[Dict]:
    """
    Parse do arquivo de padrões de lavagem e extração de informações estruturadas.

    Args:
        file_path: Caminho para o arquivo de padrões de lavagem

    Returns:
        Lista de dicionários contendo informações dos padrões identificados
    """
    import re
    from collections import Counter

    patterns = []
    current_pattern = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                # Extrair tipo do padrão e características
                match = re.search(r'BEGIN LAUNDERING ATTEMPT - (\w+):?\s*(.*)', line)
                if match:
                    pattern_type = match.group(1)
                    characteristics = match.group(2) if match.group(2) else ""
                    current_pattern = {
                        'type': pattern_type,
                        'characteristics': characteristics,
                        'transactions': []
                    }

            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current_pattern:
                    patterns.append(current_pattern)
                    current_pattern = None

            elif current_pattern and line and not line.startswith('BEGIN') and not line.startswith('END'):
                # Parse da linha de transação
                parts = line.split(',')
                if len(parts) >= 11:
                    try:
                        transaction = {
                            'timestamp': parts[0],
                            'from_bank': parts[1],
                            'from_account': parts[2],
                            'to_bank': parts[3],
                            'to_account': parts[4],
                            'amount_orig': float(parts[5]),
                            'currency_orig': parts[6],
                            'amount_dest': float(parts[7]),
                            'currency_dest': parts[8],
                            'payment_format': parts[9],
                            'is_laundering': int(parts[10])
                        }
                        current_pattern['transactions'].append(transaction)
                    except (ValueError, IndexError):
                        continue

    logger.info(f"Parsed {len(patterns)} laundering patterns from {file_path}")
    return patterns

# Implementações locais temporárias (devido a problema sklearn/numpy)

def load_raw_transactions(data_path='../data'):
    """Carrega dados transacionais brutos.
    
    Args:
        data_path: Caminho para arquivo CSV ou diretório contendo arquivos CSV
    """
    import os
    from pathlib import Path

    print(f" Tentando carregar de: {data_path}")
    data_path_obj = Path(data_path).resolve()  # Resolver caminho absoluto
    print(f" Caminho resolvido: {data_path_obj}")
    print(f" É arquivo? {data_path_obj.is_file()}")
    print(f" Existe? {data_path_obj.exists()}")

    # Se data_path for um arquivo específico, carregue diretamente
    if data_path_obj.is_file() and data_path_obj.suffix.lower() == '.csv':
        print(f" Carregado arquivo específico: {data_path_obj.name}")
        df = pd.read_csv(data_path_obj)
        
        # Verificar se é arquivo processado específico
        if data_path_obj.name == 'transactions_enriched_final.csv':
            print(f" Aplicando mapeamento para arquivo processado: {data_path_obj.name}")
            print(f" Colunas antes do mapeamento: {list(df.columns)[:5]}...")
            
            # Mapear colunas para o formato padrão
            column_mapping = {
                'Timestamp': 'timestamp',
                'From Account': 'source',
                'To Account': 'target',
                'Amount Paid': 'amount',
                'is_fraud': 'is_fraud'
            }
            df = df.rename(columns=column_mapping)
            
            print(f" Colunas após mapeamento: {list(df.columns)[:5]}...")
            
            # Garantir tipos corretos
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'is_fraud' in df.columns:
                df['is_fraud'] = df['is_fraud'].astype(int)
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            print(f" Arquivo processado mapeado: {list(df.columns)[:5]}...")
        
        # Tentar mapear colunas se for arquivo bruto
        elif 'Timestamp' in df.columns:
            column_mapping = {
                'Timestamp': 'timestamp',
                'From Bank': 'from_bank',
                'Account': 'source',  # primeira ocorrência
                'To Bank': 'to_bank',
                'Account.1': 'target',  # segunda ocorrência
                'Amount Paid': 'amount',
                'Payment Format': 'payment_format',
                'Is Laundering': 'is_fraud'
            }
            df = df.rename(columns=column_mapping)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['is_fraud'] = df['is_fraud'].astype(int)
            print(f" Colunas mapeadas para formato padrão: {list(df.columns)}")
        
        return df

    # Caso contrário, tratar como diretório
    if not data_path_obj.is_dir():
        raise FileNotFoundError(f"Caminho não encontrado ou não é arquivo/diretório válido: {data_path}")
        
    data_dir = data_path_obj
    print(f" Procurando arquivos CSV em: {data_dir}")

    # Listar todos os arquivos CSV disponíveis
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_path}")
    
    print(f" Arquivos CSV encontrados: {[f.name for f in csv_files]}")
    
    # Verificar se estamos no diretório processed e forçar carregamento do arquivo final
    processed_file = data_dir / 'transactions_enriched_final.csv'
    if processed_file.exists():
        print(f"  Carregando arquivo processado específico: transactions_enriched_final.csv")
        df = pd.read_csv(processed_file)
        
        print(f"  Aplicando mapeamento para transactions_enriched_final.csv")
        print(f"  Colunas antes do mapeamento: {list(df.columns)[:5]}...")
        
        # Mapear colunas para o formato padrão
        column_mapping = {
            'From Account': 'source',
            'To Account': 'target',
            'Amount Paid': 'amount',
            'timestamp': 'timestamp',
            'is_fraud': 'is_fraud'
        }
        df = df.rename(columns=column_mapping)
        
        print(f"  Colunas após mapeamento: {list(df.columns)[:5]}...")
        
        # Garantir tipos corretos
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype(int)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        print(f"  Arquivo processado mapeado: {list(df.columns)[:5]}...")
        return df
    
    # Fallback: carregar primeiro arquivo CSV encontrado
    df = pd.read_csv(csv_files[0])
    print(f" Carregado (fallback): {csv_files[0].name}")
    return df

def validate_data_compliance(df):
    """Validação básica de compliance."""
    required_cols = ['source', 'target', 'amount', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f" Colunas obrigatórias faltando: {missing_cols}")
        return False

    # Verificar se há dados sensíveis
    sensitive_patterns = ['cpf', 'cnpj', 'ssn', 'passport']
    sensitive_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in sensitive_patterns)]

    if sensitive_cols:
        print(f" Possíveis dados sensíveis detectados: {sensitive_cols}")

    return True

def clean_transactions(df):
    """Limpeza básica de transações."""
    print(" Aplicando limpeza de dados...")

    # Remover duplicatas
    initial_len = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_len - len(df)

    # Remover valores inválidos
    df = df.dropna(subset=['amount', 'timestamp'])

    # Garantir tipos corretos
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Remover valores negativos ou zero
    df = df[df['amount'] > 0]

    print(f" Limpeza concluída: {duplicates_removed} duplicatas removidas")
    return df

def impute_and_encode(df, target_col='is_fraud'):
    """Imputação e encoding simplificado."""
    print("Aplicando encoding categórico simplificado...")

    df_processed = df.copy()

    # Imputação simples para numéricos
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_col:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    # Encoding simples para categóricos (label encoding)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    encoders = {}

    for col in categorical_cols:
        if col != target_col:
            # Label encoding simples
            unique_vals = df_processed[col].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            df_processed[col] = df_processed[col].map(mapping)
            encoders[col] = mapping

    print(f"Encoding aplicado para {len(encoders)} colunas categóricas")
    return df_processed, encoders

def aggregate_by_entity(df, entity_col, windows=[7, 30]):
    """Agregação temporal por entidade."""
    print(f"Criando features temporais para {entity_col}...")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    features_list = []

    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].copy()
        entity_data = entity_data.sort_values('date')

        for window in windows:
            # Rolling aggregations
            entity_data[f'{entity_col}_amount_sum_{window}d'] = entity_data['amount'].rolling(
                window=window, min_periods=1
            ).sum()

            entity_data[f'{entity_col}_amount_mean_{window}d'] = entity_data['amount'].rolling(
                window=window, min_periods=1
            ).mean()

            entity_data[f'{entity_col}_amount_std_{window}d'] = entity_data['amount'].rolling(
                window=window, min_periods=1
            ).std()

            entity_data[f'{entity_col}_transaction_count_{window}d'] = entity_data['amount'].rolling(
                window=window, min_periods=1
            ).count()

        features_list.append(entity_data)

    result = pd.concat(features_list, ignore_index=True)

    # Fill NaN values
    result = result.fillna(0)

    print(f"Features temporais criadas: {len([col for col in result.columns if 'temporal' in col.lower() or 'd' in col.lower()])}")
    return result

def compute_network_features(edges_df):
    """Features de rede simplificadas."""
    print("Calculando features de rede simplificadas...")

    try:
        import networkx as nx
    except ImportError:
        print("NetworkX não disponível, criando features básicas...")
        # Features básicas sem networkx
        nodes = set(edges_df['source'].unique()) | set(edges_df['target'].unique())

        basic_features = []
        for node in nodes:
            degree = len(edges_df[(edges_df['source'] == node) | (edges_df['target'] == node)])
            basic_features.append({
                'node': node,
                'degree': degree,
                'centrality': degree / len(nodes) if len(nodes) > 0 else 0
            })

        return pd.DataFrame(basic_features)

    # Features completas com networkx
    G = nx.from_pandas_edgelist(edges_df, 'source', 'target', ['amount'])

    features = []
    for node in G.nodes():
        features.append({
            'node': node,
            'degree': G.degree(node),
            'centrality': nx.degree_centrality(G)[node]
        })

    return pd.DataFrame(features)

print("Funções locais implementadas com sucesso!")