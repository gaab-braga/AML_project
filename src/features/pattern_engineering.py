"""
Pattern-based Feature Engineering Module

This module provides functions for creating features based on known money laundering patterns.
It integrates pattern analysis with the main feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PatternFeatureEngineer:
    """
    Classe para engenharia de features baseada em padrões de lavagem de dinheiro conhecidos.
    """

    def __init__(self, patterns_file: str = None):
        """
        Inicializa o engenheiro de features baseado em patterns.

        Parameters
        ----------
        patterns_file : str, optional
            Caminho para o arquivo de patterns. Se None, usa o padrão.
        """
        if patterns_file is None:
            # Caminho padrão relativo ao projeto
            self.patterns_file = Path(__file__).parent.parent.parent / "data" / "raw" / "HI-Small_Patterns.txt"
        else:
            self.patterns_file = Path(patterns_file)

        self.patterns_df = None
        self.pattern_networks = {}
        self.pattern_features = {}

        # Carregar patterns na inicialização
        self._load_patterns()

    def _load_patterns(self) -> None:
        """
        Carrega e parseia os patterns do arquivo.
        """
        if not self.patterns_file.exists():
            print(f"Warning: Patterns file not found at {self.patterns_file}")
            return

        try:
            with open(self.patterns_file, 'r') as f:
                content = f.read()

            # Separar patterns por tipo
            pattern_blocks = content.strip().split('BEGIN LAUNDERING ATTEMPT')

            self.patterns_df = self._parse_patterns(pattern_blocks[1:])  # Skip empty first element

            print(f"Loaded {len(self.patterns_df)} pattern transactions from {len(pattern_blocks)-1} laundering attempts")

        except Exception as e:
            print(f"Error loading patterns: {e}")
            self.patterns_df = pd.DataFrame()

    def _parse_patterns(self, pattern_blocks: List[str]) -> pd.DataFrame:
        """
        Parseia os blocos de patterns em um DataFrame estruturado.
        """
        all_transactions = []
        pattern_types = []

        for block in pattern_blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue

            # Extrair tipo do pattern
            header = lines[0]
            if 'FAN-OUT' in header:
                pattern_type = 'FAN-OUT'
            elif 'CYCLE' in header:
                pattern_type = 'CYCLE'
            elif 'GATHER-SCATTER' in header:
                pattern_type = 'GATHER-SCATTER'
            elif 'STACK' in header:
                pattern_type = 'STACK'
            else:
                pattern_type = 'UNKNOWN'

            # Parsear transações
            for line in lines[1:]:
                if line.startswith('END LAUNDERING ATTEMPT'):
                    break
                if not line.strip():
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) >= 11:
                        transaction = {
                            'timestamp': parts[0],
                            'from_bank': parts[1],
                            'from_account': parts[2],
                            'to_bank': parts[3],
                            'to_account': parts[4],
                            'amount': float(parts[5]),
                            'amount_currency': parts[6],
                            'received_amount': float(parts[7]),
                            'received_currency': parts[8],
                            'payment_format': parts[9],
                            'is_fraud': int(parts[10]) if len(parts) > 10 else 1,
                            'pattern_type': pattern_type
                        }
                        all_transactions.append(transaction)
                        pattern_types.append(pattern_type)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line} - {e}")
                    continue

        df = pd.DataFrame(all_transactions)

        # Converter timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        return df

    def create_pattern_similarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas na similaridade com patterns conhecidos.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com transações para análise

        Returns
        -------
        pd.DataFrame
            DataFrame com features de similaridade adicionadas
        """
        if self.patterns_df is None or len(self.patterns_df) == 0:
            print("Warning: No patterns loaded, returning original dataframe")
            return df.copy()

        df_features = df.copy()

        # Standardize column names for consistency
        df_features = df_features.rename(columns={
            'from_account': 'source',
            'to_account': 'target'
        })

        # 1. Feature: Similaridade de valor com patterns
        df_features = self._add_amount_similarity_features(df_features)

        # 2. Feature: Similaridade de rede (conexões entre contas)
        df_features = self._add_network_similarity_features(df_features)

        # 3. Feature: Padrões de sequência temporal
        df_features = self._add_temporal_pattern_features(df_features)

        # 4. Feature: Concentração por banco
        df_features = self._add_bank_concentration_features(df_features)

        # 5. Feature: Indicadores de estrutura de lavagem
        df_features = self._add_laundering_structure_features(df_features)

        return df_features

    def _add_amount_similarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas na similaridade de valores com patterns.
        """
        # Estatísticas dos patterns por tipo
        pattern_stats = self.patterns_df.groupby('pattern_type').agg({
            'amount': ['mean', 'std', 'min', 'max', 'median']
        }).round(2)

        pattern_stats.columns = ['_'.join(col).strip() for col in pattern_stats.columns]

        # Para cada transação, calcular similaridade com cada tipo de pattern
        for pattern_type in self.patterns_df['pattern_type'].unique():
            mean_amount = pattern_stats.loc[pattern_type, 'amount_mean']
            std_amount = pattern_stats.loc[pattern_type, 'amount_std']

            if std_amount > 0:
                # Z-score da transação em relação ao pattern
                z_score_col = f'amount_zscore_{pattern_type.lower()}'
                df[z_score_col] = (df['amount'] - mean_amount) / std_amount

                # Similaridade baseada em proximidade ao centro do pattern
                similarity_col = f'amount_similarity_{pattern_type.lower()}'
                df[similarity_col] = np.exp(-0.5 * df[z_score_col]**2)  # Gaussian similarity
            else:
                df[f'amount_similarity_{pattern_type.lower()}'] = (df['amount'] == mean_amount).astype(int)

        # Feature agregada: similaridade máxima com qualquer pattern
        similarity_cols = [col for col in df.columns if col.startswith('amount_similarity_')]
        if similarity_cols:
            df['amount_pattern_similarity_max'] = df[similarity_cols].max(axis=1)
            df['amount_pattern_similarity_mean'] = df[similarity_cols].mean(axis=1)

        return df

    def _add_network_similarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas na similaridade de rede com patterns.
        """
        # Criar grafos para cada tipo de pattern
        for pattern_type in self.patterns_df['pattern_type'].unique():
            pattern_data = self.patterns_df[self.patterns_df['pattern_type'] == pattern_type]

            # Criar grafo direcionado
            G = nx.DiGraph()
            for _, row in pattern_data.iterrows():
                G.add_edge(row['from_account'], row['to_account'],
                          amount=row['amount'], bank_from=row['from_bank'], bank_to=row['to_bank'])

            self.pattern_networks[pattern_type] = G

            # Métricas de rede do pattern
            if len(G.nodes()) > 0:
                # Centralidade de grau
                degree_centrality = nx.degree_centrality(G)

                # Betweenness centrality
                betweenness_centrality = nx.betweenness_centrality(G)

                # Armazenar métricas
                self.pattern_features[pattern_type] = {
                    'avg_degree': np.mean(list(degree_centrality.values())),
                    'max_degree': max(degree_centrality.values()),
                    'avg_betweenness': np.mean(list(betweenness_centrality.values())),
                    'network_density': nx.density(G),
                    'num_nodes': len(G.nodes()),
                    'num_edges': len(G.edges())
                }

        # Para cada transação, calcular features de rede local
        # (simplificado - em produção seria mais sofisticado)
        df['account_degree_centrality'] = df.groupby('source')['target'].transform('count')
        df['account_in_degree'] = df.groupby('target')['source'].transform('count')
        df['account_out_degree'] = df.groupby('source')['target'].transform('count')

        # Razão entre graus (indicativo de fan-out patterns)
        df['degree_ratio'] = df['account_out_degree'] / (df['account_in_degree'] + 1)

        return df

    def _add_temporal_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas em padrões temporais.
        """
        if 'timestamp' not in df.columns:
            return df

        # Converter timestamp se necessário
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Features temporais
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Padrões de horário suspeitos (comparando com patterns)
        pattern_hours = self.patterns_df['timestamp'].dt.hour.value_counts(normalize=True)

        # Similaridade de distribuição de horários
        transaction_hours = df['hour'].value_counts(normalize=True)
        common_hours = set(pattern_hours.index) & set(transaction_hours.index)

        if common_hours:
            hour_similarity = 0
            for hour in common_hours:
                hour_similarity += min(pattern_hours.get(hour, 0), transaction_hours.get(hour, 0))
            df['hour_pattern_similarity'] = hour_similarity
        else:
            df['hour_pattern_similarity'] = 0

        return df

    def _add_bank_concentration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas na concentração de transações por banco.
        """
        # Concentração por banco de origem
        bank_from_counts = df['from_bank'].value_counts()
        total_transactions = len(df)

        df['from_bank_frequency'] = df['from_bank'].map(bank_from_counts) / total_transactions
        df['from_bank_is_rare'] = (df['from_bank_frequency'] < 0.01).astype(int)

        # Concentração por banco de destino
        bank_to_counts = df['to_bank'].value_counts()
        df['to_bank_frequency'] = df['to_bank'].map(bank_to_counts) / total_transactions
        df['to_bank_is_rare'] = (df['to_bank_frequency'] < 0.01).astype(int)

        # Mesmo banco (transferências internas)
        df['same_bank_transfer'] = (df['from_bank'] == df['to_bank']).astype(int)

        # Comparar com padrões de concentração dos patterns
        pattern_bank_concentration = self.patterns_df.groupby('pattern_type').agg({
            'from_bank': 'nunique',
            'to_bank': 'nunique'
        })

        return df

    def _add_laundering_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features indicativas de estruturas de lavagem.
        """
        # Feature: Transações em sequência (mesma conta origem/destino)
        df = df.sort_values(['source', 'timestamp'])

        # Rolling window features
        df['rolling_amount_mean_3'] = df.groupby('source')['amount'].rolling(3).mean().reset_index(0, drop=True)
        df['rolling_amount_std_3'] = df.groupby('source')['amount'].rolling(3).std().reset_index(0, drop=True)

        # Coeficiente de variação
        df['amount_cv_3'] = df['rolling_amount_std_3'] / (df['rolling_amount_mean_3'] + 1e-6)

        # Feature: Padrão de fan-out (uma conta enviando para múltiplas contas)
        fan_out_counts = df.groupby('source')['target'].nunique()
        df['fan_out_degree'] = df['source'].map(fan_out_counts)

        # Feature: Padrão de fan-in (uma conta recebendo de múltiplas contas)
        fan_in_counts = df.groupby('target')['source'].nunique()
        df['fan_in_degree'] = df['target'].map(fan_in_counts)

        # Feature: Razão amount/balance (se disponível)
        # Esta seria uma feature importante mas requer dados de saldo

        return df

    def calculate_pattern_iv(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Calcula Information Value para features baseadas em patterns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com features incluindo as baseadas em patterns
        target_col : str
            Nome da coluna target

        Returns
        -------
        pd.DataFrame
            DataFrame com IV das features de pattern
        """
        from src.features.iv_calculator import calculate_iv

        # Filtrar apenas colunas de features de pattern
        pattern_feature_cols = [col for col in df.columns if any(pattern in col.lower()
            for pattern in ['pattern', 'similarity', 'degree', 'concentration', 'fan_', 'temporal'])]

        if not pattern_feature_cols:
            print("Warning: No pattern-based features found")
            return pd.DataFrame()

        # Criar subset com target e features de pattern
        analysis_cols = [target_col] + pattern_feature_cols
        df_analysis = df[analysis_cols].copy()

        # Calcular IV
        iv_results = calculate_iv(df_analysis, target_col)

        return iv_results

    def get_pattern_summary(self) -> Dict:
        """
        Retorna um resumo estatístico dos patterns carregados.
        """
        if self.patterns_df is None or len(self.patterns_df) == 0:
            return {"error": "No patterns loaded"}

        summary = {
            "total_patterns": len(self.patterns_df),
            "pattern_types": self.patterns_df['pattern_type'].value_counts().to_dict(),
            "total_amount_range": {
                "min": self.patterns_df['amount'].min(),
                "max": self.patterns_df['amount'].max(),
                "mean": self.patterns_df['amount'].mean(),
                "median": self.patterns_df['amount'].median()
            },
            "unique_accounts": len(set(self.patterns_df['from_account'].tolist() +
                                     self.patterns_df['to_account'].tolist())),
            "unique_banks": len(set(self.patterns_df['from_bank'].tolist() +
                                  self.patterns_df['to_bank'].tolist())),
            "time_range": {
                "start": self.patterns_df['timestamp'].min(),
                "end": self.patterns_df['timestamp'].max()
            } if 'timestamp' in self.patterns_df.columns else None
        }

        return summary


def create_pattern_enhanced_features(df: pd.DataFrame,
                                   patterns_file: str = None) -> pd.DataFrame:
    """
    Função de conveniência para criar features baseadas em patterns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original com transações
    patterns_file : str, optional
        Caminho para arquivo de patterns

    Returns
    -------
    pd.DataFrame
        DataFrame com features de pattern adicionadas
    """
    engineer = PatternFeatureEngineer(patterns_file)
    return engineer.create_pattern_similarity_features(df)


if __name__ == "__main__":
    # Exemplo de uso
    engineer = PatternFeatureEngineer()

    # Mostrar resumo dos patterns
    summary = engineer.get_pattern_summary()
    print("Pattern Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Exemplo de criação de features (requer dados reais)
    # df_enhanced = engineer.create_pattern_similarity_features(df)