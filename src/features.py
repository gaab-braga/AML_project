"""
Features Module for AML Detection
Handles feature engineering: aggregations, network features, temporal features.
"""

import pandas as pd
import networkx as nx
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import functions from temporal module to maintain compatibility
try:
    from src.features.temporal import aggregate_by_entity as temporal_aggregate
    from src.features.temporal import compute_network_features as temporal_network
except ImportError:
    logger.warning("Temporal features module not available")
    temporal_aggregate = None
    temporal_network = None

def aggregate_by_entity(df: pd.DataFrame, group_col: str, windows: List[int] = [7, 30]) -> pd.DataFrame:
    """
    Calculate aggregations (sum, mean, count) by entity for time windows.

    Args:
        df: DataFrame with entity and temporal data
        group_col: Column to group by (e.g., 'customer_id')
        windows: List of window sizes in days

    Returns:
        pd.DataFrame: DataFrame with aggregated features
    """
    logger.info(f"Aggregating features by {group_col} for windows {windows}")

    # Ensure we have a date column
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp'])
    elif 'date' not in df.columns:
        raise ValueError("DataFrame must contain either 'date' or 'timestamp' column")

    df['date'] = pd.to_datetime(df['date'])
    features = []

    for window in windows:
        agg_df = df.set_index('date').groupby(group_col).rolling(f'{window}D').agg({
            'amount': ['sum', 'mean', 'count']
        }).reset_index()
        agg_df.columns = [f"{group_col}_{col[0]}_{col[1]}_{window}d" if col[1] else col[0] for col in agg_df.columns]
        features.append(agg_df)

    if features:
        result = pd.concat(features, axis=1)
        # Fill NaN values
        result = result.fillna(0)
    else:
        result = df.copy()

    logger.info("Aggregation completed")
    return result

def compute_network_features(df_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network features: degree, centrality from transaction edges.

    Args:
        df_edges: DataFrame with edges (source, target, weight)

    Returns:
        pd.DataFrame: DataFrame with network features
    """
    logger.info("Computing network features")
    G = nx.from_pandas_edgelist(df_edges, 'source', 'target', 'amount')
    degrees = dict(G.degree())
    centrality = nx.degree_centrality(G)

    features = pd.DataFrame({
        'node': list(degrees.keys()),
        'degree': list(degrees.values()),
        'centrality': [centrality.get(node, 0) for node in degrees.keys()]
    })
    logger.info("Network features computed")
    return features