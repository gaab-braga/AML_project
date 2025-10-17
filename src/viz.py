"""
Visualization Module for AML Detection
Handles plotting for time series, networks, and AML-specific visuals.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def plot_time_series(df, col: str, interactive: bool = False):
    """
    Plot time series for a column, with AML annotations.

    Args:
        df: DataFrame with date column
        col: Column to plot
        interactive: Use Plotly for interactivity
    """
    logger.info(f"Plotting time series for {col}")
    if interactive:
        fig = px.line(df, x='date', y=col, title=f'Time Series: {col} (AML Transactions)')
        fig.add_annotation(text="Potential AML spike", x=df['date'].iloc[len(df)//2], y=df[col].max(), showarrow=True)
        fig.show()
    else:
        df.set_index('date')[col].plot(figsize=(10, 5))
        plt.title(f'Time Series: {col}')
        plt.axhline(y=df[col].quantile(0.95), color='red', linestyle='--', label='95th Percentile (Suspicious)')
        plt.legend()
        plt.show()

def plot_network_subgraph(G, nodes, interactive: bool = False):
    """
    Plot subgraph for given nodes, highlighting AML connections.

    Args:
        G: NetworkX graph
        nodes: List of nodes to include
        interactive: Use Plotly for interactivity
    """
    logger.info("Plotting network subgraph")
    subgraph = G.subgraph(nodes)
    if interactive:
        pos = nx.spring_layout(subgraph)
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x, node_y, node_text = [], [], []
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Entity {node} (Degree: {subgraph.degree[node]})")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, marker=dict(size=10, color='lightblue')))
        fig.update_layout(title="AML Transaction Network (Suspicious Entities)")
        fig.show()
    else:
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title('Network Subgraph')
        plt.show()

def plot_risk_heatmap(df, entity_col: str = 'customer_id', risk_col: str = 'risk_score'):
    """
    Plot heatmap of risk by entity, AML-specific.

    Args:
        df: DataFrame with entity and risk columns
    """
    logger.info("Plotting risk heatmap")
    pivot = df.pivot_table(values=risk_col, index=entity_col, aggfunc='mean')
    sns.heatmap(pivot, cmap='Reds', annot=True)
    plt.title('AML Risk Heatmap by Entity')
    plt.show()

def plot_feature_importance(importances, feature_names):
    """
    Plot feature importance with AML context.

    Args:
        importances: Array of importance scores
        feature_names: List of feature names
    """
    logger.info("Plotting feature importance")
    fig = px.bar(x=feature_names, y=importances, title="Feature Importance (AML Detection)")
    fig.add_annotation(text="Top features indicate AML patterns", x=0, y=max(importances), showarrow=True)
    fig.show()