# src/visualization/plot_config.py
"""
Centralized plotting configuration for AML project notebooks.
Provides consistent colors, styles, and themes across all visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from cycler import cycler
import matplotlib.colors as mcolors

# Color palette definitions
AML_COLORS = {
    'primary': '#06445e',      # Dark blue - main theme
    'secondary': '#1f77b4',    # Medium blue
    'accent': '#ff7f0e',       # Orange
    'success': '#2ca02c',      # Green
    'warning': '#d62728',      # Red
    'neutral': '#7f7f7f',      # Gray
    'light_blue': '#aec7e8',
    'light_orange': '#ffbb78',
    'light_green': '#98df8a',
    'light_red': '#ff9896'
}

# Model comparison colors
MODEL_COLORS = {
    'XGBoost': '#1f77b4',      # Blue
    'LightGBM': '#ff7f0e',     # Orange
    'LogisticRegression': '#2ca02c',  # Green
    'RandomForest': '#d62728', # Red
    'Baseline': '#7f7f7f'      # Gray
}

# Fraud vs Non-fraud colors
FRAUD_COLORS = {
    'Fraud': '#d62728',       # Red
    'Non-Fraud': '#1f77b4',   # Blue
    'Unknown': '#7f7f7f'      # Gray
}

# Feature importance colors
FEATURE_IMPORTANCE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def set_aml_style():
    """
    Apply AML project styling to matplotlib and seaborn.
    Call this at the beginning of plotting cells.
    """
    # Set seaborn style
    sns.set_style('whitegrid', {
        'grid.color': '.8',
        'grid.linestyle': '--',
        'axes.edgecolor': '.8',
        'axes.labelcolor': AML_COLORS['primary'],
        'xtick.color': AML_COLORS['primary'],
        'ytick.color': AML_COLORS['primary']
    })

    # Matplotlib parameters
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titleweight': 'bold'
    })

    # Color cycle for multiple lines
    plt.rcParams['axes.prop_cycle'] = cycler(color=FEATURE_IMPORTANCE_COLORS)

    # Set default colormap
    plt.rcParams['image.cmap'] = 'viridis'

def get_model_palette():
    """Get color palette for model comparisons."""
    return sns.color_palette(list(MODEL_COLORS.values()))

def get_fraud_palette():
    """Get color palette for fraud/non-fraud visualizations."""
    return sns.color_palette(list(FRAUD_COLORS.values()))

def get_feature_importance_palette(n_colors=None):
    """Get color palette for feature importance plots."""
    if n_colors:
        return sns.color_palette(FEATURE_IMPORTANCE_COLORS[:n_colors])
    return sns.color_palette(FEATURE_IMPORTANCE_COLORS)

def apply_consistent_formatting(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply consistent formatting to matplotlib axes.

    Parameters:
    ax: matplotlib axes object
    title: str, plot title
    xlabel: str, x-axis label
    ylabel: str, y-axis label
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color=AML_COLORS['primary'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color=AML_COLORS['primary'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color=AML_COLORS['primary'])

    # Style the grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Style the spines
    for spine in ax.spines.values():
        spine.set_edgecolor(AML_COLORS['neutral'])
        spine.set_alpha(0.5)

def create_model_comparison_plot(data, x_col, y_col, hue_col='model',
                                title="Model Comparison", figsize=(12, 6)):
    """
    Create a standardized model comparison plot.

    Parameters:
    data: DataFrame with model comparison data
    x_col: str, column for x-axis
    y_col: str, column for y-axis
    hue_col: str, column for color grouping (default: 'model')
    title: str, plot title
    figsize: tuple, figure size

    Returns:
    fig, ax: matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Apply AML styling
    set_aml_style()

    # Create the plot
    sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col,
                palette=MODEL_COLORS, ax=ax)

    # Apply consistent formatting
    apply_consistent_formatting(ax, title=title, xlabel=x_col, ylabel=y_col)

    # Customize legend
    if hue_col in data.columns:
        ax.legend(title=hue_col.title(), title_fontsize=12, fontsize=10)

    plt.tight_layout()
    return fig, ax

def create_feature_importance_plot(feature_names, importance_values,
                                  title="Feature Importance", top_n=20,
                                  figsize=(12, 8)):
    """
    Create a standardized feature importance plot.

    Parameters:
    feature_names: list of feature names
    importance_values: list of importance values
    title: str, plot title
    top_n: int, number of top features to show
    figsize: tuple, figure size

    Returns:
    fig, ax: matplotlib figure and axes objects
    """
    # Sort and select top features
    sorted_idx = importance_values.argsort()[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = [importance_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Apply AML styling
    set_aml_style()

    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_importance,
                   color=FEATURE_IMPORTANCE_COLORS[:len(top_features)])

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_importance)):
        ax.text(value + max(top_importance) * 0.01, i,
                f'{value:.4f}', va='center', fontsize=9)

    # Customize axes
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Highest importance at top

    # Apply consistent formatting
    apply_consistent_formatting(ax, title=title, xlabel='Importance', ylabel='Features')

    plt.tight_layout()
    return fig, ax


# Optuna visualization functions with AML styling
def plot_aml_optimization_history(study, model_name: str):
    """
    Exibe o histórico de otimização do Optuna com o estilo AML padronizado.
    """
    from optuna.visualization import plot_optimization_history

    fig = plot_optimization_history(study)
    
    # Convert color to hex if it's a tuple
    primary_color = AML_COLORS['primary']
    if isinstance(primary_color, tuple):
        primary_color = mcolors.to_hex(primary_color)
    
    fig.update_traces(
        marker=dict(color=primary_color),
        line=dict(color=primary_color)
    )
    fig.update_layout(
        title_text=f'Histórico de Otimização - {model_name.upper()}',
        title_font_color='#FFFFFF',  # White title for better contrast
        title_font_size=16,
        title_font_family='Arial',
        font_color='#FFFFFF',  # White text for dark background
        plot_bgcolor='#2e2e2e',  # Dark background
        paper_bgcolor='#2e2e2e',  # Dark paper background
        xaxis=dict(
            title_font_color='#FFFFFF',  # White axis titles
            tickfont_color='#B0B0B0',  # Light gray ticks
            gridcolor='#404040',  # Dark gray grid
            linecolor='#404040'  # Dark gray axis lines
        ),
        yaxis=dict(
            title_font_color='#FFFFFF',  # White axis titles
            tickfont_color='#B0B0B0',  # Light gray ticks
            gridcolor='#404040',  # Dark gray grid
            linecolor='#404040'  # Dark gray axis lines
        )
    )
    fig.show()


def plot_aml_param_importances(study, model_name: str):
    """
    Exibe a importância dos hiperparâmetros do Optuna com o estilo AML padronizado.
    """
    from optuna.visualization import plot_param_importances

    fig = plot_param_importances(study)
    
    # Convert colors to hex if they are tuples
    num_colors = len(fig.data[0].x) if len(fig.data) > 0 else len(FEATURE_IMPORTANCE_COLORS)
    colors = [mcolors.to_hex(c) if isinstance(c, tuple) else c for c in FEATURE_IMPORTANCE_COLORS[:num_colors]]
    
    fig.update_traces(
        marker=dict(color=colors)
    )
    
    # Convert primary color
    primary_color = AML_COLORS['primary']
    if isinstance(primary_color, tuple):
        primary_color = mcolors.to_hex(primary_color)
    
    fig.update_layout(
        title_text=f'Importância dos Hiperparâmetros - {model_name.upper()}',
        title_font_color='#FFFFFF',  # White title for better contrast
        title_font_size=16,
        title_font_family='Arial',
        font_color='#FFFFFF',  # White text for dark background
        plot_bgcolor='#2e2e2e',  # Dark background
        paper_bgcolor='#2e2e2e',  # Dark paper background
        xaxis=dict(
            title_font_color='#FFFFFF',  # White axis titles
            tickfont_color='#B0B0B0',  # Light gray ticks
            gridcolor='#404040',  # Dark gray grid
            linecolor='#404040'  # Dark gray axis lines
        ),
        yaxis=dict(
            title_font_color='#FFFFFF',  # White axis titles
            tickfont_color='#B0B0B0',  # Light gray ticks
            gridcolor='#404040',  # Dark gray grid
            linecolor='#404040'  # Dark gray axis lines
        )
    )
    fig.show()