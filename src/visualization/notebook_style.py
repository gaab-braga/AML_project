"""
Configuração de estilo visual para notebooks AML.
Cores, fontes e formatação padronizada.
"""
import matplotlib.pyplot as plt
import seaborn as sns


# DICIONÁRIO CENTRAL DE CORES - ÚNICA FONTE DA VERDADE
diverging_colors = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=False, n=8)

AML_COLORS = {
    # Cores principais para gráficos
    'primary': diverging_colors[0],
    'secondary': diverging_colors[1],
    'accent': diverging_colors[2],
    'neutral': diverging_colors[3],
    # Cores semânticas
    'fraud': diverging_colors[4],
    'legit': diverging_colors[5],
    # UI
    'background': "#2E2E2E",
    'text': "#FFFFFF",
    'text_primary': "#FFFFFF",
    'text_secondary': "#B0B0B0",
    'grid': "#404040",
    # Ranking IV
    'iv_not_predictive': diverging_colors[3],
    'iv_weak': diverging_colors[0],
    'iv_medium': diverging_colors[1],
    'iv_strong': diverging_colors[2],
    'iv_very_strong': diverging_colors[4],
    # Mapeamentos para compatibilidade
    'success': diverging_colors[5],
    'danger': diverging_colors[4],
    'purple': diverging_colors[6],
    'orange': diverging_colors[7],
    'warning': "#F39C12",
    'info': "#3498DB"
}

MODEL_COLORS = {
    'XGBoost': AML_COLORS['primary'],
    'LightGBM': AML_COLORS['secondary'],
    'LogisticRegression': AML_COLORS['accent'],
    'RandomForest': AML_COLORS['danger'],
    'Baseline': AML_COLORS['neutral']
}

FRAUD_COLORS = {
    'Fraud': AML_COLORS['danger'],
    'Non-Fraud': AML_COLORS['legit'],
    'Unknown': AML_COLORS['neutral']
}

FEATURE_IMPORTANCE_COLORS = diverging_colors[:10]


def setup_notebook_style():
    """
    Configura todo o estilo visual dos notebooks AML.
    Chame esta função UMA VEZ no início do notebook.
    """
    # Aplicar tema escuro
    plt.style.use('dark_background')
    
    # Configurar cores padrão
    plt.rcParams['figure.facecolor'] = AML_COLORS['background']
    plt.rcParams['axes.facecolor'] = AML_COLORS['background']
    plt.rcParams['savefig.facecolor'] = AML_COLORS['background']
    
    # Configurar paleta seaborn
    sns.set_palette(diverging_colors[:6])
    
    # Configurar tamanhos padrão
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    # Configurar cores de texto
    plt.rcParams['text.color'] = AML_COLORS['text']
    plt.rcParams['axes.labelcolor'] = AML_COLORS['text']
    plt.rcParams['xtick.color'] = AML_COLORS['text_secondary']
    plt.rcParams['ytick.color'] = AML_COLORS['text_secondary']
    
    # Grid sutil
    plt.rcParams['grid.color'] = AML_COLORS['grid']
    plt.rcParams['grid.alpha'] = 0.3
    
    # Atualizar módulo de visualização existente
    try:
        import src.visualization.plot_config as pc
        pc.AML_COLORS.update(AML_COLORS)
        pc.MODEL_COLORS = MODEL_COLORS
        pc.FRAUD_COLORS = FRAUD_COLORS
        pc.FEATURE_IMPORTANCE_COLORS = FEATURE_IMPORTANCE_COLORS
    except ImportError:
        pass  # Módulo não existe ainda


def create_aml_figure(figsize=(12, 8)):
    """
    Cria figura com estilo AML padrão.
    
    Returns:
        tuple: (fig, ax) prontos para plotagem
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(AML_COLORS['background'])
    ax.set_facecolor(AML_COLORS['background'])
    ax.tick_params(colors=AML_COLORS['text_secondary'])
    return fig, ax


def apply_consistent_formatting(ax, title='', xlabel='', ylabel=''):
    """
    Aplica formatação consistente a um eixo matplotlib.
    
    Args:
        ax: Eixo matplotlib
        title: Título do gráfico
        xlabel: Label do eixo X
        ylabel: Label do eixo Y
    """
    if title:
        ax.set_title(title, color=AML_COLORS['text'], fontsize=12, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, color=AML_COLORS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, color=AML_COLORS['text'])
    
    ax.tick_params(colors=AML_COLORS['text_secondary'])
    ax.spines['bottom'].set_color(AML_COLORS['grid'])
    ax.spines['left'].set_color(AML_COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_legend():
        legend = ax.get_legend()
        legend.get_frame().set_facecolor(AML_COLORS['background'])
        legend.get_frame().set_edgecolor(AML_COLORS['grid'])
        for text in legend.get_texts():
            text.set_color(AML_COLORS['text'])
