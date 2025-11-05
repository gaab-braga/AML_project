def set_aml_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style for the plots
    plt.style.use('seaborn-darkgrid')
    sns.set_palette("colorblind")

def apply_consistent_formatting(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)

def plot_feature_importance(importances, feature_names, top_n=20):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=feature_importances, x='importance', y='feature', ax=ax)
    apply_consistent_formatting(ax, title='Top Feature Importances', xlabel='Importance', ylabel='Features')
    plt.tight_layout()
    plt.show()