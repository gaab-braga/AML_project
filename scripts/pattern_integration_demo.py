"""
Pattern-Aware Feature Engineering and IV Analysis

Este notebook demonstra como integrar patterns de lavagem de dinheiro conhecidos
nas análises de feature engineering e Information Value (IV).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos criados
from src.features.pattern_engineering import PatternFeatureEngineer, create_pattern_enhanced_features
from src.features.iv_calculator import calculate_iv, interpret_iv

# Configuração de paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data"
artifacts_dir = project_root / "artifacts"

def main():
    """
    Demonstração da integração de patterns nas análises.
    """
    print("🚀 Iniciando análise integrada com patterns de lavagem de dinheiro")
    print("=" * 70)

    # 1. Carregar dados processados
    print("\n📊 Carregando dados processados...")
    features_file = artifacts_dir / "features_processed.pkl"

    if not features_file.exists():
        print(f"❌ Arquivo {features_file} não encontrado. Execute o notebook de feature engineering primeiro.")
        return

    df = pd.read_pickle(features_file)
    print(f"✅ Dados carregados: {df.shape[0]} amostras, {df.shape[1]} features")

    # 2. Inicializar engenheiro de patterns
    print("\n🎯 Inicializando Pattern Feature Engineer...")
    pattern_engineer = PatternFeatureEngineer()

    # Mostrar resumo dos patterns
    summary = pattern_engineer.get_pattern_summary()
    print("\n📋 Resumo dos Patterns Carregados:")
    print(f"   • Total de transações em patterns: {summary['total_patterns']}")
    print(f"   • Tipos de patterns: {summary['pattern_types']}")
    print(f"   • Contas únicas envolvidas: {summary['unique_accounts']}")
    print(f"   • Bancos únicos envolvidos: {summary['unique_banks']}")
    print(".2f")
    print(".2f")
    # 3. Criar features baseadas em patterns
    print("\n🔧 Criando features baseadas em patterns...")
    df_enhanced = pattern_engineer.create_pattern_similarity_features(df.copy())

    new_features = set(df_enhanced.columns) - set(df.columns)
    print(f"✅ Criadas {len(new_features)} novas features baseadas em patterns:")
    for feature in sorted(new_features):
        print(f"   • {feature}")

    # 4. Calcular IV para features de pattern
    print("\n📈 Calculando Information Value (IV) para features de pattern...")
    pattern_iv = pattern_engineer.calculate_pattern_iv(df_enhanced)

    if len(pattern_iv) > 0:
        print("\n🎯 Top 10 Features de Pattern por IV:")
        top_pattern_features = pattern_iv.head(10)
        for _, row in top_pattern_features.iterrows():
            interpretation = interpret_iv(row['IV'])
            print(".4f")
            print(f"      Interpretação: {interpretation}")

        # 5. Comparar distribuição entre patterns e dados gerais
        print("\n📊 Comparando distribuições...")
        compare_pattern_distributions(df_enhanced, pattern_engineer.patterns_df)

        # 6. Análise de correlação entre features de pattern e target
        print("\n🔗 Analisando correlação com target...")
        analyze_pattern_correlations(df_enhanced, new_features)

        # 7. Salvar resultados
        print("\n💾 Salvando resultados...")
        output_file = artifacts_dir / "features_with_patterns.pkl"
        df_enhanced.to_pickle(output_file)
        print(f"✅ Features enriquecidas salvas em: {output_file}")

        iv_output_file = artifacts_dir / "pattern_iv_analysis.csv"
        pattern_iv.to_csv(iv_output_file, index=False)
        print(f"✅ Análise de IV salva em: {iv_output_file}")

    else:
        print("❌ Nenhuma feature de pattern encontrada para análise de IV")

    print("\n🎉 Análise integrada concluída!")
    print("=" * 70)


def compare_pattern_distributions(df_enhanced: pd.DataFrame, patterns_df: pd.DataFrame):
    """
    Compara distribuições entre patterns e dados gerais.
    """
    if patterns_df is None or len(patterns_df) == 0:
        return

    # Comparar distribuição de valores
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Amount distribution
    if 'amount' in df_enhanced.columns and 'amount' in patterns_df.columns:
        # Sample dos dados gerais para comparison
        sample_size = min(10000, len(df_enhanced))
        df_sample = df_enhanced.sample(sample_size, random_state=42)

        sns.histplot(df_sample['amount'], bins=50, alpha=0.7, label='Dados Gerais',
                    ax=axes[0,0], color='blue')
        sns.histplot(patterns_df['amount'], bins=30, alpha=0.7, label='Patterns',
                    ax=axes[0,0], color='red')
        axes[0,0].set_title('Distribuição de Valores')
        axes[0,0].set_xlabel('Valor')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')

    # Pattern similarity features
    similarity_cols = [col for col in df_enhanced.columns if 'similarity' in col and 'amount' in col]
    if similarity_cols:
        for i, col in enumerate(similarity_cols[:3]):  # Max 3 features
            row, col_idx = divmod(i+1, 2)
            if row < 2 and col_idx < 2:
                sns.histplot(df_enhanced[col], bins=30, alpha=0.7, ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'Distribuição: {col}')
                axes[row, col_idx].set_xlabel('Similaridade')

    plt.tight_layout()
    plt.savefig(artifacts_dir / "pattern_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Gráfico de comparação salvo em: pattern_distribution_comparison.png")


def analyze_pattern_correlations(df_enhanced: pd.DataFrame, new_features: set):
    """
    Analisa correlações entre features de pattern e target.
    """
    if 'is_fraud' not in df_enhanced.columns:
        return

    # Calcular correlações
    correlation_data = []
    target = df_enhanced['is_fraud']

    for feature in new_features:
        if feature in df_enhanced.columns:
            try:
                corr = df_enhanced[feature].corr(target)
                if not np.isnan(corr):
                    correlation_data.append({
                        'feature': feature,
                        'correlation_with_target': corr,
                        'abs_correlation': abs(corr)
                    })
            except:
                continue

    if correlation_data:
        corr_df = pd.DataFrame(correlation_data).sort_values('abs_correlation', ascending=False)

        print("🔗 Top correlações com target:")
        for _, row in corr_df.head(10).iterrows():
            print(f"   • {row['feature']}: {row['correlation_with_target']:.4f}")
        # Salvar análise de correlação
        corr_df.to_csv(artifacts_dir / "pattern_correlations.csv", index=False)


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\gafeb\OneDrive\Desktop\lavagem_dev\scripts\pattern_integration_demo.py