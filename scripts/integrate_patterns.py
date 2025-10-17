"""
Integração de Features Baseadas em Patterns no Pipeline de AML

Este script demonstra como integrar as features baseadas em patterns
no pipeline existente de feature engineering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos
from src.features.pattern_engineering import PatternFeatureEngineer
from src.features.iv_calculator import calculate_iv, interpret_iv

def main():
    """
    Integra features baseadas em patterns no pipeline existente.
    """
    print("🚀 Integrando Features Baseadas em Patterns no Pipeline AML")
    print("=" * 70)

    # Paths
    artifacts_dir = Path("../artifacts")

    # 1. Carregar dados processados existentes
    print("\n📊 Carregando dados processados existentes...")
    features_file = artifacts_dir / "features_processed.pkl"

    if not features_file.exists():
        print(f"❌ Arquivo {features_file} não encontrado. Execute primeiro o notebook de feature engineering.")
        return

    df_existing = pd.read_pickle(features_file)
    print(f"✅ Dados carregados: {df_existing.shape[0]} amostras, {df_existing.shape[1]} features")

    # 2. Inicializar Pattern Feature Engineer
    print("\n🎯 Inicializando Pattern Feature Engineer...")
    pattern_engineer = PatternFeatureEngineer()

    summary = pattern_engineer.get_pattern_summary()
    print("📋 Patterns disponíveis:")
    print(f"   • {summary['total_patterns']} transações fraudulentas")
    print(f"   • {summary['pattern_types']}")
    print(f"   • {summary['unique_accounts']} contas únicas")
    print(f"   • {summary['unique_banks']} bancos únicos")

    # 3. Criar features baseadas em patterns
    print("\n🔧 Criando features baseadas em patterns...")
    df_enhanced = pattern_engineer.create_pattern_similarity_features(df_existing.copy())

    new_features = set(df_enhanced.columns) - set(df_existing.columns)
    print(f"✅ Criadas {len(new_features)} novas features:")
    for feature in sorted(new_features)[:10]:  # Mostrar primeiras 10
        print(f"   • {feature}")
    if len(new_features) > 10:
        print(f"   ... e mais {len(new_features) - 10} features")

    # 4. Calcular IV incluindo pattern features
    print("\n📈 Calculando Information Value (IV) com pattern features...")
    pattern_iv = pattern_engineer.calculate_pattern_iv(df_enhanced)

    if len(pattern_iv) > 0:
        print("🎯 Top 10 Features de Pattern por IV:")
        for _, row in pattern_iv.head(10).iterrows():
            interpretation = interpret_iv(row['IV'])
            print(".4f")
    # 5. Comparar correlações
    print("\n🔗 Comparando correlações com e sem pattern features...")

    # Correlações originais
    numeric_cols_orig = df_existing.select_dtypes(include=[np.number]).columns
    corr_orig = df_existing[numeric_cols_orig].corr()['is_fraud'].abs().max()
    print(".4f")
    # Correlações com patterns
    numeric_cols_enhanced = df_enhanced.select_dtypes(include=[np.number]).columns
    corr_enhanced = df_enhanced[numeric_cols_enhanced].corr()['is_fraud'].abs().max()
    print(".4f")
    improvement = (corr_enhanced - corr_orig) / corr_orig * 100
    print(".1f")
    # 6. Salvar resultados
    print("\n💾 Salvando resultados...")
    enhanced_file = artifacts_dir / "features_with_patterns.pkl"
    df_enhanced.to_pickle(enhanced_file)
    print(f"✅ Features enriquecidas salvas em: {enhanced_file}")

    if len(pattern_iv) > 0:
        iv_file = artifacts_dir / "pattern_iv_analysis.csv"
        pattern_iv.to_csv(iv_file, index=False)
        print(f"✅ Análise de IV salva em: {iv_file}")

    # 7. Estatísticas finais
    print("\n📊 Estatísticas Finais:")
    print(f"   • Dataset original: {df_existing.shape[1]} features")
    print(f"   • Dataset enriquecido: {df_enhanced.shape[1]} features")
    print(f"   • Features de pattern adicionadas: {len(new_features)}")
    print(f"   • Melhoria na correlação máxima: {improvement:.1f}%")

    print("\n🎉 Integração concluída com sucesso!")
    print("💡 As features de pattern estão agora disponíveis para modelagem.")
    print("=" * 70)


if __name__ == "__main__":
    main()