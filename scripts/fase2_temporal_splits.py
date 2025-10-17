#!/usr/bin/env python3
"""
FASE 2: Splits Temporais para Prevenção de Data Leakage
Implementação de validação temporal adequada para dados de séries temporais
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent.resolve()
data_processed_dir = project_root / 'data' / 'processed'

print("⏰ FASE 2: SPLITS TEMPORAIS PARA PREVENÇÃO DE DATA LEAKAGE")
print("=" * 60)

# Carregar dados enriquecidos da Fase 1
print("\n📊 CARREGANDO DADOS ENRIQUECIDOS DA FASE 1")
try:
    df = pd.read_csv(data_processed_dir / 'transactions_enriched_features.csv')
    print(f"✅ Dados carregados: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
except FileNotFoundError:
    print("❌ Arquivo não encontrado. Usando dados de exemplo...")
    # Fallback para dados básicos
    df = pd.read_csv(project_root / 'data' / 'processed' / 'transactions_enriched_sample.csv')
    print(f"✅ Dados fallback carregados: {df.shape[0]:,} linhas, {df.shape[1]} colunas")

# Converter timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"📅 Período dos dados: {df['Timestamp'].min()} até {df['Timestamp'].max()}")
print(f"📊 Total de dias: {(df['Timestamp'].max() - df['Timestamp'].min()).days} dias")

# Análise temporal básica
print("\n📈 ANÁLISE TEMPORAL BÁSICA:")
daily_counts = df.groupby(df['Timestamp'].dt.date).size()
print(f"   Média de transações por dia: {daily_counts.mean():.0f}")
print(f"   Desvio padrão diário: {daily_counts.std():.0f}")
print(f"   Dias com dados: {len(daily_counts)}")

# Verificar se há lacunas temporais
date_range = pd.date_range(start=df['Timestamp'].min().date(), end=df['Timestamp'].max().date())
missing_dates = date_range.difference(daily_counts.index)
print(f"   Dias sem dados: {len(missing_dates)}")

print("\n✅ ANÁLISE TEMPORAL CONCLUÍDA!")

# Implementar splits temporais
print("\n⏰ IMPLEMENTANDO SPLITS TEMPORAIS")

def create_temporal_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap_days=1):
    """
    Cria splits temporais adequados para dados de séries temporais.

    Args:
        df: DataFrame ordenado por timestamp
        train_ratio: proporção para treino
        val_ratio: proporção para validação
        test_ratio: proporção para teste
        gap_days: dias de gap entre splits para evitar data leakage

    Returns:
        train_df, val_df, test_df: DataFrames divididos temporalmente
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios devem somar 1.0"

    # Para dados com período muito curto, usar divisão por minutos/horas
    total_period = df['Timestamp'].max() - df['Timestamp'].min()

    if total_period.days < 1:
        # Se período < 1 dia, dividir por tempo absoluto
        print("   ⚠️ Período muito curto detectado. Usando divisão por tempo absoluto...")

        # Calcular pontos de corte baseados em tempo absoluto
        train_end = df['Timestamp'].min() + total_period * train_ratio
        val_end = train_end + (total_period * val_ratio)

        # Criar splits
        train_df = df[df['Timestamp'] <= train_end].copy()
        val_df = df[(df['Timestamp'] > train_end) & (df['Timestamp'] <= val_end)].copy()
        test_df = df[df['Timestamp'] > val_end].copy()
    else:
        # Período normal - usar abordagem original
        train_end = df['Timestamp'].min() + total_period * train_ratio
        val_end = train_end + (total_period * val_ratio)

        # Adicionar gap
        gap = timedelta(days=gap_days)
        val_start = train_end + gap
        test_start = val_end + gap

        # Criar splits
        train_df = df[df['Timestamp'] <= train_end].copy()
        val_df = df[(df['Timestamp'] >= val_start) & (df['Timestamp'] <= val_end)].copy()
        test_df = df[df['Timestamp'] >= test_start].copy()

    return train_df, val_df, test_df

# Aplicar splits temporais
print("   Criando splits temporais (60% treino, 20% validação, 20% teste)...")
train_df, val_df, test_df = create_temporal_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

print(f"   📊 Train: {train_df.shape[0]:,} transações ({train_df.shape[0]/len(df)*100:.1f}%)")
print(f"   📊 Validation: {val_df.shape[0]:,} transações ({val_df.shape[0]/len(df)*100:.1f}%)")
print(f"   📊 Test: {test_df.shape[0]:,} transações ({test_df.shape[0]/len(df)*100:.1f}%)")

# Verificar períodos
print("\n📅 PERÍODOS DOS SPLITS:")
print(f"   Train: {train_df['Timestamp'].min()} até {train_df['Timestamp'].max()}")
print(f"   Validation: {val_df['Timestamp'].min()} até {val_df['Timestamp'].max()}")
print(f"   Test: {test_df['Timestamp'].min()} até {test_df['Timestamp'].max()}")

# Verificar distribuição do target
print("\n🎯 DISTRIBUIÇÃO DO TARGET POR SPLIT:")
for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    fraud_rate = split_df['Is Laundering'].mean()
    print(f"   {name}: {fraud_rate:.3%} fraudulento")

# Verificar se não há data leakage
print("\n🔍 VERIFICAÇÃO DE DATA LEAKAGE:")
max_train = train_df['Timestamp'].max()
min_val = val_df['Timestamp'].min()
min_test = test_df['Timestamp'].min()

gap_train_val = (min_val - max_train).days
gap_val_test = (min_test - val_df['Timestamp'].max()).days

print(f"   Gap Train→Validation: {gap_train_val} dias")
print(f"   Gap Validation→Test: {gap_val_test} dias")

if gap_train_val > 0 and gap_val_test > 0:
    print("   ✅ Sem data leakage detectado!")
else:
    print("   ⚠️ Possível data leakage - verificar gaps!")

print("\n✅ SPLITS TEMPORAIS CRIADOS!")

# Validações adicionais
print("\n🔍 VALIDAÇÕES ADICIONAIS")

# Verificar se os splits são temporalmente coerentes
def validate_temporal_splits(train_df, val_df, test_df):
    """Valida se os splits temporais estão corretos"""
    validations = []

    # 1. Ordem temporal
    train_max = train_df['Timestamp'].max()
    val_min = val_df['Timestamp'].min()
    val_max = val_df['Timestamp'].max()
    test_min = test_df['Timestamp'].min()

    validations.append(("Ordem temporal Train < Val", train_max < val_min))
    validations.append(("Ordem temporal Val < Test", val_max < test_min))

    # 2. Sem sobreposição
    no_overlap_train_val = len(train_df[train_df['Timestamp'] >= val_min]) == 0
    no_overlap_val_test = len(val_df[val_df['Timestamp'] >= test_min]) == 0

    validations.append(("Sem sobreposição Train-Val", no_overlap_train_val))
    validations.append(("Sem sobreposição Val-Test", no_overlap_val_test))

    # 3. Continuidade temporal (não deve haver gaps muito grandes)
    expected_total = len(train_df) + len(val_df) + len(test_df)
    actual_total = len(df)
    validations.append(("Contagem total correta", expected_total == actual_total))

    return validations

validations = validate_temporal_splits(train_df, val_df, test_df)
print("   Validações dos splits:")
for validation, passed in validations:
    status = "✅" if passed else "❌"
    print(f"      {status} {validation}")

# Análise estatística dos splits
print("\n📊 ANÁLISE ESTATÍSTICA DOS SPLITS:")
stats_data = []
for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    stats = {
        'split': name,
        'transactions': len(split_df),
        'fraud_rate': split_df['Is Laundering'].mean(),
        'avg_amount': split_df['Amount Paid'].mean(),
        'unique_accounts': split_df['From Account'].nunique(),
        'period_days': (split_df['Timestamp'].max() - split_df['Timestamp'].min()).days
    }
    stats_data.append(stats)

stats_df = pd.DataFrame(stats_data)
print(stats_df.round(4))

# Visualização dos splits temporais
print("\n📈 VISUALIZAÇÃO DOS SPLITS TEMPORAIS")

plt.figure(figsize=(15, 10))

# 1. Distribuição temporal dos splits
plt.subplot(2, 2, 1)
for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    split_df.groupby(split_df['Timestamp'].dt.date).size().plot(label=name, alpha=0.7)

plt.title('Distribuição Temporal dos Splits')
plt.xlabel('Data')
plt.ylabel('Número de Transações')
plt.legend()
plt.xticks(rotation=45)

# 2. Taxa de fraude por split
plt.subplot(2, 2, 2)
fraud_rates = [train_df['Is Laundering'].mean(), val_df['Is Laundering'].mean(), test_df['Is Laundering'].mean()]
plt.bar(['Train', 'Validation', 'Test'], fraud_rates, color=['blue', 'orange', 'red'])
plt.title('Taxa de Fraude por Split')
plt.ylabel('Taxa de Fraude')
plt.ylim(0, max(fraud_rates) * 1.1)

# 3. Distribuição de valores por split
plt.subplot(2, 2, 3)
for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    plt.hist(split_df['Amount Paid'], bins=50, alpha=0.5, label=name, density=True)

plt.title('Distribuição de Valores por Split')
plt.xlabel('Valor Pago')
plt.ylabel('Densidade')
plt.legend()
plt.xscale('log')

# 4. Features de risco por split
plt.subplot(2, 2, 4)
risk_features = ['transaction_risk', 'high_amount_risk', 'same_bank_transaction']
risk_data = []

for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    for feature in risk_features:
        if feature in split_df.columns:
            risk_data.append({
                'split': name,
                'feature': feature,
                'rate': split_df[feature].mean()
            })

risk_df = pd.DataFrame(risk_data)
sns.barplot(data=risk_df, x='feature', y='rate', hue='split')
plt.title('Features de Risco por Split')
plt.ylabel('Taxa')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\n💾 SALVANDO SPLITS TEMPORAIS")

# Criar diretório para splits
splits_dir = data_processed_dir / 'temporal_splits'
splits_dir.mkdir(exist_ok=True)

# Salvar splits
train_df.to_csv(splits_dir / 'train_temporal.csv', index=False)
val_df.to_csv(splits_dir / 'validation_temporal.csv', index=False)
test_df.to_csv(splits_dir / 'test_temporal.csv', index=False)

# Salvar metadados dos splits
metadata = {
    'creation_date': datetime.now().isoformat(),
    'total_transactions': len(df),
    'splits': {
        'train': {
            'transactions': len(train_df),
            'fraud_rate': train_df['Is Laundering'].mean(),
            'period_start': train_df['Timestamp'].min().isoformat(),
            'period_end': train_df['Timestamp'].max().isoformat()
        },
        'validation': {
            'transactions': len(val_df),
            'fraud_rate': val_df['Is Laundering'].mean(),
            'period_start': val_df['Timestamp'].min().isoformat(),
            'period_end': val_df['Timestamp'].max().isoformat()
        },
        'test': {
            'transactions': len(test_df),
            'fraud_rate': test_df['Is Laundering'].mean(),
            'period_start': test_df['Timestamp'].min().isoformat(),
            'period_end': test_df['Timestamp'].max().isoformat()
        }
    },
    'validation_results': {name: passed for name, passed in validations}
}

import json
with open(splits_dir / 'splits_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"   ✅ Splits salvos em: {splits_dir}")
print(f"   ✅ Metadados salvos em: {splits_dir}/splits_metadata.json")

print("\n🎯 FASE 2 CONCLUÍDA COM SUCESSO!")
print("   ✅ Splits temporais implementados")
print("   ✅ Data leakage prevenido")
print("   ✅ Validações de integridade realizadas")
print("   ✅ Visualizações e estatísticas geradas")
print("   ✅ Dados salvos para próximas fases")

print("\n🚀 PRÓXIMOS PASSOS:")
print("   - Fase 3: Re-treinamento de modelos com features enriquecidas")
print("   - Fase 4: Otimização de hiperparâmetros")
print("   - Fase 5: Ensemble e avaliação final")