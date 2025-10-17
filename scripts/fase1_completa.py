#!/usr/bin/env python3
"""
FASE 1 COMPLETA: Integração de dados e features avançadas
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("🚀 FASE 1 COMPLETA: INTEGRAÇÃO DE DADOS E FEATURES AVANÇADAS")

# Carregar dados
print("\n📊 CARREGANDO DADOS PROCESSADOS")
trans_enriched = pd.read_csv('data/processed/transactions_enriched_sample.csv')
patterns_df = pd.read_csv('data/processed/fraud_patterns.csv')

print(f"Transações enriquecidas: {trans_enriched.shape}")
print(f"Padrões de fraude: {patterns_df.shape}")

# Criar sets para lookup rápido
fraud_account_bank_pairs = set()
for _, row in patterns_df.iterrows():
    fraud_account_bank_pairs.add((row['from_account'], row['from_bank']))
    fraud_account_bank_pairs.add((row['to_account'], row['to_bank']))

print(f"Pares conta-banco fraudulentos: {len(fraud_account_bank_pairs)}")

# Features de risco baseadas em padrões conhecidos
print("\n⚠️ CRIANDO FEATURES DE RISCO")

trans_enriched['from_account_risk'] = trans_enriched.apply(
    lambda row: (row['From Account'], str(row['From Bank ID'])) in fraud_account_bank_pairs,
    axis=1
)

trans_enriched['to_account_risk'] = trans_enriched.apply(
    lambda row: (row['To Account'], str(row['To Bank ID'])) in fraud_account_bank_pairs,
    axis=1
)

trans_enriched['transaction_risk'] = trans_enriched['from_account_risk'] | trans_enriched['to_account_risk']

# Features baseadas em valores
print("\n💰 CRIANDO FEATURES DE VALOR")
amount_paid_q95 = trans_enriched['Amount Paid'].quantile(0.95)
trans_enriched['high_amount_risk'] = trans_enriched['Amount Paid'] > amount_paid_q95

# Features baseadas em entidades
print("\n🏢 CRIANDO FEATURES DE ENTIDADE")
entity_types = ['Sole Proprietorship', 'Partnership', 'Corporation', 'Individual']

for entity_type in entity_types:
    trans_enriched[f'from_{entity_type.lower().replace(" ", "_")}_flag'] = trans_enriched['Entity Name'].str.contains(entity_type, na=False)
    trans_enriched[f'to_{entity_type.lower().replace(" ", "_")}_flag'] = trans_enriched['Entity Name_to'].str.contains(entity_type, na=False)

# Features de rede básica
print("\n🔗 CRIANDO FEATURES DE REDE BÁSICA")
trans_enriched['same_bank_transaction'] = trans_enriched['From Bank ID'] == trans_enriched['To Bank ID']
trans_enriched['same_entity_transaction'] = trans_enriched['Entity ID'] == trans_enriched['Entity ID_to']

# Estatísticas das features
print("\n📊 ESTATÍSTICAS DAS FEATURES:")
print(f"   - Transações com conta origem em risco: {trans_enriched['from_account_risk'].sum()}")
print(f"   - Transações com conta destino em risco: {trans_enriched['to_account_risk'].sum()}")
print(f"   - Transações com qualquer conta em risco: {trans_enriched['transaction_risk'].sum()}")
print(f"   - Transações com valor alto: {trans_enriched['high_amount_risk'].sum()}")
print(f"   - Transações no mesmo banco: {trans_enriched['same_bank_transaction'].sum()}")
print(f"   - Transações mesma entidade: {trans_enriched['same_entity_transaction'].sum()}")

# Salvar dataset final
output_path = 'data/processed/transactions_enriched_features.csv'
trans_enriched.to_csv(output_path, index=False)

print(f"\n💾 DATASET FINAL SALVO: {output_path}")
print(f"   Shape: {trans_enriched.shape[0]:,} linhas, {trans_enriched.shape[1]} colunas")

# Resumo da Fase 1
print("\n✅ FASE 1 CONCLUÍDA COM SUCESSO!")
print("📝 RESUMO DAS MELHORIAS:")
print(f"   - ✅ Integração transações + contas: {trans_enriched.shape[1] - 11} features adicionadas")
print(f"   - ✅ Parsing de padrões: {len(patterns_df)} transações fraudulentas identificadas")
print("   - ✅ Features de risco: baseadas em padrões conhecidos e valores altos")
print("   - ✅ Features de entidade: flags para tipos de entidade (Sole Proprietorship, Partnership, etc.)")
print("   - ✅ Features de rede: transações no mesmo banco e mesma entidade")
print("\n🎯 PRÓXIMOS PASSOS DA ROADMAP:")
print("   - Fase 2: Implementar splits temporais para evitar data leakage")
print("   - Fase 3: Re-treinar modelos com features enriquecidas")
print("   - Fase 4: Otimizar hiperparâmetros por tipo de modelo")
print("   - Fase 5: Implementar ensemble e avaliação final")