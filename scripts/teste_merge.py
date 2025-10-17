#!/usr/bin/env python3
"""
Teste rápido de merge entre transações e contas
"""

import pandas as pd
from pathlib import Path

# Caminhos
data_raw_dir = Path('data/raw')

print("🔗 TESTE DE MERGE TRANSAÇÕES + CONTAS")

# Carregar dados
print("\n📊 Carregando contas...")
accounts_df = pd.read_csv(data_raw_dir / 'HI-Small_accounts.csv')
print(f"Contas: {accounts_df.shape}")

print("\n💸 Carregando transações (amostra)...")
trans_df = pd.read_csv(data_raw_dir / 'HI-Small_Trans.csv', nrows=1000)
print(f"Transações: {trans_df.shape}")

# Renomear colunas
trans_df = trans_df.rename(columns={
    'Account': 'From Account',
    'Account.1': 'To Account',
    'From Bank': 'From Bank ID',
    'To Bank': 'To Bank ID'
})

print("\n🔗 Fazendo merge...")

# Merge origem
trans_merged = trans_df.merge(
    accounts_df,
    left_on=['From Bank ID', 'From Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left',
    suffixes=('', '_from')
)

print(f"Após merge origem: {trans_merged.shape}")

# Merge destino
trans_merged = trans_merged.merge(
    accounts_df,
    left_on=['To Bank ID', 'To Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left',
    suffixes=('', '_to')
)

print(f"Após merge destino: {trans_merged.shape}")

# Verificar
missing_from = trans_merged['Entity ID'].isnull().sum()
missing_to = trans_merged['Entity ID_to'].isnull().sum()
print(f"Contas origem não encontradas: {missing_from}")
print(f"Contas destino não encontradas: {missing_to}")

print("\n✅ Merge concluído!")
print("Colunas finais:", len(trans_merged.columns))
print("Amostra:")
print(trans_merged[['From Account', 'To Account', 'Entity Name', 'Entity Name_to']].head())

# Salvar resultado
from pathlib import Path
output_path = Path('data/processed/transactions_enriched_sample.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
trans_merged.to_csv(output_path, index=False)
print(f"\n💾 Dataset salvo em: {output_path}")

print("\n📊 RESUMO DA INTEGRAÇÃO:")
print(f"   - Transações processadas: {trans_merged.shape[0]:,}")
print(f"   - Features adicionadas: {trans_merged.shape[1] - 11}")  # 11 colunas originais
print(f"   - Contas origem não encontradas: {missing_from}")
print(f"   - Contas destino não encontradas: {missing_to}")