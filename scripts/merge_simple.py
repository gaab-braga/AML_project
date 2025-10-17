#!/usr/bin/env python3
"""
Merge simples de transações e contas
"""

import pandas as pd
from pathlib import Path

print("🔗 MERGE SIMPLES - TRANSAÇÕES + CONTAS")

# Carregar dados
accounts_df = pd.read_csv('data/raw/HI-Small_accounts.csv')
trans_df = pd.read_csv('data/raw/HI-Small_Trans.csv', nrows=1000)

print(f"Contas: {accounts_df.shape}")
print(f"Transações: {trans_df.shape}")

# Renomear
trans_df = trans_df.rename(columns={
    'Account': 'From Account',
    'Account.1': 'To Account',
    'From Bank': 'From Bank ID',
    'To Bank': 'To Bank ID'
})

# Merge origem
trans_merged = trans_df.merge(
    accounts_df,
    left_on=['From Bank ID', 'From Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left'
)

# Merge destino
trans_merged = trans_merged.merge(
    accounts_df,
    left_on=['To Bank ID', 'To Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left',
    suffixes=('', '_to')
)

print(f"Resultado: {trans_merged.shape}")

# Salvar
output_path = Path('data/processed/transactions_enriched_sample.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
trans_merged.to_csv(output_path, index=False)

print(f"Salvo em: {output_path}")
print("✅ Concluído!")