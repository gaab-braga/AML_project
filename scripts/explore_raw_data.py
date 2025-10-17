#!/usr/bin/env python3
"""
Script rápido para explorar estrutura dos dados raw
"""

import pandas as pd
from pathlib import Path

# Caminhos
data_raw_dir = Path('data/raw')

print("🔍 EXPLORANDO ESTRUTURA DOS DADOS RAW")

# Carregar contas
print("\n📊 CONTAS:")
accounts_df = pd.read_csv(data_raw_dir / 'HI-Small_accounts.csv')
print(f"Shape: {accounts_df.shape}")
print("Columns:", list(accounts_df.columns))
print("Dtypes:")
print(accounts_df.dtypes)
print("\nAmostra:")
print(accounts_df.head())

# Carregar padrões
print("\n📋 PADRÕES:")
with open(data_raw_dir / 'HI-Small_Patterns.txt', 'r') as f:
    patterns = f.read()

print(f"Tamanho: {len(patterns)} caracteres")
print("Primeiras 1000 chars:")
print(patterns[:1000])

# Carregar pequena amostra de transações
print("\n💸 TRANSAÇÕES (amostra):")
trans_df = pd.read_csv(data_raw_dir / 'HI-Small_Trans.csv', nrows=5)
print(f"Shape (amostra): {trans_df.shape}")
print("Columns:", list(trans_df.columns))
print("Dtypes:")
print(trans_df.dtypes)
print("\nAmostra:")
print(trans_df.head())