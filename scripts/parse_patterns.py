#!/usr/bin/env python3
"""
Parsing eficiente dos padrões de lavagem
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

print("🎯 PARSING DE PADRÕES DE LAVAGEM")

# Carregar padrões
with open('data/raw/HI-Small_Patterns.txt', 'r') as f:
    patterns_content = f.read()

pattern_lines = patterns_content.strip().split('\n')
print(f"Total de linhas: {len(pattern_lines)}")

# Parsear transações fraudulentas
fraudulent_transactions = []
current_pattern = None

for line in pattern_lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith('BEGIN LAUNDERING ATTEMPT'):
        current_pattern = line
    else:
        parts = line.split(',')
        if len(parts) >= 6:
            fraudulent_transactions.append({
                'timestamp': parts[0],
                'from_bank': parts[1],
                'from_account': parts[2],
                'to_bank': parts[3],
                'to_account': parts[4],
                'amount': parts[5],
                'pattern_type': current_pattern
            })

patterns_df = pd.DataFrame(fraudulent_transactions)
print(f"Transações fraudulentas parseadas: {len(patterns_df)}")

# Criar sets para lookup rápido
fraud_accounts = set()
fraud_account_bank_pairs = set()

for _, row in patterns_df.iterrows():
    fraud_accounts.add(row['from_account'])
    fraud_accounts.add(row['to_account'])
    fraud_account_bank_pairs.add((row['from_account'], row['from_bank']))
    fraud_account_bank_pairs.add((row['to_account'], row['to_bank']))

print(f"Contas fraudulentas únicas: {len(fraud_accounts)}")
print(f"Pares conta-banco fraudulentos: {len(fraud_account_bank_pairs)}")

# Salvar dados processados
patterns_df.to_csv('data/processed/fraud_patterns.csv', index=False)
print("💾 Padrões salvos em: data/processed/fraud_patterns.csv")

# Estatísticas dos padrões
print("\n📊 ESTATÍSTICAS DOS PADRÕES:")
print(patterns_df['pattern_type'].value_counts().head(10))

print("✅ Parsing concluído!")