#!/usr/bin/env python3
"""
FASE 1: Integração de Dados e Engenharia de Features Avançada
Implementação do roadmap AML - Integração de dados raw para enriquecer features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("🚀 FASE 1: INTEGRAÇÃO DE DADOS E ENGENHARIA DE FEATURES")
print("=" * 60)

# Caminhos dos arquivos
data_raw_dir = project_root / 'data' / 'raw'
data_processed_dir = project_root / 'data' / 'processed'

# Verificar se arquivos existem
required_files = [
    data_raw_dir / 'HI-Small_Trans.csv',
    data_raw_dir / 'HI-Small_accounts.csv',
    data_raw_dir / 'HI-Small_Patterns.txt'
]

for file_path in required_files:
    if file_path.exists():
        print(f"✅ {file_path.name} encontrado")
    else:
        print(f"❌ {file_path.name} NÃO encontrado")
        sys.exit(1)

print("\n📊 CARREGANDO DADOS RAW...")

# 1. Carregar transações (amostra para desenvolvimento)
print("\n1️⃣ Carregando transações (amostra de 1000 linhas)...")
try:
    # Carregar apenas uma amostra para desenvolvimento - depois usar chunks para produção
    trans_df = pd.read_csv(data_raw_dir / 'HI-Small_Trans.csv', nrows=10000)  # Aumentei para 10k
    print(f"   Transações (amostra): {trans_df.shape[0]:,} linhas, {trans_df.shape[1]} colunas")
    print(f"   Colunas: {list(trans_df.columns)}")
    print(f"   Amostra:")
    print(trans_df.head(3))
except Exception as e:
    print(f"❌ Erro ao carregar transações: {e}")
    sys.exit(1)

# 2. Carregar contas
print("\n2️⃣ Carregando contas...")
try:
    accounts_df = pd.read_csv(data_raw_dir / 'HI-Small_accounts.csv')
    print(f"   Contas: {accounts_df.shape[0]:,} linhas, {accounts_df.shape[1]} colunas")
    print(f"   Colunas: {list(accounts_df.columns)}")
    print(f"   Amostra:")
    print(accounts_df.head(3))
except Exception as e:
    print(f"❌ Erro ao carregar contas: {e}")
    sys.exit(1)

# 3. Parsear padrões
print("\n3️⃣ Parseando padrões de lavagem...")
try:
    with open(data_raw_dir / 'HI-Small_Patterns.txt', 'r') as f:
        patterns_content = f.read()

    print(f"   Arquivo de padrões carregado ({len(patterns_content):,} caracteres)")
    print("   Primeiras 500 caracteres:")
    print(patterns_content[:500])

    # Parse patterns - assuming format is structured
    # This will need adjustment based on actual file format
    pattern_lines = patterns_content.strip().split('\n')
    print(f"   {len(pattern_lines)} linhas de padrão encontradas")

except Exception as e:
    print(f"❌ Erro ao carregar padrões: {e}")
    sys.exit(1)

print("\n🔍 ANÁLISE EXPLORATÓRIA...")

# Análise das transações
print("\n📈 Estatísticas das transações:")
print(trans_df.describe())

# Verificar tipos de dados
print("\n📋 Tipos de dados - Transações:")
print(trans_df.dtypes)

print("\n📋 Tipos de dados - Contas:")
print(accounts_df.dtypes)

# Verificar valores únicos e missing
print("\n🔍 Valores missing:")
print("Transações:")
print(trans_df.isnull().sum())
print("Contas:")
print(accounts_df.isnull().sum())

print("\n✅ FASE 1 - CARREGAMENTO CONCLUÍDO!")
print("📝 PRÓXIMOS PASSOS:")
print("   1. Explorar estrutura dos dados em detalhes")
print("   2. Implementar merge entre transações e contas")
print("   3. Parsear padrões e criar features de risco")
print("   4. Criar agregações temporais")
print("   5. Implementar features de rede")

print("\n🔗 INTEGRAÇÃO DE DADOS - MERGE TRANSAÇÕES + CONTAS")

# Renomear colunas para clareza
trans_df = trans_df.rename(columns={
    'Account': 'From Account',
    'Account.1': 'To Account',
    'From Bank': 'From Bank ID',
    'To Bank': 'To Bank ID'
})

accounts_df = accounts_df.rename(columns={
    'Bank ID': 'Bank ID',
    'Account Number': 'Account Number',
    'Entity ID': 'Entity ID',
    'Entity Name': 'Entity Name',
    'Bank Name': 'Bank Name'
})

print("   Colunas transações renomeadas:", list(trans_df.columns))
print("   Colunas contas renomeadas:", list(accounts_df.columns))

# Merge para conta de origem
print("\n🔗 Merge com conta de origem...")
trans_enriched = trans_df.merge(
    accounts_df,
    left_on=['From Bank ID', 'From Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left',
    suffixes=('', '_from')
)

print(f"   Após merge origem: {trans_enriched.shape[0]:,} linhas, {trans_enriched.shape[1]} colunas")

# Merge para conta de destino
print("\n🔗 Merge com conta de destino...")
trans_enriched = trans_enriched.merge(
    accounts_df,
    left_on=['To Bank ID', 'To Account'],
    right_on=['Bank ID', 'Account Number'],
    how='left',
    suffixes=('', '_to')
)

print(f"   Após merge destino: {trans_enriched.shape[0]:,} linhas, {trans_enriched.shape[1]} colunas")
print("   Novas colunas:", [col for col in trans_enriched.columns if col not in trans_df.columns])

# Verificar qualidade do merge
missing_from = trans_enriched['Entity ID'].isnull().sum()
missing_to = trans_enriched['Entity ID_to'].isnull().sum()
print(f"   Contas origem não encontradas: {missing_from}")
print(f"   Contas destino não encontradas: {missing_to}")

print("\n📊 DATASET ENRIQUECIDO - AMOSTRA:")
print(trans_enriched[['Timestamp', 'From Account', 'To Account', 'Amount Paid', 'Entity Name', 'Entity Name_to', 'Is Laundering']].head())

print("\n🎯 PARSING DE PADRÕES DE LAVAGEM")

# Parsear arquivo de padrões
patterns_lines = patterns_content.strip().split('\n')
print(f"   Total de linhas no arquivo de padrões: {len(patterns_lines)}")

# Identificar seções de padrões
pattern_sections = []
current_section = []
section_header = None

for line in pattern_lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith('BEGIN LAUNDERING ATTEMPT'):
        if current_section:
            pattern_sections.append((section_header, current_section))
        section_header = line
        current_section = []
    else:
        current_section.append(line)

if current_section:
    pattern_sections.append((section_header, current_section))

print(f"   Seções de padrões encontradas: {len(pattern_sections)}")

# Parsear transações fraudulentas
fraudulent_transactions = []

for header, transactions in pattern_sections:
    print(f"\n   📋 Seção: {header}")
    print(f"      Transações: {len(transactions)}")

    for trans in transactions[:3]:  # Mostrar apenas primeiras 3
        parts = trans.split(',')
        if len(parts) >= 6:
            timestamp = parts[0]
            from_bank = parts[1]
            from_account = parts[2]
            to_bank = parts[3]
            to_account = parts[4]
            amount = parts[5]

            fraudulent_transactions.append({
                'timestamp': timestamp,
                'from_bank': from_bank,
                'from_account': from_account,
                'to_bank': to_bank,
                'to_account': to_account,
                'amount': amount,
                'pattern_type': header
            })

print(f"\n   🎯 Total de transações fraudulentas parseadas: {len(fraudulent_transactions)}")

# Criar DataFrame de padrões
patterns_df = pd.DataFrame(fraudulent_transactions)
print("   Colunas dos padrões:", list(patterns_df.columns))
print("   Amostra dos padrões:")
print(patterns_df.head())

# Criar features de risco baseadas nos padrões
print("\n⚠️ CRIANDO FEATURES DE RISCO")

# Criar sets para lookup rápido
fraud_accounts = set()
fraud_account_bank_pairs = set()

for _, row in patterns_df.iterrows():
    # Adicionar contas individuais
    fraud_accounts.add(row['from_account'])
    fraud_accounts.add(row['to_account'])

    # Adicionar pares conta-banco
    fraud_account_bank_pairs.add((row['from_account'], str(row['from_bank'])))
    fraud_account_bank_pairs.add((row['to_account'], str(row['to_bank'])))

print(f"   Contas fraudulentas únicas: {len(fraud_accounts)}")
print(f"   Pares conta-banco fraudulentos: {len(fraud_account_bank_pairs)}")

# Features eficientes
trans_enriched['from_account_risk'] = trans_enriched.apply(
    lambda row: (row['From Account'], str(row['From Bank ID'])) in fraud_account_bank_pairs,
    axis=1
)

trans_enriched['to_account_risk'] = trans_enriched.apply(
    lambda row: (row['To Account'], str(row['To Bank ID'])) in fraud_account_bank_pairs,
    axis=1
)

trans_enriched['transaction_risk'] = trans_enriched['from_account_risk'] | trans_enriched['to_account_risk']

print("   Features de risco criadas:")
print(f"   - from_account_risk: {trans_enriched['from_account_risk'].sum()} contas origem em risco")
print(f"   - to_account_risk: {trans_enriched['to_account_risk'].sum()} contas destino em risco")
print(f"   - transaction_risk: {trans_enriched['transaction_risk'].sum()} transações em risco")

print("\n📊 DATASET FINAL - FEATURES CRIADAS:")
print(f"   Shape: {trans_enriched.shape[0]:,} linhas, {trans_enriched.shape[1]} colunas")
print("   Colunas finais:", list(trans_enriched.columns))