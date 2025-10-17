#!/usr/bin/env python3
"""
FASE 1: REVISÃO DE PARIDADE DE DADOS
Script para verificar se modelos tabulares e GNN usam os mesmos dados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import json
from datetime import datetime

def load_tabular_data():
    """Carrega dados processados usados pelos modelos tabulares"""
    print("📊 CARREGANDO DADOS TABULARES...")

    artifacts_dir = Path('artifacts')  # Caminho relativo correto
    features_pkl = artifacts_dir / 'features_processed.pkl'

    if not features_pkl.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {features_pkl}")

    df = pd.read_pickle(features_pkl)

    # Separar features e target
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)

    # Selecionar apenas colunas numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    return X, y, df

def load_gnn_data():
    """Carrega dados brutos usados pelo GNN"""
    print("🧠 CARREGANDO DADOS DO GNN...")

    data_dir = Path('data')  # Caminho relativo correto

    # Carregar transações
    trans_path = data_dir / 'processed' / 'formatted_transactions.csv'
    if trans_path.exists():
        df_trans = pd.read_csv(trans_path)
        print(f"   ✅ Transações: {df_trans.shape[0]:,} linhas")
    else:
        df_trans = None
        print("   ❌ Transações não encontradas")

    # Carregar contas
    accounts_path = data_dir / 'raw' / 'HI-Small_accounts.csv'
    if accounts_path.exists():
        df_accounts = pd.read_csv(accounts_path)
        print(f"   ✅ Contas: {df_accounts.shape[0]:,} linhas")
    else:
        df_accounts = None
        print("   ❌ Contas não encontradas")

    # Carregar padrões
    patterns_path = data_dir / 'raw' / 'HI-Small_Patterns.txt'
    if patterns_path.exists():
        with open(patterns_path, 'r') as f:
            patterns_content = f.read()
        print(f"   ✅ Padrões: {len(patterns_content.split('BEGIN LAUNDERING ATTEMPT'))-1} padrões")
    else:
        patterns_content = None
        print("   ❌ Padrões não encontrados")

    return df_trans, df_accounts, patterns_content

def compare_data_integrity(X_tabular, y_tabular, df_gnn_trans):
    """Compara integridade dos dados entre tabulares e GNN"""
    print("\n🔍 COMPARAÇÃO DE INTEGRIDADE...")

    # Verificar se target existe nos dados do GNN
    if df_gnn_trans is not None and 'Is Laundering' in df_gnn_trans.columns:
        y_gnn = (df_gnn_trans['Is Laundering'] == 1)
        print(f"   🎯 Target GNN: {y_gnn.sum()} positivos ({y_gnn.mean():.3%})")
        print(f"   🎯 Target Tabular: {y_tabular.sum()} positivos ({y_tabular.mean():.3%})")

        if y_gnn.sum() == y_tabular.sum():
            print("   ✅ Número de casos positivos idêntico!")
        else:
            print(f"   ⚠️ Diferença no número de positivos: GNN={y_gnn.sum()}, Tabular={y_tabular.sum()}")

        # Verificar se as transações são as mesmas (comparar hashes de IDs)
        if 'Account' in df_gnn_trans.columns and 'Account.1' in df_gnn_trans.columns:
            # Criar identificador único de transação
            trans_ids_gnn = (df_gnn_trans['Account'].astype(str) + '_' +
                           df_gnn_trans['Account.1'].astype(str) + '_' +
                           df_gnn_trans['Timestamp'].astype(str)).values

            # Para dados tabulares, seria necessário ter os IDs originais
            # Por enquanto, apenas verificar tamanhos
            print(f"   📊 Transações GNN: {len(trans_ids_gnn)}")
            print(f"   📊 Amostras tabulares: {len(X_tabular)}")

    else:
        print("   ⚠️ Target não encontrado nos dados do GNN")

def generate_data_report(X_tabular, y_tabular, df_gnn_trans, df_accounts, patterns):
    """Gera relatório completo da paridade de dados"""
    print("\n📋 GERANDO RELATÓRIO DE PARIDADE...")

    report = {
        'timestamp': datetime.now().isoformat(),
        'tabular_data': {
            'samples': len(X_tabular),
            'features': X_tabular.shape[1],
            'feature_names': list(X_tabular.columns),
            'positive_cases': int(y_tabular.sum()),
            'fraud_rate': float(y_tabular.mean()),
            'data_hash': hashlib.sha256(X_tabular.values.tobytes()).hexdigest(),
            'target_hash': hashlib.sha256(y_tabular.values.tobytes()).hexdigest()
        },
        'gnn_data': {
            'transactions_available': df_gnn_trans is not None,
            'accounts_available': df_accounts is not None,
            'patterns_available': patterns is not None
        }
    }

    if df_gnn_trans is not None:
        report['gnn_data'].update({
            'transaction_samples': len(df_gnn_trans),
            'transaction_columns': list(df_gnn_trans.columns),
            'has_target': 'Is Laundering' in df_gnn_trans.columns
        })

        if 'Is Laundering' in df_gnn_trans.columns:
            report['gnn_data'].update({
                'positive_cases': int((df_gnn_trans['Is Laundering'] == 1).sum()),
                'fraud_rate': float((df_gnn_trans['Is Laundering'] == 1).mean())
            })

    if df_accounts is not None:
        report['gnn_data'].update({
            'accounts_samples': len(df_accounts),
            'account_columns': list(df_accounts.columns)
        })

    if patterns is not None:
        patterns_count = len(patterns.split('BEGIN LAUNDERING ATTEMPT')) - 1
        report['gnn_data']['patterns_count'] = patterns_count

    # Verificar paridade
    report['parity_check'] = {
        'sample_count_match': (report['tabular_data']['samples'] == report['gnn_data'].get('transaction_samples', 0)),
        'positive_cases_match': (report['tabular_data']['positive_cases'] == report['gnn_data'].get('positive_cases', 0)),
        'target_available': report['gnn_data'].get('has_target', False)
    }

    # Salvar relatório
    report_path = Path('artifacts/data_parity_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"   💾 Relatório salvo: {report_path}")

    return report

def main():
    print("🔍 FASE 1: REVISÃO DE PARIDADE DE DADOS")
    print("=" * 60)

    try:
        # Carregar dados
        X_tabular, y_tabular, df_tabular = load_tabular_data()
        df_gnn_trans, df_accounts, patterns = load_gnn_data()

        # Comparar integridade
        compare_data_integrity(X_tabular, y_tabular, df_gnn_trans)

        # Gerar relatório
        report = generate_data_report(X_tabular, y_tabular, df_gnn_trans, df_accounts, patterns)

        # Resumo final
        print("\n🎯 RESUMO DA PARIDADE:")
        parity = report['parity_check']

        if parity['sample_count_match']:
            print("   ✅ Contagem de amostras idêntica")
        else:
            print("   ⚠️ Contagem de amostras diferente")

        if parity['positive_cases_match']:
            print("   ✅ Número de casos positivos idêntico")
        else:
            print("   ⚠️ Número de casos positivos diferente")

        if parity['target_available']:
            print("   ✅ Target disponível nos dados do GNN")
        else:
            print("   ❌ Target não disponível nos dados do GNN")

        print("\n✅ FASE 1 CONCLUÍDA!")
        print("📋 Próximo: Revisar pipeline do GNN para garantir uso dos mesmos dados")

    except Exception as e:
        print(f"❌ ERRO na Fase 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()