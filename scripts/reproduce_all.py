#!/usr/bin/env python3
"""
SCRIPT DE REPRODUTIBILIDADE COMPLETA
Sistema de Detecção de Lavagem de Dinheiro - AML

Este script permite reproduzir todos os resultados da validação
desde o início até o modelo final otimizado.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Executa um comando e verifica se foi bem-sucedido"""
    print(f"\n🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Falhou: {e}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        return False

def main():
    print("🔄 INICIANDO REPRODUTIBILIDADE COMPLETA")
    print("=" * 60)

    # Verificar ambiente
    print("\n🔍 VERIFICANDO AMBIENTE...")

    # Verificar Python
    if not run_command("python --version", "Verificando Python"):
        return False

    # Verificar dependências
    try:
        import pandas, numpy, sklearn, xgboost, lightgbm
        print("   ✅ Dependências principais OK")
    except ImportError as e:
        print(f"   ❌ Dependência faltando: {e}")
        return False

    # Criar diretórios necessários
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Pipeline de reprodutibilidade
    steps = [
        {
            "command": "python scripts/data_parity_check.py",
            "description": "FASE 1: Verificação de Paridade de Dados"
        },
        {
            "command": "python scripts/gnn_diagnostic.py",
            "description": "FASE 2: Diagnóstico do GNN Benchmark"
        },
        {
            "command": "python scripts/temporal_validation.py",
            "description": "FASE 3: Validação Temporal"
        },
        {
            "command": "python scripts/fine_tuning.py",
            "description": "FASE 4: Fine-tuning com Optuna"
        },
        {
            "command": "python scripts/shap_analysis.py",
            "description": "FASE 5: Análise SHAP de Interpretabilidade"
        },
        {
            "command": "python scripts/robustness_validation.py",
            "description": "FASE 6: Validação de Robustez"
        }
    ]

    success_count = 0

    for step in steps:
        if run_command(step["command"], step["description"]):
            success_count += 1
        else:
            print(f"\n❌ Pipeline interrompido na etapa: {step['description']}")
            break

    print(f"\n📊 RESULTADO DA REPRODUTIBILIDADE:")
    print(f"   • Etapas executadas com sucesso: {success_count}/{len(steps)}")

    if success_count == len(steps):
        print("   ✅ REPRODUTIBILIDADE COMPLETA VERIFICADA!")
        return True
    else:
        print("   ⚠️ REPRODUTIBILIDADE PARCIAL")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
