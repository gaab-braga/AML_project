import subprocess
import os
import time
from pathlib import Path

# --- Configuração do Benchmark ---
BENCHMARK_REPO_PATH = Path("../benchmarks") / "Multi-GNN"
CONDA_ENV_NAME = "multignn"  # Nome do ambiente criado via env.yml
DATA_NAME = "Small_HI"  # Nome do dataset conforme data_config.json
MODEL_NAME = "gin"  # Modelo GNN: gin, gat, rgcn, pna

print("🚀 INICIANDO EXECUÇÃO DO BENCHMARK MULTI-GNN DA IBM")
print("=" * 55)

# Verificar se ambiente existe
try:
    result = subprocess.run(
        ["C:\\Users\\gafeb\\Miniconda3\\Scripts\\conda.exe", "env", "list"],
        capture_output=True, text=True, timeout=10
    )
    if CONDA_ENV_NAME not in result.stdout:
        print(f"❌ ERRO: Ambiente conda '{CONDA_ENV_NAME}' não encontrado!")
        print("   Execute: conda env create -f benchmarks/Multi-GNN/env.yml")
        raise SystemExit(1)
    else:
        print(f"✅ Ambiente conda '{CONDA_ENV_NAME}' encontrado")
except subprocess.TimeoutExpired:
    print("⚠️  Timeout ao verificar ambiente conda - continuando...")

# Verificar arquivos necessários
required_files = [
    Path("benchmarks/Multi-GNN/main.py"),
    Path("benchmarks/Multi-GNN/data_config.json"),
    Path("data/processed/formatted_transactions.csv"),
    Path("data/raw/HI-Small_accounts.csv"),
    Path("data/raw/HI-Small_Patterns.txt")
]

missing_files = [f for f in required_files if not f.exists()]
if missing_files:
    print("❌ ERROS NOS ARQUIVOS NECESSÁRIOS:")
    for f in missing_files:
        print(f"   • {f} - NÃO ENCONTRADO")
    raise SystemExit(1)
else:
    print("✅ Todos os arquivos necessários encontrados")

# --- Comando para treinar o modelo de benchmark ---
# Usando --testing para evitar problemas de wandb
# Usando batch_size menor para evitar problemas de memória
command = [
    "C:\\Users\\gafeb\\Miniconda3\\Scripts\\conda.exe", "run", "-n", CONDA_ENV_NAME,
    "python", "benchmarks/Multi-GNN/main.py",
    "--data", DATA_NAME,
    "--model", MODEL_NAME,
    "--testing",  # Desabilita wandb
    "--batch_size", "1024",  # Batch menor para estabilidade
    "--n_epochs", "2"  # Apenas algumas épocas para teste rápido
]

print(f"\n🔧 Comando a executar:")
print(f"   {' '.join(command)}")
print(f"\n⏳ Executando benchmark (timeout: 300s)...")

start_time = time.time()

# --- Executando o processo externo ---
try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300  # 5 minutos timeout
    )

    execution_time = time.time() - start_time

    # --- Verificando os Resultados ---
    if result.returncode != 0:
        print(f"\n❌ ERRO APÓS {execution_time:.1f}s:")
        print("STDOUT (últimas 20 linhas):")
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-20:]:
            if line.strip():
                print(f"   {line}")

        print("\nSTDERR:")
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-10:]:
            if line.strip():
                print(f"   {line}")

        # Verificar erros comuns
        if "torch-sparse" in result.stderr or "torch_scatter" in result.stderr:
            print("\n🔧 DIAGNÓSTICO:")
            print("   Problema identificado: PyTorch Geometric incompatível com Windows")
            print("   Soluções:")
            print("   1. Usar WSL2: wsl --install")
            print("   2. Usar Docker com Linux")
            print("   3. Modificar código para CPU-only")

        elif "api_key not configured" in result.stderr:
            print("\n🔧 DIAGNÓSTICO:")
            print("   Problema: Weights & Biases requer API key")
            print("   Solução: Adicione --testing ao comando")

        elif "ImportError" in result.stderr:
            print("\n🔧 DIAGNÓSTICO:")
            print("   Problema: Biblioteca não encontrada")
            print("   Solução: Verifique instalação do ambiente conda")

    else:
        print(f"\n✅ BENCHMARK EXECUTADO COM SUCESSO EM {execution_time:.1f}s!")
        print("\n📊 RESULTADOS (últimas 10 linhas):")
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-10:]:
            if line.strip():
                print(f"   {line}")

except subprocess.TimeoutExpired:
    execution_time = time.time() - start_time
    print(f"\n⏰ TIMEOUT APÓS {execution_time:.1f}s!")
    print("   O benchmark demorou mais que 5 minutos.")
    print("   Isso pode indicar problemas de performance ou configuração.")

except Exception as e:
    execution_time = time.time() - start_time
    print(f"\n💥 ERRO INESPERADO APÓS {execution_time:.1f}s: {e}")

print(f"\n🏁 EXECUÇÃO FINALIZADA EM {time.time() - start_time:.1f}s")