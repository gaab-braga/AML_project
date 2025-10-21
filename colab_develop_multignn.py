#!/usr/bin/env python3
"""
Script Autônomo para Desenvolvimento do Multi-GNN no Google Colab

Este script automatiza todo o processo:
1. Instalação do ambiente com pip.
2. Download dos dados do Kaggle.
3. Clone do repositório IBM/multi-gnn.
4. Modificação do código para salvar predições.
5. Execução do pipeline de treinamento.
"""

import os
import sys
import subprocess
import json
import shutil
import argparse
from pathlib import Path

def run_command(command, cwd=None):
    """Executa um comando no shell e lida com erros."""
    print(f"🚀 Executando: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERRO ao executar comando: {' '.join(command)}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

class ColabMultiGNNBenchmark:
    """Gerencia o pipeline de benchmark do Multi-GNN no Colab."""

    def __init__(self):
        self.workdir = Path("/content/aml_project")
        self.repo_path = self.workdir / "multi-gnn"
        self.data_path = self.workdir / "data"
        self.raw_data_path = self.data_path / "raw"
        self.processed_data_path = self.data_path / "processed" / "Small_HI"
        self.kaggle_data_path = self.raw_data_path / "ibm-transactions-for-anti-money-laundering-aml"

        # Criar diretórios
        for path in [self.workdir, self.raw_data_path, self.processed_data_path]:
            path.mkdir(parents=True, exist_ok=True)

    def setup_environment(self):
        """Fase 1: Instalar dependências com pip."""
        print("🔧 FASE 1: Instalando ambiente...")

        commands = [
            ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
            ["pip", "install", "torch-scatter", "torch-sparse", "-f", "https://data.pyg.org/whl/torch-2.1.0+cu118.html"],
            ["pip", "install", "torch-geometric"],
            ["pip", "install", "datatable", "wandb", "tqdm", "scikit-learn", "pandas", "numpy", "munch"]
        ]

        for cmd in commands:
            if not run_command(cmd):
                return False
        return True

    def download_data(self, sample_size=None):
        """Fase 2: Baixar dados do Kaggle."""
        print("📊 FASE 2: Baixando dados do Kaggle...")

        # Configurar Kaggle API
        if not Path("/root/.kaggle/kaggle.json").exists():
            print("❌ ERRO: Arquivo kaggle.json não encontrado.")
            print("   Faça o upload do seu arquivo kaggle.json para a sessão do Colab.")
            return False

        run_command(["chmod", "600", "/root/.kaggle/kaggle.json"])

        # Baixar e descompactar
        dataset = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
        if not run_command(["kaggle", "datasets", "download", "-d", dataset, "-p", str(self.raw_data_path)]):
            return False
        if not run_command(["unzip", "-o", str(self.raw_data_path / f"{dataset.split('/')[1]}.zip"), "-d", str(self.kaggle_data_path)]):
            return False

        # Criar sample se solicitado
        if sample_size:
            print(f"🎯 Criando sample de {sample_size} registros...")
            import pandas as pd
            csv_file = self.kaggle_data_path / "HI-Small_Trans.csv"
            df = pd.read_csv(csv_file)
            df_sample = df.sample(n=sample_size, random_state=42)
            df_sample.to_csv(csv_file, index=False)
            print(f"✅ Sample criado: {len(df_sample)} registros")

        print("✅ Dados baixados com sucesso.")
        return True

    def clone_and_patch_repo(self):
        """Fase 3: Clonar e modificar o repositório."""
        print("🔧 FASE 3: Clonando e modificando o repositório...")

        # Clonar
        if not run_command(["git", "clone", "https://github.com/ibm/multi-gnn.git"], cwd=self.workdir):
            return False

        # Modificar train_util.py
        train_util_file = self.repo_path / "train_util.py"
        with open(train_util_file, 'r') as f:
            content = f.read()

        if "--- INÍCIO DA MODIFICAÇÃO ---" in content:
            print("✅ Repositório já modificado.")
            return True

        # Código de modificação
        modification_code = '''
    # --- INÍCIO DA MODIFICAÇÃO ---
    import pandas as pd
    import os
    if args.save_model:
        print("💾 Salvando predições do conjunto de teste...")
        pred_numpy = torch.cat(preds, dim=0).cpu().numpy()
        gt_numpy = torch.cat(ground_truths, dim=0).cpu().numpy()
        output_df = pd.DataFrame({'prediction_prob': pred_numpy, 'ground_truth': gt_numpy})
        output_df.to_csv("/content/multi_gnn_predictions.csv", index=False)
        print(f"✅ Predições salvas em /content/multi_gnn_predictions.csv")
    # --- FIM DA MODIFICAÇÃO ---
'''
        # Inserir antes do `return f1` na função `evaluate_homo`
        content = content.replace("    return f1", f"{modification_code}\n    return f1")

        with open(train_util_file, 'w') as f:
            f.write(content)

        print("✅ Repositório modificado com sucesso.")
        return True

    def run_pipeline(self, epochs=50, sample_size=None):
        """Fase 4: Executar o pipeline de dados e treinamento."""
        print("🚀 FASE 4: Executando pipeline...")

        # Formatar dados
        kaggle_csv = self.kaggle_data_path / "HI-Small_Trans.csv"
        if not run_command(["python", "format_kaggle_files.py", str(kaggle_csv)], cwd=self.repo_path):
            return False

        # Mover arquivo formatado
        shutil.move(
            str(self.kaggle_data_path / "formatted_transactions.csv"),
            str(self.processed_data_path / "formatted_transactions.csv")
        )

        # Atualizar data_config.json
        config_file = self.repo_path / "data_config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        config["paths"]["aml_data"] = str(self.data_path / "processed")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Treinar
        train_cmd = [
            "python", "main.py",
            "--data", "Small_HI",
            "--model", "gin",
            "--emlps", "--reverse_mp", "--ego", "--ports",
            "--testing",
            "--n_epochs", str(epochs),
            "--batch_size", "2048",
            "--save_model",
            "--unique_name", "colab_benchmark"
        ]
        if not run_command(train_cmd, cwd=self.repo_path):
            return False

        print("✅ Pipeline executado com sucesso.")
        return True

def main():
    """Orquestra todas as fases."""
    parser = argparse.ArgumentParser(description='Multi-GNN Benchmark Development')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Tamanho da amostra para teste (opcional)')
    args = parser.parse_args()

    benchmark = ColabMultiGNNBenchmark()

    phases = [
        ("Configuração do Ambiente", benchmark.setup_environment),
        ("Download dos Dados", lambda: benchmark.download_data(sample_size=args.sample_size)),
        ("Clone e Modificação do Repositório", benchmark.clone_and_patch_repo),
        ("Execução do Pipeline", lambda: benchmark.run_pipeline(sample_size=args.sample_size))
    ]

    for name, func in phases:
        if not func():
            print(f"❌ FALHA na fase: {name}")
            sys.exit(1)

    print("\n🎉 PROCESSO CONCLUÍDO! 🎉")
    print("📄 O arquivo 'multi_gnn_predictions.csv' está pronto no diretório /content/.")

if __name__ == "__main__":
    main()