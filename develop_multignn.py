#!/usr/bin/env python3
"""
Script de Desenvolvimento Completo do Multi-GNN para Benchmark AML

Baseado no roadmap do notebook 04_MultiGNN_Benchmark_Development.ipynb
Implementa todas as fases necessárias para treinar o Multi-GNN e gerar artefatos de benchmark.
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path

class MultiGNNBenchmark:
    """Classe principal para gerenciar o desenvolvimento do Multi-GNN"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or Path(__file__).parent)
        self.multignn_path = self.project_root / "benchmarks" / "Multi-GNN"
        self.data_raw_path = self.project_root / "data" / "raw"
        self.data_processed_path = self.project_root / "data" / "processed" / "multi-gnn"
        self.models_path = self.project_root / "models"
        self.results_path = self.project_root / "results" / "multi-gnn"

        # Criar diretórios necessários
        for path in [self.data_processed_path, self.models_path, self.results_path]:
            path.mkdir(parents=True, exist_ok=True)

    def check_environment(self):
        """Fase 0: Verificar ambiente WSL2/Conda"""
        print("🔧 FASE 0: Verificando ambiente...")

        # Verificar WSL2
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' not in f.read().lower():
                    print("⚠️  AVISO: Não detectado WSL2. Recomenda-se usar WSL2 para evitar problemas.")
                    return False
        except:
            print("❌ ERRO: Este script deve ser executado no WSL2/Linux.")
            return False

        # Verificar ambiente conda (mais flexível)
        try:
            # Tentar múltiplos caminhos comuns do conda
            conda_paths = [
                'conda',
                '/opt/conda/bin/conda',
                '/home/user/miniconda3/bin/conda',
                '/home/user/anaconda3/bin/conda',
                '/usr/local/miniconda3/bin/conda',
                '/usr/local/anaconda3/bin/conda'
            ]

            conda_found = False
            for conda_path in conda_paths:
                try:
                    result = subprocess.run([conda_path, 'env', 'list'],
                                          capture_output=True, text=True,
                                          timeout=5)
                    if 'multignn' in result.stdout:
                        conda_found = True
                        break
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue

            if not conda_found:
                print("⚠️  AVISO: Ambiente 'multignn' não encontrado via conda.")
                print("   Certifique-se de que está executando em um ambiente com PyTorch Geometric.")
                print("   Continuando mesmo assim...")
                return True  # Permitir continuação

        except Exception as e:
            print(f"⚠️  AVISO: Erro na verificação do conda: {e}")
            print("   Continuando mesmo assim...")
            return True

        print("✅ Ambiente OK")
        return True

    def prepare_data(self, kaggle_file=None):
        """Fase 1: Preparar dados"""
        print("📊 FASE 1: Preparando dados...")

        # Verificar arquivo de entrada
        if kaggle_file:
            input_file = Path(kaggle_file)
        else:
            input_file = self.data_raw_path / "HI-Small_Trans.csv"

        if not input_file.exists():
            print(f"❌ ERRO: Arquivo {input_file} não encontrado.")
            print("   Baixe os dados do Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data")
            return False

        # Criar diretório para dados processados
        small_hi_dir = self.data_processed_path / "Small_HI"
        small_hi_dir.mkdir(exist_ok=True)

        # Executar formatação
        cmd = [
            'python3',
            str(self.multignn_path / 'format_kaggle_files.py'),
            str(input_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.project_root)
            print("✅ Dados formatados com sucesso")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERRO na formatação: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False

        # Mover arquivo formatado
        formatted_file = input_file.parent / "formatted_transactions.csv"
        if formatted_file.exists():
            target_file = small_hi_dir / "formatted_transactions.csv"
            shutil.move(str(formatted_file), str(target_file))
            print(f"✅ Arquivo movido para: {target_file}")

        # Configurar data_config.json
        config_file = self.multignn_path / "data_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Atualizar caminhos relativos - absolutos para execução da pasta Multi-GNN
            project_root_relative = Path("../../..")
            config["paths"]["aml_data"] = str(project_root_relative / "data" / "processed" / "multi-gnn" / "Small_HI")
            config["model_to_save"] = str(project_root_relative / "models")
            config["model_to_load"] = str(project_root_relative / "models")
            config["results_dir"] = str(project_root_relative / "results" / "multi-gnn")

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print("✅ data_config.json atualizado")

        return True

    def modify_train_util(self):
        """Fase 4: Modificar train_util.py para salvar predições"""
        print("🔧 FASE 4: Modificando train_util.py para salvar predições...")

        train_util_file = self.multignn_path / "train_util.py"

        if not train_util_file.exists():
            print(f"❌ ERRO: Arquivo {train_util_file} não encontrado")
            return False

        # Ler conteúdo atual
        with open(train_util_file, 'r') as f:
            content = f.read()

        # Verificar se já foi modificado
        if "# --- INÍCIO DA MODIFICAÇÃO PARA SALVAR PREDIÇÕES ---" in content:
            print("✅ train_util.py já foi modificado anteriormente")
            return True

        # Encontrar a função evaluate_homo
        import re
        pattern = r'(def evaluate_homo\(.*?\):.*?)(\n    return f1\n)'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print("❌ ERRO: Não foi possível encontrar a função evaluate_homo")
            return False

        # Código a ser inserido
        modification_code = '''
    # --- INÍCIO DA MODIFICAÇÃO PARA SALVAR PREDIÇÕES ---
    import pandas as pd
    import os

    # Salvar apenas durante a avaliação final (não na validação)
    # Usamos uma heurística baseada no tamanho do batch
    is_test_evaluation = len(preds) > 1000  # Heurística simples

    if is_test_evaluation and hasattr(args, 'save_model') and args.save_model:
        print("💾 Salvando predições do conjunto de teste...")

        # Combinar todas as predições e ground truths
        all_preds = torch.cat(preds, dim=0).cpu().numpy()
        all_ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

        # Criar DataFrame com predições
        output_df = pd.DataFrame({
            'prediction_prob': all_preds[:, 1] if len(all_preds.shape) > 1 else all_preds,  # Probabilidade da classe 1
            'ground_truth': all_ground_truths
        })

        output_path = os.path.join(os.getcwd(), "multi_gnn_predictions.csv")
        output_df.to_csv(output_path, index=False)
        print(f"✅ Predições salvas em: {output_path}")
    # --- FIM DA MODIFICAÇÃO ---
'''

        # Inserir modificação antes do return
        modified_content = content.replace(match.group(2), modification_code + match.group(2))

        # Salvar arquivo modificado
        with open(train_util_file, 'w') as f:
            f.write(modified_content)

        print("✅ train_util.py modificado com sucesso")
        return True

    def train_model(self, model='gin', epochs=5, save_model=False):
        """Fase 3/5: Executar treinamento"""
        print(f"🚀 FASE 5: Executando treinamento do {model.upper()}...")

        cmd = [
            'python3', str(self.multignn_path / 'main.py'),
            '--data', 'Small_HI',
            '--model', model,
            '--emlps', '--reverse_mp', '--ego', '--ports', '--tds',
            '--testing',
            '--n_epochs', str(epochs),
            '--batch_size', '1024'
        ]

        if save_model:
            cmd.extend(['--save_model', '--unique_name', 'aml_benchmark'])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=self.multignn_path)
            print("✅ Treinamento concluído com sucesso")
            print("\n📊 RESULTADOS:")
            # Mostrar últimas linhas do output
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ERRO no treinamento: {e}")
            print(f"STDOUT: {e.stdout[-1000:]}")  # Últimos 1000 chars
            print(f"STDERR: {e.stderr[-1000:]}")
            return False

    def run_all_phases(self, kaggle_file=None, model='gin', epochs=50, save_model=True):
        """Executar todas as fases em sequência"""
        print("🤖 EXECUTANDO TODAS AS FASES DO MULTI-GNN BENCHMARK")
        print("=" * 60)

        phases = [
            ("Verificação de Ambiente", self.check_environment),
            ("Preparação de Dados", lambda: self.prepare_data(kaggle_file)),
            ("Modificação do Código", self.modify_train_util),
            ("Treinamento do Modelo", lambda: self.train_model(model, epochs, save_model))
        ]

        for phase_name, phase_func in phases:
            print(f"\n📍 {phase_name}")
            if not phase_func():
                print(f"❌ FALHA na fase: {phase_name}")
                return False

        print("\n🎉 TODAS AS FASES CONCLUÍDAS COM SUCESSO!")
        print("📄 Verifique o arquivo 'multi_gnn_predictions.csv' para as predições.")
        return True

def main():
    parser = argparse.ArgumentParser(description='Desenvolvimento Completo do Multi-GNN para Benchmark AML')
    parser.add_argument('--phase', choices=['check', 'data', 'modify', 'train', 'all'],
                       default='all', help='Fase específica a executar')
    parser.add_argument('--kaggle-file', help='Caminho para o arquivo CSV do Kaggle')
    parser.add_argument('--model', default='gin', choices=['gin', 'gat', 'pna', 'rgcn'],
                       help='Modelo GNN a usar')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas para treinamento')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Salvar modelo e predições')
    parser.add_argument('--project-root', help='Diretório raiz do projeto')

    args = parser.parse_args()

    benchmark = MultiGNNBenchmark(args.project_root)

    if args.phase == 'check':
        success = benchmark.check_environment()
    elif args.phase == 'data':
        success = benchmark.prepare_data(args.kaggle_file)
    elif args.phase == 'modify':
        success = benchmark.modify_train_util()
    elif args.phase == 'train':
        success = benchmark.train_model(args.model, args.epochs, args.save_model)
    elif args.phase == 'all':
        success = benchmark.run_all_phases(args.kaggle_file, args.model, args.epochs, args.save_model)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()