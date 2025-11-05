"""
Benchmark Comparison Tool
=========================
Compara resultados entre implementação local (CPU) e Kaggle (Multi-GPU).

Uso:
    python compare_results.py --local ../artifacts/gnn_results.json --kaggle ./kaggle_results/gnn_results_multigpu.json
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(path: Path) -> Dict[str, Any]:
    """Carrega arquivo de resultados JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def compare_metrics(local_results: Dict, kaggle_results: Dict) -> pd.DataFrame:
    """Compara métricas entre as duas implementações."""
    metrics = ['roc_auc', 'pr_auc', 'precision@100', 'precision@500', 
               'precision@1000', 'recall@100', 'recall@500', 'recall@1000']
    
    comparison = []
    for metric in metrics:
        local_val = local_results.get(metric, 0)
        kaggle_val = kaggle_results.get(metric, 0)
        diff = kaggle_val - local_val
        diff_pct = (diff / local_val * 100) if local_val > 0 else 0
        
        comparison.append({
            'Metric': metric,
            'Local (CPU)': f"{local_val:.4f}",
            'Kaggle (2x T4)': f"{kaggle_val:.4f}",
            'Difference': f"{diff:+.4f}",
            'Difference (%)': f"{diff_pct:+.2f}%"
        })
    
    return pd.DataFrame(comparison)


def compare_hyperparameters(local_results: Dict, kaggle_results: Dict) -> pd.DataFrame:
    """Compara hiperparâmetros encontrados."""
    local_params = local_results.get('best_params', {})
    kaggle_params = kaggle_results.get('best_params', {})
    
    all_params = set(local_params.keys()) | set(kaggle_params.keys())
    
    comparison = []
    for param in sorted(all_params):
        comparison.append({
            'Hyperparameter': param,
            'Local': local_params.get(param, 'N/A'),
            'Kaggle': kaggle_params.get(param, 'N/A')
        })
    
    return pd.DataFrame(comparison)


def plot_comparison(df_metrics: pd.DataFrame, output_path: Path):
    """Gera gráfico de comparação."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Converter strings de volta para float
    metrics = df_metrics['Metric'].values
    local_vals = [float(v) for v in df_metrics['Local (CPU)'].values]
    kaggle_vals = [float(v) for v in df_metrics['Kaggle (2x T4)'].values]
    
    x = range(len(metrics))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], local_vals, width, label='Local (CPU)', alpha=0.8)
    ax.bar([i + width/2 for i in x], kaggle_vals, width, label='Kaggle (2x T4)', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Local vs. Kaggle Multi-GPU Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")


def generate_report(local_results: Dict, kaggle_results: Dict, output_dir: Path):
    """Gera relatório completo de comparação."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("BENCHMARK COMPARISON REPORT")
    print("="*80)
    
    # Comparação de métricas
    print("\n📊 PERFORMANCE METRICS")
    print("-"*80)
    df_metrics = compare_metrics(local_results, kaggle_results)
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    
    # Comparação de hiperparâmetros
    print("\n⚙️  HYPERPARAMETERS")
    print("-"*80)
    df_params = compare_hyperparameters(local_results, kaggle_results)
    print(df_params.to_string(index=False))
    df_params.to_csv(output_dir / 'hyperparameters_comparison.csv', index=False)
    
    # Informações adicionais
    print("\n🔧 IMPLEMENTATION DETAILS")
    print("-"*80)
    print(f"Local Training Method: {local_results.get('training_method', 'N/A')}")
    print(f"Kaggle Training Method: {kaggle_results.get('training_method', 'N/A')}")
    print(f"Kaggle GPUs: {kaggle_results.get('num_gpus', 'N/A')}")
    
    kaggle_opts = kaggle_results.get('optimizations', [])
    if kaggle_opts:
        print(f"\nKaggle Optimizations:")
        for opt in kaggle_opts:
            print(f"  ✓ {opt}")
    
    # Gerar gráfico
    print("\n📈 Gerando gráfico de comparação...")
    plot_comparison(df_metrics, output_dir / 'comparison_plot.png')
    
    # Análise estatística
    print("\n📐 STATISTICAL ANALYSIS")
    print("-"*80)
    
    local_pr_auc = local_results.get('pr_auc', 0)
    kaggle_pr_auc = kaggle_results.get('pr_auc', 0)
    
    diff_abs = kaggle_pr_auc - local_pr_auc
    diff_pct = (diff_abs / local_pr_auc * 100) if local_pr_auc > 0 else 0
    
    print(f"PR-AUC Difference: {diff_abs:+.4f} ({diff_pct:+.2f}%)")
    
    if abs(diff_pct) < 0.5:
        print("✓ Resultados praticamente idênticos (< 0.5% diferença)")
        print("  As otimizações não afetaram a qualidade do modelo.")
    elif abs(diff_pct) < 2.0:
        print("⚠ Pequena diferença detectada (< 2%)")
        print("  Provavelmente devido a precisão numérica (float16 vs. float32).")
    else:
        print("⚠️ Diferença significativa detectada (> 2%)")
        print("  Recomenda-se investigar possíveis problemas na implementação.")
    
    # Salvar relatório completo
    report_text = f"""
BENCHMARK COMPARISON REPORT
===========================

Generated: {pd.Timestamp.now()}

METRICS COMPARISON
------------------
{df_metrics.to_string(index=False)}

HYPERPARAMETERS COMPARISON
--------------------------
{df_params.to_string(index=False)}

IMPLEMENTATION DETAILS
----------------------
Local Training Method: {local_results.get('training_method', 'N/A')}
Kaggle Training Method: {kaggle_results.get('training_method', 'N/A')}
Kaggle GPUs: {kaggle_results.get('num_gpus', 'N/A')}

Kaggle Optimizations:
{chr(10).join(['  ✓ ' + opt for opt in kaggle_opts])}

STATISTICAL ANALYSIS
--------------------
PR-AUC Difference: {diff_abs:+.4f} ({diff_pct:+.2f}%)

Conclusion:
{_get_conclusion(diff_pct)}
"""
    
    with open(output_dir / 'full_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n{'='*80}")
    print(f"Relatório completo salvo em: {output_dir / 'full_report.txt'}")
    print(f"{'='*80}")


def _get_conclusion(diff_pct: float) -> str:
    """Retorna conclusão baseada na diferença percentual."""
    abs_diff = abs(diff_pct)
    if abs_diff < 0.5:
        return "✓ As otimizações de GPU mantiveram a qualidade do modelo intacta."
    elif abs_diff < 2.0:
        return "⚠ Pequena variação numérica esperada devido a mixed precision training."
    else:
        return "⚠️ Diferença significativa. Recomenda-se investigação adicional."


def main():
    parser = argparse.ArgumentParser(
        description='Compara resultados entre implementações local e Kaggle'
    )
    parser.add_argument(
        '--local',
        type=Path,
        default=Path('../artifacts/gnn_results.json'),
        help='Caminho para o arquivo de resultados local'
    )
    parser.add_argument(
        '--kaggle',
        type=Path,
        default=Path('./kaggle_results/gnn_results_multigpu.json'),
        help='Caminho para o arquivo de resultados Kaggle'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./comparison_report'),
        help='Diretório para salvar o relatório'
    )
    
    args = parser.parse_args()
    
    # Verificar se arquivos existem
    if not args.local.exists():
        print(f"❌ Arquivo local não encontrado: {args.local}")
        print("Execute primeiro o notebook local para gerar resultados.")
        return
    
    if not args.kaggle.exists():
        print(f"❌ Arquivo Kaggle não encontrado: {args.kaggle}")
        print("Baixe os resultados do Kaggle e coloque em: {args.kaggle}")
        return
    
    # Carregar resultados
    print("Carregando resultados...")
    local_results = load_results(args.local)
    kaggle_results = load_results(args.kaggle)
    
    # Gerar relatório
    generate_report(local_results, kaggle_results, args.output)


if __name__ == '__main__':
    main()
