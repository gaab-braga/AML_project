# AML_project

Projeto de detecção de lavagem de dinheiro usando técnicas de Machine Learning e Graph Neural Networks.

## Scripts Disponíveis

### Conversão de Dados para MultiGNN
- `scripts/convert_to_multignn_csv.py`: Converte o dataset processado (.pkl) para o formato CSV esperado pelo MultiGNN da IBM
- Uso: `python scripts/convert_to_multignn_csv.py`
- Saída: `benchmarks/Multi-GNN/data/aml/formatted_transactions.csv`

### Outros Scripts
- `scripts/benchmark_models.py`: Benchmark de modelos tradicionais
- `scripts/clean_data.py`: Limpeza e pré-processamento de dados
- `scripts/fine_tuning.py`: Fine-tuning de modelos
- `scripts/gnn_diagnostic.py`: Diagnóstico do modelo GNN
- `scripts/predict_sample.py`: Predições em dados de exemplo

## Estrutura do Projeto

```
AML_project/
├── data/
│   ├── processed/
│   └── raw/
├── scripts/           # Scripts utilitários
├── notebooks/         # Notebooks Jupyter
├── benchmarks/        # Modelos de benchmark (Multi-GNN, etc.)
├── models/           # Modelos treinados
├── artifacts/        # Resultados e métricas
└── logs/             # Logs de execução
```

## Como Usar

1. **Preparação de Dados**: Execute `python scripts/convert_to_multignn_csv.py` para converter dados para MultiGNN
2. **Treinamento**: Use os notebooks em `notebooks/` para treinar modelos
3. **Avaliação**: Verifique métricas em `artifacts/`

## Dependências

- PyTorch
- PyTorch Geometric
- Pandas
- Scikit-learn
- Hydra/OmegaConf (para MultiGNN)