# Notebooks Organization

Esta pasta contém os Jupyter notebooks organizados por módulo, refletindo a arquitetura modular da plataforma AML.

## Estrutura

```
notebooks/
├── data/                          # Notebooks de ingestão e processamento de dados
│   ├── 01_data_ingestion_and_split.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_exploratory_analysis.ipynb
├── features/                      # Notebooks de engenharia de features
│   ├── 04_feature_audit.ipynb
│   └── 06_validacao_correcao_leakage.ipynb
├── modeling/                      # Notebooks de desenvolvimento de modelos
│   ├── 01_feature_engineering.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_ensemble_modeling.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_final_results.ipynb
│   └── 07_integrated_workflow.ipynb
├── evaluation/                    # Notebooks de avaliação e reporting
│   └── 11_reporting_dashboard.ipynb
├── monitoring/                    # Notebooks de monitoramento
│   └── 10_monitoring_drift.ipynb
├── orchestration/                 # Notebooks de orquestração (futuro)
├── artifacts/                     # Artefatos gerados pelos notebooks
├── catboost_info/                 # Informações específicas do CatBoost
```

## Convenções

- **Numeração**: Notebooks são numerados sequencialmente (01_, 02_, etc.)
- **Módulos**: Organizados por módulo da arquitetura (`data/`, `features/`, `modeling/`, etc.)
- **Artefatos**: Outputs dos notebooks ficam em `artifacts/`

## Relacionamento com Código Fonte

- **`src/`**: Contém o código Python de produção (módulos, classes, funções)
- **`notebooks/`**: Contém notebooks Jupyter para exploração, análise e documentação
- **Separação clara**: Código reutilizável em `src/`, experimentação em `notebooks/`

## Fluxo de Desenvolvimento

1. **Exploração** → Notebooks em `notebooks/`
2. **Refatoração** → Código movido para `src/`
3. **Produção** → Uso dos módulos de `src/`

Esta organização mantém a separação clara entre código de exploração (notebooks) e código de produção (src/), facilitando manutenção e deployment.