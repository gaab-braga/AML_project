# Script: convert_to_multignn_csv.py

Este script converte o dataset processado do projeto AML para o formato CSV esperado pelo MultiGNN da IBM.

## Funcionalidades

- Carrega dataset do arquivo `.pkl` processado
- Mapeia contas para IDs numéricos
- Renomeia colunas para o padrão MultiGNN
- Adiciona colunas `from_id` e `to_id` necessárias
- Salva dataset completo em formato CSV

## Uso

```bash
cd /caminho/para/AML_project
python scripts/convert_to_multignn_csv.py
```

## Arquivos de Entrada

- `data/processed/features_with_patterns_sampled.pkl`: Dataset processado com features

## Arquivos de Saída

- `benchmarks/Multi-GNN/data/aml/formatted_transactions.csv`: Dataset em formato CSV

## Formato de Saída

O CSV contém todas as colunas originais mais:
- `from_id`: ID numérico da conta de origem
- `to_id`: ID numérico da conta de destino
- `Timestamp`: Timestamp da transação
- `Is Laundering`: Label binária (0/1)

## Exemplo de Execução

```
🚀 Iniciando conversão para formato MultiGNN CSV
📂 Carregando dataset...
✅ Dataset carregado: 1019808 transações, 75 colunas
🏦 Criando mapeamento de contas...
✅ 332344 contas únicas mapeadas
🔄 Preparando dados para CSV...
✅ CONVERSÃO CONCLUÍDA!
📄 Arquivo salvo em: c:\Users\...\formatted_transactions.csv
📊 Shape: (1019808, 77)
📈 Estatísticas do Dataset:
   • Transações totais: 1019808
   • Transações ilícitas: 5177
   • Taxa de ilícitas: 0.51%
   • Contas únicas: 332344
   • Tamanho do arquivo: 805.25 MB
```

## Próximos Passos

Após executar este script, você pode treinar o MultiGNN:

```bash
cd benchmarks/Multi-GNN
python main.py --data aml --model gin
```