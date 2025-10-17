import sys
sys.path.insert(0, '.')

# Verificar estrutura do training_results
from pathlib import Path
import joblib

# Carregar dados como no notebook
artifacts_dir = Path('artifacts')
cache_dir = artifacts_dir / 'cache' / 'experiments'

try:
    # Tentar carregar algum resultado de treinamento
    import os
    cache_files = list(cache_dir.glob('*.pkl'))
    if cache_files:
        print(f'Encontrados {len(cache_files)} arquivos de cache')
        for f in cache_files[:3]:  # Mostrar primeiros 3
            print(f'  {f.name}')

        # Carregar um arquivo de exemplo
        example_file = cache_files[0]
        result = joblib.load(example_file)
        print(f'\nEstrutura do resultado carregado:')
        print(f'Keys: {list(result.keys())}')
        if 'results' in result:
            for model_name, model_result in result['results'].items():
                print(f'\nModelo {model_name}:')
                print(f'  Keys: {list(model_result.keys())}')
                if 'pipeline' in model_result:
                    print(f'  Pipeline type: {type(model_result["pipeline"])}')
                if 'evaluation_results' in model_result:
                    eval_keys = list(model_result['evaluation_results'].keys())
                    print(f'  Evaluation keys: {eval_keys}')
                    if 'probabilities' in model_result['evaluation_results']:
                        prob_shape = model_result['evaluation_results']['probabilities'].shape
                        print(f'  Probabilities shape: {prob_shape}')
    else:
        print('Nenhum arquivo de cache encontrado')

except Exception as e:
    print(f'Erro: {e}')
    import traceback
    traceback.print_exc()