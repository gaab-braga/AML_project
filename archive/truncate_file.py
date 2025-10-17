with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')
    print(f'Linhas após split: {len(lines)}')
    print(f'Últimas 5 linhas:')
    for i in range(max(0, len(lines)-5), len(lines)):
        print(f'{i+1:4d}: {repr(lines[i])}')
    
    # Remover linhas vazias do final
    while lines and not lines[-1].strip():
        lines.pop()
    
    print(f'\nApós remoção de vazias: {len(lines)} linhas')
    print(f'Última linha: {repr(lines[-1])}')
    
    # Reescrever arquivo
    with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Arquivo reescrito')