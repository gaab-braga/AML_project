with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Antes da correção:')
print(f'Linha 1839: {repr(lines[1838])}')

# Corrigir removendo a vírgula do final
if lines[1838].strip().endswith(','):
    lines[1838] = lines[1838].rstrip()[:-1] + '\n'  # Remove a vírgula
    print('Vírgula removida')

    with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('Arquivo corrigido salvo')

print(f'Linha 1839 após correção: {repr(lines[1838])}')