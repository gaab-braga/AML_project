with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Verificando linhas ao redor do erro:')
for i in range(1837, min(1845, len(lines))):
    marker = '>>>' if i == 1839 else '   '
    print(f'{marker} {i+1:4d}: {repr(lines[i])}')

# Se linha 1840 estiver vazia, removê-la
if 1839 < len(lines) and lines[1839].strip() == '':
    print('Removendo linha vazia 1840')
    lines.pop(1839)
    
    with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print('Linha vazia removida')
else:
    print('Linha 1840 não está vazia ou não existe')