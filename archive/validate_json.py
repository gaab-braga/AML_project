import json

try:
    with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'r', encoding='utf-8') as f:
        content = f.read()
        print(f'Arquivo lido, tamanho: {len(content)} caracteres')
        nb = json.loads(content)
        print(f'JSON válido! {len(nb["cells"])} células')
except json.JSONDecodeError as e:
    print(f'Erro JSON na linha {e.lineno}, coluna {e.colno}: {e.msg}')
    # Mostrar mais contexto
    lines = content.split('\n')
    start = max(0, e.lineno - 15)
    end = min(len(lines), e.lineno + 5)
    for i in range(start, end):
        marker = '>>>' if i == e.lineno - 1 else '   '
        print(f'{marker} {i+1:4d}: {repr(lines[i])}')
    
    # Verificar linhas seguintes
    print('\nVerificando linhas seguintes...')
    for i in range(e.lineno, min(e.lineno + 15, len(lines))):
        if i < len(lines):
            print(f'Linha {i+1}: {repr(lines[i])}')
            if ']' in lines[i] or '}' in lines[i]:
                print(f'Encontrado fechamento na linha {i+1}')
                break
    # Tentar corrigir automaticamente
    print('\nTentando corrigir...')
    target_line = lines[e.lineno - 1]
    print(f'Linha problemática: {repr(target_line)}')
    if target_line.strip() == '"    \\"\\n\\""':
        lines[e.lineno - 1] = '    "    print(\\"   🚀 Use \'gpu_accelerated\' se GPU disponível\\")\\n",\n    "\\n"\n'
        print('Correção aplicada: adicionada vírgula na linha anterior')
        # Salvar arquivo corrigido
        with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Arquivo salvo. Testando novamente...')
        try:
            nb = json.loads(''.join(lines))
            print(f'Sucesso! JSON válido com {len(nb["cells"])} células')
        except json.JSONDecodeError as e2:
            print(f'Ainda erro: {e2}')
    else:
        print('Tentando correção simples...')
        # Simplesmente adicionar vírgula na linha anterior
        if e.lineno - 2 >= 0:
            prev_line = lines[e.lineno - 2]
            if not prev_line.strip().endswith(','):
                lines[e.lineno - 2] = prev_line.rstrip() + ',\n'
                print('Adicionada vírgula na linha anterior')
                with open('notebooks/03_Modelagem_e_Avaliacao.ipynb', 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                try:
                    nb = json.loads(''.join(lines))
                    print(f'Sucesso! JSON válido com {len(nb["cells"])} células')
                except json.JSONDecodeError as e2:
                    print(f'Ainda erro: {e2}')