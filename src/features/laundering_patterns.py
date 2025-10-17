import re
from typing import List, Dict, Any


def parse_laundering_patterns(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse do arquivo de padrões de lavagem e extração de informações estruturadas

    Args:
        file_path: Caminho para o arquivo de padrões

    Returns:
        Lista de padrões parseados com suas transações
    """
    patterns = []
    current_pattern = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                # Extrair tipo do padrão e características
                match = re.search(r'BEGIN LAUNDERING ATTEMPT - (\w+):?\s*(.*)', line)
                if match:
                    pattern_type = match.group(1)
                    characteristics = match.group(2) if match.group(2) else ""
                    current_pattern = {
                        'type': pattern_type,
                        'characteristics': characteristics,
                        'transactions': []
                    }

            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current_pattern:
                    patterns.append(current_pattern)
                    current_pattern = None

            elif current_pattern and line and not line.startswith('BEGIN') and not line.startswith('END'):
                # Parse da linha de transação
                parts = line.split(',')
                if len(parts) >= 11:
                    try:
                        transaction = {
                            'timestamp': parts[0],
                            'from_bank': parts[1],
                            'from_account': parts[2],
                            'to_bank': parts[3],
                            'to_account': parts[4],
                            'amount_orig': float(parts[5]),
                            'currency_orig': parts[6],
                            'amount_dest': float(parts[7]),
                            'currency_dest': parts[8],
                            'payment_format': parts[9],
                            'is_laundering': int(parts[10])
                        }
                        current_pattern['transactions'].append(transaction)
                    except (ValueError, IndexError):
                        continue

    return patterns