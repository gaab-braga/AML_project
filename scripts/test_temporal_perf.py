import time
import pandas as pd
import numpy as np
from pathlib import Path

# Carregar o módulo diretamente para evitar import de pacote que exige dependências extras
import importlib.util
import importlib.machinery
import os

aml_features_path = os.path.join('src', 'features', 'aml_features.py')
spec = importlib.util.spec_from_file_location('aml_features', aml_features_path)
aml_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aml_features)
create_temporal_features = aml_features.create_temporal_features

# Load a sample from the raw transactions
DATA_PATH = Path('data/raw/HI-Small_Trans.csv')
print('Loading data...')
# Read a sample to avoid OOM
nrows = 100000

# Use pandas to read first nrows as a sample
df = pd.read_csv(DATA_PATH, nrows=nrows)

# Prepare minimal columns expected by the function
# ensure columns present
if 'source' not in df.columns:
    # try to map existing columns
    if 'From Account' in df.columns:
        df['source'] = df['From Account'].astype(str)
    else:
        df['source'] = df.index.astype(str)

if 'timestamp' not in df.columns:
    if 'Timestamp' in df.columns:
        df['timestamp'] = df['Timestamp']
    else:
        df['timestamp'] = pd.Timestamp.now()

if 'amount' not in df.columns:
    if 'Amount Paid' in df.columns:
        df['amount'] = df['Amount Paid']
    else:
        df['amount'] = np.random.rand(len(df)) * 1000

print('Sample prepared:', df.shape)

start = time.time()
res = create_temporal_features(df, windows=[7,30])
end = time.time()
print('Result shape:', res.shape)
print('Elapsed seconds:', end - start)

# Print sample columns and head
print('Columns:', res.columns.tolist())
print(res.head(2))
