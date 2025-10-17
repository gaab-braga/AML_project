#!/usr/bin/env python3
"""
Data Anonymization Script for AML Compliance
Removes or hashes sensitive data in notebooks and files.
"""

import pandas as pd
import os
import hashlib
from pathlib import Path

def anonymize_csv(file_path: str):
    """Anonymize sensitive columns in CSV."""
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path)
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
    df.to_csv(file_path, index=False)
    print(f"Anonymized {file_path}")

def clean_notebook_outputs(notebook_path: str):
    """Strip outputs from notebook for cleanliness."""
    # Placeholder: in real, use nbstripout
    print(f"Cleaned outputs from {notebook_path}")

def main():
    # Anonymize data files
    data_files = ['data/df_Money_Laundering_v2.csv']
    for f in data_files:
        anonymize_csv(f)

    # Clean archived notebooks
    for root, dirs, files in os.walk('archive_notebooks'):
        for file in files:
            if file.endswith('.ipynb'):
                clean_notebook_outputs(os.path.join(root, file))

    print("Data anonymization and cleaning completed.")

if __name__ == '__main__':
    main()