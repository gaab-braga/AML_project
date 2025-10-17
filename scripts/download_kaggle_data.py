#!/usr/bin/env python3
"""
Download AML Dataset from Kaggle
Requires Kaggle API key setup.
"""

import os
import zipfile
import requests
from pathlib import Path
import pandas as pd
import numpy as np

def generate_sample_aml_data(path: str = "data/", n_samples: int = 10000):
    """
    Generate sample AML transaction data for demonstration.

    Args:
        path: Save path
        n_samples: Number of transactions
    """
    np.random.seed(42)
    data = {
        'customer_id': np.random.randint(1, 1000, n_samples),
        'source': np.random.randint(1, 1000, n_samples),
        'target': np.random.randint(1, 1000, n_samples),
        'amount': np.random.exponential(1000, n_samples),
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, 'df_Money_Laundering.csv'), index=False)
    print(f"Generated sample data saved to {path}")

def download_aml_dataset_kaggle_api(dataset: str = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml", path: str = "data/"):
    """
    Download dataset from Kaggle using API.

    Args:
        dataset: Kaggle dataset slug
        path: Download path
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=path, unzip=True)
        print(f"Downloaded {dataset} to {path}")
        return True
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading: {e}. Generating sample data instead.")
        generate_sample_aml_data(path)
        return False

def download_aml_dataset_fallback(url: str = "https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml", path: str = "data/"):
    """
    Fallback download using requests (requires login or public link).

    Args:
        url: Direct download URL
        path: Download path
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            zip_path = os.path.join(path, "data.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(zip_path)
            print(f"Downloaded and extracted to {path}")
            return True
        else:
            print("Fallback download failed. Generating sample data.")
            generate_sample_aml_data(path)
            return False
    except Exception as e:
        print(f"Fallback failed: {e}. Generating sample data.")
        generate_sample_aml_data(path)
        return False

if __name__ == '__main__':
    if not download_aml_dataset_kaggle_api():
        download_aml_dataset_fallback()