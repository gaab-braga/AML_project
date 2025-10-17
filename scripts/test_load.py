#!/usr/bin/env python3
"""
Test data loading
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly from module
import data_io

if __name__ == '__main__':
    df = data_io.load_raw_transactions("data/")
    print(f"Loaded {len(df)} rows")
    print(df.columns)
    print(df.head())