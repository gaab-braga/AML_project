"""
Tests for preprocessing module
"""

import pandas as pd
from src.preprocessing import clean_transactions

def test_clean_transactions():
    df = pd.DataFrame({'amount': [100, None, 200], 'date': ['2023-01-01', '2023-01-02', None]})
    result = clean_transactions(df)
    assert len(result) == 1  # Should drop rows with missing critical fields (amount=None and date=None)