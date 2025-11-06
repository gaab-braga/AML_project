"""
Tests for batch processing.
"""
import pytest
import pandas as pd
from pathlib import Path

from entrypoints.batch import process_batch


def test_process_batch_csv(tmp_path, sample_data):
    """Test batch processing with CSV file."""
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    output_dir = tmp_path / "predictions"
    
    result_path = process_batch(str(input_file), str(output_dir))
    
    assert result_path.exists()
    
    results = pd.read_csv(result_path)
    assert "prediction" in results.columns
    assert "probability" in results.columns
    assert len(results) > 0


def test_process_batch_with_date(tmp_path, sample_data):
    """Test batch processing with specific date."""
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    output_dir = tmp_path / "predictions"
    date = "2024-01-15"
    
    result_path = process_batch(str(input_file), str(output_dir), date=date)
    
    assert result_path.exists()
    assert date in result_path.name
