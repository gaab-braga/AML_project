"""
Batch processing entrypoint.
"""
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessing import clean_data
from src.features.engineering import build_features
from src.models.predict import predict_batch, load_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_batch(input_file: str, output_dir: str = "data/predictions", date: str = None):
    """
    Process batch of transactions and save predictions.
    
    Args:
        input_file: Input CSV file
        output_dir: Output directory
        date: Batch date (YYYY-MM-DD)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Processing batch for date: {date}")
    
    df = pd.read_csv(input_file) if input_file.endswith('.csv') else load_raw_data(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    df_clean = clean_data(df)
    df_features = build_features(df_clean)
    
    model = load_model()
    
    logger.info("Making batch predictions")
    results = predict_batch(df_features, model)
    
    output_path = Path(output_dir) / f"predictions_{date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved to {output_path}")
    
    suspicious = results[results['prediction'] == 1]
    logger.info(f"Total: {len(results)} | Suspicious: {len(suspicious)} ({len(suspicious)/len(results)*100:.2f}%)")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Batch Processing")
    parser.add_argument("--input", "-i", required=True, help="Input file")
    parser.add_argument("--output", "-o", default="data/predictions", help="Output directory")
    parser.add_argument("--date", "-d", help="Batch date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    process_batch(args.input, args.output, args.date)
