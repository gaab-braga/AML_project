#!/usr/bin/env python3
"""
Full Pipeline Runner for AML Detection
Executes ETL -> Feature Engineering -> Training -> Evaluation -> Model Export
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score, precision_score

# Import src modules
from src.data_io import load_raw_transactions, save_model, validate_data_compliance
from src.preprocessing import clean_transactions, impute_and_encode
from src.features import aggregate_by_entity, compute_network_features
from src.modeling import build_pipeline, train_pipeline
from src.evaluation import compute_metrics

# Import download script
from scripts.download_kaggle_data import download_aml_dataset_kaggle_api

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path: str = 'config/pipeline_config.yaml'):
    """Run the full AML detection pipeline."""
    logger.info("🚀 Starting AML Detection Pipeline")

    # Load config
    config = load_config(config_path)
    logger.info(f"📋 Loaded config from {config_path}")

    # 0. Download Data if not present
    data_path = config['data']['path']
    if not Path(data_path).exists():
        logger.info("📥 Data not found, downloading from Kaggle...")
        download_aml_dataset_kaggle_api(path="data/")
    else:
        logger.info("✅ Data already exists, skipping download")

    # 1. Data Loading
    logger.info("📊 Loading and validating data...")
    df = load_raw_transactions(data_path)
    if not validate_data_compliance(df):
        raise ValueError("❌ Data failed compliance check")
    logger.info(f"✅ Loaded {len(df)} transactions")

    # 2. Preprocessing
    logger.info("🧹 Preprocessing data...")
    df_clean = clean_transactions(df)
    df_processed = impute_and_encode(df_clean, config['preprocessing'])
    logger.info(f"✅ Preprocessing completed: {len(df_processed)} samples")

    # 3. Feature Engineering
    logger.info("🔧 Feature engineering...")
    # For demo, use simple features from processed data
    # In production, would use aggregate_by_entity and compute_network_features
    X = df_processed[['amount', 'from_bank', 'to_bank']].copy()
    if 'type_transfer' in df_processed.columns:
        X['type_transfer'] = df_processed['type_transfer']
    y = df_processed['is_fraud']

    logger.info(f"✅ Features ready: {len(X)} samples with {len(X.columns)} features")

    # 4. Modeling
    logger.info("🤖 Training model...")
    pipeline = build_pipeline(config['model'])
    results = train_pipeline(X, y, pipeline, cv=config['validation']['cv_folds'], temporal=config['validation'].get('temporal_split', False))
    logger.info(f"✅ Training completed - CV Score: {results['mean_score']:.3f}")

    # Custom AML metrics: Recall for top K% (high-risk transactions)
    logger.info("📊 Computing AML-specific metrics...")
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    top_k_percent = int(config['aml_config']['top_k_percent'] * len(y_pred_proba))
    top_indices = y_pred_proba.argsort()[-top_k_percent:]
    recall_top_k = recall_score(y.iloc[top_indices], (y_pred_proba[top_indices] > 0.5).astype(int))
    logger.info(f"✅ Recall at top {config['aml_config']['top_k_percent']*100}%: {recall_top_k:.3f}")

    # 5. Evaluation
    metrics = compute_metrics(y, y_pred_proba)
    logger.info(f"✅ Final Metrics - ROC-AUC: {metrics['roc_auc']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}")

    # 6. Save Model and Metadata
    model_path = config['output']['model_path']
    save_model(pipeline, model_path)

    # Save metadata
    metadata = {
        'config': config,
        'metrics': metrics,
        'recall_top_k': recall_top_k,
        'cv_results': results,
        'data_info': {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'fraud_rate': float(y.mean())
        }
    }
    with open(config['output']['metadata_path'], 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info("✅ Pipeline completed successfully!")
    logger.info(f"📁 Model saved to: {model_path}")
    logger.info(f"📋 Metadata saved to: {config['output']['metadata_path']}")

if __name__ == '__main__':
    main()