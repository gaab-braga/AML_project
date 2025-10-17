#!/usr/bin/env python3
"""
Sample Prediction Script for AML Model
Loads model and predicts on sample data.
"""

import pandas as pd
from src.modeling import load_model
from src.preprocessing import impute_and_encode
from src.features import aggregate_by_entity

def predict_sample(model_path: str, sample_data: dict) -> dict:
    """Predict on sample transaction data."""
    # Load model
    model = load_model(model_path)

    # Process sample
    df = pd.DataFrame([sample_data])
    df_processed = impute_and_encode(df, {'categorical_cols': ['type']})
    features = aggregate_by_entity(df_processed, 'customer_id', [7])

    # Predict
    proba = model.predict_proba(features)[:, 1][0]
    prediction = model.predict(features)[0]

    return {
        'prediction': int(prediction),
        'probability': float(proba),
        'threshold': 0.5,
        'risk_level': 'High' if proba > 0.5 else 'Low'
    }

if __name__ == '__main__':
    sample = {
        'customer_id': '12345',
        'amount': 10000,
        'type': 'transfer',
        'date': '2023-10-01'
    }
    result = predict_sample('models/aml_model.pkl', sample)
    print(result)