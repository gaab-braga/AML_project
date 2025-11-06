"""
Command-line interface for AML system.
"""
import typer
from typing import Optional
import pandas as pd
from pathlib import Path

from src.data.loader import load_raw_data, save_processed_data, load_processed_data
from src.data.preprocessing import clean_data, split_data
from src.features.engineering import build_features
from src.models.train import train_model, save_model, load_model
from src.models.evaluate import evaluate_model, save_evaluation_report, print_evaluation_summary
from src.models.predict import predict, predict_batch
from src.config import config
from src.utils.logger import setup_logger

app = typer.Typer()
logger = setup_logger(__name__)


@app.command()
def train(
    data_file: Optional[str] = typer.Option(None, "--data", "-d"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
    use_temporal_split: bool = typer.Option(True, "--temporal/--random"),
):
    """Train AML detection model."""
    
    logger.info("Starting training pipeline")
    
    df = load_raw_data(data_file) if data_file else load_raw_data()
    df_clean = clean_data(df)
    df_features = build_features(df_clean)
    
    target_col = config.get('model.target_column')
    if use_temporal_split:
        from src.data.preprocessing import temporal_split
        X_train, X_test, y_train, y_test = temporal_split(df_features, target_col)
    else:
        from sklearn.model_selection import train_test_split
        test_size = config.get('model.test_size', 0.2)
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    save_processed_data(train_data, 'train')
    save_processed_data(test_data, 'test')
    
    model = train_model(X_train, y_train, model_name)
    
    y_pred = model.predict(X_test)
    y_proba = predict(X_test, model, return_proba=True)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_evaluation_summary(metrics)
    
    save_model(model)
    save_evaluation_report(metrics)
    
    logger.info("Training pipeline completed")


@app.command()
def predict_cmd(
    input_file: str = typer.Argument(...),
    output_file: str = typer.Option("predictions.csv", "--output", "-o"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Make predictions on new data."""
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    model = load_model(model_path) if model_path else load_model()
    df_features = build_features(df)
    
    results = predict_batch(df_features, model)
    results.to_csv(output_file, index=False)
    
    suspicious_count = results['prediction'].sum()
    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Suspicious transactions: {suspicious_count}/{len(results)}")


@app.command()
def evaluate(
    test_data: Optional[str] = typer.Option(None, "--test-data", "-t"),
    model_path: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Evaluate trained model."""
    
    if test_data:
        df_test = pd.read_csv(test_data)
    else:
        df_test = load_processed_data('test')
    
    target_col = config.get('model.target_column')
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    model = load_model(model_path) if model_path else load_model()
    
    y_pred = model.predict(X_test)
    y_proba = predict(X_test, model, return_proba=True)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_evaluation_summary(metrics)
    
    save_evaluation_report(metrics, "artifacts/evaluation_report.json")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start API server."""
    
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "entrypoints.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    app()
