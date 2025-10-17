#!/usr/bin/env python3
"""
AML Pipeline MLflow Example Script

This script demonstrates how to use the AML Pipeline MLflow integration
for experiment tracking, model logging, and registry management.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

from mlops.model_registry.mlflow_integration import AMLPipelineMLflow, quick_experiment


def load_sample_data():
    """Load sample fraud detection data."""
    # Generate synthetic fraud detection data
    np.random.seed(42)
    n_samples = 10000

    # Features
    features = {
        'amount': np.random.exponential(100, n_samples),
        'customer_age': np.random.normal(35, 10, n_samples),
        'transaction_frequency': np.random.poisson(5, n_samples),
        'is_international': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'is_night_transaction': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'risk_score': np.random.beta(2, 5, n_samples),
    }

    # Create DataFrame
    df = pd.DataFrame(features)

    # Generate target (fraud label) based on features
    fraud_probability = (
        0.1 +  # base rate
        df['amount'] / 1000 * 0.3 +  # high amounts more suspicious
        df['is_international'] * 0.2 +  # international more suspicious
        df['is_night_transaction'] * 0.15 +  # night transactions more suspicious
        df['risk_score'] * 0.25  # risk score directly contributes
    ).clip(0, 1)

    df['is_fraud'] = np.random.binomial(1, fraud_probability)

    return df


def train_model(X_train, X_test, y_train, y_test, model_params):
    """Train a fraud detection model."""
    # Initialize model
    model = RandomForestClassifier(**model_params, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }

    return model, metrics, y_pred, y_pred_proba


def run_experiment():
    """Run a complete ML experiment with MLflow tracking."""
    print("🚀 Starting AML Pipeline MLflow Experiment...")

    # Initialize MLflow integration
    mlflow_integration = AMLPipelineMLflow()

    # Load data
    print("📊 Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} transactions")

    # Prepare features and target
    feature_cols = ['amount', 'customer_age', 'transaction_frequency',
                   'is_international', 'is_night_transaction', 'risk_score']
    X = df[feature_cols]
    y = df['is_fraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate: {y.mean():.1f}")

    # Define experiment parameters
    experiment_configs = [
        {
            'name': 'rf_baseline',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
            }
        },
        {
            'name': 'rf_tuned',
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
            }
        },
        {
            'name': 'rf_complex',
            'params': {
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
            }
        }
    ]

    best_model = None
    best_metrics = None
    best_run_id = None

    # Run experiments
    for config in experiment_configs:
        print(f"\n🔬 Running experiment: {config['name']}")

        with mlflow_integration.start_run("fraud_detection_experiments", config['name']) as run:

            # Log parameters
            mlflow_integration.log_parameters({
                'model_type': 'RandomForestClassifier',
                'data_size': len(X_train),
                'feature_count': len(feature_cols),
                'test_size': len(X_test),
                **config['params']
            })

            # Log data info
            mlflow.log_param("features", feature_cols)
            mlflow.log_param("fraud_rate_train", y_train.mean())
            mlflow.log_param("fraud_rate_test", y_test.mean())

            # Train model
            print("   Training model...")
            model, metrics, y_pred, y_pred_proba = train_model(
                X_train, X_test, y_train, y_test, config['params']
            )

            # Log metrics
            mlflow_integration.log_metrics(metrics)

            # Log model
            print("   Logging model...")
            mlflow_integration.log_model(model, "model", "sklearn")

            # Log additional artifacts
            print("   Logging artifacts...")

            # Create feature importance plot
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Save feature importance
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

            # Log confusion matrix data
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, columns=['Predicted_Negative', 'Predicted_Positive'],
                               index=['Actual_Negative', 'Actual_Positive'])
            cm_df.to_csv("confusion_matrix.csv")
            mlflow.log_artifact("confusion_matrix.csv")

            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            # Track best model
            if best_metrics is None or metrics['f1_score'] > best_metrics['f1_score']:
                best_model = model
                best_metrics = metrics
                best_run_id = run.info.run_id

    # Register best model
    if best_model is not None:
        print("\n🏆 Registering best model...")
        with mlflow_integration.start_run("model_registration", "best_model_registration"):
            # Log best model
            mlflow_integration.log_model(best_model, "model", "sklearn")
            mlflow_integration.log_metrics(best_metrics)

            # Register model
            model_version = mlflow_integration.register_model("fraud_detection_model")

            print(f"✅ Model registered as version: {model_version}")

            # Transition to staging
            mlflow_integration.transition_model_stage(
                "fraud_detection_model", model_version, "staging"
            )
            print("✅ Model transitioned to staging")

    # Demonstrate model loading
    print("\n📥 Testing model loading...")
    try:
        model_uri = mlflow_integration.get_model_uri("fraud_detection_model", stage="staging")
        loaded_model = mlflow_integration.load_model(model_uri)

        # Test prediction
        sample_prediction = loaded_model.predict(X_test.iloc[:5])
        print(f"✅ Model loaded successfully. Sample predictions: {sample_prediction}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

    print("\n🎉 MLflow experiment completed!")
    print(f"📊 Best model F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"🏷️  Best run ID: {best_run_id}")

    # Show experiment summary
    print("\n📈 Experiment Summary:")
    experiments = mlflow_integration.list_experiments()
    for exp in experiments:
        if "fraud_detection" in exp['name']:
            print(f"   Experiment: {exp['name']} (ID: {exp['experiment_id']})")

    runs = mlflow_integration.list_runs()
    print(f"   Total runs: {len(runs)}")

    models = mlflow_integration.list_registered_models()
    print(f"   Registered models: {len(models)}")


def demonstrate_quick_experiment():
    """Demonstrate quick experiment functionality."""
    print("\n⚡ Demonstrating Quick Experiment...")
    from mlops.model_registry.mlflow_integration import log_model_metrics

    # Quick logging example
    log_model_metrics(
        "quick_test_model",
        {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90
        },
        {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10
        }
    )

    print("✅ Quick experiment logged!")


if __name__ == "__main__":
    try:
        # Run main experiment
        run_experiment()

        # Demonstrate quick functionality
        demonstrate_quick_experiment()

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)