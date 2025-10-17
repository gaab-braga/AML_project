"""
AML Pipeline Enterprise Automated Retraining Demonstration

This script demonstrates the enterprise-grade automated retraining system
with advanced features like circuit breakers, data validation, and health checks.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add mlops module to path
sys.path.append(str(Path(__file__).parent.parent))

from retraining.automated_retraining import (
    AutomatedRetrainingPipeline,
    create_retraining_pipeline,
    quick_retraining_check,
    CircuitBreakerOpenException
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_enterprise_sample_data(n_samples: int = 10000, with_anomalies: bool = False) -> pd.DataFrame:
    """
    Create enterprise-grade sample data with realistic patterns and optional anomalies.

    Args:
        n_samples: Number of samples to generate
        with_anomalies: Whether to include data quality issues

    Returns:
        Enterprise-quality dataset
    """
    np.random.seed(42)

    # Base features with realistic distributions
    data = {
        'transaction_amount': np.random.exponential(500, n_samples),
        'customer_age': np.random.normal(35, 12, n_samples).clip(18, 80),
        'account_balance': np.random.exponential(5000, n_samples),
        'transaction_frequency': np.random.poisson(8, n_samples),
        'is_international': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'is_night_transaction': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'merchant_category': np.random.choice(['retail', 'online', 'restaurant', 'travel', 'entertainment'],
                                            n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'device_type': np.random.choice(['mobile', 'web', 'atm', 'pos'], n_samples,
                                      p=[0.5, 0.3, 0.15, 0.05]),
        'location_risk_score': np.random.beta(1.5, 3, n_samples),
        'customer_risk_score': np.random.beta(2, 5, n_samples),
        'transaction_velocity': np.random.poisson(3, n_samples),
        'amount_to_balance_ratio': np.zeros(n_samples),  # Will be calculated
    }

    df = pd.DataFrame(data)

    # Calculate derived features
    df['amount_to_balance_ratio'] = df['transaction_amount'] / (df['account_balance'] + 1)

    # Create sophisticated target variable
    fraud_probability = (
        df['transaction_amount'] * 0.0005 +
        df['is_international'] * 0.25 +
        df['is_night_transaction'] * 0.15 +
        df['location_risk_score'] * 0.4 +
        df['customer_risk_score'] * 0.6 +
        df['transaction_velocity'] * 0.1 +
        df['amount_to_balance_ratio'] * 0.3 +
        (df['merchant_category'] == 'online').astype(int) * 0.2
    ).clip(0, 1)

    df['is_fraud'] = np.random.binomial(1, fraud_probability.values)

    # Add anomalies if requested
    if with_anomalies:
        # Add some missing values
        anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df.loc[anomaly_indices[:len(anomaly_indices)//2], 'customer_age'] = np.nan

        # Add some outliers
        outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
        df.loc[outlier_indices, 'transaction_amount'] *= 100  # Extreme values

        # Add duplicates
        duplicate_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
        duplicate_rows = df.iloc[duplicate_indices].copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Encode categorical variables
    df['merchant_category'] = df['merchant_category'].map({
        'retail': 0, 'online': 1, 'restaurant': 2, 'travel': 3, 'entertainment': 4
    })
    df['device_type'] = df['device_type'].map({
        'mobile': 0, 'web': 1, 'atm': 2, 'pos': 3
    })

    return df


def simulate_enterprise_data_drift(original_data: pd.DataFrame, drift_type: str = 'gradual',
                                 drift_intensity: float = 0.3) -> pd.DataFrame:
    """
    Simulate different types of enterprise data drift.

    Args:
        original_data: Original dataset
        drift_type: Type of drift ('gradual', 'sudden', 'recurring', 'covariate_shift')
        drift_intensity: Intensity of drift (0-1)

    Returns:
        Drifted dataset
    """
    drifted_data = original_data.copy()

    if drift_type == 'gradual':
        # Gradual changes over time (e.g., economic inflation)
        drifted_data['transaction_amount'] *= (1 + drift_intensity * np.random.normal(0.1, 0.05, len(drifted_data)))

        # Demographic shifts
        age_shift = np.random.normal(0, drift_intensity * 3, len(drifted_data))
        drifted_data['customer_age'] = (drifted_data['customer_age'] + age_shift).clip(18, 80)

    elif drift_type == 'sudden':
        # Sudden changes (e.g., new regulations, market events)
        sudden_mask = np.random.random(len(drifted_data)) < drift_intensity
        drifted_data.loc[sudden_mask, 'is_international'] = 1
        drifted_data.loc[sudden_mask, 'transaction_amount'] *= 2

    elif drift_type == 'recurring':
        # Recurring patterns (e.g., seasonal effects)
        time_component = np.sin(np.arange(len(drifted_data)) * 0.01) * drift_intensity
        drifted_data['transaction_frequency'] = (drifted_data['transaction_frequency'] *
                                               (1 + time_component)).clip(0)

    elif drift_type == 'covariate_shift':
        # Covariate shift (change in feature distribution without label change)
        # Increase online transactions (digital transformation)
        online_boost = np.random.random(len(drifted_data)) < drift_intensity * 0.5
        drifted_data.loc[online_boost, 'merchant_category'] = 1  # online category

        # Change device preferences
        mobile_boost = np.random.random(len(drifted_data)) < drift_intensity * 0.3
        drifted_data.loc[mobile_boost, 'device_type'] = 0  # mobile

    logger.info(f"Simulated {drift_type} drift with intensity {drift_intensity:.1f}")
    return drifted_data


def demonstrate_enterprise_data_validation():
    """Demonstrate enterprise data validation capabilities."""
    logger.info("=== Enterprise Data Validation Demonstration ===")

    # Create pipeline
    pipeline = create_retraining_pipeline()

    # Test with clean data
    clean_data = create_enterprise_sample_data(1000)
    logger.info("Testing with clean data...")
    clean_validation = pipeline.data_validator.validate_dataset(clean_data)
    logger.info(f"Clean data validation: {'✅ Valid' if clean_validation['is_valid'] else '❌ Invalid'}")

    # Test with anomalous data
    anomalous_data = create_enterprise_sample_data(1000, with_anomalies=True)
    logger.info("Testing with anomalous data...")
    anomalous_validation = pipeline.data_validator.validate_dataset(anomalous_data)
    logger.info(f"Anomalous data validation: {'✅ Valid' if anomalous_validation['is_valid'] else '❌ Invalid'}")

    if not anomalous_validation['is_valid']:
        logger.info("Issues found:")
        for issue in anomalous_validation['issues']:
            logger.info(f"  - {issue}")

        logger.info("Cleaning data...")
        cleaned_data = pipeline.data_validator.clean_dataset(anomalous_data, anomalous_validation)
        logger.info(f"Data cleaned: {len(anomalous_data)} → {len(cleaned_data)} rows")


def demonstrate_enterprise_drift_detection():
    """Demonstrate enterprise-grade drift detection."""
    logger.info("\n=== Enterprise Drift Detection Demonstration ===")

    # Create pipeline
    pipeline = create_retraining_pipeline()

    # Initialize with baseline
    baseline_data = create_enterprise_sample_data(2000)
    pipeline.initialize_baseline(baseline_data.drop('is_fraud', axis=1))

    # Test different drift types
    drift_types = ['gradual', 'sudden', 'covariate_shift']

    for drift_type in drift_types:
        logger.info(f"\nTesting {drift_type} drift...")

        # Create drifted data
        drifted_data = simulate_enterprise_data_drift(baseline_data, drift_type, drift_intensity=0.4)

        # Detect drift
        drift_results = pipeline.drift_detector.detect_drift(drifted_data.drop('is_fraud', axis=1))

        logger.info(f"Drift detected: {drift_results['drift_detected']}")
        logger.info(f"Overall drift score: {drift_results['overall_drift_score']:.3f}")
        logger.info(f"Severity: {drift_results.get('severity', 'unknown')}")
        logger.info(f"Methods used: {', '.join(drift_results.get('methods_used', []))}")

        if drift_results['drift_detected']:
            logger.info("Significant drift in features:")
            for feature, info in drift_results['feature_drift'].items():
                if info['significant_drift']:
                    logger.info(f"  - {feature}: {info['drift_score']:.3f}")


def demonstrate_enterprise_performance_monitoring():
    """Demonstrate enterprise performance monitoring."""
    logger.info("\n=== Enterprise Performance Monitoring Demonstration ===")

    # Create pipeline
    pipeline = create_retraining_pipeline()

    # Simulate performance history with degradation
    logger.info("Simulating performance history...")

    base_metrics = {
        'accuracy': 0.92,
        'precision': 0.88,
        'recall': 0.85,
        'f1_score': 0.86,
        'auc_roc': 0.91
    }

    # Generate 50 performance evaluations with gradual degradation
    for i in range(50):
        # Add noise and gradual degradation
        noise = np.random.normal(0, 0.02, len(base_metrics))
        degradation = -i * 0.0005  # Gradual degradation

        current_metrics = {}
        for j, (metric, base_value) in enumerate(base_metrics.items()):
            current_metrics[metric] = max(0, base_value + noise[j] + degradation)

        # Simulate predictions (simplified)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.1, n_samples)
        y_pred = np.random.binomial(1, current_metrics['precision'] * 0.1, n_samples)

        # Evaluate performance
        metrics = pipeline.performance_monitor.evaluate_performance(y_true, y_pred)

        if i % 10 == 0:
            auc_value = metrics.get('auc_roc', 'N/A')
            auc_str = f"{auc_value:.3f}" if isinstance(auc_value, (int, float)) else str(auc_value)
            logger.info(f"Evaluation {i+1}: F1-Score = {metrics['f1_score']:.3f}, AUC = {auc_str}")

    # Check for degradation
    degradation_check = pipeline.performance_monitor.check_performance_degradation()
    logger.info(f"\nPerformance degradation detected: {degradation_check['degradation_detected']}")
    logger.info(f"Confidence: {degradation_check['confidence']:.2f}")

    if degradation_check['degradation_detected']:
        logger.info("Degradation details:")
        for metric, details in degradation_check['details'].items():
            logger.info(f"  - {metric}: {details['relative_drop']:.1%} drop")

    # Analyze trends
    trends = pipeline.performance_monitor.get_performance_trends(days=90)
    logger.info("\nPerformance trends (last 90 days):")
    for metric, trend_info in trends.get('trends', {}).items():
        logger.info(f"  - {metric}: {trend_info['trend_direction']} ({trend_info['trend_strength']})")


def demonstrate_enterprise_retraining_pipeline():
    """Demonstrate the complete enterprise retraining pipeline."""
    logger.info("\n=== Enterprise Retraining Pipeline Demonstration ===")

    # Create pipeline
    pipeline = create_retraining_pipeline()

    # Initialize with baseline
    baseline_data = create_enterprise_sample_data(3000)
    pipeline.initialize_baseline(baseline_data.drop('is_fraud', axis=1))

    # Perform health check
    logger.info("Performing initial health check...")
    health_status = pipeline.perform_health_check()
    logger.info(f"System health: {health_status['overall_health']}")

    # Simulate production scenario with drift
    logger.info("Simulating production scenario with data drift...")
    production_data = simulate_enterprise_data_drift(baseline_data, 'covariate_shift', 0.3)

    # Check retraining decision
    decision = pipeline.check_retraining_needed(production_data.drop('is_fraud', axis=1))
    logger.info(f"Retraining needed: {decision['retraining_needed']}")
    logger.info(f"Confidence: {decision['confidence']:.2f}")
    logger.info(f"Triggers: {decision['triggers']}")

    if decision['retraining_needed']:
        logger.info("Triggering enterprise retraining...")

        # Trigger retraining
        results = pipeline.trigger_retraining(baseline_data)

        if results['success']:
            logger.info("✅ Enterprise retraining completed successfully!")
            logger.info(f"Model version: {results['model_version']}")
            logger.info("Performance metrics:")
            for metric, value in results['metrics'].items():
                logger.info(f"  - {metric}: {value:.4f}")
            logger.info(f"Training duration: {results['duration']:.1f}s")
        else:
            logger.error(f"❌ Retraining failed: {results.get('error', 'Unknown error')}")
    else:
        logger.info("No retraining needed at this time")


def demonstrate_circuit_breaker_resilience():
    """Demonstrate circuit breaker resilience."""
    logger.info("\n=== Circuit Breaker Resilience Demonstration ===")

    from retraining.automated_retraining import CircuitBreaker

    # Create circuit breaker
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)  # 5 seconds for demo

    def failing_operation():
        raise Exception("Simulated failure")

    def successful_operation():
        return "Success!"

    # Simulate failures
    logger.info("Simulating operation failures...")
    for i in range(5):
        try:
            result = cb.call(failing_operation)
            logger.info(f"Attempt {i+1}: {result}")
        except CircuitBreakerOpenException:
            logger.info(f"Attempt {i+1}: Circuit breaker OPEN")
            break
        except Exception as e:
            logger.info(f"Attempt {i+1}: Failed - {e}")

    # Wait for recovery
    logger.info("Waiting for circuit breaker recovery...")
    import time
    time.sleep(6)

    # Try successful operation
    try:
        result = cb.call(successful_operation)
        logger.info(f"After recovery: {result}")
    except Exception as e:
        logger.info(f"After recovery: Failed - {e}")

    # Check status
    status = cb.get_status()
    logger.info(f"Circuit breaker status: {status}")


def demonstrate_enterprise_monitoring():
    """Demonstrate enterprise monitoring capabilities."""
    logger.info("\n=== Enterprise Monitoring Demonstration ===")

    # Create pipeline
    pipeline = create_retraining_pipeline()

    # Initialize
    baseline_data = create_enterprise_sample_data(1000)
    pipeline.initialize_baseline(baseline_data.drop('is_fraud', axis=1))

    # Get comprehensive status
    status = pipeline.get_monitoring_status()

    logger.info("Enterprise monitoring status:")
    logger.info(f"  - Model: {status.get('model_name', 'N/A')}")
    logger.info(f"  - Performance history: {status['performance_history_count']} evaluations")
    logger.info(f"  - Drift detector: {'✅ Active' if status['drift_detector_active'] else '❌ Inactive'}")
    logger.info(f"  - Retraining triggers: {', '.join(status['retraining_triggers'])}")
    logger.info(f"  - Data validation: {'✅ Enabled' if status['data_validation_enabled'] else '❌ Disabled'}")
    logger.info(f"  - Last retraining: {status.get('last_retraining', 'Never')}")

    # Health check details
    health = status['health_status']
    logger.info(f"  - System health: {health['overall_health']}")

    if health['overall_health'] != 'healthy':
        logger.info("  - Health issues:")
        for check_name, check_info in health['checks'].items():
            if check_info['status'] != 'healthy':
                logger.info(f"    * {check_name}: {check_info['status']}")

    # Circuit breaker status
    cb_status = status['circuit_breakers']
    logger.info("  - Circuit breakers:")
    for cb_name, cb_info in cb_status.items():
        logger.info(f"    * {cb_name}: {cb_info['state']}")


def main():
    """Main enterprise demonstration function."""
    logger.info("🚀 Starting AML Pipeline Enterprise Retraining Demonstration")
    logger.info("=" * 70)

    try:
        # Run enterprise demonstrations
        demonstrate_enterprise_data_validation()
        demonstrate_enterprise_drift_detection()
        demonstrate_enterprise_performance_monitoring()
        demonstrate_enterprise_retraining_pipeline()
        demonstrate_circuit_breaker_resilience()
        demonstrate_enterprise_monitoring()

        logger.info("\n" + "=" * 70)
        logger.info("🎉 Enterprise Retraining Demonstration Completed Successfully!")
        logger.info("\n✨ Key Enterprise Features Demonstrated:")
        logger.info("  • Robust statistical drift detection")
        logger.info("  • Circuit breaker resilience patterns")
        logger.info("  • Comprehensive data validation")
        logger.info("  • Advanced performance monitoring")
        logger.info("  • Enterprise-grade health checks")
        logger.info("  • Sophisticated retraining triggers")

    except Exception as e:
        logger.error(f"❌ Enterprise demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()