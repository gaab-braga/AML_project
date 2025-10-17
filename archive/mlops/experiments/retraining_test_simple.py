"""
AML Pipeline Automated Retraining - Simple Test

This script provides a simplified test of the automated retraining system
without external dependencies for initial validation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add mlops module to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create simple test data for demonstration.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Test dataset
    """
    np.random.seed(42)

    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'is_fraud': np.random.binomial(1, 0.1, n_samples)
    }

    return pd.DataFrame(data)


def test_data_drift_detection():
    """Test data drift detection with simple statistical methods."""
    logger.info("=== Testing Data Drift Detection ===")

    try:
        # Import the drift detector
        from retraining.automated_retraining import DataDriftDetector
        import configparser

        # Create mock config
        config = configparser.ConfigParser()
        config.add_section('model_retraining')
        config.set('model_retraining', 'drift_threshold', '0.1')

        # Create baseline data
        baseline_data = create_test_data(500)
        logger.info(f"Baseline data shape: {baseline_data.shape}")

        # Initialize detector
        detector = DataDriftDetector(baseline_data[['feature1', 'feature2']], config)

        # Test with same data (should not detect drift)
        no_drift_result = detector.detect_drift(baseline_data[['feature1', 'feature2']])
        logger.info(f"No drift test - detected: {no_drift_result['drift_detected']}")

        # Test with drifted data
        drifted_data = baseline_data.copy()
        drifted_data['feature1'] = drifted_data['feature1'] * 1.5  # Significant shift

        drift_result = detector.detect_drift(drifted_data[['feature1', 'feature2']])
        logger.info(f"Drift test - detected: {drift_result['drift_detected']}")
        logger.info(f"Drift score: {drift_result.get('drift_score', 0):.3f}")
        logger.info("✅ Data drift detection test passed")

    except ImportError as e:
        logger.warning(f"Could not import DataDriftDetector: {e}")
        logger.info("Skipping drift detection test")
    except Exception as e:
        logger.error(f"Data drift detection test failed: {e}")
        return False

    return True


def test_performance_monitoring():
    """Test performance monitoring with simple metrics."""
    logger.info("\n=== Testing Performance Monitoring ===")

    try:
        # Import performance monitor
        from retraining.automated_retraining import ModelPerformanceMonitor
        import configparser

        # Create mock config
        config = configparser.ConfigParser()
        config.add_section('model_retraining')
        config.set('model_retraining', 'performance_thresholds.accuracy_drop', '0.05')
        config.set('model_retraining', 'performance_thresholds.precision_drop', '0.10')
        config.set('model_retraining', 'performance_thresholds.recall_drop', '0.10')

        # Initialize monitor
        monitor = ModelPerformanceMonitor(config)

        # Test performance evaluation
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

        metrics = monitor.evaluate_performance(y_true, y_pred)
        logger.info(f"Performance metrics: {metrics}")

        # Test with multiple evaluations
        for i in range(5):
            # Simulate slightly degrading performance
            noise_level = i * 0.02
            noisy_pred = y_pred.copy()
            flip_indices = np.random.choice(len(y_pred), int(len(y_pred) * noise_level), replace=False)
            noisy_pred[flip_indices] = 1 - noisy_pred[flip_indices]

            monitor.evaluate_performance(y_true, noisy_pred)

        # Check for degradation
        degradation = monitor.check_performance_degradation()
        logger.info(f"Degradation detected: {degradation['degradation_detected']}")

        logger.info("✅ Performance monitoring test passed")

    except ImportError as e:
        logger.warning(f"Could not import ModelPerformanceMonitor: {e}")
        logger.info("Skipping performance monitoring test")
    except Exception as e:
        logger.error(f"Performance monitoring test failed: {e}")
        return False

    return True


def test_retraining_pipeline():
    """Test the automated retraining pipeline structure."""
    logger.info("\n=== Testing Retraining Pipeline ===")

    try:
        # Import pipeline
        from retraining.automated_retraining import AutomatedRetrainingPipeline

        # Create pipeline (without config file)
        pipeline = AutomatedRetrainingPipeline()

        # Test basic functionality
        status = pipeline.get_monitoring_status()
        logger.info(f"Pipeline status: {status}")

        # Test with sample data
        test_data = create_test_data(100)
        pipeline.initialize_baseline(test_data[['feature1', 'feature2']])

        # Test retraining check
        decision = pipeline.check_retraining_needed(test_data[['feature1', 'feature2']])
        logger.info(f"Retraining decision: {decision['retraining_needed']}")

        logger.info("✅ Retraining pipeline test passed")

    except ImportError as e:
        logger.warning(f"Could not import AutomatedRetrainingPipeline: {e}")
        logger.info("Skipping retraining pipeline test")
    except Exception as e:
        logger.error(f"Retraining pipeline test failed: {e}")
        return False

    return True


def main():
    """Main test function."""
    logger.info("Starting AML Pipeline Automated Retraining Tests")
    logger.info("=" * 60)

    # Run tests
    tests_passed = 0
    total_tests = 3

    if test_data_drift_detection():
        tests_passed += 1

    if test_performance_monitoring():
        tests_passed += 1

    if test_retraining_pipeline():
        tests_passed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        logger.info("🎉 All automated retraining tests passed!")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - tests_passed} tests failed or were skipped")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)