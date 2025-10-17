#!/bin/bash

# AML Pipeline Automated Retraining Setup and Testing Script
# This script sets up and tests the automated retraining system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/mlops-config.ini"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if configuration file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Please run the main MLOps setup script first"
        exit 1
    fi
    log_success "Configuration file found: $CONFIG_FILE"
}

# Check Python environment and dependencies
check_python_environment() {
    log_info "Checking Python environment..."

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"

    # Check required packages
    REQUIRED_PACKAGES=("pandas" "numpy" "scikit-learn" "mlflow" "schedule")
    MISSING_PACKAGES=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        log_warning "Missing required packages: ${MISSING_PACKAGES[*]}"
        log_info "Installing missing packages..."
        pip install "${MISSING_PACKAGES[@]}"
        log_success "Packages installed successfully"
    else
        log_success "All required packages are installed"
    fi
}

# Validate retraining configuration
validate_config() {
    log_info "Validating retraining configuration..."

    # Check if retraining section exists
    if ! grep -q "\[model_retraining\]" "$CONFIG_FILE"; then
        log_error "model_retraining section not found in config file"
        exit 1
    fi

    # Check required parameters
    REQUIRED_PARAMS=("retraining_triggers" "drift_threshold" "time_based_retraining_days")
    for param in "${REQUIRED_PARAMS[@]}"; do
        if ! grep -q "^${param} =" "$CONFIG_FILE"; then
            log_error "Required parameter '$param' not found in config"
            exit 1
        fi
    done

    log_success "Retraining configuration is valid"
}

# Test automated retraining components
test_retraining_components() {
    log_info "Testing automated retraining components..."

    # Test data drift detector
    log_info "Testing data drift detector..."
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from mlops.retraining.automated_retraining import DataDriftDetector
import pandas as pd
import numpy as np
import configparser

# Load config
config = configparser.ConfigParser()
config.read('${CONFIG_FILE}')

# Create test data
np.random.seed(42)
baseline_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.exponential(1, 1000),
    'feature3': np.random.choice(['A', 'B', 'C'], 1000)
})

# Initialize detector
detector = DataDriftDetector(baseline_data, config)

# Test with same data (should not detect drift)
no_drift_result = detector.detect_drift(baseline_data)
assert not no_drift_result['drift_detected'], 'False positive drift detection'

# Test with drifted data
drifted_data = baseline_data.copy()
drifted_data['feature1'] = drifted_data['feature1'] * 2 + 1  # Significant shift
drift_result = detector.detect_drift(drifted_data)
assert drift_result['drift_detected'], 'Failed to detect significant drift'

print('Data drift detector test passed')
" && log_success "Data drift detector test passed" || { log_error "Data drift detector test failed"; exit 1; }

    # Test performance monitor
    log_info "Testing performance monitor..."
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from mlops.retraining.automated_retraining import ModelPerformanceMonitor
import numpy as np
import configparser

# Load config
config = configparser.ConfigParser()
config.read('${CONFIG_FILE}')

# Initialize monitor
monitor = ModelPerformanceMonitor(config)

# Test performance evaluation
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
y_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.1, 0.7, 0.3])

metrics = monitor.evaluate_performance(y_true, y_pred, y_proba)
required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in required_metrics:
    assert metric in metrics, f'Missing metric: {metric}'

print('Performance monitor test passed')
" && log_success "Performance monitor test passed" || { log_error "Performance monitor test failed"; exit 1; }

    # Test automated retraining pipeline
    log_info "Testing automated retraining pipeline..."
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from mlops.retraining.automated_retraining import AutomatedRetrainingPipeline
import pandas as pd
import numpy as np

# Create test data
np.random.seed(42)
test_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.exponential(1, 100),
    'is_fraud': np.random.binomial(1, 0.1, 100)
})

# Initialize pipeline
pipeline = AutomatedRetrainingPipeline('${CONFIG_FILE}')
pipeline.initialize_baseline(test_data[['feature1', 'feature2']])

# Test retraining check
decision = pipeline.check_retraining_needed(test_data[['feature1', 'feature2']])
assert 'retraining_needed' in decision, 'Missing retraining decision'
assert 'triggers' in decision, 'Missing triggers'
assert 'confidence' in decision, 'Missing confidence'

print('Automated retraining pipeline test passed')
" && log_success "Automated retraining pipeline test passed" || { log_error "Automated retraining pipeline test failed"; exit 1; }
}

# Run demonstration script
run_demo() {
    log_info "Running retraining demonstration..."

    if [ -f "${PROJECT_ROOT}/mlops/experiments/retraining_demo.py" ]; then
        cd "${PROJECT_ROOT}"
        python mlops/experiments/retraining_demo.py
        log_success "Retraining demonstration completed"
    else
        log_error "Demo script not found: ${PROJECT_ROOT}/mlops/experiments/retraining_demo.py"
        exit 1
    fi
}

# Setup monitoring daemon (optional)
setup_monitoring_daemon() {
    log_info "Setting up monitoring daemon..."

    # Create systemd service file (Linux)
    if command -v systemctl &> /dev/null; then
        SERVICE_FILE="/etc/systemd/system/aml-retraining-monitor.service"
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AML Pipeline Retraining Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${PROJECT_ROOT}
ExecStart=/usr/bin/python ${PROJECT_ROOT}/mlops/retraining/automated_retraining.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        log_info "Created systemd service: $SERVICE_FILE"
        log_info "To enable: sudo systemctl enable aml-retraining-monitor"
        log_info "To start: sudo systemctl start aml-retraining-monitor"
    else
        log_warning "Systemd not available, skipping daemon setup"
        log_info "For manual daemon execution, run:"
        log_info "python ${PROJECT_ROOT}/mlops/retraining/automated_retraining.py"
    fi
}

# Main setup function
main() {
    log_info "Starting AML Pipeline Automated Retraining Setup"
    echo "=================================================="

    check_config
    check_python_environment
    validate_config
    test_retraining_components
    run_demo

    echo
    log_success "Automated Retraining Setup Completed Successfully!"
    echo
    log_info "Next steps:"
    log_info "1. Review the demonstration output above"
    log_info "2. Configure your data sources in the pipeline"
    log_info "3. Set up production monitoring (optional)"
    log_info "4. Integrate with your existing ML pipeline"
    echo
    log_info "To start monitoring manually:"
    log_info "cd ${PROJECT_ROOT} && python mlops/experiments/retraining_demo.py"
    echo
    log_info "For production deployment:"
    log_info "1. Update data source connections in the pipeline"
    log_info "2. Configure alerting and notification callbacks"
    log_info "3. Set up proper logging and monitoring infrastructure"
    log_info "4. Test with real production data before deployment"

    # Ask about daemon setup
    echo
    read -p "Do you want to set up the monitoring daemon? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_monitoring_daemon
    fi
}

# Run main function
main "$@"