#!/bin/bash

# AML Pipeline MLOps Setup and Test Script
# This script sets up the complete MLOps environment and runs tests

set -e

echo "🚀 Setting up AML Pipeline MLOps Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
MLOPS_DIR="./mlops"
CONFIG_FILE="$MLOPS_DIR/mlops-config.ini"
MLFLOW_SERVER_SCRIPT="$MLOPS_DIR/mlflow-server.sh"

# Function to check Python environment
check_python() {
    print_status "Checking Python environment..."

    if ! command -v python > /dev/null 2>&1; then
        print_error "Python is not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python version: $PYTHON_VERSION"

    # Check if required packages are installed
    python -c "import mlflow" 2>/dev/null || {
        print_warning "MLflow not found. Installing..."
        pip install mlflow scikit-learn pandas numpy
        print_success "MLflow and dependencies installed"
    }
}

# Function to validate configuration
validate_config() {
    print_status "Validating MLOps configuration..."

    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi

    # Basic config validation
    if ! python -c "
import configparser
config = configparser.ConfigParser()
config.read('$CONFIG_FILE')
print('Configuration loaded successfully')
required_sections = ['mlflow', 'experiment_tracking', 'model_registry']
for section in required_sections:
    if not config.has_section(section):
        raise ValueError(f'Missing required section: {section}')
"; then
        print_error "Configuration validation failed"
        exit 1
    fi

    print_success "Configuration validated"
}

# Function to initialize directories
init_directories() {
    print_status "Initializing MLOps directories..."

    mkdir -p "$MLOPS_DIR/artifacts"
    mkdir -p "$MLOPS_DIR/experiments"
    mkdir -p "$MLOPS_DIR/models"
    mkdir -p "$MLOPS_DIR/retraining"
    mkdir -p "$MLOPS_DIR/logs"

    print_success "Directories initialized"
}

# Function to test MLflow integration
test_mlflow_integration() {
    print_status "Testing MLflow integration..."

    # Start MLflow server
    print_status "Starting MLflow server..."
    bash "$MLFLOW_SERVER_SCRIPT" start

    # Wait for server to be ready
    sleep 5

    # Test basic functionality
    if ! python -c "
import sys
sys.path.insert(0, '.')
from mlops.model_registry.mlflow_integration import AMLPipelineMLflow

# Test basic integration
mlflow_int = AMLPipelineMLflow()
print('MLflow integration initialized')

# Test experiment creation
exp_id = mlflow_int.create_experiment('test_experiment', 'Test experiment for validation')
print(f'Created experiment: {exp_id}')

# Test experiment listing
experiments = mlflow_int.list_experiments()
print(f'Found {len(experiments)} experiments')

print('MLflow integration test passed')
"; then
        print_error "MLflow integration test failed"
        # Stop server on failure
        bash "$MLFLOW_SERVER_SCRIPT" stop
        exit 1
    fi

    print_success "MLflow integration test passed"

    # Stop server
    bash "$MLFLOW_SERVER_SCRIPT" stop
}

# Function to run sample experiment
run_sample_experiment() {
    print_status "Running sample experiment..."

    # Start MLflow server
    bash "$MLFLOW_SERVER_SCRIPT" start
    sleep 5

    # Run experiment
    if python mlops/experiments/fraud_detection_experiment.py; then
        print_success "Sample experiment completed successfully"
    else
        print_error "Sample experiment failed"
        bash "$MLFLOW_SERVER_SCRIPT" stop
        exit 1
    fi

    # Stop server
    bash "$MLFLOW_SERVER_SCRIPT" stop
}

# Function to create demo dashboard
create_demo_dashboard() {
    print_status "Creating demo dashboard..."

    # Start MLflow server
    bash "$MLFLOW_SERVER_SCRIPT" start
    sleep 5

    # Run experiment to generate data
    python mlops/experiments/fraud_detection_experiment.py > /dev/null 2>&1

    print_success "Demo dashboard data created"
    print_status "MLflow UI available at: http://localhost:5000"

    # Keep server running for demo
    print_status "MLflow server is running. Press Ctrl+C to stop."
    trap 'bash "$MLFLOW_SERVER_SCRIPT" stop' INT
    wait
}

# Function to show usage
usage() {
    echo "AML Pipeline MLOps Setup and Test Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     Complete MLOps environment setup"
    echo "  test      Run MLOps integration tests"
    echo "  demo      Run sample experiment and show dashboard"
    echo "  experiment Run sample fraud detection experiment"
    echo "  clean     Clean up MLOps environment"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Complete setup and validation"
    echo "  $0 demo       # Run demo with UI"
    echo "  $0 test       # Run integration tests"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up MLOps environment..."

    # Stop MLflow server
    if [ -f "$MLFLOW_SERVER_SCRIPT" ]; then
        bash "$MLFLOW_SERVER_SCRIPT" stop
    fi

    # Remove generated files
    rm -rf "$MLOPS_DIR/artifacts" 2>/dev/null || true
    rm -rf "$MLOPS_DIR/logs" 2>/dev/null || true
    rm -f "$MLOPS_DIR/mlflow.db" 2>/dev/null || true
    rm -f "$MLOPS_DIR/mlflow.log" 2>/dev/null || true
    rm -f "$MLOPS_DIR/mlflow.pid" 2>/dev/null || true

    print_success "Cleanup completed"
}

# Main script logic
case "${1:-}" in
    "setup")
        check_python
        validate_config
        init_directories
        test_mlflow_integration
        print_success "MLOps environment setup completed!"
        ;;
    "test")
        check_python
        validate_config
        test_mlflow_integration
        print_success "MLOps tests passed!"
        ;;
    "demo")
        check_python
        validate_config
        init_directories
        create_demo_dashboard
        ;;
    "experiment")
        check_python
        validate_config
        run_sample_experiment
        ;;
    "clean")
        cleanup
        ;;
    *)
        usage
        exit 1
        ;;
esac