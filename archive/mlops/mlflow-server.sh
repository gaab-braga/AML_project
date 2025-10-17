#!/bin/bash

# AML Pipeline MLflow Setup and Management Script
# This script manages the MLflow tracking server and model registry

set -e

echo "🧪 Setting up AML Pipeline MLflow Environment..."

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
MLFLOW_DIR="./mlops"
CONFIG_FILE="$MLFLOW_DIR/mlops-config.ini"
BACKEND_STORE="$MLFLOW_DIR/mlflow.db"
ARTIFACT_STORE="$MLFLOW_DIR/artifacts"
HOST="0.0.0.0"
PORT="5000"

# Function to check if MLflow is installed
check_mlflow() {
    if ! command -v mlflow > /dev/null 2>&1; then
        print_error "MLflow is not installed. Installing..."
        pip install mlflow
        print_success "MLflow installed successfully"
    else
        print_success "MLflow is already installed"
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating MLflow directories..."

    mkdir -p "$ARTIFACT_STORE"
    mkdir -p "$MLFLOW_DIR/experiments"
    mkdir -p "$MLFLOW_DIR/models"

    print_success "Directories created"
}

# Function to initialize MLflow database
init_database() {
    print_status "Initializing MLflow database..."

    if [ ! -f "$BACKEND_STORE" ]; then
        # Create database schema
        python -c "
import sqlite3
import os

db_path = '$BACKEND_STORE'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create basic schema for model registry
cursor.execute('''
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT,
    version INTEGER,
    source TEXT,
    run_id TEXT,
    status TEXT,
    creation_timestamp INTEGER,
    last_updated_timestamp INTEGER,
    user_id TEXT,
    description TEXT,
    PRIMARY KEY (name, version)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS registered_models (
    name TEXT PRIMARY KEY,
    creation_timestamp INTEGER,
    last_updated_timestamp INTEGER,
    description TEXT
)
''')

conn.commit()
conn.close()
print('MLflow database initialized')
"
        print_success "MLflow database initialized"
    else
        print_success "MLflow database already exists"
    fi
}

# Function to start MLflow server
start_server() {
    print_status "Starting MLflow Tracking Server..."

    # Check if port is available
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $PORT is already in use. MLflow server might already be running."
        return 1
    fi

    # Set environment variables
    export MLFLOW_TRACKING_URI="http://$HOST:$PORT"
    export MLFLOW_BACKEND_STORE_URI="sqlite:///$BACKEND_STORE"
    export MLFLOW_ARTIFACT_STORE_URI="file:$ARTIFACT_STORE"

    # Start server in background
    nohup mlflow server \
        --host $HOST \
        --port $PORT \
        --backend-store-uri "sqlite:///$BACKEND_STORE" \
        --default-artifact-root "file:$ARTIFACT_STORE" \
        --workers 4 \
        > "$MLFLOW_DIR/mlflow.log" 2>&1 &

    SERVER_PID=$!

    # Save PID
    echo $SERVER_PID > "$MLFLOW_DIR/mlflow.pid"

    print_success "MLflow server started (PID: $SERVER_PID)"

    # Wait for server to be ready
    print_status "Waiting for MLflow server to be ready..."
    for i in {1..30}; do
        if curl -f -s "http://localhost:$PORT/api/2.0/mlflow/experiments/list" > /dev/null 2>&1; then
            print_success "MLflow server is ready!"
            echo ""
            echo "🌐 MLflow UI: http://localhost:$PORT"
            echo "📊 API Endpoint: http://localhost:$PORT/api/2.0/mlflow"
            echo "🗄️  Database: $BACKEND_STORE"
            echo "📁 Artifacts: $ARTIFACT_STORE"
            return 0
        fi
        sleep 2
    done

    print_error "MLflow server failed to start properly"
    return 1
}

# Function to stop MLflow server
stop_server() {
    print_status "Stopping MLflow server..."

    if [ -f "$MLFLOW_DIR/mlflow.pid" ]; then
        PID=$(cat "$MLFLOW_DIR/mlflow.pid")

        if kill -0 $PID 2>/dev/null; then
            kill $PID
            sleep 2

            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID
                print_warning "Force killed MLflow server"
            else
                print_success "MLflow server stopped gracefully"
            fi
        else
            print_warning "MLflow server was not running"
        fi

        rm -f "$MLFLOW_DIR/mlflow.pid"
    else
        print_warning "No PID file found. MLflow server may not be running."
    fi
}

# Function to check server status
check_status() {
    if [ -f "$MLFLOW_DIR/mlflow.pid" ]; then
        PID=$(cat "$MLFLOW_DIR/mlflow.pid")

        if kill -0 $PID 2>/dev/null; then
            print_success "MLflow server is running (PID: $PID)"

            # Check if responsive
            if curl -f -s "http://localhost:$PORT/api/2.0/mlflow/experiments/list" > /dev/null 2>&1; then
                print_success "MLflow server is responsive"
            else
                print_warning "MLflow server is running but not responsive"
            fi
        else
            print_error "MLflow server is not running (stale PID file)"
            rm -f "$MLFLOW_DIR/mlflow.pid"
        fi
    else
        print_status "MLflow server is not running"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$MLFLOW_DIR/mlflow.log" ]; then
        tail -f "$MLFLOW_DIR/mlflow.log"
    else
        print_error "No log file found"
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up MLflow environment..."

    stop_server

    if [ -f "$BACKEND_STORE" ]; then
        read -p "Delete MLflow database? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$BACKEND_STORE"
            print_success "Database deleted"
        fi
    fi

    if [ -d "$ARTIFACT_STORE" ]; then
        read -p "Delete artifacts directory? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ARTIFACT_STORE"
            print_success "Artifacts deleted"
        fi
    fi

    print_success "Cleanup completed"
}

# Function to show usage
usage() {
    echo "AML Pipeline MLflow Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start MLflow tracking server"
    echo "  stop      Stop MLflow tracking server"
    echo "  restart   Restart MLflow tracking server"
    echo "  status    Check MLflow server status"
    echo "  logs      Show MLflow server logs"
    echo "  cleanup   Clean up MLflow environment"
    echo "  init      Initialize MLflow environment (first time setup)"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start the MLflow server"
    echo "  $0 status    # Check if server is running"
    echo "  $0 logs      # View server logs"
}

# Main script logic
case "${1:-}" in
    "start")
        check_mlflow
        create_directories
        init_database
        start_server
        ;;
    "stop")
        stop_server
        ;;
    "restart")
        stop_server
        sleep 2
        start_server
        ;;
    "status")
        check_status
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "init")
        check_mlflow
        create_directories
        init_database
        print_success "MLflow environment initialized. Run '$0 start' to start the server."
        ;;
    *)
        usage
        exit 1
        ;;
esac