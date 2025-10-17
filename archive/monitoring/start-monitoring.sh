#!/bin/bash

# AML Pipeline Monitoring Stack Startup Script
# This script initializes the complete monitoring stack for the AML Pipeline

set -e

echo "🚀 Starting AML Pipeline Monitoring Stack..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose file exists
if [ ! -f "docker-compose.monitoring.yml" ]; then
    print_error "docker-compose.monitoring.yml not found in current directory"
    exit 1
fi

print_status "Checking for port conflicts..."

# Check if ports are available
PORTS=(9090 3000 9093 9091)
for port in "${PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $port is already in use. This might cause issues."
    fi
done

print_status "Starting monitoring services..."

# Start the monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

print_status "Waiting for services to be healthy..."

# Wait for services to be ready
sleep 10

# Check service health
services=("prometheus" "grafana" "alertmanager" "node-exporter" "cadvisor")
for service in "${services[@]}"; do
    if docker-compose -f docker-compose.monitoring.yml ps $service | grep -q "Up"; then
        print_success "$service is running"
    else
        print_error "$service failed to start"
    fi
done

print_success "Monitoring stack started successfully!"
echo ""
echo "📊 Access your monitoring dashboards:"
echo "   Grafana:        http://localhost:3000 (admin/admin)"
echo "   Prometheus:     http://localhost:9090"
echo "   Alertmanager:   http://localhost:9093"
echo "   cAdvisor:       http://localhost:8080"
echo ""
echo "🔗 Pre-configured dashboards:"
echo "   - AML Pipeline Overview"
echo "   - AML Business Metrics"
echo "   - AML Alerts & Incidents"
echo ""
print_warning "Remember to update default passwords in production!"