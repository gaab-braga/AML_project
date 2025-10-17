#!/bin/bash
# AML Pipeline Deployment Script
# Supports blue-green deployment to AWS ECS

set -e

# Configuration
CLUSTER_NAME="aml-pipeline-prod"
SERVICE_NAME_BASE="aml-api-service"
REGION="us-east-1"
PROFILE=${AWS_PROFILE:-default}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Check AWS CLI configuration
check_aws_config() {
    log_info "Checking AWS configuration..."
    if ! aws sts get-caller-identity --profile $PROFILE >/dev/null 2>&1; then
        log_error "AWS CLI not configured or invalid credentials"
        exit 1
    fi
    log_success "AWS CLI configured correctly"
}

# Get current color (blue or green)
get_current_color() {
    log_info "Determining current active color..."

    # Get task definition ARN from current service
    CURRENT_TASK_DEF=$(aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME_BASE \
        --profile $PROFILE \
        --region $REGION \
        --query 'services[0].taskDefinition' \
        --output text 2>/dev/null || echo "none")

    if [ "$CURRENT_TASK_DEF" = "none" ]; then
        log_info "No active service found, defaulting to blue"
        echo "blue"
        return
    fi

    # Extract color from task definition
    if echo "$CURRENT_TASK_DEF" | grep -q "blue"; then
        echo "blue"
    elif echo "$CURRENT_TASK_DEF" | grep -q "green"; then
        echo "green"
    else
        log_warning "Could not determine color from task definition, defaulting to blue"
        echo "blue"
    fi
}

# Update task definition with new image
update_task_definition() {
    local color=$1
    local image_tag=$2

    log_info "Updating task definition for $color environment..."

    # Register new task definition
    aws ecs register-task-definition \
        --cli-input-json file://deploy/ecs-task-definition.json \
        --profile $PROFILE \
        --region $REGION

    # Update the image in the task definition
    aws ecs describe-task-definition \
        --task-definition aml-pipeline-api \
        --profile $PROFILE \
        --region $REGION \
        --query 'taskDefinition' \
        | jq --arg IMAGE "ghcr.io/$GITHUB_REPOSITORY:$image_tag-api" \
             '.containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)' \
        > deploy/ecs-task-definition-updated.json

    # Register updated task definition
    aws ecs register-task-definition \
        --cli-input-json file://deploy/ecs-task-definition-updated.json \
        --profile $PROFILE \
        --region $REGION

    log_success "Task definition updated for $color environment"
}

# Deploy to specified color
deploy_to_color() {
    local color=$1
    local service_name="$SERVICE_NAME_BASE-$color"

    log_info "Deploying to $color environment (service: $service_name)..."

    # Update service to use new task definition
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $service_name \
        --task-definition aml-pipeline-api \
        --force-new-deployment \
        --profile $PROFILE \
        --region $REGION

    log_success "Deployment initiated for $color environment"

    # Wait for service to be stable
    log_info "Waiting for deployment to complete..."
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $service_name \
        --profile $PROFILE \
        --region $REGION

    log_success "Deployment completed for $color environment"
}

# Switch traffic using Route 53 weighted routing
switch_traffic() {
    local new_color=$1
    local hosted_zone_id=${ROUTE53_HOSTED_ZONE_ID}

    log_info "Switching traffic to $new_color environment..."

    # Update Route 53 weights
    aws route53 change-resource-record-sets \
        --hosted-zone-id $hosted_zone_id \
        --change-batch "{
            \"Changes\": [{
                \"Action\": \"UPSERT\",
                \"ResourceRecordSet\": {
                    \"Name\": \"api.aml-pipeline.com\",
                    \"Type\": \"CNAME\",
                    \"SetIdentifier\": \"$new_color\",
                    \"Weight\": 100,
                    \"TTL\": 60,
                    \"ResourceRecords\": [{\"Value\": \"$new_color-api.aml-pipeline.com\"}]
                }
            }]
        }" \
        --profile $PROFILE \
        --region $REGION

    log_success "Traffic switched to $new_color environment"
}

# Run smoke tests
run_smoke_tests() {
    local color=$1
    local url="https://$color-api.aml-pipeline.com"

    log_info "Running smoke tests on $url..."

    # Wait for DNS propagation
    sleep 60

    # Test health endpoint
    if curl -f --max-time 30 "$url/health" >/dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi

    # Test API docs
    if curl -f --max-time 30 "$url/docs" >/dev/null 2>&1; then
        log_success "API docs accessible"
    else
        log_error "API docs not accessible"
        return 1
    fi

    log_success "All smoke tests passed"
}

# Rollback function
rollback() {
    local from_color=$1
    local to_color=$2

    log_warning "Rolling back from $from_color to $to_color..."

    switch_traffic $to_color

    # Scale down the failed environment
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME_BASE-$from_color \
        --desired-count 0 \
        --profile $PROFILE \
        --region $REGION

    log_info "Rollback completed"
}

# Main deployment function
main() {
    local environment=${1:-production}
    local image_tag=${2:-latest}

    log_info "Starting AML Pipeline deployment to $environment"
    log_info "Image tag: $image_tag"

    # Validate inputs
    if [ "$environment" != "staging" ] && [ "$environment" != "production" ]; then
        log_error "Invalid environment. Must be 'staging' or 'production'"
        exit 1
    fi

    # Check AWS configuration
    check_aws_config

    # For staging, deploy directly
    if [ "$environment" = "staging" ]; then
        update_task_definition "staging" $image_tag
        deploy_to_color "staging"

        if run_smoke_tests "staging"; then
            log_success "Staging deployment completed successfully!"
        else
            log_error "Staging deployment failed smoke tests"
            exit 1
        fi
        exit 0
    fi

    # Production blue-green deployment
    local current_color=$(get_current_color)
    local new_color="blue"

    if [ "$current_color" = "blue" ]; then
        new_color="green"
    fi

    log_info "Current active color: $current_color"
    log_info "Deploying to color: $new_color"

    # Update task definition
    update_task_definition $new_color $image_tag

    # Deploy to new color
    deploy_to_color $new_color

    # Run smoke tests on new environment
    if run_smoke_tests $new_color; then
        # Switch traffic
        switch_traffic $new_color

        # Scale down old environment after successful traffic switch
        log_info "Scaling down $current_color environment..."
        aws ecs update-service \
            --cluster $CLUSTER_NAME \
            --service $SERVICE_NAME_BASE-$current_color \
            --desired-count 0 \
            --profile $PROFILE \
            --region $REGION

        log_success "Production deployment completed successfully!"
        log_success "New active environment: $new_color"

    else
        log_error "Smoke tests failed on $new_color environment"
        log_info "Initiating rollback..."

        rollback $new_color $current_color
        log_error "Deployment failed and rolled back"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [environment] [image_tag]"
    echo ""
    echo "Arguments:"
    echo "  environment    Target environment (staging|production) [default: production]"
    echo "  image_tag      Docker image tag [default: latest]"
    echo ""
    echo "Examples:"
    echo "  $0 staging v1.2.3"
    echo "  $0 production latest"
    echo ""
    echo "Environment variables:"
    echo "  AWS_PROFILE                 AWS CLI profile to use"
    echo "  ROUTE53_HOSTED_ZONE_ID      Route 53 hosted zone ID for production"
    echo "  GITHUB_REPOSITORY           GitHub repository (owner/repo format)"
}

# Parse arguments
case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac