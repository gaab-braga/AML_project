#!/bin/bash

# AML Pipeline Production Secrets Setup Script
# This script helps configure AWS Secrets Manager and GitHub secrets for production deployment

set -e

echo "🔐 Setting up AML Pipeline Production Secrets..."

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

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    print_error "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

print_status "AWS CLI is configured. Proceeding with secrets setup..."

# Get AWS account ID and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)

print_status "Using AWS Account: $AWS_ACCOUNT_ID"
print_status "Using AWS Region: $AWS_REGION"

# Create AWS Secrets Manager secrets
create_secret() {
    local secret_name=$1
    local secret_value=$2
    local description=$3

    print_status "Creating secret: $secret_name"

    if aws secretsmanager describe-secret --secret-id "$secret_name" > /dev/null 2>&1; then
        print_warning "Secret $secret_name already exists. Skipping..."
    else
        aws secretsmanager create-secret \
            --name "$secret_name" \
            --description "$description" \
            --secret-string "$secret_value" \
            --region "$AWS_REGION"

        print_success "Created secret: $secret_name"
    fi
}

# Database secrets
print_status "Setting up database secrets..."
read -p "Enter PostgreSQL host: " DB_HOST
read -p "Enter PostgreSQL port (default: 5432): " DB_PORT
DB_PORT=${DB_PORT:-5432}
read -p "Enter PostgreSQL database name: " DB_NAME
read -p "Enter PostgreSQL username: " DB_USER
read -s -p "Enter PostgreSQL password: " DB_PASSWORD
echo ""

DB_SECRET=$(cat <<EOF
{
  "host": "$DB_HOST",
  "port": $DB_PORT,
  "database": "$DB_NAME",
  "username": "$DB_USER",
  "password": "$DB_PASSWORD"
}
EOF
)

create_secret "aml-pipeline/database" "$DB_SECRET" "AML Pipeline PostgreSQL database credentials"

# Redis secrets
print_status "Setting up Redis secrets..."
read -p "Enter Redis host: " REDIS_HOST
read -p "Enter Redis port (default: 6379): " REDIS_PORT
REDIS_PORT=${REDIS_PORT:-6379}
read -p "Enter Redis password (leave empty if no password): " REDIS_PASSWORD

REDIS_SECRET=$(cat <<EOF
{
  "host": "$REDIS_HOST",
  "port": $REDIS_PORT,
  "password": "$REDIS_PASSWORD"
}
EOF
)

create_secret "aml-pipeline/redis" "$REDIS_SECRET" "AML Pipeline Redis cache credentials"

# JWT secrets
print_status "Setting up JWT secrets..."
JWT_SECRET=$(openssl rand -hex 32)
JWT_REFRESH_SECRET=$(openssl rand -hex 32)

JWT_SECRETS=$(cat <<EOF
{
  "jwt_secret": "$JWT_SECRET",
  "jwt_refresh_secret": "$JWT_REFRESH_SECRET"
}
EOF
)

create_secret "aml-pipeline/jwt" "$JWT_SECRETS" "AML Pipeline JWT signing secrets"

# API keys and external services
print_status "Setting up external service API keys..."
read -p "Enter OpenAI API key (leave empty if not using): " OPENAI_API_KEY
read -p "Enter Slack webhook URL for alerts (leave empty if not using): " SLACK_WEBHOOK_URL
read -p "Enter email SMTP server (leave empty if not using): " SMTP_SERVER
read -p "Enter email SMTP port (default: 587): " SMTP_PORT
SMTP_PORT=${SMTP_PORT:-587}
read -p "Enter email SMTP username: " SMTP_USER
read -s -p "Enter email SMTP password: " SMTP_PASSWORD
echo ""

EXTERNAL_SECRETS=$(cat <<EOF
{
  "openai_api_key": "$OPENAI_API_KEY",
  "slack_webhook_url": "$SLACK_WEBHOOK_URL",
  "smtp_server": "$SMTP_SERVER",
  "smtp_port": $SMTP_PORT,
  "smtp_user": "$SMTP_USER",
  "smtp_password": "$SMTP_PASSWORD"
}
EOF
)

create_secret "aml-pipeline/external-services" "$EXTERNAL_SECRETS" "AML Pipeline external service API keys"

# Docker registry secrets
print_status "Setting up Docker registry secrets..."
read -p "Enter Docker registry URL (e.g., your-registry.com): " DOCKER_REGISTRY
read -p "Enter Docker registry username: " DOCKER_USER
read -s -p "Enter Docker registry password/token: " DOCKER_PASSWORD
echo ""

DOCKER_SECRET=$(cat <<EOF
{
  "registry": "$DOCKER_REGISTRY",
  "username": "$DOCKER_USER",
  "password": "$DOCKER_PASSWORD"
}
EOF
)

create_secret "aml-pipeline/docker-registry" "$DOCKER_SECRET" "AML Pipeline Docker registry credentials"

print_success "AWS Secrets Manager setup completed!"
echo ""
print_status "Next steps:"
echo "1. Note down the secret ARNs for use in Terraform"
echo "2. Configure GitHub repository secrets"
echo "3. Update your CI/CD pipeline with the new secrets"
echo ""

# Display secret ARNs
echo "📋 Secret ARNs:"
aws secretsmanager list-secrets --query 'SecretList[?Name==`aml-pipeline/database` || Name==`aml-pipeline/redis` || Name==`aml-pipeline/jwt` || Name==`aml-pipeline/external-services` || Name==`aml-pipeline/docker-registry`].{Name:Name,ARN:ARN}' --output table

echo ""
print_warning "⚠️  IMPORTANT SECURITY NOTES:"
echo "• Never commit secrets to version control"
echo "• Rotate secrets regularly"
echo "• Use IAM roles with minimal required permissions"
echo "• Enable AWS Secrets Manager automatic rotation where possible"
echo "• Monitor secret access logs for suspicious activity"