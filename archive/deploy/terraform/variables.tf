# ==============================================================================
# AML Pipeline Infrastructure Variables
# ==============================================================================

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "default"
}

variable "environment" {
  description = "Environment name (staging/production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be either 'staging' or 'production'."
  }
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDRs"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDRs"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# ECS Configuration
variable "task_cpu" {
  description = "CPU units for ECS task"
  type        = string
  default     = "1024"
}

variable "task_memory" {
  description = "Memory for ECS task (MB)"
  type        = string
  default     = "2048"
}

variable "container_image" {
  description = "Container image for deployment"
  type        = string
  default     = "ghcr.io/owner/aml-pipeline:latest-api"
}

# Load Balancer Configuration
variable "traffic_weights" {
  description = "Traffic weights for blue-green deployment"
  type        = map(number)
  default = {
    blue  = 100
    green = 0
  }
}

# Route 53 Configuration
variable "route53_zone_id" {
  description = "Route 53 hosted zone ID for production"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch value."
  }
}

# ==============================================================================
# ENVIRONMENT SPECIFIC VARIABLES
# ==============================================================================

# Staging Environment
variable "staging_task_cpu" {
  description = "CPU for staging tasks"
  type        = string
  default     = "512"
}

variable "staging_task_memory" {
  description = "Memory for staging tasks"
  type        = string
  default     = "1024"
}

# Production Environment
variable "production_task_cpu" {
  description = "CPU for production tasks"
  type        = string
  default     = "2048"
}

variable "production_task_memory" {
  description = "Memory for production tasks"
  type        = string
  default     = "4096"
}

# ==============================================================================
# SECRETS (Should be provided via environment or .tfvars)
# ==============================================================================
variable "database_url" {
  description = "Database connection URL"
  type        = string
  sensitive   = true
  default     = ""
}

variable "api_key" {
  description = "API authentication key"
  type        = string
  sensitive   = true
  default     = ""
}

# ==============================================================================
# TAGS
# ==============================================================================
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}