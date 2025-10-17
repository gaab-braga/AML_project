# AML Pipeline Monitoring Stack

This directory contains the complete monitoring and observability stack for the AML Pipeline, providing enterprise-grade monitoring, alerting, and visualization capabilities.

## 🏗️ Architecture

The monitoring stack consists of:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard visualization and alerting
- **Alertmanager**: Alert routing and notification management
- **Node Exporter**: System metrics collection
- **cAdvisor**: Container metrics collection
- **PostgreSQL Exporter**: Database metrics (when configured)
- **Redis Exporter**: Cache metrics (when configured)

## 📊 Dashboards

Three pre-configured dashboards are included:

### 1. AML Pipeline Overview
- API health status and response times
- System resource utilization (CPU, memory)
- Container performance metrics
- Database and cache monitoring
- Active alerts summary

### 2. AML Business Metrics
- Transaction processing volumes
- Fraud detection rates and alerts
- Model performance metrics (accuracy, recall)
- Processing time distributions
- Risk category analysis

### 3. AML Alerts & Incidents
- Active alerts table with severity levels
- Alert trends and distributions
- System health scoring
- Performance alert indicators
- Incident response tracking

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Ports 9090, 3000, 9093, 9091, 8080 available

### Starting the Stack

1. Navigate to the monitoring directory:
   ```bash
   cd monitoring
   ```

2. Start all services:
   ```bash
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

3. Or use the convenience script:
   ```bash
   # On Linux/Mac
   ./start-monitoring.sh

   # On Windows PowerShell
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

### Accessing Services

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
  - ⚠️ **Change default password in production!**

- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **cAdvisor**: http://localhost:8080

## 🔧 Configuration

### Alert Rules

Alert rules are defined in `prometheus/alert_rules.yml` and include:

- **API Health**: Monitors service availability
- **High Error Rate**: Detects elevated 5xx error rates
- **High Latency**: Monitors response time degradation
- **Resource Usage**: CPU, memory, and disk alerts
- **Model Performance**: Accuracy and drift detection
- **Business Metrics**: Transaction volume and fraud alerts

### Alertmanager Configuration

Notifications are configured in `alertmanager/alertmanager.yml`:

- **Email**: SMTP configuration for email alerts
- **Slack**: Webhook integration for Slack notifications
- **PagerDuty**: Enterprise alerting integration

### Data Sources

Grafana is pre-configured with:
- Prometheus (metrics)
- CloudWatch (AWS metrics, when applicable)
- PostgreSQL (database metrics, when configured)

## 📈 Metrics Collection

The stack collects metrics from:

### Application Metrics
- HTTP request/response metrics
- Business KPIs (transactions, fraud alerts)
- Model performance indicators
- Processing latency histograms

### System Metrics
- CPU, memory, disk, and network usage
- Container resource consumption
- Host system health

### Infrastructure Metrics
- Database connection pools and performance
- Cache hit rates and latency
- Load balancer metrics (when applicable)

## 🔒 Security Considerations

### Production Deployment
- Change default Grafana credentials
- Configure proper authentication (LDAP, OAuth)
- Use HTTPS/TLS for all services
- Implement network segmentation
- Set up proper firewall rules
- Use secrets management for sensitive configuration

### Alert Configuration
- Configure appropriate notification channels
- Set up escalation policies
- Define alert thresholds based on your environment
- Test alert delivery regularly

## 🛠️ Maintenance

### Updating Dashboards
1. Modify JSON files in `grafana/dashboards/`
2. Restart Grafana or wait for auto-reload (10s interval)

### Adding New Metrics
1. Update Prometheus configuration in `prometheus/prometheus.yml`
2. Add scrape targets as needed
3. Create new panels in Grafana dashboards

### Backup and Recovery
- Prometheus data: Configure persistent volumes
- Grafana dashboards: Export JSON configurations
- Alert rules: Version control all configuration files

## 📚 Troubleshooting

### Common Issues

**Services not starting:**
- Check port availability
- Verify Docker resource allocation
- Review container logs: `docker-compose logs`

**Metrics not appearing:**
- Check Prometheus targets status
- Verify service discovery configuration
- Ensure applications are exposing metrics on correct endpoints

**Alerts not firing:**
- Validate alert rule syntax
- Check Prometheus rule evaluation
- Review alert thresholds

### Logs and Debugging
```bash
# View all service logs
docker-compose -f docker-compose.monitoring.yml logs

# View specific service logs
docker-compose -f docker-compose.monitoring.yml logs prometheus

# Check service health
docker-compose -f docker-compose.monitoring.yml ps
```

## 📞 Support

For issues with the monitoring stack:
1. Check the troubleshooting section above
2. Review service logs for error messages
3. Validate configuration files syntax
4. Ensure all prerequisites are met

## 🔄 Integration with AML Pipeline

The monitoring stack is designed to integrate seamlessly with the AML Pipeline:

- **API Metrics**: Automatic collection from FastAPI instrumentation
- **Business Metrics**: Custom metrics from fraud detection logic
- **Model Monitoring**: Performance and drift detection metrics
- **Alert Integration**: Automated incident response workflows

For detailed integration instructions, see the main AML Pipeline documentation.