# FeatureStore SLO Deployment Guide

## Overview

This guide covers the deployment of the FeatureStore hit ratio SLO monitoring system, providing 95% cache hit ratio observability and alerting.

## Components

### 1. Metrics Instrumentation
- **File**: `src/shared/feature_store.py`
- **Metrics**: 
  - `featurestore_hits_total` - Counter for cache hits
  - `featurestore_misses_total` - Counter for cache misses  
  - `featurestore_hit_ratio` - Gauge for current hit ratio

### 2. API Metrics Endpoint
- **Endpoint**: `GET /metrics`
- **Port**: 8000 (default FastAPI port)
- **Format**: Prometheus exposition format

### 3. Prometheus Rules
- **File**: `config/prometheus/featurestore_rules.yml`
- **Rules**: Hit ratio calculation and alerting thresholds

### 4. Grafana Dashboard
- **File**: `config/grafana/dashboards/featurestore_slo_dashboard.json`
- **Panels**: Hit ratio, request rate, per-symbol breakdown, SLO status

### 5. Runbook Documentation
- **File**: `docs/runbooks/featurestore_slo.md`
- **Content**: Incident response procedures and troubleshooting

## Deployment Steps

### Step 1: Deploy Application Changes

```bash
# 1. Deploy updated FeatureStore with metrics
git pull origin main
pip install prometheus-client  # If not already installed

# 2. Restart trading system to load new metrics
systemctl restart trading-system

# 3. Verify metrics endpoint
curl http://localhost:8000/metrics | grep featurestore
```

### Step 2: Configure Prometheus

```bash
# 1. Copy rules file to Prometheus config directory
sudo cp config/prometheus/featurestore_rules.yml /etc/prometheus/rules/

# 2. Update prometheus.yml to include new rules
sudo tee -a /etc/prometheus/prometheus.yml << EOF

rule_files:
  - "rules/featurestore_rules.yml"

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
EOF

# 3. Validate configuration
promtool check config /etc/prometheus/prometheus.yml

# 4. Reload Prometheus configuration
sudo systemctl reload prometheus

# 5. Verify rules are loaded
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name | contains("featurestore"))'
```

### Step 3: Import Grafana Dashboard

```bash
# Option 1: Import via Grafana UI
# 1. Open Grafana: http://localhost:3000
# 2. Go to Dashboards → Import
# 3. Upload config/grafana/dashboards/featurestore_slo_dashboard.json

# Option 2: Import via API
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @config/grafana/dashboards/featurestore_slo_dashboard.json

# 3. Verify dashboard is accessible
curl -s http://admin:admin@localhost:3000/api/dashboards/uid/featurestore-slo
```

### Step 4: Configure Alerting

```bash
# 1. Verify alerts are loaded in Prometheus
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name == "featurestore.alerts")'

# 2. Configure Alertmanager (if not already configured)
sudo tee /etc/alertmanager/alertmanager.yml << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'dev-data-oncall@company.com'
    subject: 'FeatureStore SLO Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'component']
EOF

# 3. Restart Alertmanager
sudo systemctl restart alertmanager
```

### Step 5: Validation and Testing

```bash
# 1. Generate test traffic to create metrics
python << EOF
import sys
sys.path.append('src')
from shared.feature_store import FeatureStore
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='1H')
data = pd.DataFrame({
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.uniform(1000, 10000, 100)
}, index=dates)

# Create FeatureStore and generate cache operations
fs = FeatureStore()
config = {'test': True}

def dummy_compute(df, config):
    return pd.DataFrame({'feature': [1, 2, 3]}, index=df.index[:3])

# Generate hits and misses
for symbol in ['AAPL', 'GOOGL', 'MSFT']:
    # First call - miss
    fs.get_or_compute(symbol, data, config, dummy_compute)
    # Second call - hit  
    fs.get_or_compute(symbol, data, config, dummy_compute)

print("Test traffic generated successfully")
EOF

# 2. Verify metrics are being collected
curl http://localhost:8000/metrics | grep featurestore

# 3. Check Prometheus is scraping metrics
curl "http://localhost:9090/api/v1/query?query=featurestore_hits_total"

# 4. Verify Grafana dashboard shows data
curl -s "http://admin:admin@localhost:3000/api/dashboards/uid/featurestore-slo" | jq '.dashboard.title'

# 5. Test alert firing (simulate low hit ratio)
# This would require generating many cache misses in production
```

## Production Deployment Checklist

### Pre-Deployment
- [ ] Code review completed for metrics instrumentation
- [ ] Unit tests passing for metrics functionality
- [ ] Integration tests validated
- [ ] Prometheus rules syntax validated
- [ ] Grafana dashboard JSON validated
- [ ] Runbook documentation reviewed

### Deployment
- [ ] Application deployed with metrics instrumentation
- [ ] Prometheus rules deployed and loaded
- [ ] Grafana dashboard imported and accessible
- [ ] Alertmanager configured for notifications
- [ ] Metrics endpoint accessible and returning data

### Post-Deployment
- [ ] Metrics collection verified in Prometheus
- [ ] Dashboard displaying data correctly
- [ ] Alert rules evaluated without errors
- [ ] Test alert fired and received by oncall
- [ ] Runbook procedures tested
- [ ] SLO baseline established

## Monitoring and Maintenance

### Daily Checks
```bash
# Check metrics endpoint health
curl -f http://localhost:8000/metrics > /dev/null && echo "✅ Metrics endpoint healthy"

# Check current hit ratio
curl -s "http://localhost:9090/api/v1/query?query=featurestore_hit_ratio" | jq '.data.result[0].value[1]'

# Check for any firing alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.component == "featurestore")'
```

### Weekly Reviews
- Review SLO compliance over the past week
- Analyze hit ratio trends and patterns
- Check for any alert fatigue or false positives
- Update capacity planning based on cache usage

### Monthly Tasks
- Review and update SLO targets if needed
- Optimize cache configuration based on performance data
- Update runbook procedures based on incidents
- Review alert thresholds and timing

## Troubleshooting

### Common Issues

**1. Metrics Not Appearing**
```bash
# Check if prometheus-client is installed
python -c "import prometheus_client; print('✅ Prometheus client available')"

# Check application logs for errors
grep -i "prometheus\|metrics" /var/log/trading-system.log

# Verify endpoint is accessible
curl -v http://localhost:8000/metrics
```

**2. Rules Not Loading**
```bash
# Check Prometheus logs
sudo journalctl -u prometheus -f

# Validate rules syntax
promtool check rules config/prometheus/featurestore_rules.yml

# Check Prometheus configuration
promtool check config /etc/prometheus/prometheus.yml
```

**3. Dashboard Not Showing Data**
```bash
# Check Prometheus data source in Grafana
curl -s "http://admin:admin@localhost:3000/api/datasources" | jq '.[] | select(.type == "prometheus")'

# Test query directly in Prometheus
curl "http://localhost:9090/api/v1/query?query=featurestore:hit_ratio_5m"

# Check dashboard configuration
curl -s "http://admin:admin@localhost:3000/api/dashboards/uid/featurestore-slo" | jq '.dashboard.panels[0].targets'
```

## Rollback Procedures

If issues occur during deployment:

```bash
# 1. Rollback application changes
git revert <commit-hash>
systemctl restart trading-system

# 2. Remove Prometheus rules
sudo rm /etc/prometheus/rules/featurestore_rules.yml
sudo systemctl reload prometheus

# 3. Remove Grafana dashboard
curl -X DELETE http://admin:admin@localhost:3000/api/dashboards/uid/featurestore-slo

# 4. Verify system is stable
curl http://localhost:8000/health
```

## Performance Impact

The FeatureStore SLO monitoring has minimal performance impact:

- **Metrics Collection**: ~0.1ms overhead per cache operation
- **Memory Usage**: ~1MB for metrics storage
- **Network**: ~1KB/s additional Prometheus scraping traffic
- **Storage**: ~10MB/day for metrics retention

## Security Considerations

- Metrics endpoint exposes cache performance data (not sensitive)
- No authentication required for metrics endpoint (standard practice)
- Grafana dashboard should use appropriate access controls
- Alert notifications should use secure channels

## Related Documentation

- [FeatureStore Architecture](../architecture/featurestore.md)
- [Prometheus Configuration](../monitoring/prometheus.md)
- [Grafana Setup Guide](../monitoring/grafana.md)
- [SLO Runbook](../runbooks/featurestore_slo.md)