# FeatureStore SLO Runbook

## Overview

This runbook covers the FeatureStore cache hit ratio Service Level Objective (SLO) monitoring and incident response procedures.

**SLO Target**: ≥ 95% of feature requests served from cache (not recomputed)

## Alerts

### FeatureStoreHitRatioLow
- **Severity**: Warning
- **Threshold**: Hit ratio < 95% for 10 minutes
- **First Responder**: dev-data-oncall
- **Escalation**: Platform team after 30 minutes

### FeatureStoreHitRatioCritical
- **Severity**: Critical  
- **Threshold**: Hit ratio < 80% for 5 minutes
- **First Responder**: dev-data-oncall + platform-oncall
- **Escalation**: Immediate escalation to engineering leadership

## Incident Response

### Initial Assessment (First 5 minutes)

1. **Check Grafana Dashboard**
   - Navigate to: `http://grafana.company.com/d/featurestore-slo`
   - Review current hit ratio and request rate
   - Identify if issue is global or symbol-specific

2. **Quick Health Checks**
   ```bash
   # Check API metrics endpoint
   curl http://trading-api:8000/metrics | grep featurestore
   
   # Check disk usage
   df -h /mnt/feature_cache
   
   # Check DuckDB manifest health
   ls -la ~/.feature_cache/manifest.duckdb*
   ```

### Root Cause Analysis

#### Common Causes and Solutions

**1. Disk Space Issues**
```bash
# Check disk usage
df -h /mnt/feature_cache

# If disk is full (>90%), clean old cache files
find /mnt/feature_cache -name "*.parquet.zst" -mtime +7 -delete

# Check manifest database size
du -h ~/.feature_cache/manifest.duckdb
```

**2. DuckDB Manifest Database Issues**
```bash
# Check for database locks
lsof ~/.feature_cache/manifest.duckdb

# Check database integrity
sqlite3 ~/.feature_cache/manifest.duckdb "PRAGMA integrity_check;"

# If corrupted, backup and rebuild
cp ~/.feature_cache/manifest.duckdb ~/.feature_cache/manifest.duckdb.backup
# Restart application to rebuild manifest
```

**3. High Data Churn (Expected)**
- New symbols being added to trading universe
- Market data updates causing cache invalidation
- Feature configuration changes

**Actions for High Data Churn:**
```bash
# Check recent symbol additions
grep "Cache MISS" /var/log/trading-system.log | tail -100

# Temporarily silence alert if expected
# (Use Grafana alert silencing for 30 minutes)
```

**4. Memory Pressure**
```bash
# Check system memory
free -h

# Check application memory usage
ps aux | grep -E "(trading|feature)" | sort -k4 -nr

# Check for memory leaks in feature store
grep -i "memory\|oom" /var/log/trading-system.log
```

**5. Cache Thrashing**
```bash
# Check request patterns
grep "get_or_compute" /var/log/trading-system.log | tail -50

# Look for rapid cache invalidation
grep "Cache key.*invalidated" /var/log/trading-system.log
```

### Mitigation Strategies

#### Immediate Actions (< 15 minutes)

1. **Emergency Cache Warming**
   ```python
   # Connect to trading system
   from src.shared.feature_store import FeatureStore
   
   fs = FeatureStore()
   
   # Warm cache for critical symbols
   critical_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
   for symbol in critical_symbols:
       # Trigger cache warming (implementation specific)
       pass
   ```

2. **Increase Cache Retention**
   ```bash
   # Temporarily disable cache cleanup
   systemctl stop feature-cache-cleanup.timer
   
   # Or increase retention period
   export FEATURE_CACHE_RETENTION_DAYS=14
   systemctl restart trading-system
   ```

3. **Scale Resources**
   ```bash
   # Increase disk space (if cloud)
   aws ec2 modify-volume --volume-id vol-xxx --size 500
   
   # Add memory if needed
   # (Requires instance restart in most cases)
   ```

#### Medium-term Actions (< 1 hour)

1. **Optimize Cache Configuration**
   ```yaml
   # Update feature store config
   feature_store:
     compression_level: 1  # Reduce from 3 for faster I/O
     max_cache_size_gb: 100  # Increase cache size
     cleanup_threshold: 0.95  # Increase cleanup threshold
   ```

2. **Database Optimization**
   ```sql
   -- Optimize manifest database
   VACUUM;
   ANALYZE;
   
   -- Add missing indexes if needed
   CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
   ON manifest(symbol, start_ts, end_ts);
   ```

3. **Application Tuning**
   ```python
   # Increase connection pool size
   # Optimize feature computation algorithms
   # Implement feature computation batching
   ```

### Monitoring and Verification

#### Key Metrics to Monitor

1. **Hit Ratio Recovery**
   - Target: Return to >95% within 30 minutes
   - Monitor: `featurestore:hit_ratio_5m`

2. **Request Rate Stability**
   - Monitor: `featurestore:request_rate_5m`
   - Look for: Stable or decreasing miss rate

3. **System Resources**
   - Disk usage: Should remain <85%
   - Memory usage: Should be stable
   - CPU usage: Should not spike during cache operations

#### Verification Commands

```bash
# Check current hit ratio
curl -s http://trading-api:8000/metrics | grep featurestore_hit_ratio

# Monitor cache file creation
watch "ls -la /mnt/feature_cache/*.zst | wc -l"

# Check application logs for errors
tail -f /var/log/trading-system.log | grep -i "feature\|cache"
```

### Post-Incident Actions

1. **Document Root Cause**
   - Update incident tracking system
   - Record timeline and actions taken
   - Identify prevention measures

2. **Review SLO Compliance**
   - Calculate SLO breach duration
   - Update error budget calculations
   - Review if SLO target needs adjustment

3. **Implement Preventive Measures**
   - Add monitoring for identified gaps
   - Implement automated remediation where possible
   - Update capacity planning

## Escalation Procedures

### Level 1: Warning Alert (Hit Ratio < 95%)
- **Time**: 0-10 minutes
- **Owner**: dev-data-oncall
- **Actions**: Initial assessment and basic mitigation

### Level 2: Critical Alert (Hit Ratio < 80%)
- **Time**: 0-5 minutes  
- **Owner**: dev-data-oncall + platform-oncall
- **Actions**: Immediate escalation, emergency procedures

### Level 3: Extended Outage (>30 minutes)
- **Time**: 30+ minutes
- **Owner**: Engineering leadership
- **Actions**: War room, external communication, major incident procedures

## Contact Information

- **dev-data-oncall**: Slack @data-oncall, PagerDuty escalation
- **platform-oncall**: Slack @platform-oncall, PagerDuty escalation  
- **Engineering Leadership**: Slack @eng-leadership

## Related Documentation

- [FeatureStore Architecture](../architecture/featurestore.md)
- [Prometheus Alerting Guide](../monitoring/prometheus.md)
- [Grafana Dashboard Guide](../monitoring/grafana.md)
- [Incident Response Procedures](../procedures/incident-response.md)

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2024-01-XX | Initial runbook creation | System |
| 2024-01-XX | Added cache thrashing procedures | System |