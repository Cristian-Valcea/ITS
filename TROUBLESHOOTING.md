# 🛠️ **TROUBLESHOOTING GUIDE**

**Common issues and solutions for the Stress Testing Platform**

---

## 🚨 **COMMON ISSUES**

### **1. Platform Won't Start**

#### **Symptoms**
- `./scripts/start_platform.sh` fails
- Services not responding on expected ports
- Error messages about missing dependencies

#### **Diagnosis**
```bash
# Check pre-flight status
./stress_testing/ci/guards.sh

# Check port availability
netstat -tlnp | grep -E "(8000|9090|3000|5432|6379)"

# Check system resources
free -h
df -h
```

#### **Solutions**
```bash
# Kill conflicting processes
sudo pkill -f prometheus
sudo pkill -f grafana
sudo pkill -f "python.*stress"

# Restart Docker services
docker-compose restart timescaledb_primary

# Clear port conflicts
sudo lsof -ti:8000 | xargs sudo kill -9
sudo lsof -ti:9090 | xargs sudo kill -9
sudo lsof -ti:3000 | xargs sudo kill -9

# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

---

### **2. Prometheus Not Scraping Metrics**

#### **Symptoms**
- Grafana shows "No data"
- Prometheus targets page shows services as "DOWN"
- Metrics endpoint returns empty results

#### **Diagnosis**
```bash
# Check metrics endpoint
curl -v http://localhost:8000/metrics

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, lastError: .lastError}'

# Check Prometheus logs
tail -f logs/stress_testing/prometheus.log
```

#### **Solutions**
```bash
# Restart metrics server
pkill -f "python.*stress.*metrics"
./scripts/start_platform.sh

# Validate Prometheus config
./bin/promtool check config monitoring/prometheus/prometheus.yml

# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload

# Check firewall/network
sudo ufw status
telnet localhost 8000
```

---

### **3. Grafana Dashboard Empty**

#### **Symptoms**
- Dashboard loads but shows no data
- "No data points" messages
- Queries return empty results

#### **Diagnosis**
```bash
# Check Grafana logs
tail -f logs/stress_testing/grafana.log

# Test Prometheus connection from Grafana
curl -s http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up

# Check if metrics exist in Prometheus
curl -s "http://localhost:9090/api/v1/query?query=risk_governor_decision_latency_seconds"
```

#### **Solutions**
```bash
# Restart Grafana
pkill -f grafana-server
./scripts/start_grafana.sh

# Re-import dashboard
curl -X POST \
    -H "Content-Type: application/json" \
    -d @monitoring/grafana/dashboards/stress_testing_dashboard.json \
    http://admin:stress_testing_2025@localhost:3000/api/dashboards/db

# Generate some test metrics
python -c "
from stress_testing.core.metrics import get_metrics
import time
metrics = get_metrics()
for i in range(100):
    metrics.timing('decision_latency_ns', 5000000 + i*10000)
    metrics.counter('decisions_total').inc()
    time.sleep(0.1)
"
```

---

### **4. High Latency Alerts**

#### **Symptoms**
- P99 latency > 15ms consistently
- Grafana shows red thresholds
- Performance degradation

#### **Diagnosis**
```bash
# Check system load
top
htop
iostat 1

# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check disk I/O
iotop
df -h

# Monitor real-time latency
./scripts/monitor.sh latency
```

#### **Solutions**
```bash
# Restart services to clear memory leaks
./scripts/daily_shutdown.sh
./scripts/daily_startup.sh

# Optimize system
# Increase file descriptors
ulimit -n 65536

# Clear system caches
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches

# Check for resource contention
# Stop non-essential processes
# Increase system resources if needed
```

---

### **5. Database Connection Issues**

#### **Symptoms**
- TimescaleDB connection failures
- Data validation errors
- Historical data not available

#### **Diagnosis**
```bash
# Check Docker container
docker ps | grep timescaledb
docker logs timescaledb_primary

# Test database connection
docker exec timescaledb_primary pg_isready -U postgres -d trading_data

# Check database size and health
docker exec timescaledb_primary psql -U postgres -d trading_data -c "\dt+"
```

#### **Solutions**
```bash
# Restart TimescaleDB container
docker-compose restart timescaledb_primary

# Check database logs for errors
docker logs timescaledb_primary | tail -50

# Recreate container if corrupted
docker-compose down
docker volume rm its_timescale_data  # WARNING: This deletes data
docker-compose up -d timescaledb_primary

# Restore from backup if available
# Follow database recovery procedures
```

---

### **6. Redis Connection Issues**

#### **Symptoms**
- Redis connection refused
- Governor pause flag not working
- State synchronization issues

#### **Diagnosis**
```bash
# Check Redis status
redis-cli ping
redis-cli info server

# Check Redis logs
tail -f /var/log/redis/redis-server.log

# Test Redis operations
redis-cli set test_key test_value
redis-cli get test_key
```

#### **Solutions**
```bash
# Restart Redis
sudo systemctl restart redis-server
# OR
redis-server --daemonize yes --port 6379

# Clear Redis if corrupted
redis-cli flushall  # WARNING: This deletes all data

# Check Redis configuration
redis-cli config get "*"

# Increase Redis memory if needed
redis-cli config set maxmemory 1gb
```

---

## 🔧 **DIAGNOSTIC COMMANDS**

### **Quick Health Check**
```bash
# One-command health check
./scripts/monitor.sh health

# Service status
./scripts/monitor.sh status

# Key metrics
./scripts/monitor.sh metrics
```

### **Log Analysis**
```bash
# Check all recent errors
find logs/stress_testing -name "*.log" -mtime -1 -exec grep -l "ERROR\|FAIL" {} \;

# Monitor logs in real-time
tail -f logs/stress_testing/*.log

# Search for specific issues
grep -r "connection refused" logs/stress_testing/
grep -r "timeout" logs/stress_testing/
grep -r "memory" logs/stress_testing/
```

### **Performance Analysis**
```bash
# System performance
vmstat 1 10
iostat -x 1 10
sar -u 1 10

# Network performance
netstat -i
ss -tuln

# Process analysis
ps aux --sort=-%cpu | head -10
ps aux --sort=-%mem | head -10
```

---

## 🚨 **EMERGENCY PROCEDURES**

### **Complete Platform Reset**
```bash
# EMERGENCY: Complete platform reset
echo "⚠️  EMERGENCY RESET - This will stop all services and clear temporary data"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    # Stop all services
    ./scripts/daily_shutdown.sh
    
    # Kill any remaining processes
    sudo pkill -f prometheus
    sudo pkill -f grafana
    sudo pkill -f "python.*stress"
    
    # Clear temporary data (keep archives)
    rm -rf logs/stress_testing/*.log
    rm -rf monitoring/prometheus/data/*
    rm -rf monitoring/grafana/data/*
    
    # Restart platform
    ./scripts/daily_startup.sh
    
    echo "✅ Emergency reset complete"
fi
```

### **Data Recovery**
```bash
# Restore from latest archive
LATEST_ARCHIVE=$(ls -1t archives/ | head -1)
echo "Restoring from: $LATEST_ARCHIVE"

# Copy certification data
cp "archives/$LATEST_ARCHIVE/certification_*.json" stress_testing/results/certification_report.json

# Restore configuration if needed
# cp "archives/$LATEST_ARCHIVE/config_backup.yml" stress_testing/config/
```

---

## 📞 **ESCALATION PROCEDURES**

### **Level 1: Self-Service** (0-15 minutes)
- Check this troubleshooting guide
- Run diagnostic commands
- Restart individual services
- Check logs for obvious errors

### **Level 2: Team Support** (15-30 minutes)
- Post in #trading-alerts Slack channel
- Include diagnostic output
- Mention specific error messages
- Tag on-call engineer if urgent

### **Level 3: Emergency Response** (30+ minutes)
- Call team lead directly
- Initiate emergency procedures
- Consider platform shutdown
- Document incident for post-mortem

---

## 📋 **MAINTENANCE CHECKLIST**

### **Daily Maintenance**
- [ ] Check service health: `./scripts/monitor.sh health`
- [ ] Review error logs: `./scripts/monitor.sh errors`
- [ ] Verify certification status
- [ ] Monitor disk space usage
- [ ] Check alert status

### **Weekly Maintenance**
- [ ] Archive old logs and data
- [ ] Update system packages
- [ ] Review performance trends
- [ ] Test backup/recovery procedures
- [ ] Update documentation

### **Monthly Maintenance**
- [ ] Full platform restart
- [ ] Database maintenance and optimization
- [ ] Security updates
- [ ] Performance baseline review
- [ ] Disaster recovery test

---

**🛠️ Remember: When in doubt, check the logs first, then restart services, then escalate if needed.**