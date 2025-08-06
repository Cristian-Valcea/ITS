# 🚀 **STRESS TEST PLATFORM USER GUIDE**

**Complete guide for operating the Stress Testing Platform in production**

---

## 📋 **TABLE OF CONTENTS**

1. [Quick Start](#quick-start)
2. [Platform Architecture](#platform-architecture)
3. [Daily Operations](#daily-operations)
4. [Monitoring & Dashboards](#monitoring--dashboards)
5. [Stress Testing Scenarios](#stress-testing-scenarios)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Operations](#advanced-operations)
8. [Reference](#reference)

---

## 🚀 **QUICK START**

### **First Time Setup**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# Validate platform readiness
./stress_testing/ci/guards.sh
# Must show: ✅ All CI guards passed

# Create required directories
mkdir -p {reports,archives,bin,logs/stress_testing}
```

### **Start Platform (Daily)**
```bash
# Complete morning startup routine
./scripts/daily_startup.sh

# Expected output:
# ✅ Pre-flight checks passed
# ✅ Platform components started
# ✅ Daily certification PASSED
# 🚀 Platform ready for trading day operations!
```

### **Access Dashboards**
- **📊 Grafana Dashboard**: http://localhost:3000 (admin/stress_testing_2025)
- **📈 Prometheus Metrics**: http://localhost:9090
- **🔍 Raw Metrics**: http://localhost:8000/metrics

### **Quick Health Check**
```bash
# One-command health check
./scripts/monitor.sh health

# Expected output:
# ✅ All services running
# ✅ Decision Latency P99: < 15ms
# ✅ Pipeline Latency P99: < 20ms
# ✅ Error Rate: < 0.1%
```

### **Stop Platform (Daily)**
```bash
# Complete evening shutdown routine
./scripts/daily_shutdown.sh

# Archives data, generates reports, stops services gracefully
```

---

## 🏗️ **PLATFORM ARCHITECTURE**

### **Core Components**

#### **🎯 Stress Testing Engine**
- **Location**: `stress_testing/`
- **Purpose**: Executes stress test scenarios
- **Key Files**:
  - `run_full_suite.py` - Main test runner
  - `scenarios/` - Test scenario definitions
  - `simulators/` - Market condition simulators
  - `injectors/` - Fault injection tools

#### **📊 Monitoring Stack**
- **Prometheus** (Port 9090): Metrics collection and alerting
- **Grafana** (Port 3000): Real-time dashboards and visualization
- **Metrics Server** (Port 8000): Stress testing metrics endpoint

#### **💾 Data Layer**
- **TimescaleDB**: Historical market data and test results
- **Redis**: Real-time state management and governor controls
- **File System**: Logs, reports, and archived data

#### **🛡️ Safety Systems**
- **Risk Governor**: Real-time risk management and position limits
- **Emergency Stop**: Immediate halt capabilities
- **Alert System**: Multi-level escalation procedures

### **Data Flow**
```
Market Data → Stress Test Engine → Risk Governor → Metrics → Prometheus → Grafana
                     ↓                    ↓
                TimescaleDB           Redis State
                     ↓                    ↓
                 Archives            Emergency Stop
```

---

## 📅 **DAILY OPERATIONS**

### **🌅 Morning Startup Routine (06:30 - 07:00 ET)**

#### **1. Platform Startup**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# Complete platform startup
./scripts/daily_startup.sh
```

**What this does:**
- ✅ Runs pre-flight checks
- ✅ Starts TimescaleDB and Redis
- ✅ Launches metrics server (port 8000)
- ✅ Starts Prometheus (port 9090)
- ✅ Starts Grafana (port 3000)
- ✅ Runs daily certification
- ✅ Generates daily report

#### **2. Verify Critical Systems**
```bash
# Quick health check
./scripts/monitor.sh health

# Check service status
./scripts/monitor.sh status

# Verify key metrics
./scripts/monitor.sh metrics
```

**Success Criteria:**
- ✅ All services responding
- ✅ Decision Latency P99 < 15ms
- ✅ Pipeline Latency P99 < 20ms
- ✅ Daily certification: PASSED
- ✅ No active alerts

### **📈 Trading Hours Monitoring (09:30 - 16:00 ET)**

#### **Primary Monitoring: Grafana Dashboard**
- **URL**: http://localhost:3000
- **Dashboard**: "Stress Testing Platform - Real-Time Monitoring"
- **Refresh**: Auto-refresh every 5 seconds

**Key Panels to Watch:**
1. **Decision Latency P99** (Critical < 15ms) - Red if exceeded
2. **Pipeline Latency P99** (Critical < 20ms) - Red if exceeded
3. **Hard Limit Breaches** (Must be 0) - Red if any breaches
4. **Position Delta** (Warning > $1000) - Yellow/Red thresholds
5. **Error Rate** (Critical > 0.1%) - Red if exceeded

#### **Command-Line Monitoring (Optional)**
```bash
# Real-time continuous monitoring
./scripts/monitor.sh watch

# Real-time latency monitoring
./scripts/monitor.sh latency

# Check for alerts
./scripts/monitor.sh alerts
```

#### **Hourly Health Checks**
```bash
# Run every hour during trading hours
./scripts/monitor.sh status
./scripts/monitor.sh metrics

# Log any anomalies
echo "$(date): Hourly check - Status OK" >> logs/hourly_checks.log
```

### **🌙 End of Day Routine (16:30 - 17:00 ET)**

#### **1. Final Certification**
```bash
# Run final stress test certification
python stress_testing/run_full_suite.py --certification

# Verify results
cat stress_testing/results/certification_report.json | jq '.certified'
# Should show: true
```

#### **2. Platform Shutdown**
```bash
# Complete shutdown with archiving
./scripts/daily_shutdown.sh
```

**What this does:**
- ✅ Archives today's logs and reports
- ✅ Generates end-of-day summary
- ✅ Stops services gracefully
- ✅ Performs database maintenance
- ✅ Cleans up temporary files
- ✅ Creates shutdown report

#### **3. Review Daily Report**
```bash
# View today's report
cat reports/daily_report_$(date +%Y%m%d).md

# Check archived data
ls -la archives/$(date +%Y%m%d)/
```

---

## 📊 **MONITORING & DASHBOARDS**

### **🎯 Grafana Dashboard**

#### **Access & Login**
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: stress_testing_2025
- **Dashboard**: "Stress Testing Platform - Real-Time Monitoring"

#### **Critical Metrics Panels**

**1. Decision Latency P99 (Top Priority)**
- **Threshold**: < 15ms (Green), > 15ms (Red)
- **What it measures**: Time from decision request to response
- **Action if red**: Check system load, restart services if needed

**2. Pipeline Latency P99**
- **Threshold**: < 20ms (Green), > 20ms (Red)
- **What it measures**: Full pipeline from tick-arrival to Prometheus scrape
- **Action if red**: Check Redis/Prometheus connectivity

**3. Hard Limit Breaches**
- **Threshold**: Must be 0 (Green), Any breach (Red)
- **What it measures**: Risk limit violations
- **Action if red**: IMMEDIATE INVESTIGATION - Check risk parameters

**4. Position Delta**
- **Threshold**: < $500 (Green), $500-$1000 (Yellow), > $1000 (Red)
- **What it measures**: Unexpected position changes
- **Action if yellow/red**: Verify position reconciliation

**5. Error Rate**
- **Threshold**: < 0.05% (Green), 0.05-0.1% (Yellow), > 0.1% (Red)
- **What it measures**: Decision processing errors
- **Action if yellow/red**: Check error logs

#### **Time Series Panels**
- **Decision Latency Over Time**: P99, P95, P50 trends
- **Decision Rate**: Decisions per second
- **Memory Usage**: Process memory consumption
- **Recovery Time**: Time to recover from faults

### **📈 Prometheus Metrics**

#### **Access**
- **URL**: http://localhost:9090
- **Key Sections**:
  - **Targets**: http://localhost:9090/targets (service health)
  - **Alerts**: http://localhost:9090/alerts (active alerts)
  - **Graph**: http://localhost:9090/graph (custom queries)

#### **Key Metrics Queries**
```promql
# Decision latency P99
histogram_quantile(0.99, rate(risk_governor_decision_latency_seconds_bucket[1m])) * 1000

# Pipeline latency P99
histogram_quantile(0.99, rate(pipeline_latency_seconds_bucket[1m])) * 1000

# Error rate
rate(risk_governor_decision_errors_total[5m]) / rate(risk_governor_decisions_total[5m]) * 100

# Hard limit breaches
increase(risk_governor_hard_limit_breaches_total[1h])

# Position delta
abs(risk_governor_position_delta_usd)
```

### **🔍 Command-Line Monitoring**

#### **Monitor Script Usage**
```bash
# Show help
./scripts/monitor.sh help

# Quick status check
./scripts/monitor.sh status

# Key performance metrics
./scripts/monitor.sh metrics

# Active alerts
./scripts/monitor.sh alerts

# Real-time latency monitoring
./scripts/monitor.sh latency

# Recent errors
./scripts/monitor.sh errors

# Comprehensive health check
./scripts/monitor.sh health

# Continuous monitoring (Ctrl+C to stop)
./scripts/monitor.sh watch
```

#### **Example Output**
```bash
$ ./scripts/monitor.sh metrics

📊 Key Performance Metrics
==========================
Timestamp: Tue Aug  5 14:30:15 EEST 2025

🎯 Critical Metrics:
✅ Decision Latency P99: 12.3ms
✅ Pipeline Latency P99: 18.7ms
✅ Error Rate: 0.02%
✅ Position Delta: $245

📈 System Metrics:
✅ Memory Usage: 456MB
✅ Disk Usage: 23%
```

---

## 🧪 **STRESS TESTING SCENARIOS**

### **Available Test Scenarios**

#### **1. Flash Crash Scenario**
```bash
# Basic flash crash test
python stress_testing/run_full_suite.py --scenario flash_crash

# Enhanced flash crash (when implemented)
python stress_testing/run_full_suite.py --scenario flash_crash_enhanced
```

**What it tests:**
- Response to sudden 15% price drop
- Slippage handling and depth impact
- Recovery time and position management
- Risk limit enforcement

**Success Criteria:**
- Max drawdown < 15%
- P99 latency < 15ms
- Zero hard limit breaches
- Final position flat

#### **2. Decision Flood Scenario**
```bash
# Basic decision flood test
python stress_testing/run_full_suite.py --scenario decision_flood

# Enhanced decision flood (when implemented)
python stress_testing/run_full_suite.py --scenario decision_flood_enhanced
```

**What it tests:**
- High-frequency decision processing (1000/sec)
- Pipeline latency under load
- Redis performance and backlog
- Memory leak detection

**Success Criteria:**
- Pipeline P99 < 20ms
- Redis backlog = 0
- Memory leak < 50MB
- No decision errors

#### **3. Broker Failure Scenario**
```bash
# Basic broker failure test
python stress_testing/run_full_suite.py --scenario broker_failure

# Hard-kill governance test (when implemented)
python stress_testing/run_full_suite.py --scenario hard_kill_governance
```

**What it tests:**
- Broker disconnect handling
- Network latency spikes
- Governor pause mechanisms
- Emergency position flattening

**Success Criteria:**
- Recovery time < 25s
- Critical pause flag raised
- Zero order leaks during pause
- Position flattening triggered

#### **4. Portfolio Integrity Scenario**
```bash
# Portfolio integrity test
python stress_testing/run_full_suite.py --scenario portfolio_integrity

# Concurrent order burst (when implemented)
python stress_testing/run_full_suite.py --scenario concurrent_order_burst
```

**What it tests:**
- Concurrent order processing
- Position reconciliation
- Fee calculation accuracy
- State consistency

**Success Criteria:**
- Zero risk limit breaches
- Fee-adjusted PnL delta ≤ $1
- Order execution rate > 95%
- Redis ↔ PostgreSQL sync maintained

### **🎯 Full Certification Suite**
```bash
# Run all scenarios with certification
python stress_testing/run_full_suite.py --certification

# Enhanced certification (when implemented)
python stress_testing/run_full_suite.py --certification --enhanced
```

**Certification Criteria:**
- All scenarios must pass
- Overall pass rate: 100%
- No critical alerts during testing
- Complete audit trail generated

### **📊 Test Results**

#### **Results Location**
- **JSON Report**: `stress_testing/results/certification_report.json`
- **HTML Dashboard**: `stress_testing/results/dashboard.html`
- **Detailed Logs**: `logs/stress_testing/`

#### **Reading Results**
```bash
# Check certification status
cat stress_testing/results/certification_report.json | jq '.certified'

# View summary
cat stress_testing/results/certification_report.json | jq '.summary'

# Check individual scenario results
cat stress_testing/results/certification_report.json | jq '.scenarios'
```

#### **Example Results**
```json
{
  "certified": true,
  "certification_timestamp": 1691234567,
  "summary": {
    "total_scenarios": 4,
    "passed_scenarios": 4,
    "failed_scenarios": 0,
    "pass_rate_pct": 100.0
  },
  "scenarios": {
    "flash_crash": {"overall_pass": true, "max_drawdown_pct": 12.3},
    "decision_flood": {"overall_pass": true, "pipeline_latency_p99_ms": 18.7},
    "broker_failure": {"overall_pass": true, "recovery_time_s": 22.1},
    "portfolio_integrity": {"overall_pass": true, "pnl_delta_usd": 0.45}
  }
}
```

---

## 🛠️ **TROUBLESHOOTING**

### **🚨 Common Issues**

#### **1. Platform Won't Start**

**Symptoms:**
- `./scripts/daily_startup.sh` fails
- Services not responding on expected ports
- Error messages about missing dependencies

**Quick Diagnosis:**
```bash
# Check pre-flight status
./stress_testing/ci/guards.sh

# Check port availability
netstat -tlnp | grep -E "(8000|9090|3000|5432|6379)"

# Check system resources
free -h && df -h
```

**Solutions:**
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

#### **2. Grafana Dashboard Shows "No Data"**

**Symptoms:**
- Dashboard loads but panels are empty
- "No data points" messages
- Queries return empty results

**Quick Diagnosis:**
```bash
# Check if metrics endpoint is working
curl -v http://localhost:8000/metrics

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Test Prometheus query
curl -s "http://localhost:9090/api/v1/query?query=up"
```

**Solutions:**
```bash
# Restart metrics server
pkill -f "python.*stress.*metrics"
./scripts/start_platform.sh

# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload

# Generate test metrics
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

#### **3. High Latency Alerts**

**Symptoms:**
- P99 latency > 15ms consistently
- Grafana shows red thresholds
- Performance degradation alerts

**Quick Diagnosis:**
```bash
# Check system load
top
free -h
iostat 1 5

# Monitor real-time latency
./scripts/monitor.sh latency

# Check for resource contention
ps aux --sort=-%cpu | head -10
ps aux --sort=-%mem | head -10
```

**Solutions:**
```bash
# Restart services to clear memory leaks
./scripts/daily_shutdown.sh
./scripts/daily_startup.sh

# Optimize system resources
ulimit -n 65536
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches

# Check for competing processes
# Stop non-essential services if needed
```

### **🚨 Emergency Procedures**

#### **Complete Platform Reset**
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

#### **Emergency Stop Trading**
```bash
# Immediate trading halt
redis-cli SET governor.emergency_stop true
redis-cli SET governor.pause true
redis-cli PUBLISH alerts "EMERGENCY STOP ACTIVATED"

# Flatten all positions
python -c "
from src.agents.risk_agent import RiskAgent
risk_agent = RiskAgent()
risk_agent.flatten_all_positions()
print('All positions flattened')
"
```

### **📞 Escalation Procedures**

#### **Level 1: Self-Service (0-15 minutes)**
- Check this troubleshooting guide
- Run diagnostic commands
- Restart individual services
- Check logs for obvious errors

#### **Level 2: Team Support (15-30 minutes)**
- Post in #trading-alerts Slack channel
- Include diagnostic output
- Mention specific error messages
- Tag on-call engineer if urgent

#### **Level 3: Emergency Response (30+ minutes)**
- Call team lead directly
- Initiate emergency procedures
- Consider platform shutdown
- Document incident for post-mortem

---

## 🔧 **ADVANCED OPERATIONS**

### **🔄 Automated Scheduling**

#### **Cron Job Setup**
```bash
# Edit crontab
crontab -e

# Add these entries:
# Daily startup at 6:30 AM
30 6 * * * cd /home/cristian/IntradayTrading/ITS && ./scripts/daily_startup.sh

# Hourly health checks during trading hours
0 9-16 * * 1-5 cd /home/cristian/IntradayTrading/ITS && ./scripts/monitor.sh health >> logs/hourly_health.log

# Daily shutdown at 5:00 PM
0 17 * * * cd /home/cristian/IntradayTrading/ITS && ./scripts/daily_shutdown.sh

# Weekly maintenance on Sunday at 2:00 AM
0 2 * * 0 cd /home/cristian/IntradayTrading/ITS && ./scripts/weekly_maintenance.sh
```

### **📊 Performance Analysis**

#### **Weekly Performance Review**
```bash
# Generate weekly latency trend
curl -s "http://localhost:9090/api/v1/query_range?query=histogram_quantile(0.99, rate(risk_governor_decision_latency_seconds_bucket[1h]))&start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600" | jq '.data.result[0].values' > reports/weekly_latency_trend.json

# Generate throughput analysis
curl -s "http://localhost:9090/api/v1/query_range?query=rate(risk_governor_decisions_total[1h])&start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600" | jq '.data.result[0].values' > reports/weekly_throughput.json

# Create performance summary
cat > reports/weekly_performance_$(date +%Y%m%d).md << EOF
# Weekly Performance Report - Week $(date +%U), $(date +%Y)

## Key Metrics
- Average P99 Latency: $(cat reports/weekly_latency_trend.json | jq 'map(.[1] | tonumber) | add / length')ms
- Peak Throughput: $(cat reports/weekly_throughput.json | jq 'map(.[1] | tonumber) | max') decisions/sec
- Uptime: $(uptime | awk '{print $3,$4}' | sed 's/,//')

## Trends
- Latency trend: $([ $(cat reports/weekly_latency_trend.json | jq 'length') -gt 0 ] && echo "Data available" || echo "No data")
- Throughput trend: $([ $(cat reports/weekly_throughput.json | jq 'length') -gt 0 ] && echo "Data available" || echo "No data")

## Recommendations
- Monitor for any upward latency trends
- Consider capacity planning if throughput approaching limits
- Review any performance anomalies
EOF
```

### **🔐 Security & Maintenance**

#### **Weekly Maintenance Script**
```bash
# Create weekly maintenance script
cat > scripts/weekly_maintenance.sh << 'EOF'
#!/bin/bash
echo "🔧 Weekly Maintenance - $(date)"

# 1. Log rotation and cleanup
find logs/stress_testing -name "*.log" -mtime +14 -delete
find archives -type d -mtime +60 -exec rm -rf {} + 2>/dev/null || true

# 2. Database maintenance
docker exec timescaledb_primary psql -U postgres -d trading_data -c "VACUUM ANALYZE;"
redis-cli BGSAVE

# 3. Performance baseline update
./scripts/monitor.sh metrics > reports/weekly_baseline_$(date +%Y%m%d).log

# 4. System health check
df -h
free -h
docker system df

echo "✅ Weekly maintenance complete"
EOF

chmod +x scripts/weekly_maintenance.sh
```

#### **Security Checklist**
```bash
# Monthly security review
cat > scripts/security_check.sh << 'EOF'
#!/bin/bash
echo "🔐 Security Check - $(date)"

# 1. Check for unauthorized access
sudo grep "Failed password" /var/log/auth.log | tail -10

# 2. Review open ports
netstat -tlnp | grep LISTEN

# 3. Check file permissions
find /home/cristian/IntradayTrading/ITS -type f -perm /o+w -ls

# 4. Review Docker security
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# 5. Check for updates
apt list --upgradable

echo "✅ Security check complete"
EOF

chmod +x scripts/security_check.sh
```

---

## 📚 **REFERENCE**

### **🎯 Success Criteria**

#### **Daily Success Metrics**
- [ ] Platform startup completed without errors
- [ ] All services responding on expected ports
- [ ] Daily certification passes (100% pass rate)
- [ ] No critical alerts during trading hours
- [ ] Clean shutdown with data archived

#### **Performance Thresholds**
| Metric | Green | Yellow | Red | Action |
|--------|-------|--------|-----|---------|
| Decision Latency P99 | < 10ms | 10-15ms | > 15ms | Investigate/Restart |
| Pipeline Latency P99 | < 15ms | 15-20ms | > 20ms | Check connectivity |
| Error Rate | < 0.05% | 0.05-0.1% | > 0.1% | Check logs |
| Position Delta | < $500 | $500-$1000 | > $1000 | Verify positions |
| Hard Limit Breaches | 0 | 0 | > 0 | IMMEDIATE ACTION |

### **📁 File Structure**
```
stress_testing/
├── StressTestUserGuide.md          # This guide
├── SUCCESS_GATES.md                # 5-day sprint criteria
├── PAPER_TRADING_CHECKLIST.md      # Launch procedures
├── CRITICAL_REFINEMENTS_COMPLETE.md # Enhancement summary
├── run_full_suite.py               # Main test runner
├── scenarios/                      # Test scenarios
├── simulators/                     # Market simulators
├── injectors/                      # Fault injectors
├── core/                          # Core framework
├── ci/                            # CI/CD scripts
└── results/                       # Test results

scripts/
├── start_platform.sh              # Platform startup
├── daily_startup.sh               # Morning routine
├── daily_shutdown.sh              # Evening routine
├── monitor.sh                     # Monitoring commands
├── start_prometheus.sh            # Prometheus startup
└── start_grafana.sh               # Grafana startup

monitoring/
├── prometheus/
│   ├── prometheus.yml             # Prometheus config
│   └── rules/                     # Alert rules
└── grafana/
    ├── grafana.ini               # Grafana config
    ├── dashboards/               # Dashboard definitions
    └── provisioning/             # Auto-provisioning

logs/stress_testing/              # All log files
reports/                          # Daily/weekly reports
archives/                         # Historical data
```

### **🌐 URLs & Ports**
| Service | URL | Port | Purpose |
|---------|-----|------|---------|
| Grafana Dashboard | http://localhost:3000 | 3000 | Real-time monitoring |
| Prometheus | http://localhost:9090 | 9090 | Metrics & alerts |
| Stress Test Metrics | http://localhost:8000/metrics | 8000 | Raw metrics |
| TimescaleDB | localhost:5432 | 5432 | Historical data |
| Redis | localhost:6379 | 6379 | State management |

### **🔑 Credentials**
| Service | Username | Password | Notes |
|---------|----------|----------|-------|
| Grafana | admin | stress_testing_2025 | Web UI access |
| TimescaleDB | postgres | (configured) | Database access |
| Redis | (none) | (none) | No auth by default |

### **📞 Support & Escalation**
1. **Self-Service**: Use this guide and troubleshooting procedures
2. **Team Chat**: #trading-alerts Slack channel
3. **On-Call Engineer**: [Configure phone number]
4. **Team Lead**: [Configure phone number]
5. **Emergency**: [Configure emergency contact]

### **📖 Additional Documentation**
- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Operational Guide**: `OPERATIONAL_GUIDE.md`
- **Paper Trading Checklist**: `PAPER_TRADING_CHECKLIST.md`
- **Success Gates**: `SUCCESS_GATES.md`

---

## 🎯 **QUICK REFERENCE COMMANDS**

### **Daily Operations**
```bash
# Morning startup
./scripts/daily_startup.sh

# Health check
./scripts/monitor.sh health

# Run certification
python stress_testing/run_full_suite.py --certification

# Evening shutdown
./scripts/daily_shutdown.sh
```

### **Monitoring**
```bash
# Real-time monitoring
./scripts/monitor.sh watch

# Check alerts
./scripts/monitor.sh alerts

# View metrics
./scripts/monitor.sh metrics

# Monitor latency
./scripts/monitor.sh latency
```

### **Troubleshooting**
```bash
# Check service status
./scripts/monitor.sh status

# View recent errors
./scripts/monitor.sh errors

# Emergency reset
./scripts/daily_shutdown.sh && ./scripts/daily_startup.sh

# Check logs
tail -f logs/stress_testing/*.log
```

### **Testing**
```bash
# Individual scenarios
python stress_testing/run_full_suite.py --scenario flash_crash
python stress_testing/run_full_suite.py --scenario decision_flood
python stress_testing/run_full_suite.py --scenario broker_failure
python stress_testing/run_full_suite.py --scenario portfolio_integrity

# Full certification
python stress_testing/run_full_suite.py --certification
```

---

**🚀 The Stress Testing Platform is now ready for production use with complete monitoring, alerting, and operational procedures!**

---

*Last Updated: $(date)*  
*Version: 1.0*  
*Platform: IntradayTrading Stress Testing System*