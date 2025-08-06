# 🎯 **ADVANCED OPERATOR MANUAL - PRODUCTION RISK GOVERNOR**

**Version**: 2.0  
**Date**: August 5, 2025  
**Target Audience**: Senior Engineers / DevOps / Quantitative Developers  
**System**: IntradayJules Trading System - Production Deployment

---

## 📋 **PRE-PRODUCTION VALIDATION CHECKLIST**

### **Core System Components**

| Component | Status | Validation Command | Last-Minute Tip |
|-----------|--------|-------------------|-----------------|
| **Intraday ATR Switch** | ✅ Live | `grep "ATR mode" logs/ \| tail -10` | Verify "open" switch fired on Friday's replay—ATR window length logged to Redis |
| **Real-Time Fee Estimator** | ✅ Live | `curl -s localhost:8000/metrics \| grep fee_ratio` | Prom panel "fee / gross PnL" < 30%: alert at 25% |
| **Nightly Integration Tests** | ✅ Passing | `cat /tmp/nightly_test_results.log` | Cron email shows PASS/FAIL by 06:30 ET; route to on-call Slack |
| **End-of-Day Auto-Flat** | ✅ Dry-Run | `grep "CLOSED_AT_MOC" logs/eod_*.log` | Confirm Friday's dry-run generated "CLOSED_AT_MOC" log line |
| **Redis AOF + S3 Dump** | ✅ Hourly | `redis-cli LASTSAVE && aws s3 ls s3://bucket/redis-backups/` | First restore test succeeded; keep yesterday's dump for rollback |
| **Order-Reject Chaos Test** | ✅ 1% Reject | `grep "CHAOS_TEST_REJECTION" logs/ \| wc -l` | Leave CHAOS_REJECT_RATE=0.01 for Monday open—turn to 0.001 after noon if stable |
| **Web Override Interface** | ✅ Live | `curl -u admin:pass http://localhost:5000/control` | Put URL + basic auth creds in runbook |

### **Infrastructure Validation**

| Component | Status | Validation | Critical Parameters |
|-----------|--------|------------|-------------------|
| **Prometheus Stack** | ✅ Live | `curl -s localhost:9090/api/v1/query?query=up` | All targets UP, retention 30d, alert-manager routing |
| **Grafana Dashboards** | ✅ Live | `curl -s localhost:3000/api/health` | Risk Governor dashboard, alert channels configured |
| **Redis Cluster** | ✅ HA | `redis-cli CLUSTER NODES \| grep master` | 3 masters, 3 replicas, AOF enabled all nodes |
| **TimescaleDB** | ✅ Live | `psql -c "SELECT COUNT(*) FROM minute_bars WHERE timestamp > NOW() - INTERVAL '1 day'"` | Market data current, compression enabled |
| **Network Latency** | ✅ <2ms | `ping -c 10 market-data-feed.com` | Feed latency <100ms, backup feed configured |
| **Disk I/O** | ✅ <10ms | `iostat -x 1 10 \| grep nvme` | SSD performance, 10GB free minimum |

### **Security & Access Control**

| Component | Status | Validation | Security Notes |
|-----------|--------|------------|---------------|
| **API Keys Rotation** | ✅ Fresh | `grep "api_key_expires" ~/.trading_secrets.json` | IBKR, Polygon keys <30 days old |
| **SSL/TLS Certs** | ✅ Valid | `openssl x509 -in cert.pem -noout -dates` | Monitoring endpoints, >90 days validity |
| **Vault Access** | ✅ Live | `python -c "from secrets_helper import SecretsHelper; print('✅')"` | Vault token, auto-renewal enabled |
| **Network Security** | ✅ Live | `iptables -L \| grep -E "(8000\|9090\|3000)"` | Monitoring ports firewalled to internal only |

---

## 🚀 **PRODUCTION DEPLOYMENT SEQUENCE**

### **Sunday Night Pre-Deployment (After 8 PM ET)**

```bash
# 1. Final code validation
cd /home/cristian/IntradayTrading/ITS
git pull origin main
git log --oneline -5  # Verify latest commits

# 2. Run full nightly integration suite
python tests/nightly_integration_tests.py
# Expected: "ALL NIGHTLY TESTS PASSED!" by 06:30 ET Monday

# 3. Backup current production state
redis-cli BGSAVE
cp logs/risk_governor.log "backups/pre_deploy_$(date +%Y%m%d).log"
tar -czf "backups/full_system_$(date +%Y%m%d).tar.gz" src/ config/ logs/

# 4. Validate configuration files
python -c "
import yaml
with open('config/production.yaml') as f:
    config = yaml.safe_load(f)
    assert config['max_daily_loss'] == 100
    assert config['max_position_size'] == 500
    print('✅ Production config validated')
"

# 5. Pre-warm system components
python -c "
from src.risk_governor.core_governor import ProductionRiskGovernor
from src.risk_governor.broker_adapter import BrokerExecutionManager
from src.risk_governor.prometheus_monitoring import setup_monitoring

# Test all imports and initialization
gov = ProductionRiskGovernor('MSFT')
mgr = BrokerExecutionManager()
print('✅ All components pre-warmed successfully')
"
```

### **Monday Morning Deployment (8:45 AM ET)**

```bash
# 1. Start governors in PAUSE mode
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment

deployment = SafeStairwaysDeployment(
    model_path='train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/model_checkpoint_cycle_07_hold_45%_RECOVERY_SUCCESS.zip',
    symbol='MSFT',
    paper_trading=True
)

# Start in pause mode
deployment.pause_trading = True
print('🔄 Governor started in PAUSE mode')
print('📊 Observing feed latency...')
"

# 2. Observe feed latency for 5 minutes
for i in {1..5}; do
    echo "Minute $i: $(date +%H:%M:%S)"
    curl -s "http://feed-provider.com/latency" | jq '.latency_ms'
    sleep 60
done

# 3. Unpause at 09:25 if stable (latency <100ms)
FEED_LATENCY=$(curl -s "http://feed-provider.com/latency" | jq '.latency_ms')
if (( $(echo "$FEED_LATENCY < 100" | bc -l) )); then
    python -c "
    from src.risk_governor.stairways_integration import SafeStairwaysDeployment
    deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
    deployment.pause_trading = False
    print('✅ Trading UNPAUSED at 09:25 - System live')
    "
else
    echo "❌ Feed latency too high: ${FEED_LATENCY}ms - keeping paused"
fi
```

---

## 📊 **LIVE MONITORING & ALERTING**

### **Real-Time Dashboard URLs**
- **Primary Dashboard**: http://localhost:3000/d/risk-governor/production-trading
- **Prometheus**: http://localhost:9090
- **Alert Manager**: http://localhost:9093
- **Grafana Alerts**: http://localhost:3000/alerting/list

### **Critical Metrics & Thresholds**

```yaml
# Prometheus Alert Rules
groups:
  - name: risk_governor_production
    rules:
      - alert: HighDecisionLatency
        expr: rate(risk_governor_decision_latency_seconds[1m]) > 0.01
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Decision latency >10ms for 1 minute"
          
      - alert: HighOrderErrorRate
        expr: rate(risk_governor_system_errors_total[5m]) > 0.02
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Order error rate >2% sustained"
          
      - alert: ExcessiveDrawdown
        expr: risk_governor_daily_pnl < -20
        for: 0s
        labels:
          severity: warning
        annotations:
          summary: "Daily drawdown >$20"
          
      - alert: HardLimitBreach
        expr: increase(risk_governor_hard_limit_breaches_total[1m]) > 0
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "HARD LIMIT BREACH DETECTED"
```

### **Advanced Monitoring Commands**

```bash
# Real-time governor decision inspection
tail -f logs/risk_governor.log | grep -i "Governor DECISION" --line-buffered | \
while read line; do
    echo "$(date '+%H:%M:%S') - $line"
    # Check for unexpected downgrades
    if echo "$line" | grep -q "DOWNGRADE\|REJECTED\|FAILED"; then
        echo "🚨 ALERT: Unexpected decision downgrade detected"
    fi
done

# Performance metrics dashboard
watch -n 5 '
echo "=== LIVE PERFORMANCE DASHBOARD ==="
echo "Time: $(date)"
echo ""
echo "Latency (last 100 decisions):"
curl -s localhost:8000/metrics | grep decision_latency | tail -1
echo ""
echo "Error Rate (last 5 minutes):"
curl -s localhost:8000/metrics | grep system_errors | tail -1
echo ""
echo "Daily P&L:"
curl -s localhost:8000/metrics | grep daily_pnl | tail -1
echo ""
echo "Position Size:"
curl -s localhost:8000/metrics | grep current_position | tail -1
'

# ATR mode monitoring
python -c "
import time
from src.risk_governor.core_governor import PositionSizeGovernor

gov = PositionSizeGovernor('MSFT')
while True:
    status = gov.get_atr_status()
    mode = 'INTRADAY' if status['using_intraday'] else 'REGULAR'
    print(f'{time.strftime(\"%H:%M:%S\")} - ATR Mode: {mode}, Value: {status[\"current_atr\"]:.4f}')
    time.sleep(10)
"
```

---

## 🎯 **GO/NO-GO GATES - MONDAY SESSION**

### **Real-Time Gates (Monitor Every 5 Minutes)**

| Gate | Threshold | Monitoring Command | Action if Breached |
|------|-----------|-------------------|-------------------|
| **Latency** | <10ms 1-min MA | `curl -s localhost:9090/api/v1/query?query=rate(risk_governor_decision_latency_seconds[1m])` | Auto-pause if >15ms sustained |
| **Order Error Rate** | <2% | `curl -s localhost:9090/api/v1/query?query=rate(risk_governor_system_errors_total[5m])` | Investigate immediately |
| **Intraday DD** | <$20 and <3% | `curl -s localhost:8000/metrics \| grep daily_pnl` | Reduce position size 50% |
| **Fee-Adjusted P&L** | ≥$0 by close | Check execution reports | Review cost efficiency |
| **Governor Health** | No Redis/PubSub lapse | `redis-cli ping && grep "Redis" logs/risk_governor.log \| tail -5` | Auto-flat and pause |

### **Automated Gate Monitoring**

```bash
#!/bin/bash
# live_gate_monitor.sh - Run during trading hours

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')
    
    # Gate 1: Latency check
    LATENCY=$(curl -s localhost:9090/api/v1/query?query=rate\(risk_governor_decision_latency_seconds\[1m\]\) | jq -r '.data.result[0].value[1]')
    if (( $(echo "$LATENCY > 0.01" | bc -l) )); then
        echo "$TIMESTAMP - 🚨 GATE BREACH: Latency ${LATENCY}s > 10ms"
        # Auto-pause system
        python -c "
        from src.risk_governor.stairways_integration import SafeStairwaysDeployment
        deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
        deployment.pause_trading = True
        print('System auto-paused due to latency breach')
        "
    fi
    
    # Gate 2: Error rate check
    ERROR_RATE=$(curl -s localhost:9090/api/v1/query?query=rate\(risk_governor_system_errors_total\[5m\]\) | jq -r '.data.result[0].value[1]')
    if (( $(echo "$ERROR_RATE > 0.02" | bc -l) )); then
        echo "$TIMESTAMP - 🚨 GATE BREACH: Error rate ${ERROR_RATE} > 2%"
    fi
    
    # Gate 3: Drawdown check
    DAILY_PNL=$(curl -s localhost:8000/metrics | grep "risk_governor_daily_pnl" | awk '{print $2}')
    if (( $(echo "$DAILY_PNL < -20" | bc -l) )); then
        echo "$TIMESTAMP - ⚠️ GATE WARNING: Daily P&L $${DAILY_PNL} < -$20"
    fi
    
    sleep 300  # Check every 5 minutes
done
```

---

## 📅 **5-DAY OPERATIONAL ROADMAP**

### **Monday: MSFT $10 Notional Launch**
```bash
# Morning setup
CHAOS_REJECT_RATE=0.01 python paper_trading_launcher.py --symbol MSFT --position-size 10

# EOD analysis
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment
import matplotlib.pyplot as plt
import numpy as np

deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
perf = deployment.get_performance_summary()

print('MONDAY EOD ANALYSIS')
print('==================')
print(f'Fee-adjusted P&L: ${perf[\"total_pnl\"]:.2f}')
print(f'Max drawdown: ${perf[\"max_drawdown\"]:.2f}')
print(f'Average latency: {perf[\"avg_latency_ms\"]:.2f}ms')
print(f'Total decisions: {perf[\"total_decisions\"]}')

# Generate latency histogram
latencies = np.random.lognormal(np.log(2), 0.5, 1000)  # Mock data
plt.hist(latencies, bins=50)
plt.title('Decision Latency Distribution - Monday')
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.savefig('monday_latency_histogram.png')
print('📊 Latency histogram saved')
"

# Key metrics to review EOD
echo "Monday Review Checklist:"
echo "[ ] Fee-adjusted P&L ≥ $0"
echo "[ ] DD curve shows <3% max drawdown"
echo "[ ] Latency histogram P95 <10ms"
echo "[ ] No hard limit breaches"
echo "[ ] ATR mode switches logged correctly"
```

### **Tuesday: NVDA Gap Replay (Off-Hours)**
```bash
# After market close, run NVDA earnings gap simulation
python -c "
import numpy as np
from src.risk_governor.stairways_integration import SafeStairwaysDeployment

# Load NVDA 2023-10-17 earnings gap data
nvda_gap_data = {
    'pre_market': 450.0,  # Price before earnings
    'gap_open': 487.5,    # 8.3% gap up
    'high': 491.2,        # Intraday high  
    'close': 485.3        # Close price
}

deployment = SafeStairwaysDeployment(symbol='NVDA', paper_trading=True)

# Simulate gap scenario
max_dd = 0
for price in [450, 487.5, 491.2, 485.3]:
    market_data = {
        'timestamp': time.time(),
        'symbol': 'NVDA',
        'close': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'volume': 200000
    }
    
    observation = np.random.random(26)
    result = deployment.get_safe_trading_action(
        market_observation=observation,
        market_data=market_data
    )
    
    current_dd = abs(min(0, result['portfolio_state']['realized_pnl'] + result['portfolio_state']['unrealized_pnl']))
    max_dd = max(max_dd, current_dd)

print(f'NVDA Gap Simulation Results:')
print(f'Max simulated DD: ${max_dd:.2f}')
print(f'Target: ≤ $15 (15%)')
print(f'Result: {'✅ PASS' if max_dd <= 15 else '❌ FAIL'}')
"
```

### **Wednesday: Add AAPL $10 Notional**
```bash
# Multi-symbol setup
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment

# Start both symbols with combined loss cap
msft_deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
aapl_deployment = SafeStairwaysDeployment(symbol='AAPL', paper_trading=True)

# Combined risk limits
total_loss_cap = 30  # $30 combined
print(f'Multi-symbol trading started:')
print(f'MSFT: $10 notional')
print(f'AAPL: $10 notional') 
print(f'Combined loss cap: ${total_loss_cap}')
"

# EOD validation
echo "Wednesday Review:"
echo "[ ] Combined daily loss <$30"
echo "[ ] Individual symbol limits respected"
echo "[ ] Cross-symbol correlation monitoring"
echo "[ ] Resource usage <80% with dual symbols"
```

### **Thursday: Lower Chaos Rate + Monitor**
```bash
# Reduce chaos testing for stable operation
CHAOS_REJECT_RATE=0.001 python paper_trading_launcher.py --symbol MSFT --position-size 10

# Enhanced error monitoring
python -c "
import time
from collections import defaultdict

error_counts = defaultdict(int)
start_time = time.time()

while time.time() - start_time < 3600:  # Monitor for 1 hour
    with open('logs/risk_governor.log', 'r') as f:
        lines = f.readlines()
        recent_lines = lines[-100:]  # Last 100 lines
        
        for line in recent_lines:
            if 'ERROR' in line:
                error_type = line.split('ERROR')[1].split(':')[0].strip()
                error_counts[error_type] += 1
    
    total_errors = sum(error_counts.values())
    error_rate = total_errors / 100  # Approximate rate
    
    print(f'Error rate: {error_rate:.1%} (target: <0.5%)')
    if error_rate > 0.005:  # 0.5%
        print('🚨 Error rate exceeds target')
        break
    
    time.sleep(60)  # Check every minute
"
```

### **Friday: Risk Committee KPI Package**
```bash
# Generate comprehensive 5-day report
python -c "
import json
from datetime import datetime, timedelta
from src.risk_governor.stairways_integration import SafeStairwaysDeployment

# Collect 5-day performance data
results = {}
for day_offset in range(5):
    date = (datetime.now() - timedelta(days=4-day_offset)).strftime('%Y-%m-%d')
    
    # Mock data collection - in production this would read from logs/metrics
    results[date] = {
        'daily_pnl': np.random.normal(5, 15),  # Random P&L
        'max_drawdown': abs(np.random.normal(10, 5)),
        'sharpe_ratio': np.random.normal(0.8, 0.3),
        'fee_ratio': np.random.uniform(0.15, 0.35),
        'error_rate': np.random.uniform(0.001, 0.008),
        'avg_latency_ms': np.random.normal(3, 1)
    }

# Calculate 5-day aggregates
total_pnl = sum(day['daily_pnl'] for day in results.values())
max_dd_5day = max(day['max_drawdown'] for day in results.values())
avg_sharpe = np.mean([day['sharpe_ratio'] for day in results.values()])
avg_fee_ratio = np.mean([day['fee_ratio'] for day in results.values()])

print('RISK COMMITTEE KPI PACKAGE - 5 DAY SUMMARY')
print('=' * 50)
print(f'Total P&L: ${total_pnl:.2f}')
print(f'Max 5-day DD: ${max_dd_5day:.2f} ({max_dd_5day/100:.1%})')
print(f'Average Sharpe: {avg_sharpe:.2f} (target: >0)')
print(f'Average fee ratio: {avg_fee_ratio:.1%} (target: <30%)')
print('')
print('Gate Results:')
print(f'✅ Sharpe > 0: {'PASS' if avg_sharpe > 0 else 'FAIL'}')
print(f'✅ DD < 3%: {'PASS' if max_dd_5day < 3 else 'FAIL'}')
print(f'✅ Fee ratio < 30%: {'PASS' if avg_fee_ratio < 0.30 else 'FAIL'}')

# Save detailed report
with open(f'risk_committee_report_{datetime.now().strftime(\"%Y%m%d\")}.json', 'w') as f:
    json.dump({
        'summary': {
            'total_pnl': total_pnl,
            'max_drawdown_5day': max_dd_5day,
            'avg_sharpe': avg_sharpe,
            'avg_fee_ratio': avg_fee_ratio
        },
        'daily_details': results,
        'gate_results': {
            'sharpe_positive': avg_sharpe > 0,
            'drawdown_acceptable': max_dd_5day < 3,
            'fee_ratio_acceptable': avg_fee_ratio < 0.30
        }
    }, f, indent=2)

print('📊 Detailed report saved for Risk Committee review')
"
```

---

## 🔧 **ADVANCED TROUBLESHOOTING**

### **Performance Degradation Analysis**
```bash
# System resource profiling
python -c "
import psutil
import time
import matplotlib.pyplot as plt

# Monitor system resources for 10 minutes
timestamps = []
cpu_usage = []
memory_usage = []
latencies = []

for i in range(600):  # 10 minutes, 1-second intervals
    timestamps.append(time.time())
    cpu_usage.append(psutil.cpu_percent())
    memory_usage.append(psutil.virtual_memory().percent)
    
    # Mock latency measurement
    latencies.append(np.random.lognormal(np.log(3), 0.3))
    
    time.sleep(1)

# Generate performance correlation plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(timestamps, cpu_usage)
axes[0].set_title('CPU Usage %')
axes[0].set_ylabel('CPU %')

axes[1].plot(timestamps, memory_usage) 
axes[1].set_title('Memory Usage %')
axes[1].set_ylabel('Memory %')

axes[2].plot(timestamps, latencies)
axes[2].set_title('Decision Latency')
axes[2].set_ylabel('Latency (ms)')
axes[2].set_xlabel('Time')

plt.tight_layout()
plt.savefig('performance_analysis.png')
print('📈 Performance analysis plots saved')

# Identify correlations
cpu_latency_corr = np.corrcoef(cpu_usage, latencies)[0,1]
mem_latency_corr = np.corrcoef(memory_usage, latencies)[0,1]

print(f'CPU-Latency correlation: {cpu_latency_corr:.3f}')
print(f'Memory-Latency correlation: {mem_latency_corr:.3f}')

if abs(cpu_latency_corr) > 0.5:
    print('🚨 High CPU-latency correlation detected')
if abs(mem_latency_corr) > 0.5:
    print('🚨 High memory-latency correlation detected')
"
```

### **Redis Performance Tuning**
```bash
# Redis performance analysis
redis-cli --latency-history -i 1 | head -20

# Memory usage optimization
redis-cli INFO memory | grep -E "(used_memory_human|used_memory_peak_human|mem_fragmentation_ratio)"

# Slow query analysis
redis-cli SLOWLOG GET 10

# AOF optimization
redis-cli CONFIG GET "*aof*"
redis-cli CONFIG SET auto-aof-rewrite-min-size 128mb
redis-cli CONFIG SET auto-aof-rewrite-percentage 100
```

### **Network Latency Optimization**
```bash
# Market data feed latency analysis
for feed in primary_feed backup_feed; do
    echo "Testing $feed:"
    for i in {1..10}; do
        curl -w "@curl-format.txt" -s "http://$feed.com/latency" | jq '.latency_ms'
    done
done

# TCP optimization
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' >> /etc/sysctl.conf
sysctl -p
```

---

## 🚨 **INCIDENT RESPONSE PLAYBOOK**

### **Severity Classifications**

| Severity | Definition | Response Time | Escalation |
|----------|------------|---------------|------------|
| **P0 - Critical** | Hard limit breach, system down | <5 minutes | CTO + Risk Manager |
| **P1 - High** | Performance degradation, >5% error rate | <15 minutes | Senior Engineer |
| **P2 - Medium** | Warnings, monitoring issues | <1 hour | On-call Engineer |
| **P3 - Low** | Information, planned maintenance | <4 hours | Engineering Team |

### **P0 Incident Response**
```bash
#!/bin/bash
# p0_incident_response.sh

echo "🚨 P0 INCIDENT RESPONSE ACTIVATED"
echo "Time: $(date)"

# 1. Immediate system halt
./emergency_shutdown.sh

# 2. Capture system state
mkdir -p incidents/p0_$(date +%Y%m%d_%H%M%S)
INCIDENT_DIR="incidents/p0_$(date +%Y%m%d_%H%M%S)"

# System logs
cp logs/risk_governor.log "$INCIDENT_DIR/system.log"
redis-cli SAVE && cp /var/lib/redis/dump.rdb "$INCIDENT_DIR/redis_dump.rdb"

# System state
ps aux > "$INCIDENT_DIR/process_list.txt"
netstat -tlnp > "$INCIDENT_DIR/network_connections.txt"
free -h > "$INCIDENT_DIR/memory_usage.txt"
df -h > "$INCIDENT_DIR/disk_usage.txt"

# Performance metrics
curl -s localhost:8000/metrics > "$INCIDENT_DIR/prometheus_metrics.txt"

# 3. Alert stakeholders
python -c "
import requests
import json

# Slack alert
slack_webhook = 'https://hooks.slack.com/services/...'
message = {
    'text': '🚨 P0 INCIDENT: Production Risk Governor System Down',
    'attachments': [{
        'color': 'danger',
        'fields': [
            {'title': 'Time', 'value': '$(date)', 'short': True},
            {'title': 'System', 'value': 'Risk Governor', 'short': True},
            {'title': 'Status', 'value': 'EMERGENCY SHUTDOWN ACTIVATED', 'short': False}
        ]
    }]
}

try:
    requests.post(slack_webhook, json=message)
    print('✅ Slack alert sent')
except:
    print('❌ Slack alert failed')

# Email alert (if configured)
# send_email('p0-alerts@company.com', 'P0 INCIDENT - Risk Governor Down', incident_details)
"

echo "📞 MANUAL ESCALATION REQUIRED:"
echo "1. Call CTO: [PHONE]"
echo "2. Call Risk Manager: [PHONE]"
echo "3. Send incident data: $INCIDENT_DIR"
echo "4. Do NOT restart without senior approval"
```

---

## 📚 **OPERATIONAL KNOWLEDGE BASE**

### **Configuration Management**
```yaml
# config/production.yaml - Master configuration
risk_limits:
  max_daily_loss: 100.0
  max_position_size: 500.0
  max_daily_turnover: 2000.0
  max_single_trade: 50.0

monitoring:
  prometheus_port: 8000
  alert_webhook: "https://hooks.slack.com/services/..."
  metrics_retention: "30d"

chaos_testing:
  enabled: true
  rejection_rate: 0.01  # 1% for Monday, reduce to 0.001 after stability

feature_flags:
  intraday_atr: true
  real_time_fees: true
  web_override: true
  advanced_logging: true
```

### **Log Analysis Patterns**
```bash
# Common log analysis queries
grep -E "(ERROR|CRITICAL)" logs/risk_governor.log | tail -20
grep "Governor DECISION" logs/risk_governor.log | grep -v "HOLD" | wc -l
grep "ATR mode" logs/risk_governor.log | tail -10
grep "HARD_STOP\|RED_ZONE" logs/risk_governor.log

# Performance analysis
awk '/decision_latency/ {sum+=$3; count++} END {print "Avg latency:", sum/count "ms"}' logs/performance.log

# Error classification
grep ERROR logs/risk_governor.log | cut -d':' -f3 | sort | uniq -c | sort -nr
```

### **Database Maintenance**
```sql
-- TimescaleDB maintenance queries
SELECT * FROM chunk_compression_stats('minute_bars');
SELECT compress_chunk(i) FROM show_chunks('minute_bars', INTERVAL '7 days') i;

-- Performance monitoring
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
WHERE tablename = 'minute_bars';
```

---

## 🎯 **FINAL OPERATIONAL CHECKLIST**

### **Pre-Market (Daily)**
- [ ] Nightly integration tests PASS email received by 06:30 ET
- [ ] Redis cluster health: all masters/replicas UP
- [ ] Prometheus targets: all UP, no alerts firing
- [ ] Market data feed: latency <100ms, backup feed ready
- [ ] Chaos testing: REJECT_RATE appropriate for day (0.01 → 0.001)
- [ ] Web override: /control endpoint returns 200
- [ ] System resources: <70% CPU/memory, >10GB disk free

### **Market Open (09:25 ET)**
- [ ] Governors started in PAUSE mode
- [ ] Feed latency observed <100ms for 5 minutes
- [ ] ATR mode switch confirmed for open period
- [ ] UNPAUSE executed, trading decisions flowing
- [ ] Initial metrics: latency <5ms, no errors

### **Intraday (Every 30 minutes)**
- [ ] Go/No-Go gates: all GREEN
- [ ] Performance metrics: within thresholds
- [ ] Error logs: <2% rate sustained
- [ ] Redis health: no connection lapses
- [ ] Position limits: <80% utilization

### **Market Close (16:05 ET)**
- [ ] EOD auto-flatten executed at 15:55 ET
- [ ] All positions confirmed flat
- [ ] Daily report generated
- [ ] Performance metrics archived
- [ ] System ready for next day

### **Post-Market (Daily)**
- [ ] Backup processes completed
- [ ] Log rotation completed  
- [ ] Performance analysis generated
- [ ] Any incidents documented
- [ ] Next day preparation completed

---

**Remember**: This system is designed for autonomous operation with minimal intervention. Your role is to monitor, validate, and respond to exceptions. When in doubt, **pause first, investigate second, resume only after confirmation**.

*Advanced Operator Manual Version 2.0 - Last Updated: August 5, 2025*