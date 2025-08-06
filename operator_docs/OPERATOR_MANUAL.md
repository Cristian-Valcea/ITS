# 🎯 **PRODUCTION RISK GOVERNOR - OPERATOR MANUAL**

**Version**: 1.0  
**Date**: August 5, 2025  
**Target Audience**: Junior Programmers / Operations Team  
**System**: IntradayJules Trading System with Production Risk Governor

---

## 📋 **DAILY OPERATIONS CHECKLIST**

### **Every Trading Day - Pre-Market (8:00 AM ET)**

```bash
# 1. Activate Python environment
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# 2. Check nightly test results
cat /tmp/nightly_test_results.log
# ✅ Look for: "ALL NIGHTLY TESTS PASSED!"
# ❌ If failed: DO NOT start trading - call senior developer

# 3. Verify Redis is running
redis-cli ping
# ✅ Expected response: PONG
# ❌ If fails: sudo systemctl start redis-server

# 4. Check system health
python -c "
from src.risk_governor.core_governor import ProductionRiskGovernor
from src.risk_governor.broker_adapter import BrokerExecutionManager
print('✅ Risk Governor: System components loaded successfully')
"

# 5. Start monitoring (if not already running)
python start_monitoring.py
# ✅ Expected: "Monitoring system ready on port 8000"
```

---

## 🚀 **STARTING THE TRADING SYSTEM**

### **Step 1: Start Core Monitoring**
```bash
# Start Prometheus metrics server
python -c "
from src.risk_governor.prometheus_monitoring import setup_monitoring
monitoring = setup_monitoring(prometheus_port=8000)
print('📊 Monitoring started at http://localhost:8000')
"
```

### **Step 2: Launch Paper Trading**
```bash
# Start paper trading (safe mode)
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment
from src.risk_governor.eod_manager import create_eod_system

# Initialize deployment
deployment = SafeStairwaysDeployment(
    model_path='train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/model_checkpoint_cycle_07_hold_45%_RECOVERY_SUCCESS.zip',
    symbol='MSFT',
    paper_trading=True
)

# Start EOD monitoring
eod_system = create_eod_system(deployment.broker_manager)

print('🚀 Paper trading system started')
print('💰 Position size: \$10 notional')
print('🛡️ All risk limits active')
print('⏰ EOD auto-flatten at 15:55 ET')
"
```

### **Step 3: Verify System Status**
```bash
# Check all components are healthy
curl -s http://localhost:8000/metrics | grep -E "(risk_governor|decision_latency)" | head -5
# ✅ Should see metrics flowing

# Test Redis connection
redis-cli get risk_governor:MSFT:daily:$(date +%Y-%m-%d)
# ✅ Should return JSON data or null
```

---

## 📊 **MONITORING DASHBOARDS**

### **Primary Monitoring URLs**
- **Prometheus Metrics**: http://localhost:8000/metrics
- **Grafana Dashboard**: http://localhost:3000 (if configured)
- **System Logs**: `tail -f logs/risk_governor.log`

### **Key Metrics to Watch**

#### **🟢 HEALTHY INDICATORS**
- **Decision Latency**: <5ms average, <10ms max
- **Daily P&L**: Between -$20 and +$50  
- **Position Size**: ≤$500 total notional
- **Daily Turnover**: ≤$2000
- **Error Rate**: <2%
- **System Uptime**: >99%

#### **🟡 WARNING THRESHOLDS**
- **Decision Latency**: 5-10ms average
- **Daily P&L**: -$20 to -$50 loss
- **Position Usage**: 80-95% of limits
- **Error Rate**: 2-5%

#### **🔴 CRITICAL ALERTS**
- **Decision Latency**: >10ms sustained
- **Daily P&L**: <-$50 loss
- **Hard Limit Breach**: ANY occurrence
- **System Errors**: >5% rate
- **Monitoring Offline**: No metrics for >5 minutes

---

## 🔍 **SYSTEM HEALTH CHECKS**

### **Every 30 Minutes During Market Hours**
```bash
# Quick health check script
python -c "
import time
import requests
from src.risk_governor.broker_adapter import BrokerExecutionManager

print(f'🕐 Health Check: {time.strftime(\"%H:%M:%S\")}')

# Check metrics endpoint
try:
    resp = requests.get('http://localhost:8000/metrics', timeout=5)
    print(f'📊 Metrics: {\"✅ OK\" if resp.status_code == 200 else \"❌ FAIL\"}')
except:
    print('📊 Metrics: ❌ UNREACHABLE')

# Check Redis
try:
    import redis
    r = redis.Redis(decode_responses=True)
    r.ping()
    print('💾 Redis: ✅ OK')
except:
    print('💾 Redis: ❌ FAIL')

# Check broker manager
try:
    mgr = BrokerExecutionManager()
    stats = mgr.get_daily_stats()
    print(f'💼 Orders Today: {stats[\"daily_order_count\"]}/{stats[\"max_daily_orders\"]}')
    print(f'💰 Daily Cost: \${stats[\"daily_effective_cost\"]:.2f}/\${stats[\"max_daily_effective_cost\"]:.2f}')
except Exception as e:
    print(f'💼 Broker: ❌ {str(e)}')
"
```

### **Red Flags - Call Senior Developer Immediately**
- 💥 **Hard limit breach alert**
- 🔥 **System error rate >10%**
- ⏰ **Monitoring down for >10 minutes**
- 📉 **Daily loss >$75**
- 🚫 **Cannot connect to broker**

---

## ⚠️ **ALERT RESPONSE PROCEDURES**

### **🟡 WARNING ALERTS**

#### **High Latency (5-10ms)**
```bash
# Check system load
top -n 1 | head -5

# Check recent errors
tail -20 logs/risk_governor.log | grep ERROR

# If persistent, restart monitoring
pkill -f prometheus_monitoring
python start_monitoring.py
```

#### **Daily Loss $20-50**
```bash
# Check current positions
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment
deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
perf = deployment.get_performance_summary()
print(f'📊 Current P&L: \${perf[\"total_pnl\"]:.2f}')
print(f'📈 Position: \${perf[\"current_position\"]:.2f}')
print(f'💫 Turnover: \${perf[\"daily_turnover\"]:.2f}')
"

# Monitor more closely - check every 15 minutes
```

#### **High Position Usage (80-95%)**
```bash
# Check position breakdown
python -c "
from src.risk_governor.broker_adapter import BrokerExecutionManager
mgr = BrokerExecutionManager()
stats = mgr.get_daily_stats()
position_pct = (stats['daily_effective_cost'] / stats['max_daily_effective_cost']) * 100
print(f'📊 Position Usage: {position_pct:.1f}%')
if position_pct > 90:
    print('⚠️ WARNING: Approaching position limits')
"
```

### **🔴 CRITICAL ALERTS**

#### **Hard Limit Breach - IMMEDIATE ACTION**
```bash
# STOP EVERYTHING IMMEDIATELY
echo "🚨 HARD LIMIT BREACH - STOPPING SYSTEM"

# 1. Kill all trading processes
pkill -f stairways_integration
pkill -f broker_adapter  
pkill -f risk_governor

# 2. Force flatten all positions
python -c "
from src.risk_governor.broker_adapter import BrokerExecutionManager
mgr = BrokerExecutionManager()
reports = mgr.broker.flatten_all_positions()
print(f'🚑 Emergency flatten: {len(reports)} positions closed')
"

# 3. Call senior developer IMMEDIATELY
echo "📞 CALL SENIOR DEVELOPER NOW"
echo "📧 Send email with system logs"

# 4. Document incident
echo "$(date): HARD LIMIT BREACH - System halted" >> incident_log.txt
```

#### **System Error Rate >5%**
```bash
# Check error details
tail -50 logs/risk_governor.log | grep ERROR

# Restart core components
python restart_system.py

# If errors persist - STOP TRADING
echo "🛑 STOPPING TRADING DUE TO HIGH ERROR RATE"
```

#### **Daily Loss >$75**
```bash
echo "🚨 EXCESSIVE DAILY LOSS - EMERGENCY STOP"

# 1. Immediately stop new orders
python -c "
from src.risk_governor.eod_manager import create_eod_system
from src.risk_governor.broker_adapter import BrokerExecutionManager
mgr = BrokerExecutionManager()
eod = create_eod_system(mgr)
result = eod.force_flatten_now()
print(f'🚑 Emergency flatten executed: {result}')
"

# 2. Alert management
echo "📞 CALL RISK MANAGER IMMEDIATELY"
echo "📧 SEND P&L REPORT TO MANAGEMENT"
```

---

## 🛑 **EMERGENCY SHUTDOWN PROCEDURES**

### **PANIC BUTTON - Complete System Shutdown**
```bash
#!/bin/bash
# emergency_shutdown.sh - Run this if everything goes wrong

echo "🚨 EMERGENCY SHUTDOWN INITIATED"

# 1. Stop all trading processes
echo "⏹️ Stopping trading processes..."
pkill -f stairways
pkill -f risk_governor
pkill -f broker
pkill -f prometheus

# 2. Flatten all positions immediately
echo "📉 Flattening all positions..."
python -c "
try:
    from src.risk_governor.broker_adapter import BrokerExecutionManager
    mgr = BrokerExecutionManager()
    reports = mgr.broker.flatten_all_positions()
    print(f'✅ Flattened {len(reports)} positions')
except Exception as e:
    print(f'❌ Flatten failed: {e}')
"

# 3. Save system state
echo "💾 Saving system state..."
redis-cli bgsave
cp logs/risk_governor.log "emergency_logs_$(date +%Y%m%d_%H%M%S).log"

# 4. Alert everyone
echo "📢 EMERGENCY SHUTDOWN COMPLETE"
echo "📞 CALL SENIOR DEVELOPER: [PHONE NUMBER]"
echo "📧 EMAIL MANAGEMENT: [EMAIL ADDRESS]"
echo "📝 LOG INCIDENT IN: incident_log.txt"

# 5. Document shutdown
echo "$(date): EMERGENCY SHUTDOWN - All systems halted" >> incident_log.txt
```

### **Manual Position Flattening**
```bash
# If automated flattening fails, manual override:
python -c "
from src.risk_governor.broker_adapter import IBKRPaperAdapter
broker = IBKRPaperAdapter()

# Cancel all pending orders
open_orders = broker.get_open_orders()
for order in open_orders:
    broker.cancel_order(order.order_id)
    print(f'❌ Cancelled order: {order.order_id}')

print('✅ All pending orders cancelled')
"
```

---

## 🔧 **ROUTINE MAINTENANCE**

### **Daily Tasks (End of Trading Day)**
```bash
# After market close (4:30 PM ET)

# 1. Generate daily report
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment
deployment = SafeStairwaysDeployment(symbol='MSFT', paper_trading=True)
perf = deployment.get_performance_summary()

print('📊 DAILY TRADING REPORT')
print(f'💰 Total P&L: \${perf[\"total_pnl\"]:.2f}')
print(f'📈 Final Position: \${perf[\"current_position\"]:.2f}')
print(f'💫 Daily Turnover: \${perf[\"daily_turnover\"]:.2f}')
print(f'⚡ Avg Latency: {perf[\"avg_latency_ms\"]:.2f}ms')
print(f'🎯 Total Decisions: {perf[\"total_decisions\"]}')
" > daily_report_$(date +%Y%m%d).txt

# 2. Backup important data
redis-cli bgsave
cp logs/risk_governor.log "daily_logs/log_$(date +%Y%m%d).log"

# 3. Reset daily counters (automatic at midnight)
# 4. Run nightly tests (automatic at 2 AM)
```

### **Weekly Tasks (Friday After Close)**
```bash
# 1. Generate weekly summary
python generate_weekly_report.py

# 2. Check disk space
df -h | grep -E "(logs|backups|redis)"

# 3. Clean old log files (keep 30 days)
find logs/ -name "*.log" -mtime +30 -delete
find daily_logs/ -name "*.log" -mtime +30 -delete

# 4. Backup Redis data
redis-cli save
cp /var/lib/redis/dump.rdb "backups/redis_backup_$(date +%Y%m%d).rdb"

# 5. Test emergency procedures (simulation)
echo "🧪 Testing emergency shutdown (simulation)"
# Run emergency_shutdown.sh in test mode
```

---

## 📞 **CONTACT INFORMATION**

### **Escalation Chain**
1. **Warnings/Questions**: Junior Developer → Senior Developer
2. **System Errors**: Direct to Senior Developer  
3. **Hard Limit Breach**: Senior Developer + Risk Manager
4. **Emergency Loss**: Risk Manager + CTO + Management

### **Contact List**
```
Senior Developer: [PHONE] [EMAIL]
Risk Manager: [PHONE] [EMAIL] 
CTO: [PHONE] [EMAIL]
Operations Manager: [PHONE] [EMAIL]

Emergency Slack: #trading-alerts
System Logs: /trading/logs/
Incident Reports: incident_log.txt
```

---

## 📚 **COMMON TROUBLESHOOTING**

### **"Cannot connect to Redis"**
```bash
# Check if Redis is running
sudo systemctl status redis-server

# If stopped, start it
sudo systemctl start redis-server

# Test connection
redis-cli ping
```

### **"Prometheus metrics not found"**
```bash
# Restart monitoring
pkill -f prometheus_monitoring
python start_monitoring.py

# Check metrics endpoint
curl http://localhost:8000/metrics | head -10
```

### **"High latency warnings"**
```bash
# Check system resources
top -n 1
free -h
df -h

# Restart system if resources are low
python restart_system.py
```

### **"Model prediction errors"**
```bash
# Check model file exists
ls -la train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/

# If model missing, use fallback
python -c "
from src.risk_governor.stairways_integration import SafeStairwaysDeployment
deployment = SafeStairwaysDeployment(model_path=None, symbol='MSFT', paper_trading=True)
print('✅ Using fallback mock model')
"
```

### **"EOD auto-flatten not working"**
```bash
# Check current time
date

# Manually trigger flatten
python -c "
from src.risk_governor.eod_manager import create_eod_system
from src.risk_governor.broker_adapter import BrokerExecutionManager
mgr = BrokerExecutionManager()
eod = create_eod_system(mgr)
result = eod.force_flatten_now()
print(f'Manual flatten result: {result}')
"
```

---

## 🎯 **SUCCESS CRITERIA - WHAT GOOD LOOKS LIKE**

### **Healthy System Indicators**
- ✅ All health checks return green
- ✅ Latency consistently <5ms
- ✅ Daily P&L between -$20 and +$50
- ✅ No error alerts for >1 hour
- ✅ Position sizes stay within limits
- ✅ EOD flatten executes automatically at 15:55

### **End of Day Success**
- ✅ All positions flattened
- ✅ Daily report generated
- ✅ No hard limit breaches
- ✅ System ready for next day
- ✅ Logs backed up
- ✅ Nightly tests scheduled

---

## 🚨 **REMEMBER - WHEN IN DOUBT**

1. **🛑 STOP FIRST** - Better safe than sorry
2. **📞 CALL SENIOR DEVELOPER** - Don't guess  
3. **📝 DOCUMENT EVERYTHING** - What happened, when, what you did
4. **💾 SAVE LOGS** - Before restarting anything
5. **🚫 NEVER OVERRIDE HARD LIMITS** - They exist for a reason

---

**This manual covers 95% of operational scenarios. For anything not covered here, immediately contact the senior developer. Remember: The system is designed to fail safe, so when in doubt, shut it down.**

*Document Version: 1.0 - Updated: August 5, 2025*