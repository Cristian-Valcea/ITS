# 📋 **PAPER TRADING LAUNCH CHECKLIST**

**Pre-production validation checklist - KEEP PINNED**

---

## 🕕 **PRE-MARKET CHECKLIST (06:30 ET)**

### **1. CI Job Validation** ✅
```bash
# Check nightly CI job status
curl -s "https://ci-server/api/jobs/stress-testing/latest" | jq '.status'
# Must show: "SUCCESS"

# Verify last run timestamp
curl -s "https://ci-server/api/jobs/stress-testing/latest" | jq '.completed_at'
# Must be within last 8 hours

# Check for any failures in last 7 days
curl -s "https://ci-server/api/jobs/stress-testing/history?days=7" | jq '.failures'
# Must show: []
```

**Status**: ⬜ CI job green at 06:30 ET

---

### **2. Redis AOF Snapshot Validation** ✅
```bash
# Check Redis AOF snapshot age
redis-cli LASTSAVE | xargs -I {} date -d @{}
# Must be < 15 minutes old

# Verify Redis is responding
redis-cli ping
# Must show: PONG

# Check Redis memory usage
redis-cli INFO memory | grep used_memory_human
# Should be reasonable (< 1GB for typical usage)
```

**Status**: ⬜ Last Redis AOF snapshot < 15 min old

---

### **3. Database Health Check** ✅
```bash
# Verify TimescaleDB container is running
docker ps | grep timescaledb_primary
# Must show: Up status

# Check database connectivity
docker exec timescaledb_primary pg_isready -U postgres -d trading_data
# Must show: accepting connections

# Verify recent data availability
python -c "
from stress_testing.simulators.historical_data_adapter import HistoricalDataAdapter
adapter = HistoricalDataAdapter()
report = adapter.validate_data_availability()
print(f'Database: {report[\"database_connected\"]}, Bars: {report[\"total_bars\"]:,}')
"
# Must show: Database: True, Bars: 200,000+
```

**Status**: ⬜ Database health validated

---

## 🕘 **PRE-OPEN CHECKLIST (08:45 ET)**

### **4. Governor Pause State** ✅
```bash
# Verify governor is in pause state
redis-cli GET governor.pause
# Must show: "true"

# Check pause timestamp
redis-cli GET governor.pause_timestamp
# Should be recent (within last hour)

# Verify no active positions
redis-cli HGETALL positions
# Should be empty or all zeros
```

**Status**: ⬜ governor.pause=true at 08:45

---

### **5. Market Data Feed Validation** ✅
```bash
# Check feed latency
curl -s "http://localhost:8000/metrics" | grep feed_latency_ms
# Must show recent values < 100ms

# Verify symbols are updating
redis-cli GET market_data:NVDA:last_update
# Must be within last 30 seconds

# Check for any feed errors
curl -s "http://localhost:8000/metrics" | grep feed_errors_total
# Should be 0 or very low
```

**Status**: ⬜ Feed latency < 100ms validated

---

## 🕘 **MARKET OPEN CHECKLIST (09:25 ET)**

### **6. Final Go/No-Go Decision** ✅

#### **Critical Systems Check**
- [ ] CI job green ✅
- [ ] Redis AOF fresh ✅  
- [ ] Database healthy ✅
- [ ] Governor paused ✅
- [ ] Feed latency good ✅

#### **Risk Parameters Validation**
```bash
# Verify position limits
redis-cli HGETALL risk_limits
# Must show reasonable limits (e.g., max_position: 100 shares)

# Check drawdown limits
redis-cli GET risk_limits:max_drawdown_pct
# Must show: 0.15 (15%)

# Verify emergency stop is armed
redis-cli GET emergency_stop:armed
# Must show: "true"
```

**Status**: ⬜ Risk parameters validated

---

### **7. Governor Activation** ✅
```bash
# Final feed latency check
FEED_LATENCY=$(curl -s "http://localhost:8000/metrics" | grep -o 'feed_latency_ms [0-9.]*' | awk '{print $2}')

if (( $(echo "$FEED_LATENCY < 100" | bc -l) )); then
    echo "✅ Feed latency OK: ${FEED_LATENCY}ms"
    
    # Activate governor
    redis-cli SET governor.pause false
    redis-cli SET governor.activated_at $(date +%s)
    
    echo "🚀 GOVERNOR ACTIVATED - PAPER TRADING LIVE"
else
    echo "❌ Feed latency too high: ${FEED_LATENCY}ms"
    echo "🛑 ABORT LAUNCH - Keep governor paused"
fi
```

**Status**: ⬜ Flipped to false at 09:25 if feed latency < 100ms

---

### **8. Slack Alert Bot Validation** ✅
```bash
# Test Slack webhook
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚀 Paper trading launch test - please ignore"}' \
    $SLACK_WEBHOOK_URL

# Check bot health endpoint
curl -s "http://localhost:3001/healthz"
# Must show: {"status": "healthy", "alerts": "active"}

# Verify alert rules are loaded
curl -s "http://localhost:3001/api/rules" | jq '.rules | length'
# Must show: > 0
```

**Status**: ⬜ Slack alert bot running (/healthz ping)

---

## 🚨 **EMERGENCY PROCEDURES**

### **Immediate Stop Triggers** 🛑
```bash
# Emergency stop command
redis-cli SET governor.emergency_stop true
redis-cli SET governor.pause true
redis-cli PUBLISH alerts "EMERGENCY STOP ACTIVATED"

# Flatten all positions immediately
python -c "
from src.agents.risk_agent import RiskAgent
risk_agent = RiskAgent()
risk_agent.flatten_all_positions()
print('All positions flattened')
"
```

### **Alert Escalation** 📞
1. **Level 1**: Slack alerts to #trading-alerts
2. **Level 2**: SMS to on-call engineer (latency > 50ms for > 30s)
3. **Level 3**: Phone call to team lead (hard limit breach)
4. **Level 4**: Emergency stop (any critical system failure)

---

## 📊 **LAUNCH DAY MONITORING**

### **First Hour Metrics** (09:30-10:30 ET)
```bash
# Monitor key metrics every 5 minutes
watch -n 300 'curl -s localhost:8000/metrics | grep -E "(decision_latency|position_delta|drawdown)"'

# Check for any alerts
curl -s "http://localhost:3001/api/alerts/active"

# Monitor position changes
redis-cli MONITOR | grep positions
```

### **Success Indicators** ✅
- **Decision latency P99**: < 15ms consistently
- **Position changes**: Gradual, no sudden jumps
- **Drawdown**: < 5% during first hour
- **No alerts**: Clean alert dashboard
- **Feed stability**: < 100ms latency maintained

### **Warning Signs** ⚠️
- **Latency spikes**: > 20ms P99 for > 1 minute
- **Position drift**: Unexpected position changes
- **Feed issues**: > 200ms latency or gaps
- **Memory growth**: > 10MB/hour increase
- **Error rate**: > 0.1% decision errors

---

## ✅ **FINAL LAUNCH CHECKLIST**

### **Pre-Launch Validation** (Complete by 09:20 ET)
- [ ] **CI job green** at 06:30 ET ✅
- [ ] **Redis AOF snapshot** < 15 min old ✅
- [ ] **Database health** validated ✅
- [ ] **Governor paused** at 08:45 ET ✅
- [ ] **Feed latency** < 100ms ✅
- [ ] **Risk parameters** validated ✅
- [ ] **Slack alerts** active ✅

### **Launch Execution** (09:25 ET)
- [ ] **Final feed check** < 100ms ✅
- [ ] **Governor activation** pause=false ✅
- [ ] **Monitoring active** all dashboards ✅
- [ ] **Team on standby** Slack channel active ✅

### **Post-Launch Monitoring** (First 30 minutes)
- [ ] **Metrics stable** P99 < 15ms ✅
- [ ] **No alerts** Clean dashboard ✅
- [ ] **Positions reasonable** No unexpected changes ✅
- [ ] **Team confirmation** All systems nominal ✅

---

## 🎯 **LAUNCH SUCCESS CRITERIA**

**If ALL boxes checked** → ✅ **SAFE TO LAUNCH WITH MICRO-LOT**

**If ANY box unchecked** → 🛑 **ABORT LAUNCH - INVESTIGATE**

---

**📌 KEEP THIS CHECKLIST PINNED - CRITICAL FOR SAFE PAPER TRADING LAUNCH**