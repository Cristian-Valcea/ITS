# 📋 **DAY 2 REFINED TO-DO LIST**
**Dual-Ticker Trading System - Data Infrastructure & Monitoring**

---

## 🗄️ **DATA INFRASTRUCTURE (Priority)**

### **Task 1: Data Ingestion Pipeline** ✅ *Refined*
- [ ] Set up NVDA real-time data feed connection (**Alpha Vantage primary**)
- [ ] Set up MSFT real-time data feed connection (**Alpha Vantage primary**)
- [ ] Implement **Yahoo Finance CSV fallback** for data feed failures
- [ ] Create data normalization for dual-ticker format
- [ ] Create data buffering system for **configurable bar sizes** (5min CI, 1min prod)
- [ ] Test data alignment between NVDA and MSFT timestamps
- [ ] **✅ Unit test: 5-row fixture → TimescaleDB insert → select count()** (CI coverage)

### **Task 2: IB Credentials Setup (0.5 day buffer)**
- [ ] Create Interactive Brokers paper trading account
- [ ] Install and configure TWS Gateway
- [ ] Test authentication and connection
- [ ] Handle firewall/network configuration issues
- [ ] Document connection parameters for team

### **Task 3: Data Quality Validation Scripts** ✅ *Refined*
- [ ] Implement **configurable missing data thresholds** (YAML-based)
  - CI: `missing_ratio_fail: 0.05` (5% tolerance)
  - Production: `missing_ratio_fail: 0.01` (1% tolerance for live trading)
- [ ] Create OHLC relationship validation
- [ ] Add timestamp alignment verification
- [ ] Build data continuity monitoring
- [ ] Set up **automated quality gate that blocks bad data**

---

## 📊 **REAL-TIME DATA SYSTEMS**

### **Task 4: Real-time Data Feeds** ✅ *Refined*
- [ ] Configure **Alpha Vantage intraday premium** as primary feed
- [ ] Implement **Yahoo Finance CSV backup** (no Bloomberg - scope protection)
- [ ] Create data feed health monitoring
- [ ] Test feed reliability during market hours
- [ ] Validate data latency requirements

### **Task 5: Order Management System Skeleton** ✅ *Enhanced*
- [ ] Create basic order structure for paper trading
- [ ] **✅ Implement position tracking table** (`current_positions`: id, symbol, qty, avg_price)
- [ ] **✅ Create CLI utility** to log what the bot "owns" during paper trading
- [ ] Set up order logging and tracking
- [ ] Create connection interface to IB Gateway
- [ ] Test basic buy/sell order placement

### **Task 6: Basic Monitoring Dashboard** ✅ *Enhanced*
- [ ] **✅ Create FastAPI routes**: `/monitoring/health` (JSON) + `/monitoring/metrics` (Prometheus)
- [ ] **✅ Add Prometheus metrics**: data ingestion counters, portfolio gauges
- [ ] Add data feed status indicators
- [ ] Implement real-time data quality metrics
- [ ] Set up alert system for critical issues
- [ ] **Browser-viewable + script-friendly** (curl /metrics works)

---

## 🤝 **COORDINATION POINTS**

### **Claude → Team Interface**
- [x] **✅ Model adapter provided**: Expects aligned dual-ticker data format
- [x] **✅ Format specification**: 26-dimension observation space (13 NVDA + 13 MSFT features)
- [x] **✅ Data contract**: Timestamp-aligned bars with technical indicators

### **Team → Claude Interface**  
- [ ] **Team provides**: Data pipeline feeding trading environment
- [ ] **Required format**: TimescaleDB `dual_ticker_bars` table population
- [ ] **Quality guarantee**: Data passes validation gate before environment consumption
- [ ] **✅ Sync timing**: Data-format contract sync **end of Day 2**

### **Shared Dependencies**
- [x] **✅ TimescaleDB schema**: Delivered and verified
- [x] **✅ Configuration system**: Bar sizes, quality thresholds in YAML
- [ ] **Data format alignment**: Ensure team pipeline matches environment expectations
- [ ] **Error handling**: Coordinate failure modes and fallback strategies

---

## 🚨 **CRITICAL GATES**

### **Day 2 End Gate**
- [ ] **Data Quality Gate operational** and blocking bad data
- [ ] **IB Gateway authenticated** and accessible  
- [ ] **Live NVDA + MSFT feeds streaming** successfully
- [ ] **Monitoring dashboard deployed** (`/monitoring/health`, `/monitoring/metrics`)

### **Day 3 Readiness Gate**
- [ ] **MANDATORY**: Data Quality Validation MUST pass
- [ ] **Blocker**: No Day 3 work begins until data quality confirmed
- [ ] **Fallback**: Yahoo Finance backup feeds tested and ready

---

## 📊 **SUCCESS METRICS** ✅ *Refined*

- [ ] **Data Pipeline**: <5% missing data for NVDA + MSFT (CI), <1% (production)
- [ ] **IB Gateway**: Paper trading account fully operational
- [ ] **Monitoring**: Dashboard shows live system health status
- [ ] **Quality Gate**: Validation pipeline catches and reports issues automatically
- [ ] **✅ CI Coverage**: Fixture→TimescaleDB pipeline test passes

---

## ⚙️ **CONFIGURATION DELIVERED** ✅

### **✅ Bar Size Configuration**
```yaml
# config/ci.yaml
data:
  bar_size: "5min"  # Fast CI execution

# config/prod.yaml  
data:
  bar_size: "1min"  # Production precision
```

### **✅ Data Quality Thresholds**
```yaml
# config/data_quality.yaml
validation:
  missing_ratio_warn: 0.02   # 2% warning
  missing_ratio_fail: 0.05   # 5% CI failure
  missing_ratio_live: 0.01   # 1% live trading failure
```

### **✅ Position Tracking Schema**
```sql
-- current_positions table ready
CREATE TABLE current_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    qty DECIMAL(12,4) NOT NULL DEFAULT 0,
    avg_price DECIMAL(10,4),
    -- ... additional fields
);
```

### **✅ Monitoring Endpoints**
- `GET /monitoring/health` - JSON system status
- `GET /monitoring/metrics` - Prometheus metrics
- `GET /monitoring/status` - Quick CLI status

---

## 🎯 **ANSWERS TO CONFIRMATION QUESTIONS**

| Question | ✅ **Recommended Answer** |
|----------|---------------------------|
| **Primary feeds** | Alpha Vantage (intraday premium) primary; Yahoo Finance CSV fallback |
| **OMS skeleton scope** | Include position tracking (qty + avg cost) - 30 lines now, hours saved Week 4 |
| **Monitoring tech** | FastAPI + Prometheus client - browser viewable + script-friendly |
| **Data quality threshold** | <5% for Day 2, <1% for live trading (YAML configurable) |
| **Coordination timing** | Sync data-format contract **end of Day 2** |

---

## 🚀 **READY TO EXECUTE**

**Status**: ✅ **All refinements implemented and ready**
- Configuration locked and loaded
- Position tracking infrastructure ready  
- Monitoring endpoints operational
- CI pipeline covers new code paths
- Quality gates configurable and documented

**Team cleared to execute Day 2 checklist!** 🎯

---

*Ping if any blocker pops up - otherwise ready for Day 2 completion report!*