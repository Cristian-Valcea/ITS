# 🚀 **DAY 2 TEAM EXECUTION GUIDE**
**Dual-Ticker Trading System - Data Infrastructure & Real-Time Systems**

---

## 📋 **EXECUTIVE SUMMARY**

**Objective**: Build production-ready data infrastructure for NVDA+MSFT dual-ticker trading system with real-time feeds, quality gates, and monitoring dashboard.

**Critical Path**: Data Quality Gate MUST be operational by end of Day 2 - this blocks Day 3 progress.

**Success Criteria**: Live NVDA+MSFT data streaming → Quality validation → TimescaleDB storage → Monitoring dashboard operational

---

## 🎯 **DAY 2 DELIVERABLES CHECKLIST**

### **🗄️ DATA INFRASTRUCTURE (Priority 1)**

#### **✅ Task 1: Data Ingestion Pipeline**
- [ ] **Primary Feed**: Alpha Vantage intraday premium API setup
  - NVDA real-time 1-minute bars
  - MSFT real-time 1-minute bars
  - API key configuration and rate limiting
- [ ] **Fallback Feed**: Yahoo Finance CSV backup implementation
  - Automatic failover when Alpha Vantage unavailable
  - Data format normalization to match primary feed
- [ ] **Data Buffering**: Configurable bar aggregation
  - 5-minute bars for CI/testing (`config/ci.yaml`)
  - 1-minute bars for production (`config/prod.yaml`)
- [ ] **Timestamp Alignment**: Ensure NVDA/MSFT data synchronization
- [ ] **✅ Unit Test Coverage**: 5-row fixture → TimescaleDB → count verification

#### **✅ Task 2: IB Credentials Setup (0.5 day buffer)**
- [ ] Interactive Brokers paper trading account creation
- [ ] TWS Gateway installation and configuration
- [ ] Authentication testing and firewall resolution
- [ ] Connection parameters documentation
- [ ] **Buffer Time**: 4 hours allocated for auth/network issues

#### **✅ Task 3: Data Quality Validation Scripts**
- [ ] **Quality Gate Implementation**: Automated data validation
  - Missing data threshold: <5% (CI), <1% (production)
  - OHLC relationship validation
  - Volume sanity checks
  - Technical indicator bounds checking
- [ ] **Configuration**: YAML-based thresholds (`config/data_quality.yaml`)
- [ ] **Blocking Behavior**: Pipeline stops on quality failure
- [ ] **Fallback Logic**: Switch to Yahoo Finance on primary feed failure

### **📊 REAL-TIME DATA SYSTEMS (Priority 2)**

#### **✅ Task 4: Real-Time Data Feeds**
- [ ] **Alpha Vantage Integration**: Live market data streaming
- [ ] **Feed Health Monitoring**: Connection status and latency tracking
- [ ] **Reliability Testing**: Market hours data continuity validation
- [ ] **Backup System**: Yahoo Finance CSV integration tested

#### **✅ Task 5: Order Management System Skeleton**
- [ ] **Position Tracking**: `current_positions` table operational
  - Schema: `id, symbol, qty, avg_price, market_value, unrealized_pnl`
  - NVDA and MSFT positions initialized
- [ ] **CLI Utility**: Portfolio status logging (`python -m src.oms.position_tracker`)
- [ ] **Order Framework**: Basic buy/sell order structure
- [ ] **IB Gateway Interface**: Connection and order placement testing

#### **✅ Task 6: Monitoring Dashboard**
- [ ] **FastAPI Endpoints**: 
  - `GET /monitoring/health` - JSON system status
  - `GET /monitoring/metrics` - Prometheus metrics
  - `GET /monitoring/status` - Quick CLI status
- [ ] **Prometheus Metrics**: Data ingestion counters, portfolio gauges
- [ ] **Real-Time Monitoring**: Data quality and feed health indicators
- [ ] **Alert System**: Critical issue notifications

---

## 🤝 **TEAM COORDINATION INTERFACE**

### **Claude → Team Handoff** ✅ **COMPLETE**
```python
# Model expects this exact format:
observation_space = {
    'nvda_features': [13 dimensions],  # OHLC, volume, RSI, EMA, VWAP, time features
    'msft_features': [13 dimensions],  # Same structure as NVDA
    'positions': [2 dimensions]        # Current NVDA/MSFT position sizes
}
# Total: 26 + 2 = 28 dimensions
```

### **Team → Claude Handoff** ⏳ **DAY 2 TARGET**
```sql
-- Required: TimescaleDB dual_ticker_bars populated
INSERT INTO dual_ticker_bars (
    timestamp, symbol, open, high, low, close, volume,
    rsi, ema_short, ema_long, vwap,
    hour_sin, hour_cos, minute_sin, minute_cos
) VALUES (...);
```

### **Shared Dependencies** ✅ **READY**
- **Database Schema**: TimescaleDB with hypertables deployed
- **Configuration System**: YAML-based settings operational
- **Test Infrastructure**: CI pipeline with database integration

---

## 🚨 **CRITICAL GATES & BLOCKERS**

### **Day 2 End Gate (MANDATORY)**
- [ ] **Data Quality Gate**: Operational and blocking bad data
- [ ] **Live Data Feeds**: NVDA + MSFT streaming successfully
- [ ] **IB Gateway**: Paper trading account authenticated
- [ ] **Monitoring**: Dashboard showing real-time system health

### **Day 3 Readiness Gate (BLOCKER)**
- [ ] **Quality Validation**: Data passes <5% missing threshold
- [ ] **Feed Reliability**: Primary + fallback feeds tested
- [ ] **Position Tracking**: OMS logging portfolio status
- [ ] **Format Contract**: Team data matches Claude environment expectations

### **Risk Mitigation**
- **IB Auth Issues**: 0.5 day buffer allocated
- **Feed Failures**: Yahoo Finance backup ready
- **Quality Problems**: Configurable thresholds prevent hard blocks
- **Integration Issues**: Early monitoring catches problems immediately

---

## 📊 **SUCCESS METRICS & VALIDATION**

### **Data Pipeline Health**
```bash
# Test data ingestion
curl http://localhost:8000/monitoring/health
# Expected: {"status": "healthy", "components": {"database": "healthy"}}

# Check data quality
curl http://localhost:8000/monitoring/metrics | grep data_quality_failures
# Expected: data_quality_failures_total{reason="missing_data"} 0
```

### **Position Tracking Validation**
```bash
# Test position logging
python -m src.oms.position_tracker
# Expected: 
# 🏦 CURRENT PORTFOLIO STATUS
# 📊 No active positions
# 💰 Total Portfolio Value: $0.00
```

### **Database Integration Test**
```bash
# Run full pipeline test
python -m pytest tests/dual_ticker/test_smoke.py::test_fixture_to_timescale_pipeline -v
# Expected: ✅ Full pipeline test: Fixture → TimescaleDB → Count verification PASSED
```

---

## ⚙️ **CONFIGURATION REFERENCE**

### **Environment-Specific Settings**
```yaml
# config/ci.yaml (Fast CI execution)
data:
  bar_size: "5min"
  max_bars_per_test: 5

# config/prod.yaml (Production precision)  
data:
  bar_size: "1min"
  max_bars_per_session: 10000
```

### **Data Quality Thresholds**
```yaml
# config/data_quality.yaml
validation:
  missing_ratio_warn: 0.02   # 2% warning
  missing_ratio_fail: 0.05   # 5% CI failure  
  missing_ratio_live: 0.01   # 1% live trading failure
```

### **Feed Configuration**
```yaml
# config/prod.yaml
feeds:
  primary: "alpha_vantage"
  fallback: "yahoo_finance"
  retry_attempts: 3
  retry_delay: 5
```

---

## 🔧 **TECHNICAL IMPLEMENTATION DETAILS**

### **Database Schema Ready**
```sql
-- Dual-ticker bars (hypertable)
CREATE TABLE dual_ticker_bars (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open, high, low, close DECIMAL(10,4),
    volume BIGINT,
    rsi, ema_short, ema_long, vwap DECIMAL(8,4),
    hour_sin, hour_cos, minute_sin, minute_cos DECIMAL(8,6)
);

-- Current positions (OMS)
CREATE TABLE current_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE,
    qty DECIMAL(12,4) DEFAULT 0,
    avg_price DECIMAL(10,4)
);
```

### **API Endpoints Available**
- `GET /monitoring/health` - System health JSON
- `GET /monitoring/metrics` - Prometheus metrics
- `GET /monitoring/status` - Quick status check
- `POST /orchestrator/*` - Existing training endpoints

### **Dependencies Installed**
```txt
# New Day 2 dependencies
prometheus-client>=0.17.0  # Monitoring metrics
psycopg2-binary>=2.9.0     # TimescaleDB connection
alpha-vantage>=2.3.1       # Primary data feed
yfinance>=0.1.70           # Fallback data feed
```

---

## 🚀 **EXECUTION WORKFLOW**

### **Morning (Hours 1-4)**
1. **Alpha Vantage Setup**: API key, rate limits, NVDA/MSFT feeds
2. **Quality Gate Implementation**: Validation scripts with YAML config
3. **IB Gateway Setup**: Paper account + TWS installation (buffer time)

### **Afternoon (Hours 5-8)**  
1. **Real-Time Integration**: Live data streaming + monitoring
2. **Position Tracking**: OMS skeleton with CLI logging
3. **End-to-End Testing**: Full pipeline validation

### **End of Day Sync**
1. **Data Format Contract**: Confirm team pipeline → Claude environment
2. **Quality Gate Verification**: All thresholds operational
3. **Day 3 Readiness**: Green light for training implementation

---

## 📞 **ESCALATION & SUPPORT**

### **Immediate Blockers**
- **IB Authentication Issues**: Use 0.5 day buffer, escalate to network team
- **Alpha Vantage API Problems**: Switch to Yahoo Finance fallback immediately
- **Database Connection Failures**: Check TimescaleDB service, restart if needed

### **Quality Gate Failures**
- **>5% Missing Data**: Investigate feed reliability, adjust thresholds if needed
- **OHLC Validation Errors**: Check data source quality, enable fallback
- **Technical Indicator Issues**: Verify calculation logic, use backup data

### **Coordination Points**
- **Claude Interface**: Data format questions → check `DAY2_REFINED_TODO.md`
- **Database Schema**: Issues → reference `sql/docker-entrypoint-initdb.d/01_schema.sql`
- **Configuration**: Settings questions → check `config/*.yaml` files

---

## ✅ **DAY 2 COMPLETION CRITERIA**

### **Green Light Indicators**
- [ ] **Data Streaming**: NVDA + MSFT live feeds operational
- [ ] **Quality Gate**: <5% missing data threshold enforced
- [ ] **Position Tracking**: CLI shows portfolio status
- [ ] **Monitoring**: All endpoints returning healthy status
- [ ] **IB Gateway**: Paper trading account connected
- [ ] **CI Pipeline**: All tests passing with database integration

### **Ready for Day 3**
- [ ] **Data Contract**: Team pipeline format matches Claude environment
- [ ] **Quality Assurance**: Validation gates operational and tested
- [ ] **Infrastructure**: All systems green and monitored
- [ ] **Documentation**: Implementation details captured for handoff

---

## 🎯 **SUCCESS STATEMENT**

**"By end of Day 2, we have live NVDA+MSFT data streaming through quality gates into TimescaleDB, with position tracking and monitoring operational, ready for Day 3 training implementation."**

---

*This document serves as the definitive Day 2 execution guide. All team members should reference this for coordination, validation, and completion criteria.*