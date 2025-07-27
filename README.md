# 🚀 IntradayJules Trading System

[![Dual-Ticker CI](https://github.com/cristian/IntradayTrading/actions/workflows/dual_ticker_ci.yml/badge.svg)](https://github.com/cristian/IntradayTrading/actions/workflows/dual_ticker_ci.yml)

**Sophisticated intraday trading system with dual-ticker (NVDA+MSFT) reinforcement learning**

## 🎯 **Quick Start**

### **Day 2 Team Execution**
```bash
# 1. Validate credentials setup
python scripts/validate_credentials.py

# 2. Check system health
curl http://localhost:8000/monitoring/health

# 3. Follow execution guide
cat DAY2_TEAM_EXECUTION_GUIDE.md
```

### **Development Setup**
```bash
# Start infrastructure
docker-compose up timescaledb -d

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/dual_ticker/test_smoke.py -v
```

## 📊 **System Status**

- **✅ Data Infrastructure**: TimescaleDB with dual-ticker schema
- **✅ Trading Environment**: OpenAI Gym compatible NVDA+MSFT environment  
- **✅ Model Architecture**: 28-dimension observation space (26 features + 2 positions)
- **✅ Monitoring**: FastAPI + Prometheus metrics (`/monitoring/*`)
- **✅ Position Tracking**: OMS with portfolio status logging
- **⏳ Live Data Feeds**: Alpha Vantage + Yahoo Finance fallback (Day 2)

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Feeds    │───▶│  Quality Gates   │───▶│   TimescaleDB   │
│ Alpha Vantage   │    │ <5% missing data │    │ dual_ticker_bars│
│ Yahoo Finance   │    │ OHLC validation  │    │ current_positions│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │    │ Trading Environment│◀───│  Feature Eng   │
│ /health /metrics│    │ NVDA+MSFT Gym Env │    │ RSI, EMA, VWAP │
│ Prometheus      │    │ 28-dim obs space  │    │ Time features   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 **Day 2 Progress**

**Current Status**: 🟡 **In Progress** → Target: 🟢 **Green Finish**

| Component | Owner | Status | ETA |
|-----------|-------|--------|-----|
| Data Ingestion | DataEng | ⏳ | 11:00 |
| IB Gateway | TradingOps | ⏳ | 15:00 |
| Quality Gates | QualityEng | ⏳ | 12:30 |
| Real-Time Feeds | DataEng | ⏳ | 14:00 |
| OMS Skeleton | TradingDev | ⏳ | 16:00 |
| Monitoring | DevOps | ⏳ | 13:30 |

**Critical Gates**: Data Quality + Live Feeds + IB Auth + Monitoring → Day 3 Ready

## 🔧 **Configuration**

### **Environment-Specific Settings**
- **CI**: 5-minute bars (`config/ci.yaml`)
- **Production**: 1-minute bars (`config/prod.yaml`)
- **Quality**: Configurable thresholds (`config/data_quality.yaml`)

### **Required Secrets**
```bash
# GitHub Repository Settings → Secrets
ALPHA_VANTAGE_KEY=****
IB_USERNAME=****
IB_PASSWORD=****
```

## 📊 **Monitoring Endpoints**

- **Health Check**: `GET /monitoring/health` (JSON system status)
- **Metrics**: `GET /monitoring/metrics` (Prometheus format)
- **Quick Status**: `GET /monitoring/status` (CLI friendly)

## 🧪 **Testing**

```bash
# Smoke tests
pytest tests/dual_ticker/test_smoke.py -v

# Database pipeline
pytest tests/dual_ticker/test_smoke.py::test_fixture_to_timescale_pipeline -v

# Full test suite
pytest tests/gym_env/test_dual_ticker_env_enhanced.py -v
```

## 📚 **Documentation**

- **[Day 2 Execution Guide](DAY2_TEAM_EXECUTION_GUIDE.md)**: Complete roadmap
- **[Completion Tracker](DAY2_COMPLETION_TRACKER.md)**: Real-time progress
- **[Credentials Setup](DAY2_CREDENTIALS_SETUP.md)**: Security configuration
- **[Implementation Summary](DAY2_IMPLEMENTATION_SUMMARY.md)**: Current status

## 🚀 **Next Steps**

1. **Complete Day 2**: Data infrastructure + monitoring operational
2. **Day 3**: Claude training implementation begins
3. **Week 2**: Live trading deployment with risk management

---

**Built with**: Python 3.10+ • FastAPI • TimescaleDB • OpenAI Gym • Prometheus • Docker