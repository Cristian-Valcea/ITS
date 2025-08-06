# CLAUDE V4.md - Detailed System Documentation

**Archive Date**: August 6, 2025  
**Status**: Complete historical documentation from Phase 1 + Team A deliverables

---

## 🏗️ **PRODUCTION ARCHITECTURE (PHASE 1)**

### **✅ Risk Management System**
- **Production Risk Governor**: `src/risk_governor/core_governor.py` (three-layer defense)
- **Broker Integration**: `src/risk_governor/broker_adapter.py` (IBKR paper trading + chaos testing)
- **EOD Management**: `src/risk_governor/eod_manager.py` (auto-flattening at 15:55 ET)
- **Monitoring**: `src/risk_governor/prometheus_monitoring.py` (metrics + alerting)

### **✅ Operational Infrastructure**
- **Operator Documentation**: `operator_docs/` (complete manuals + scripts)
- **Security**: `~/.trading_secrets.json` (Argon2id + AES-256-GCM encryption)
- **Database**: TimescaleDB with real market data (2022-2024)
- **Backup System**: Redis AOF + PostgreSQL + S3 with hourly snapshots

---

## 💻 **OPERATIONAL COMMANDS**

### **⚠️ PYTHON ENVIRONMENT - CRITICAL**
**ALWAYS USE VENV**: The project uses an encapsulated Python environment.

```bash
# ✅ REQUIRED: Always activate venv first
source venv/bin/activate
```

### **🚀 Production Operations**
```bash
# Daily system startup
./operator_docs/system_health_check.py      # Check all systems
./operator_docs/paper_trading_launcher.py   # Start paper trading

# Monitoring (every 30 minutes)
./operator_docs/system_health_check.py      # Health status
curl -s localhost:8000/metrics | grep decision_latency | tail -1

# Emergency shutdown
./operator_docs/emergency_shutdown.sh       # Immediate stop + flatten positions
```

### **📊 Key Metrics Dashboard**
- **Prometheus**: http://localhost:8000/metrics
- **Grafana**: http://localhost:3000/d/risk-governor/production-trading
- **Health Check**: `./operator_docs/system_health_check.py`

---

## 🔐 **SECURITY SYSTEM**

### **✅ Vault Management (Operational)**
```python
# Database access (secure)
from secrets_helper import SecretsHelper
db_url = SecretsHelper.get_database_url()
password = SecretsHelper.get_timescaledb_password()
```

### **✅ Current Secrets in Vault**
- `POLYGON_API_KEY`: Market data API access
- `TIMESCALEDB_PASSWORD`: Database authentication  
- `POSTGRES_PASSWORD`: PostgreSQL access

---

## 📊 **CURRENT CONFIGURATION**

### **✅ Dual-Ticker Setup (Production Ready)**
```yaml
# Environment Configuration
assets: ["NVDA", "MSFT"]                # Dual-ticker portfolio
observation_space: Box(26,)             # 12 features × 2 assets + 2 positions
action_space: Discrete(5)               # Fixed 5-action system
lookback_window: 50                     # Feature history
episode_length: 1000                    # Steps per episode

# Risk Limits (Conservative for Demo)
max_position_size: 1000                 # $1000 max per asset
daily_loss_limit: 50                    # $50 daily limit
total_drawdown_limit: 100               # $100 total limit
```

### **✅ Database Config**
```yaml
# Database (Operational)
timescaledb:
  host: localhost
  port: 5432
  database: trading_data
  table: minute_bars
  date_range: 2022-01-03 to 2024-12-31
```

---

## 📚 **KEY FILES & LOCATIONS**

### **📋 Operator Documentation (AUGUST 5, 2025)**
- `operator_docs/README.md` - **Complete documentation index**
- `operator_docs/OPERATOR_MANUAL.md` - **50-page manual for junior operators**
- `operator_docs/ADVANCED_OPERATOR_MANUAL.md` - **Technical manual for senior engineers**
- `operator_docs/OPERATOR_QUICK_REFERENCE.md` - **Daily operations quick reference**

### **🛠️ Operational Scripts**
- `operator_docs/system_health_check.py` - **Complete system health monitoring**
- `operator_docs/paper_trading_launcher.py` - **Safe startup with pre-flight checks**
- `operator_docs/emergency_shutdown.sh` - **Emergency stop with position flattening**
- `operator_docs/start_monitoring.py` - **Monitoring system startup**

### **🏗️ Production Risk Governor**
- `src/risk_governor/core_governor.py` - **Three-layer defense architecture**
- `src/risk_governor/broker_adapter.py` - **IBKR integration + chaos testing**
- `src/risk_governor/stairways_integration.py` - **Safe model deployment wrapper**
- `train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/` - **Production model**

---

## 🎯 **HISTORICAL DEPLOYMENT ROADMAP (PHASE 1)**

### **📅 5-Day Paper Trading Launch**
1. **Monday**: MSFT $10 notional launch with 1% chaos testing
2. **Tuesday**: NVDA earnings gap replay simulation (off-hours)
3. **Wednesday**: Add AAPL $10 notional, dual-symbol monitoring
4. **Thursday**: Reduce chaos rate to 0.1%, performance validation
5. **Friday**: Risk Committee KPI package with 5-day performance analysis

### **🎯 Go/No-Go Gates (Real-Time)**
| Gate | Threshold | Action if Breached |
|------|-----------|-------------------|
| **Latency** | <10ms 1-min MA | Auto-pause if >15ms sustained |
| **Error Rate** | <2% | Investigate immediately |
| **Daily DD** | <$20 and <3% | Reduce position size 50% |
| **Fee P&L** | ≥$0 by close | Review cost efficiency |

### **📋 Success Criteria**
- **Sharpe ratio >0** across 5-day period
- **Max drawdown <3%** during any single day
- **Fee ratio <30%** of gross P&L
- **Zero hard limit breaches** throughout testing

---

## 🚨 **OPERATIONAL SAFEGUARDS**

### **✅ Risk Management Features**
- **Three-layer defense**: Position → Drawdown → Production governors
- **Real-time monitoring**: Decision latency, error rates, P&L tracking
- **Automated safeguards**: EOD auto-flattening, hard limit circuit breakers
- **Chaos testing**: 1% order rejection for production resilience validation

### **📞 Emergency Contacts**
- **System Issues**: Senior Developer
- **Hard Limit Breach**: Senior Developer + Risk Manager  
- **Daily Loss >$75**: Risk Manager + CTO
- **Emergency**: `./operator_docs/emergency_shutdown.sh`

---

## 📊 **TEAM A DELIVERABLES (AUGUST 6, 2025)**

### **✅ Files Created & Tested**
- **scripts/evaluate_phase2.py** - Phase 2 OOS evaluation (306 lines)
- **scripts/run_seed_variance.py** - Multi-seed variance testing (394 lines)
- **train.py** - Unified training entry point (268 lines)
- **config/curriculum/phase2_oos.yaml** - Phase 2 configuration (109 lines)
- **test_all_deliverables.sh** - Test suite (78 lines)

### **✅ Files Modified**
- **operator_docs/paper_trading_launcher.py** - Added paper account reset (+29 lines)

### **✅ Test Results**
- **10/10 Tests Passing** - All scripts functional and ready
- **Mock Training Data Created** - 120 episodes, Sharpe 3.475, Reward 0.242
- **Integration Verified** - IBKR imports, training environment, risk governor

### **✅ Production Commands Ready**
```bash
# Phase 2 OOS Training (3 seeds × 50K steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config config/curriculum/phase2_oos.yaml --seed 0 --steps 50000
CUDA_VISIBLE_DEVICES=1 python train.py --config config/curriculum/phase2_oos.yaml --seed 1 --steps 50000
CUDA_VISIBLE_DEVICES=2 python train.py --config config/curriculum/phase2_oos.yaml --seed 2 --steps 50000

# Phase 2 Evaluation
python scripts/evaluate_phase2.py --run-pattern "train_runs/phase2_oos_seed*" --output phase2_results.json

# Seed Variance Testing
python scripts/run_seed_variance.py --config config/curriculum/phase2_oos.yaml --steps 10000 --seeds 0 1 2 3
```

### **✅ Success Criteria Implemented**
- **Sharpe ≥ 0.3**: Automated calculation in evaluation script
- **ep_rew_mean ≥ 0.1**: 2024 OOS data filtering
- **Seed Variance σ/μ < 30%**: Coefficient of variation analysis
- **Early Exit Tax**: 5.0 configured in phase2_oos.yaml

---

**Status**: Complete historical documentation + Team A Phase 2 implementation  
**Next**: Phase 2 OOS training launch Monday 09:00