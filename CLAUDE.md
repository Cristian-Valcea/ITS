# CLAUDE.md - IntradayJules Development Guide

**Current Status**: 🚀 **PHASE 2 READY** - Team A Deliverables Complete + Production Risk Governor  
**Strategy**: Phase 2 OOS Training → Grid Search → Micro-Lot Live Trading  
**Phase**: Phase 2 Launch Ready - Monday 09:00 Europe/Bucharest

---

## 🚀 **CURRENT SESSION STATUS - AUGUST 6, 2025**

### **✅ TEAM A PHASE 2 DELIVERABLES COMPLETE**

**Mission**: 3-week sprint to live demo - Phase 2 OOS training implementation

#### **Team A Status (Quant Dev)**
- **✅ Phase 2 Scripts**: 5 core scripts created and tested (10/10 tests passing)
- **✅ Training Pipeline**: Unified entry point with Phase 2 configuration
- **✅ Evaluation System**: Automated success criteria validation (Sharpe ≥0.3, ep_rew_mean ≥0.1)
- **✅ Variance Testing**: Multi-seed robustness analysis (σ/μ < 30%)
- **✅ Paper Trading Enhancement**: Account reset integration in pre-flight checks

#### **Production Foundation (Phase 1)**
- **✅ Production Risk Governor**: Three-layer defense architecture
- **✅ IBKR Integration**: Live connection through IBKR Workstation established
- **✅ Operator Documentation**: Complete manuals + operational scripts
- **✅ Monitoring**: Prometheus metrics, Grafana dashboards, health checks

---

## 📊 **PROJECT OVERVIEW**

**IntradayJules** is a production-ready algorithmic trading system with reinforcement learning, three-layer risk management, and comprehensive operational safeguards for intraday strategies.

### **Current Sprint: 3-Week Live Demo Path**
- **Week 1 (Aug 11-15)**: Phase 2 OOS training completion & success report
- **Week 2 (Aug 18-22)**: Grid search + reward audit (Team B)
- **Week 3 (Aug 25-29)**: 7-day micro-lot paper run validation
- **Week 4 (Sep 1-5)**: Risk Committee scale-up decision

### **Production Architecture**
- **Core Model**: Stairways V3 → Phase 2 OOS enhancement
- **Risk Management**: Three-layer defense + micro-lot governors
- **Trading Environment**: 5-action dual-ticker system (NVDA/MSFT)
- **Deployment**: Paper → Micro-lot ($10) → Scale-up ($50)
- **Base Model**: `train_runs/stairways_8cycle_20250803_193928/cycle_07_hold_45%_FIXED/`

---

## 💻 **PHASE 2 OPERATIONAL COMMANDS**

### **⚠️ PYTHON ENVIRONMENT - CRITICAL**
```bash
# ✅ REQUIRED: Always activate venv first
source venv/bin/activate
cd /home/cristian/IntradayTrading/ITS
```

### **🚀 Phase 2 Launch (Monday 09:00)**
```bash
# Pre-flight checks (includes account reset)
python operator_docs/paper_trading_launcher.py --skip-checks

# Phase 2 OOS Training (3 seeds × 50K steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config config/curriculum/phase2_oos.yaml --seed 0 --steps 50000
CUDA_VISIBLE_DEVICES=1 python train.py --config config/curriculum/phase2_oos.yaml --seed 1 --steps 50000  
CUDA_VISIBLE_DEVICES=2 python train.py --config config/curriculum/phase2_oos.yaml --seed 2 --steps 50000

# Phase 2 Evaluation
python scripts/evaluate_phase2.py --run-pattern "train_runs/phase2_oos_seed*" --output phase2_results.json

# Seed Variance Testing (Week 1)
python scripts/run_seed_variance.py --config config/curriculum/phase2_oos.yaml --steps 10000 --seeds 0 1 2 3
```

---

## 📊 **PHASE 2 SUCCESS CRITERIA**

### **✅ Automated Gates (Week 1)**
- **OOS Sharpe Ratio**: ≥ 0.30 (calculated in `evaluate_phase2.py`)
- **Episode Reward Mean**: ≥ 0.10 (2024 test data)
- **Episode Length**: ≥ 80 steps (vs. current ~50)
- **Seed Variance**: σ/μ < 30% (coefficient of variation)

### **✅ Phase 2 Configuration**
- **Training Steps**: 50,000 per seed
- **Early Exit Tax**: 5.0 (Phase 2 requirement)
- **Base Model**: Stairways V3 from Phase 1
- **Environment**: Dual-ticker NVDA/MSFT, 390 episode steps

---

## 📚 **KEY FILES & TEAM A DELIVERABLES**

### **✅ Phase 2 Scripts Created (Tested & Ready)**
- **`scripts/evaluate_phase2.py`** - OOS evaluation with success criteria (306 lines)
- **`scripts/run_seed_variance.py`** - Multi-seed variance testing (394 lines)  
- **`train.py`** - Unified training entry point (268 lines)
- **`config/curriculum/phase2_oos.yaml`** - Phase 2 configuration (109 lines)
- **`test_all_deliverables.sh`** - Complete test suite (78 lines)

### **✅ Enhanced Files**
- **`operator_docs/paper_trading_launcher.py`** - Account reset integration (+29 lines)

### **✅ Production Foundation (See CLAUDE_V4.md for details)**
- **Risk Governor**: `src/risk_governor/` (three-layer defense)
- **Operator Docs**: `operator_docs/` (complete manuals)
- **Monitoring**: Prometheus + Grafana dashboards

---

## 🎯 **3-WEEK SPRINT TIMELINE**

### **Week 1 (Aug 11-15): Phase 2 OOS Training**
- **Team A (Me)**: Complete 3×50K training runs + evaluation
- **Target**: Sharpe ≥0.3, ep_rew_mean ≥0.1, signed report
- **Status**: ✅ All scripts ready, launch Monday 09:00

### **Week 2 (Aug 18-22): Grid Search + Audit**  
- **Team B**: Grid search infrastructure + reward audits
- **Target**: Best config with ep_len ≥80, ep_rew ≥0
- **Status**: Waiting for Team B action_trace.ipynb template

### **Week 3 (Aug 25-29): Micro-Lot Paper Run**
- **DevOps/SRE**: 7-day $10 notional validation
- **Target**: P&L >0, drawdown <2%, success gates pass
- **Status**: Ready after Week 1-2 completion

---

## 🚨 **CURRENT PRIORITIES**

### **📅 Tomorrow (Monday Aug 11) - Phase 2 Launch**
1. **09:00**: Pre-flight checks + paper account reset
2. **09:30**: Launch 3×50K Phase 2 OOS training runs
3. **Monitor**: GPU utilization, training progress, system health
4. **Target**: Complete all 3 runs by Wednesday

### **📊 Success Metrics (Week 1)**
| Metric | Target | Status |
|---------|--------|---------|
| **Phase 2 Sharpe** | ≥ 0.30 | Ready to measure |
| **Episode Reward** | ≥ 0.10 | Ready to measure |
| **Seed Variance** | σ/μ < 30% | Script ready |
| **Report Sign-off** | By Friday | Template ready |

### **🔗 Dependencies**
- **Team B**: action_trace.ipynb template (for Phase 2.4)
- **DevOps**: CI integration for validation
- **Data Sci**: Weekly report format

---

## 📞 **TEAM COORDINATION**

### **✅ Team A Status (Me - Quant Dev)**
- **Phase 2 Implementation**: ✅ Complete (10/10 tests passing)
- **Launch Readiness**: ✅ Ready for Monday 09:00
- **Next Milestone**: Week 1 completion by Friday Aug 15

### **⏳ Waiting On**
- **Team B**: action_trace.ipynb template for Phase 2.4
- **DevOps**: CI integration setup
- **Risk Committee**: Micro-lot approval process

### **📋 Communication Channels**
- **#stairways-ops**: Daily standups (🟢 on track | 🟠 at risk | 🔴 blocked)
- **Grafana**: Real-time system health monitoring
- **Confluence**: All reports and documentation

---

**Current Phase**: 🚀 **PHASE 2 LAUNCH READY** → OOS Training Monday  
**Team A Status**: All deliverables complete, 100% launch readiness  
**Timeline**: Week 1 OOS → Week 2 Grid Search → Week 3 Micro-Lot → Week 4 Scale Decision

**Detailed Documentation**: See `CLAUDE_V4.md` for complete system architecture and Phase 1 details