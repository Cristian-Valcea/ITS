# 📚 **OPERATOR DOCUMENTATION DIRECTORY**

**Production Risk Governor - Complete Operator Resources**

---

## 📋 **DOCUMENTATION INDEX**

### **👨‍💻 For Junior Operators (Non-Technical)**
- **[OPERATOR_MANUAL.md](./OPERATOR_MANUAL.md)** - Complete 50-page manual with step-by-step procedures
- **[OPERATOR_QUICK_REFERENCE.md](./OPERATOR_QUICK_REFERENCE.md)** - Quick reference card for daily operations
- **[userManual.md](./userManual.md)** - Original system user manual

### **🔧 For Senior Engineers (Technical)**
- **[ADVANCED_OPERATOR_MANUAL.md](./ADVANCED_OPERATOR_MANUAL.md)** - Technical manual with validation checklists and troubleshooting

### **🛠️ Operational Scripts**
- **[system_health_check.py](./system_health_check.py)** - Complete system health monitoring
- **[paper_trading_launcher.py](./paper_trading_launcher.py)** - Safe startup with pre-flight checks
- **[emergency_shutdown.sh](./emergency_shutdown.sh)** - Emergency stop with position flattening
- **[start_monitoring.py](./start_monitoring.py)** - Monitoring system startup

---

## 🚀 **QUICK START GUIDE**

### **Daily Startup (8:00 AM ET)**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
./operator_docs/system_health_check.py
./operator_docs/paper_trading_launcher.py
```

### **Health Monitoring (Every 30 minutes)**
```bash
./operator_docs/system_health_check.py
```

### **Emergency Stop**
```bash
./operator_docs/emergency_shutdown.sh
```

---

## 📞 **EMERGENCY CONTACTS**

- **System Issues**: Senior Developer
- **Hard Limit Breach**: Senior Developer + Risk Manager  
- **Daily Loss >$75**: Risk Manager + CTO

---

## 🎯 **KEY METRICS TO MONITOR**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **Latency** | <5ms | 5-10ms | >10ms |
| **Daily P&L** | -$20 to +$50 | -$20 to -$50 | <-$50 |
| **Position** | <$400 | $400-$475 | >$475 |
| **Error Rate** | <2% | 2-5% | >5% |

---

**All documentation is designed for safe, autonomous operation with minimal intervention.**

*Last Updated: August 5, 2025*