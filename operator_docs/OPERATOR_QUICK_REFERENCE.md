# 🎯 **OPERATOR QUICK REFERENCE CARD**

*Keep this handy during trading hours*

---

## 🚀 **DAILY STARTUP** (8:00 AM ET)
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate
./system_health_check.py    # Check everything is OK
./paper_trading_launcher.py  # Start trading system
```

---

## 🔍 **HEALTH CHECKS** (Every 30 minutes)
```bash
./system_health_check.py  # Quick system status
curl -s http://localhost:8000/metrics | grep decision_latency | tail -1
tail -5 logs/risk_governor.log  # Recent log entries
```

---

## 🚨 **EMERGENCY SHUTDOWN**
```bash
./emergency_shutdown.sh
# Then call: [SENIOR DEVELOPER PHONE]
```

---

## 📊 **KEY METRICS TO WATCH**

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **Latency** | <5ms | 5-10ms | >10ms |
| **Daily P&L** | -$20 to +$50 | -$20 to -$50 | <-$50 |
| **Position** | <$400 | $400-$475 | >$475 |
| **Error Rate** | <2% | 2-5% | >5% |

---

## 📞 **WHO TO CALL**
- **Warnings**: Senior Developer  
- **Hard Limit Breach**: Senior Developer + Risk Manager  
- **System Down**: Senior Developer IMMEDIATELY  
- **Daily Loss >$75**: Risk Manager + CTO  

---

## 🛑 **NEVER DO THIS**
- ❌ Override hard limits  
- ❌ Restart without saving logs  
- ❌ Change position sizes  
- ❌ Ignore hard limit breach alerts  

---

## ✅ **WHAT GOOD LOOKS LIKE**
- 🟢 Health check shows all green  
- 🟢 Latency <5ms consistently  
- 🟢 P&L between -$20 and +$50  
- 🟢 No errors in last hour  
- 🟢 Positions flatten at 15:55 ET  

---

**Emergency Phone**: [INSERT NUMBER]  
**System Logs**: `logs/risk_governor.log`  
**Health Status**: `./system_health_check.py`