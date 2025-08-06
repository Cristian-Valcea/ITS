# 🚀 **OPERATIONAL GUIDE**

**Complete guide for daily operations of the Stress Testing Platform**

---

## 📅 **DAILY WORKFLOW**

### **Morning Routine (06:30 - 07:00 ET)**

#### **1. Platform Startup**
```bash
cd /home/cristian/IntradayTrading/ITS
source venv/bin/activate

# Complete platform startup
./scripts/daily_startup.sh

# Expected output:
# ✅ Pre-flight checks passed
# ✅ Platform components started
# ✅ Daily certification PASSED
# 🚀 Platform ready for trading day operations!
```

#### **2. Verify Dashboard Access**
- **Grafana**: http://localhost:3000 (admin/stress_testing_2025)
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/metrics

#### **3. Check Critical Metrics**
```bash
# Quick health check
./scripts/monitor.sh health

# Monitor key metrics
./scripts/monitor.sh metrics

# Expected thresholds:
# ✅ Decision Latency P99: < 15ms
# ✅ Pipeline Latency P99: < 20ms  
# ✅ Error Rate: < 0.1%
# ✅ Position Delta: < $1000
```

---

### **Trading Hours Monitoring (09:30 - 16:00 ET)**

#### **Continuous Monitoring**
```bash
# Real-time dashboard monitoring
# Keep Grafana dashboard open: http://localhost:3000

# Command-line monitoring (optional)
./scripts/monitor.sh watch

# Real-time latency monitoring
./scripts/monitor.sh latency
```

#### **Hourly Checks**
```bash
# Every hour during trading:
./scripts/monitor.sh status
./scripts/monitor.sh alerts

# Log any anomalies in daily report
```

#### **Alert Response**
```bash
# If alerts trigger:
./scripts/monitor.sh alerts

# Check specific issues:
./scripts/monitor.sh errors

# Escalate if critical (see TROUBLESHOOTING.md)
```

---

### **End of Day Routine (16:30 - 17:00 ET)**

#### **1. Final Certification Run**
```bash
# Run final stress test certification
python stress_testing/run_full_suite.py --certification

# Verify results
cat stress_testing/results/certification_report.json | jq '.certified'
# Should show: true
```

#### **2. Generate Daily Report**
```bash
# Daily report is auto-generated during shutdown
./scripts/daily_shutdown.sh

# Review report
cat reports/daily_report_$(date +%Y%m%d).md
```

#### **3. Archive and Cleanup**
```bash
# Shutdown handles archiving automatically
# Check archived data
ls -la archives/$(date +%Y%m%d)/

# Verify shutdown completed
./scripts/monitor.sh status
# Should show services stopped
```

---

## 🔄 **WEEKLY WORKFLOWS**

### **Monday: Weekly Startup**
```bash
# Extended startup with full validation
./scripts/daily_startup.sh

# Run comprehensive health check
./scripts/monitor.sh health

# Review weekend logs
find logs/stress_testing -name "*.log" -mtime -3 -exec grep -l "ERROR\|WARN" {} \;

# Update weekly tracking
echo "Week $(date +%U) - Platform Status: Operational" >> reports/weekly_status.log
```

### **Wednesday: Mid-Week Maintenance**
```bash
# Performance review
./scripts/monitor.sh metrics > reports/midweek_performance_$(date +%Y%m%d).log

# Log rotation
find logs/stress_testing -name "*.log" -mtime +7 -delete

# Database maintenance
docker exec timescaledb_primary psql -U postgres -d trading_data -c "VACUUM ANALYZE;"
redis-cli BGSAVE
```

### **Friday: Weekly Wrap-up**
```bash
# Generate weekly summary
cat > reports/weekly_summary_$(date +%Y%m%d).md << EOF
# Weekly Summary - Week $(date +%U), $(date +%Y)

## Platform Uptime
$(grep "Platform Status" reports/weekly_status.log | tail -5)

## Key Metrics This Week
$(find reports -name "daily_report_*.md" -mtime -7 -exec grep -H "Pass Rate\|Memory Usage\|Disk Usage" {} \;)

## Issues Resolved
$(find logs/stress_testing -name "*.log" -mtime -7 -exec grep -l "ERROR" {} \; | wc -l) error incidents

## Next Week Preparation
- [ ] Review performance trends
- [ ] Update documentation
- [ ] Plan any maintenance windows
EOF

# Archive weekly data
mkdir -p archives/weekly/$(date +%Y_W%U)
cp reports/weekly_summary_$(date +%Y%m%d).md archives/weekly/$(date +%Y_W%U)/
```

---

## 🎯 **STRESS TESTING WORKFLOWS**

### **On-Demand Testing**
```bash
# Individual scenario testing
python stress_testing/run_full_suite.py --scenario flash_crash
python stress_testing/run_full_suite.py --scenario decision_flood
python stress_testing/run_full_suite.py --scenario broker_failure
python stress_testing/run_full_suite.py --scenario portfolio_integrity

# Full certification
python stress_testing/run_full_suite.py --certification

# Enhanced scenarios (when implemented)
python stress_testing/run_full_suite.py --scenario flash_crash_enhanced
python stress_testing/run_full_suite.py --scenario decision_flood_enhanced
python stress_testing/run_full_suite.py --scenario hard_kill_governance
```

### **Automated Testing Schedule**
```bash
# Add to crontab for automated execution
crontab -e

# Daily certification at 6:30 AM
30 6 * * * cd /home/cristian/IntradayTrading/ITS && ./scripts/daily_startup.sh

# Hourly health checks during trading hours
0 9-16 * * 1-5 cd /home/cristian/IntradayTrading/ITS && ./scripts/monitor.sh health >> logs/hourly_health.log

# Daily shutdown at 5:00 PM
0 17 * * * cd /home/cristian/IntradayTrading/ITS && ./scripts/daily_shutdown.sh

# Weekly maintenance on Sunday at 2:00 AM
0 2 * * 0 cd /home/cristian/IntradayTrading/ITS && ./scripts/weekly_maintenance.sh
```

---

## 📊 **MONITORING WORKFLOWS**

### **Real-Time Monitoring**
```bash
# Primary monitoring: Grafana Dashboard
# URL: http://localhost:3000
# Dashboard: "Stress Testing Platform - Real-Time Monitoring"

# Key panels to watch:
# 1. Decision Latency P99 (must be < 15ms)
# 2. Pipeline Latency P99 (must be < 20ms)  
# 3. Hard Limit Breaches (must be 0)
# 4. Position Delta (must be < $1000)
# 5. Error Rate (must be < 0.1%)
```

### **Alert Management**
```bash
# Check active alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state == "firing")'

# Alert escalation levels:
# INFO: Log only
# WARNING: Slack notification
# CRITICAL: Immediate action required

# Silence alerts (if needed)
curl -X POST http://localhost:9090/api/v1/alerts \
    -H "Content-Type: application/json" \
    -d '{"matchers":[{"name":"alertname","value":"DecisionLatencyP99High"}],"startsAt":"2025-01-01T00:00:00Z","endsAt":"2025-01-01T01:00:00Z","comment":"Planned maintenance"}'
```

### **Performance Trending**
```bash
# Weekly performance analysis
curl -s "http://localhost:9090/api/v1/query_range?query=histogram_quantile(0.99, rate(risk_governor_decision_latency_seconds_bucket[1h]))&start=$(date -d '7 days ago' +%s)&end=$(date +%s)&step=3600" | jq '.data.result[0].values' > reports/weekly_latency_trend.json

# Monthly capacity planning
curl -s "http://localhost:9090/api/v1/query_range?query=rate(risk_governor_decisions_total[1h])&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=86400" | jq '.data.result[0].values' > reports/monthly_throughput_trend.json
```

---

## 🚨 **INCIDENT RESPONSE WORKFLOWS**

### **Level 1: Performance Degradation**
```bash
# Symptoms: Latency > thresholds, error rate elevated
# Response time: 5 minutes

# 1. Immediate assessment
./scripts/monitor.sh health
./scripts/monitor.sh metrics
./scripts/monitor.sh errors

# 2. Quick fixes
# Restart services if needed
pkill -f "python.*stress.*metrics"
./scripts/start_platform.sh

# 3. Monitor for improvement
./scripts/monitor.sh latency  # Watch for 5 minutes

# 4. Document incident
echo "$(date): Level 1 incident - Performance degradation - Actions taken: Service restart" >> logs/incidents.log
```

### **Level 2: Service Outage**
```bash
# Symptoms: Services not responding, dashboard down
# Response time: 15 minutes

# 1. Full platform restart
./scripts/daily_shutdown.sh
./scripts/daily_startup.sh

# 2. Verify recovery
./scripts/monitor.sh health

# 3. Check data integrity
python stress_testing/run_full_suite.py --scenario portfolio_integrity

# 4. Notify team
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🚨 Level 2 incident resolved - Platform restarted and operational"}' \
    $SLACK_WEBHOOK_URL
```

### **Level 3: Critical System Failure**
```bash
# Symptoms: Hard limit breaches, data corruption, security issues
# Response time: Immediate

# 1. Emergency stop
redis-cli SET governor.emergency_stop true
redis-cli SET governor.pause true

# 2. Isolate system
./scripts/daily_shutdown.sh

# 3. Assess damage
# Check data integrity
# Review security logs
# Document all findings

# 4. Escalate immediately
# Call team lead
# Initiate incident response procedure
# Consider external notification if needed
```

---

## 📋 **MAINTENANCE WORKFLOWS**

### **Daily Maintenance**
```bash
# Automated via daily_startup.sh and daily_shutdown.sh
# Manual checks:

# 1. Log review
tail -100 logs/stress_testing/*.log | grep -i error

# 2. Disk space check
df -h | grep -E "(80%|90%|95%)"

# 3. Memory usage check
free -h
ps aux --sort=-%mem | head -5

# 4. Database health
docker exec timescaledb_primary pg_isready -U postgres -d trading_data
redis-cli ping
```

### **Weekly Maintenance**
```bash
# Create weekly maintenance script
cat > scripts/weekly_maintenance.sh << 'EOF'
#!/bin/bash
echo "🔧 Weekly Maintenance - $(date)"

# 1. System updates (if approved)
# sudo apt update && sudo apt upgrade -y

# 2. Log rotation and cleanup
find logs/stress_testing -name "*.log" -mtime +14 -delete
find archives -type d -mtime +60 -exec rm -rf {} + 2>/dev/null || true

# 3. Database maintenance
docker exec timescaledb_primary psql -U postgres -d trading_data -c "VACUUM ANALYZE;"
redis-cli BGSAVE

# 4. Performance baseline update
./scripts/monitor.sh metrics > reports/weekly_baseline_$(date +%Y%m%d).log

# 5. Security scan (if tools available)
# nmap localhost
# chkrootkit

echo "✅ Weekly maintenance complete"
EOF

chmod +x scripts/weekly_maintenance.sh
```

### **Monthly Maintenance**
```bash
# 1. Full platform rebuild test
# Test complete shutdown and startup
# Verify all data recovery procedures

# 2. Capacity planning review
# Analyze monthly trends
# Plan resource scaling if needed

# 3. Documentation updates
# Update this guide with any changes
# Review and update troubleshooting procedures

# 4. Disaster recovery test
# Test backup/restore procedures
# Verify off-site backup integrity

# 5. Security review
# Review access logs
# Update passwords if needed
# Check for security updates
```

---

## 🎯 **SUCCESS METRICS**

### **Daily Success Criteria**
- [ ] Platform startup completed without errors
- [ ] All services responding on expected ports
- [ ] Daily certification passes (100% pass rate)
- [ ] No critical alerts during trading hours
- [ ] Clean shutdown with data archived

### **Weekly Success Criteria**
- [ ] 5/5 successful daily operations
- [ ] Average P99 latency < 15ms
- [ ] Zero hard limit breaches
- [ ] All incidents resolved within SLA
- [ ] Weekly maintenance completed

### **Monthly Success Criteria**
- [ ] 20+ successful trading days
- [ ] Platform uptime > 99.5%
- [ ] Performance within baseline thresholds
- [ ] All maintenance windows completed
- [ ] Documentation kept current

---

## 📞 **CONTACT INFORMATION**

### **Escalation Chain**
1. **Self-Service**: Use this guide and troubleshooting procedures
2. **Team Chat**: #trading-alerts Slack channel
3. **On-Call Engineer**: [Phone number]
4. **Team Lead**: [Phone number]
5. **Emergency**: [Emergency contact]

### **Key Resources**
- **Platform Documentation**: `/home/cristian/IntradayTrading/ITS/stress_testing/`
- **Troubleshooting Guide**: `TROUBLESHOOTING.md`
- **Success Gates**: `SUCCESS_GATES.md`
- **Paper Trading Checklist**: `PAPER_TRADING_CHECKLIST.md`

---

**🚀 This operational guide ensures consistent, reliable operation of the Stress Testing Platform for production trading environments.**