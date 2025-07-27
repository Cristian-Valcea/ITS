# Flash-Crash Lite Stress Runner - Operations Runbook

## Overview

The Flash-Crash Lite stress testing system runs automated hourly stress tests during market hours to validate risk management systems. It simulates 60-second liquidity shock scenarios and automatically alerts via PagerDuty and triggers KILL_SWITCH on risk limit breaches.

## System Architecture

### Components
- **StressRunner**: Core stress testing engine
- **HourlyStressScheduler**: Automated hourly execution
- **PagerDutyAlerter**: Critical alert notifications
- **StressScenario**: YAML-based scenario configuration

### Integration Points
- **RiskEventBus**: Event publishing for KILL_SWITCH
- **RiskAgentV2**: Risk limit validation
- **Prometheus**: Metrics and monitoring
- **PagerDuty**: Critical alerting

## Operational Schedule

### Timing
- **Frequency**: Every hour during market hours
- **Execution Time**: :00:05 UTC (5 seconds past the hour)
- **Market Hours**: 14:30 - 21:00 UTC (9:30 AM - 4:00 PM ET)
- **Weekends**: Disabled

### Performance Requirements
- **Max Runtime**: 50ms for up to 100 symbols
- **Max Symbols**: 100 symbols per test
- **Zero Latency Impact**: No impact on live trading path

## Stress Scenario Details

### Default Scenario (flash_crash.yaml)
```yaml
scenario_name: "flash_crash_lite"
price_shock_pct: -0.03              # 3% down-spike
spread_mult: 3.0                    # 3x bid-ask spread
duration_sec: 60                    # 60-second scenario
recovery_type: "linear"             # Linear price recovery
```

### Risk Thresholds
- **Max Drawdown**: 5% portfolio drawdown
- **VaR Multiplier**: 2x normal VaR threshold
- **Position Delta**: 10% position change

## Alert Handling

### PagerDuty Integration
- **Service**: Live-Risk team
- **Severity**: Critical
- **Auto-Escalation**: 5 minutes if not acknowledged
- **Routing Key**: Stored in `PD_ROUTING_KEY` environment variable

### Alert Payload
```json
{
  "summary": "Stress test breach: flash_crash_lite on MES, MNQ",
  "severity": "critical",
  "source": "stress_runner",
  "custom_details": {
    "scenario": "flash_crash_lite",
    "symbols": ["MES", "MNQ"],
    "breach_count": 2,
    "timestamp": "2024-01-15T15:00:05Z"
  }
}
```

## Incident Response

### Immediate Actions (0-5 minutes)
1. **Acknowledge PagerDuty alert**
2. **Verify trading halt status** in Grafana dashboard
3. **Check system health** via monitoring dashboards
4. **Review stress test logs** for breach details

### Investigation (5-15 minutes)
1. **Analyze breach symbols** and risk metrics
2. **Check market conditions** for unusual volatility
3. **Review position sizes** and exposure levels
4. **Validate risk limits** are appropriate

### Resolution Steps

#### If Breach is Valid (Real Risk Issue)
1. **Keep trading halted** until risk is mitigated
2. **Adjust position sizes** or close risky positions
3. **Update risk limits** if necessary
4. **Test with manual stress run** before resuming

#### If Breach is False Positive
1. **Review scenario parameters** in `flash_crash.yaml`
2. **Adjust risk thresholds** if too conservative
3. **Update stress scenario** if market conditions changed
4. **Resume trading** after validation

### Resuming Trading
1. **Fix underlying risk issues**
2. **Call REST endpoint**: `POST /api/v1/risk/reset_circuit`
3. **Enable trading toggle** in Grafana dashboard
4. **Monitor closely** for next 30 minutes

## Monitoring and Dashboards

### Prometheus Metrics
```
# Stress test execution
stress_runs_total                    # Total stress test runs
stress_breaches_total               # Total breaches detected
stress_runtime_seconds              # Runtime distribution
stress_symbols_tested               # Symbols tested per run

# System health
stress_runner_up                    # Stress runner health status
stress_scheduler_next_run           # Next scheduled run time
```

### Grafana Panels
- **Hourly Stress Breach Count** (Green=0, Red>0)
- **Stress Test Runtime** (Target: <50ms)
- **Symbols Tested per Run** (Typical: 10-50)
- **Time Since Last Stress Test** (Alert if >2 hours)

### Log Monitoring
```bash
# View stress test logs
kubectl logs -f deployment/trading-system | grep "stress_runner"

# Check for breaches
kubectl logs deployment/trading-system | grep "KILL_SWITCH triggered"

# Monitor performance
kubectl logs deployment/trading-system | grep "exceeded max runtime"
```

## Configuration Management

### Scenario Configuration
```bash
# Edit stress scenario
vim risk/stress_packs/flash_crash.yaml

# Validate configuration
python -c "from src.risk.stress_runner import StressScenario; StressScenario('risk/stress_packs/flash_crash.yaml')"

# Deploy changes
kubectl apply -f k8s/trading-system.yaml
```

### Environment Variables
```bash
# Required environment variables
PD_ROUTING_KEY=<pagerduty-routing-key>
STRESS_ENABLED=true
MARKET_HOURS_ONLY=true

# Optional tuning
STRESS_MAX_SYMBOLS=100
STRESS_MAX_RUNTIME_MS=50
```

## Troubleshooting

### Common Issues

#### Stress Tests Not Running
```bash
# Check scheduler status
curl http://localhost:8000/api/v1/stress/status

# Check logs for errors
kubectl logs deployment/trading-system | grep "stress.*error"

# Verify environment variables
kubectl exec deployment/trading-system -- env | grep STRESS
```

#### Performance Issues (Runtime > 50ms)
```bash
# Check symbol count
kubectl logs deployment/trading-system | grep "symbols_tested"

# Monitor resource usage
kubectl top pods | grep trading-system

# Reduce max symbols if needed
kubectl set env deployment/trading-system STRESS_MAX_SYMBOLS=50
```

#### PagerDuty Alerts Not Sending
```bash
# Verify routing key
kubectl get secret pagerduty-config -o yaml

# Test PagerDuty integration
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{"routing_key":"'$PD_ROUTING_KEY'","event_action":"trigger","payload":{"summary":"Test alert"}}'

# Check network connectivity
kubectl exec deployment/trading-system -- curl -I https://events.pagerduty.com
```

#### False Positive Breaches
```bash
# Review recent market volatility
curl "http://localhost:8000/api/v1/market/volatility?hours=24"

# Check risk limit history
curl "http://localhost:8000/api/v1/risk/limits/history"

# Adjust scenario if needed
vim risk/stress_packs/flash_crash.yaml
# Increase price_shock_pct or adjust thresholds
```

### Emergency Procedures

#### Disable Stress Testing
```bash
# Temporary disable
kubectl set env deployment/trading-system STRESS_ENABLED=false

# Or scale down
kubectl scale deployment/trading-system --replicas=0
```

#### Manual Stress Test
```bash
# Run single stress test
curl -X POST http://localhost:8000/api/v1/stress/run

# Run with custom scenario
curl -X POST http://localhost:8000/api/v1/stress/run \
  -H "Content-Type: application/json" \
  -d '{"scenario":"custom_scenario","symbols":["MES","MNQ"]}'
```

## Maintenance

### Weekly Tasks
- [ ] Review stress test breach history
- [ ] Validate PagerDuty alert routing
- [ ] Check performance metrics trends
- [ ] Update scenario parameters if needed

### Monthly Tasks
- [ ] Review and update risk thresholds
- [ ] Analyze false positive rate
- [ ] Update documentation
- [ ] Test disaster recovery procedures

### Quarterly Tasks
- [ ] Full system stress test
- [ ] Review and update runbook
- [ ] Validate monitoring dashboards
- [ ] Update team training materials

## API Endpoints

### Stress Testing Control
```bash
# Get status
GET /api/v1/stress/status

# Run manual stress test
POST /api/v1/stress/run

# Get recent results
GET /api/v1/stress/results?hours=24

# Update configuration
PUT /api/v1/stress/config
```

### Risk Management
```bash
# Reset circuit breaker
POST /api/v1/risk/reset_circuit

# Get current risk status
GET /api/v1/risk/status

# Update risk limits
PUT /api/v1/risk/limits
```

## Contact Information

### Primary Contacts
- **Live-Risk Team**: live-risk@company.com
- **Trading Operations**: trading-ops@company.com
- **Platform Engineering**: platform@company.com

### Escalation
- **Level 1**: Trading desk (immediate)
- **Level 2**: Risk management team (5 minutes)
- **Level 3**: CTO/Head of Trading (15 minutes)

### External Contacts
- **PagerDuty Support**: support@pagerduty.com
- **CME Technical Support**: 1-800-CME-HELP

---

**Document Version**: 1.0  
**Last Updated**: January 2024  
**Next Review**: April 2024  
**Owner**: Live-Risk Team