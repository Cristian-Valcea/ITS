#!/bin/bash

# Daily Startup Routine for Stress Testing Platform
set -e

PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK") echo -e "${GREEN}✅ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "FAIL") echo -e "${RED}❌ $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
    esac
}

echo "🌅 Daily Startup Routine"
echo "========================"
echo "Date: $(date)"
echo "Time: $(date +%H:%M:%S)"
echo ""

# 1. Pre-flight checks
print_status "INFO" "Running pre-flight checks..."
if ./stress_testing/ci/guards.sh > /dev/null 2>&1; then
    print_status "OK" "Pre-flight checks passed"
else
    print_status "FAIL" "Pre-flight checks failed - check logs"
    exit 1
fi

# 2. Start platform
print_status "INFO" "Starting platform components..."
./scripts/start_platform.sh

# 3. Wait for stabilization
print_status "INFO" "Waiting for platform stabilization..."
sleep 30

# 4. Run daily certification
print_status "INFO" "Running daily certification..."
source venv/bin/activate

if python stress_testing/run_full_suite.py --certification > logs/stress_testing/daily_cert_$(date +%Y%m%d).log 2>&1; then
    print_status "OK" "Daily certification PASSED"
    
    # Extract key metrics
    CERT_RESULT=$(cat stress_testing/results/certification_report.json | jq -r '.certified')
    PASS_RATE=$(cat stress_testing/results/certification_report.json | jq -r '.summary.pass_rate_pct')
    
    print_status "OK" "Certification Status: $CERT_RESULT"
    print_status "OK" "Pass Rate: $PASS_RATE%"
    
else
    print_status "FAIL" "Daily certification FAILED"
    print_status "WARN" "Check logs: logs/stress_testing/daily_cert_$(date +%Y%m%d).log"
    
    # Send alert (if configured)
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 Daily stress test certification FAILED on $(date)\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
fi

# 5. System health check
print_status "INFO" "Performing system health check..."

# Check memory usage
MEMORY_USAGE=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    print_status "WARN" "High memory usage: $MEMORY_USAGE%"
else
    print_status "OK" "Memory usage: $MEMORY_USAGE%"
fi

# Check disk space
DISK_USAGE=$(df "$PROJECT_ROOT" | awk 'NR==2 {printf "%.1f", $5}' | sed 's/%//')
if (( $(echo "$DISK_USAGE > 80" | bc -l) )); then
    print_status "WARN" "High disk usage: $DISK_USAGE%"
else
    print_status "OK" "Disk usage: $DISK_USAGE%"
fi

# Check service status
SERVICES=("TimescaleDB:5432" "Redis:6379" "Prometheus:9090" "Grafana:3000" "Metrics:8000")
for service in "${SERVICES[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if nc -z localhost $port > /dev/null 2>&1; then
        print_status "OK" "$name service running"
    else
        print_status "FAIL" "$name service not responding on port $port"
    fi
done

# 6. Generate daily report
print_status "INFO" "Generating daily report..."
cat > "reports/daily_report_$(date +%Y%m%d).md" << EOF
# Daily Stress Testing Report - $(date +%Y-%m-%d)

## Platform Status
- **Startup Time**: $(date +%H:%M:%S)
- **Certification**: $CERT_RESULT
- **Pass Rate**: $PASS_RATE%
- **Memory Usage**: $MEMORY_USAGE%
- **Disk Usage**: $DISK_USAGE%

## Service Health
$(for service in "${SERVICES[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    if nc -z localhost $port > /dev/null 2>&1; then
        echo "- ✅ $name: Running"
    else
        echo "- ❌ $name: Down"
    fi
done)

## Key Metrics
$(curl -s http://localhost:8000/metrics | grep -E "(decision_latency|decisions_total|recovery_time)" | head -5)

## Next Steps
- Monitor throughout trading day
- Check alerts in Grafana
- Review any performance anomalies
- Prepare for evening shutdown

---
Generated at $(date)
EOF

print_status "OK" "Daily report generated: reports/daily_report_$(date +%Y%m%d).md"

echo ""
echo "🎯 Daily Startup Complete"
echo "========================="
echo "📊 Grafana Dashboard: http://localhost:3000"
echo "📈 Prometheus: http://localhost:9090"
echo "📋 Daily Report: reports/daily_report_$(date +%Y%m%d).md"
echo ""
echo "🚀 Platform ready for trading day operations!"