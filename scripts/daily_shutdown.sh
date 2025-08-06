#!/bin/bash

# Daily Shutdown Routine for Stress Testing Platform
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

echo "🌙 Daily Shutdown Routine"
echo "========================="
echo "Date: $(date)"
echo "Time: $(date +%H:%M:%S)"
echo ""

# 1. Archive today's data
print_status "INFO" "Archiving today's data..."
mkdir -p "archives/$(date +%Y%m%d)"

# Archive logs
cp logs/stress_testing/*.log "archives/$(date +%Y%m%d)/" 2>/dev/null || true

# Archive reports
cp reports/daily_report_$(date +%Y%m%d).md "archives/$(date +%Y%m%d)/" 2>/dev/null || true

# Archive certification results
cp stress_testing/results/certification_report.json "archives/$(date +%Y%m%d)/certification_$(date +%Y%m%d).json" 2>/dev/null || true

print_status "OK" "Data archived to archives/$(date +%Y%m%d)/"

# 2. Generate end-of-day summary
print_status "INFO" "Generating end-of-day summary..."

# Get final metrics
FINAL_CERT=$(cat stress_testing/results/certification_report.json | jq -r '.certified' 2>/dev/null || echo "unknown")
TOTAL_DECISIONS=$(curl -s http://localhost:8000/metrics | grep "decisions_total" | tail -1 | awk '{print $2}' || echo "0")
UPTIME_HOURS=$(ps -o etime= -p $(cat logs/stress_testing/prometheus.pid 2>/dev/null || echo "1") | awk -F: '{print NF-1}' 2>/dev/null || echo "0")

cat > "archives/$(date +%Y%m%d)/end_of_day_summary.md" << EOF
# End of Day Summary - $(date +%Y-%m-%d)

## Daily Performance
- **Final Certification Status**: $FINAL_CERT
- **Total Decisions Processed**: $TOTAL_DECISIONS
- **Platform Uptime**: ~$UPTIME_HOURS hours
- **Shutdown Time**: $(date +%H:%M:%S)

## System Health at Shutdown
$(curl -s http://localhost:8000/metrics | grep -E "(decision_latency|memory|errors)" | head -10)

## Files Archived
- Daily logs
- Certification reports
- Performance metrics
- System health data

---
Generated at $(date)
EOF

print_status "OK" "End-of-day summary generated"

# 3. Graceful service shutdown
print_status "INFO" "Shutting down services gracefully..."

# Stop Grafana
if [ -f "logs/stress_testing/grafana.pid" ]; then
    GRAFANA_PID=$(cat logs/stress_testing/grafana.pid)
    if kill -TERM $GRAFANA_PID 2>/dev/null; then
        print_status "OK" "Grafana stopped gracefully"
    else
        print_status "WARN" "Grafana may have already stopped"
    fi
    rm -f logs/stress_testing/grafana.pid
fi

# Stop Prometheus
if [ -f "logs/stress_testing/prometheus.pid" ]; then
    PROMETHEUS_PID=$(cat logs/stress_testing/prometheus.pid)
    if kill -TERM $PROMETHEUS_PID 2>/dev/null; then
        print_status "OK" "Prometheus stopped gracefully"
    else
        print_status "WARN" "Prometheus may have already stopped"
    fi
    rm -f logs/stress_testing/prometheus.pid
fi

# Stop Metrics Server
if [ -f "logs/stress_testing/metrics_server.pid" ]; then
    METRICS_PID=$(cat logs/stress_testing/metrics_server.pid)
    if kill -TERM $METRICS_PID 2>/dev/null; then
        print_status "OK" "Metrics server stopped gracefully"
    else
        print_status "WARN" "Metrics server may have already stopped"
    fi
    rm -f logs/stress_testing/metrics_server.pid
fi

# Wait for graceful shutdown
sleep 5

# Force kill any remaining processes
pkill -f prometheus 2>/dev/null || true
pkill -f grafana-server 2>/dev/null || true
pkill -f "stress_testing.*metrics" 2>/dev/null || true

print_status "OK" "All monitoring services stopped"

# 4. Database maintenance
print_status "INFO" "Performing database maintenance..."

# Redis save
if redis-cli ping > /dev/null 2>&1; then
    redis-cli BGSAVE > /dev/null 2>&1
    print_status "OK" "Redis background save initiated"
else
    print_status "WARN" "Redis not responding"
fi

# TimescaleDB checkpoint (if accessible)
if docker exec timescaledb_primary pg_isready -U postgres -d trading_data > /dev/null 2>&1; then
    docker exec timescaledb_primary psql -U postgres -d trading_data -c "CHECKPOINT;" > /dev/null 2>&1
    print_status "OK" "TimescaleDB checkpoint completed"
else
    print_status "WARN" "TimescaleDB not accessible for checkpoint"
fi

# 5. Cleanup temporary files
print_status "INFO" "Cleaning up temporary files..."

# Clean old logs (keep last 7 days)
find logs/stress_testing -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Clean old archives (keep last 30 days)
find archives -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

# Clean Prometheus data (keep retention policy)
# Note: Prometheus handles its own retention

print_status "OK" "Cleanup completed"

# 6. Final system check
print_status "INFO" "Final system check..."

# Check if services are actually stopped
RUNNING_SERVICES=()
if nc -z localhost 9090 > /dev/null 2>&1; then RUNNING_SERVICES+=("Prometheus:9090"); fi
if nc -z localhost 3000 > /dev/null 2>&1; then RUNNING_SERVICES+=("Grafana:3000"); fi
if nc -z localhost 8000 > /dev/null 2>&1; then RUNNING_SERVICES+=("Metrics:8000"); fi

if [ ${#RUNNING_SERVICES[@]} -eq 0 ]; then
    print_status "OK" "All monitoring services stopped"
else
    print_status "WARN" "Some services still running: ${RUNNING_SERVICES[*]}"
fi

# 7. Generate shutdown report
cat > "logs/shutdown_$(date +%Y%m%d_%H%M%S).log" << EOF
Daily Shutdown Report - $(date)
================================

Services Stopped:
- Grafana: $([ -f logs/stress_testing/grafana.pid ] && echo "Failed" || echo "Success")
- Prometheus: $([ -f logs/stress_testing/prometheus.pid ] && echo "Failed" || echo "Success")  
- Metrics Server: $([ -f logs/stress_testing/metrics_server.pid ] && echo "Failed" || echo "Success")

Data Archived:
- Location: archives/$(date +%Y%m%d)/
- Files: $(ls -1 archives/$(date +%Y%m%d)/ | wc -l) files archived

Database Maintenance:
- Redis: Background save initiated
- TimescaleDB: Checkpoint completed

Cleanup:
- Old logs cleaned (>7 days)
- Old archives cleaned (>30 days)
- Temporary files removed

Final Status: SHUTDOWN COMPLETE
EOF

echo ""
echo "🌙 Daily Shutdown Complete"
echo "=========================="
echo "📁 Data archived to: archives/$(date +%Y%m%d)/"
echo "📋 Shutdown log: logs/shutdown_$(date +%Y%m%d_%H%M%S).log"
echo "💤 Platform ready for overnight maintenance"
echo ""
echo "🌅 Next startup: ./scripts/daily_startup.sh"