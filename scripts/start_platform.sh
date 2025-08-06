#!/bin/bash

# Stress Testing Platform Startup Script
set -e

PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "🚀 Starting Stress Testing Platform"
echo "=================================="
echo "Timestamp: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo ""

# 1. Activate virtual environment
print_status "INFO" "Activating virtual environment..."
source venv/bin/activate

# 2. Start TimescaleDB (if not running)
print_status "INFO" "Checking TimescaleDB..."
if ! docker ps | grep -q timescaledb_primary; then
    print_status "INFO" "Starting TimescaleDB container..."
    docker-compose -f docker-compose.yml up -d timescaledb_primary
    sleep 10
fi
print_status "OK" "TimescaleDB running"

# 3. Start Redis (if not running)
print_status "INFO" "Checking Redis..."
if ! pgrep redis-server > /dev/null; then
    print_status "INFO" "Starting Redis server..."
    redis-server --daemonize yes --port 6379
    sleep 2
fi
print_status "OK" "Redis running"

# 4. Start Stress Testing Metrics Server
print_status "INFO" "Starting stress testing metrics server..."
nohup python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from stress_testing.core.metrics import init_metrics
from prometheus_client import start_http_server
import time

# Initialize metrics
metrics = init_metrics(enable_prometheus=True, port=8000)
print('Stress testing metrics server started on port 8000')

# Keep server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Metrics server stopped')
" > logs/stress_testing/metrics_server.log 2>&1 &

METRICS_PID=$!
echo $METRICS_PID > logs/stress_testing/metrics_server.pid
sleep 3

if curl -s http://localhost:8000/metrics > /dev/null; then
    print_status "OK" "Stress testing metrics server running (PID: $METRICS_PID)"
else
    print_status "FAIL" "Failed to start metrics server"
    exit 1
fi

# 5. Start Prometheus
print_status "INFO" "Starting Prometheus..."
./scripts/start_prometheus.sh

# 6. Start Grafana
print_status "INFO" "Starting Grafana..."
./scripts/start_grafana.sh

# 7. Validate all services
print_status "INFO" "Validating all services..."
sleep 5

# Check TimescaleDB
if docker exec timescaledb_primary pg_isready -U postgres -d trading_data > /dev/null 2>&1; then
    print_status "OK" "TimescaleDB: Ready"
else
    print_status "FAIL" "TimescaleDB: Not ready"
fi

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    print_status "OK" "Redis: Ready"
else
    print_status "FAIL" "Redis: Not ready"
fi

# Check Stress Testing Metrics
if curl -s http://localhost:8000/metrics | grep -q "risk_governor"; then
    print_status "OK" "Stress Testing Metrics: Ready"
else
    print_status "WARN" "Stress Testing Metrics: Limited data"
fi

# Check Prometheus
if curl -s http://localhost:9090/-/ready > /dev/null 2>&1; then
    print_status "OK" "Prometheus: Ready"
else
    print_status "FAIL" "Prometheus: Not ready"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    print_status "OK" "Grafana: Ready"
else
    print_status "FAIL" "Grafana: Not ready"
fi

echo ""
echo "🎯 Platform Status Summary"
echo "=========================="
echo "📊 Stress Testing Metrics: http://localhost:8000/metrics"
echo "📈 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3000 (admin/stress_testing_2025)"
echo ""
echo "🚀 Platform ready for stress testing!"
echo "Run: python stress_testing/run_full_suite.py --certification"