#!/bin/bash

# Prometheus Startup Script
set -e

PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
PROMETHEUS_DIR="$PROJECT_ROOT/monitoring/prometheus"
PROMETHEUS_VERSION="2.45.0"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK") echo -e "${GREEN}✅ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "FAIL") echo -e "${RED}❌ $message${NC}" ;;
    esac
}

echo "📈 Starting Prometheus"
echo "====================="

# Check if Prometheus is already running
if pgrep -f prometheus > /dev/null; then
    print_status "WARN" "Prometheus already running, stopping existing instance..."
    pkill -f prometheus || true
    sleep 2
fi

# Download Prometheus if not exists
PROMETHEUS_BINARY="$PROJECT_ROOT/bin/prometheus"
if [ ! -f "$PROMETHEUS_BINARY" ]; then
    print_status "INFO" "Downloading Prometheus $PROMETHEUS_VERSION..."
    mkdir -p "$PROJECT_ROOT/bin"
    cd "$PROJECT_ROOT/bin"
    
    wget -q "https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz"
    tar xzf "prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz"
    cp "prometheus-$PROMETHEUS_VERSION.linux-amd64/prometheus" .
    cp "prometheus-$PROMETHEUS_VERSION.linux-amd64/promtool" .
    rm -rf "prometheus-$PROMETHEUS_VERSION.linux-amd64"*
    chmod +x prometheus promtool
    
    print_status "OK" "Prometheus downloaded"
fi

# Validate configuration
print_status "INFO" "Validating Prometheus configuration..."
if "$PROJECT_ROOT/bin/promtool" check config "$PROMETHEUS_DIR/prometheus.yml"; then
    print_status "OK" "Configuration valid"
else
    print_status "FAIL" "Configuration invalid"
    exit 1
fi

# Create data directory
mkdir -p "$PROMETHEUS_DIR/data"

# Start Prometheus
print_status "INFO" "Starting Prometheus server..."
nohup "$PROMETHEUS_BINARY" \
    --config.file="$PROMETHEUS_DIR/prometheus.yml" \
    --storage.tsdb.path="$PROMETHEUS_DIR/data" \
    --storage.tsdb.retention.time=30d \
    --storage.tsdb.retention.size=10GB \
    --web.console.libraries="$PROJECT_ROOT/bin/console_libraries" \
    --web.console.templates="$PROJECT_ROOT/bin/consoles" \
    --web.listen-address="0.0.0.0:9090" \
    --web.enable-lifecycle \
    --log.level=info \
    > "$PROJECT_ROOT/logs/stress_testing/prometheus.log" 2>&1 &

PROMETHEUS_PID=$!
echo $PROMETHEUS_PID > "$PROJECT_ROOT/logs/stress_testing/prometheus.pid"

# Wait for startup
print_status "INFO" "Waiting for Prometheus to start..."
for i in {1..30}; do
    if curl -s http://localhost:9090/-/ready > /dev/null 2>&1; then
        print_status "OK" "Prometheus started successfully (PID: $PROMETHEUS_PID)"
        print_status "OK" "Prometheus UI: http://localhost:9090"
        break
    fi
    sleep 1
done

# Verify startup
if ! curl -s http://localhost:9090/-/ready > /dev/null 2>&1; then
    print_status "FAIL" "Prometheus failed to start"
    cat "$PROJECT_ROOT/logs/stress_testing/prometheus.log"
    exit 1
fi

# Check targets
print_status "INFO" "Checking Prometheus targets..."
sleep 5
TARGETS_UP=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | map(select(.health == "up")) | length')
TARGETS_TOTAL=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length')

print_status "OK" "Prometheus targets: $TARGETS_UP/$TARGETS_TOTAL up"

echo ""
echo "📈 Prometheus Ready"
echo "=================="
echo "🌐 Web UI: http://localhost:9090"
echo "📊 Targets: http://localhost:9090/targets"
echo "🚨 Alerts: http://localhost:9090/alerts"
echo "📋 Config: http://localhost:9090/config"