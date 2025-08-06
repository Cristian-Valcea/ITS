#!/bin/bash

# Grafana Startup Script
set -e

PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
GRAFANA_DIR="$PROJECT_ROOT/monitoring/grafana"
GRAFANA_VERSION="10.0.3"

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

echo "📊 Starting Grafana"
echo "==================="

# Check if Grafana is already running
if pgrep -f grafana-server > /dev/null; then
    print_status "WARN" "Grafana already running, stopping existing instance..."
    pkill -f grafana-server || true
    sleep 2
fi

# Download Grafana if not exists
GRAFANA_BINARY="$PROJECT_ROOT/bin/grafana-server"
if [ ! -f "$GRAFANA_BINARY" ]; then
    print_status "INFO" "Downloading Grafana $GRAFANA_VERSION..."
    mkdir -p "$PROJECT_ROOT/bin"
    cd "$PROJECT_ROOT/bin"
    
    wget -q "https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.linux-amd64.tar.gz"
    tar xzf "grafana-$GRAFANA_VERSION.linux-amd64.tar.gz"
    cp "grafana-$GRAFANA_VERSION/bin/grafana-server" .
    cp "grafana-$GRAFANA_VERSION/bin/grafana-cli" .
    cp -r "grafana-$GRAFANA_VERSION/public" "$GRAFANA_DIR/"
    cp -r "grafana-$GRAFANA_VERSION/conf" "$GRAFANA_DIR/"
    rm -rf "grafana-$GRAFANA_VERSION"*
    chmod +x grafana-server grafana-cli
    
    print_status "OK" "Grafana downloaded"
fi

# Create necessary directories
mkdir -p "$GRAFANA_DIR/data"
mkdir -p "$GRAFANA_DIR/logs"
mkdir -p "$GRAFANA_DIR/plugins"
mkdir -p "$GRAFANA_DIR/provisioning/datasources"
mkdir -p "$GRAFANA_DIR/provisioning/dashboards"

# Copy dashboard files
cp "$GRAFANA_DIR/dashboards/stress_testing_dashboard.json" "$GRAFANA_DIR/data/"

# Start Grafana
print_status "INFO" "Starting Grafana server..."
cd "$GRAFANA_DIR"

nohup "$GRAFANA_BINARY" \
    --config="$GRAFANA_DIR/grafana.ini" \
    --homepath="$GRAFANA_DIR" \
    web \
    > "$PROJECT_ROOT/logs/stress_testing/grafana.log" 2>&1 &

GRAFANA_PID=$!
echo $GRAFANA_PID > "$PROJECT_ROOT/logs/stress_testing/grafana.pid"

# Wait for startup
print_status "INFO" "Waiting for Grafana to start..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        print_status "OK" "Grafana started successfully (PID: $GRAFANA_PID)"
        break
    fi
    sleep 2
done

# Verify startup
if ! curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    print_status "FAIL" "Grafana failed to start"
    cat "$PROJECT_ROOT/logs/stress_testing/grafana.log"
    exit 1
fi

# Import dashboard
print_status "INFO" "Importing stress testing dashboard..."
sleep 5

curl -X POST \
    -H "Content-Type: application/json" \
    -d @"$GRAFANA_DIR/dashboards/stress_testing_dashboard.json" \
    http://admin:stress_testing_2025@localhost:3000/api/dashboards/db \
    > /dev/null 2>&1 || print_status "WARN" "Dashboard import may have failed"

print_status "OK" "Dashboard import attempted"

echo ""
echo "📊 Grafana Ready"
echo "================"
echo "🌐 Web UI: http://localhost:3000"
echo "👤 Login: admin / stress_testing_2025"
echo "📊 Dashboard: Stress Testing Platform - Real-Time Monitoring"
echo "🔧 Admin: http://localhost:3000/admin"