#!/bin/bash
# Stress Testing CI Guards
# Ensures all prerequisites are met before running stress tests

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/stress_testing/data/historical"
REQUIRED_DATA_FILES=(
    "nvda_l2_20231017.parquet"
)

# Backup data sources
BACKUP_SOURCES=(
    "https://data-host/nvda_l2_20231017.parquet"
    "s3://trading-data/historical/nvda_l2_20231017.parquet"
)

echo -e "${GREEN}🛡️  Stress Testing CI Guards${NC}"
echo "=================================================="

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "✅ ${GREEN}$message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "⚠️  ${YELLOW}$message${NC}"
    else
        echo -e "❌ ${RED}$message${NC}"
    fi
}

# Function to check command availability
check_command() {
    local cmd=$1
    local description=$2
    
    if command -v "$cmd" >/dev/null 2>&1; then
        print_status "OK" "$description available"
        return 0
    else
        print_status "FAIL" "$description not found"
        return 1
    fi
}

# Function to check Python environment
check_python_env() {
    echo "🐍 Checking Python environment..."
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        print_status "FAIL" "Virtual environment not activated"
        echo "   Please run: source venv/bin/activate"
        return 1
    fi
    
    print_status "OK" "Virtual environment active: $VIRTUAL_ENV"
    
    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    if python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        print_status "OK" "Python version: $python_version"
    else
        print_status "FAIL" "Python version $python_version < 3.10 required"
        return 1
    fi
    
    return 0
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    local requirements_file="$SCRIPT_DIR/requirements.txt"
    if [[ ! -f "$requirements_file" ]]; then
        print_status "FAIL" "Requirements file not found: $requirements_file"
        return 1
    fi
    
    if pip install -r "$requirements_file" --quiet; then
        print_status "OK" "Dependencies installed successfully"
    else
        print_status "FAIL" "Failed to install dependencies"
        return 1
    fi
    
    return 0
}

# Function to check data availability
check_data_availability() {
    echo "📊 Checking historical data availability..."
    
    # Check if TimescaleDB container is running
    if ! docker ps | grep -q timescaledb_primary; then
        print_status "FAIL" "TimescaleDB container not running"
        echo "   Please start with: docker start timescaledb_primary"
        return 1
    fi
    
    # Check if database is ready
    if ! docker exec timescaledb_primary pg_isready -U postgres -d trading_data >/dev/null 2>&1; then
        print_status "FAIL" "TimescaleDB not ready"
        echo "   Database may still be starting up..."
        return 1
    fi
    
    print_status "OK" "TimescaleDB container running and ready"
    
    # Validate historical data using Python adapter
    if python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from stress_testing.simulators.historical_data_adapter import HistoricalDataAdapter

try:
    adapter = HistoricalDataAdapter()
    report = adapter.validate_data_availability()
    
    if report['validation_passed']:
        print(f'✅ Database validation passed')
        print(f'   Symbols: {report[\"symbols_available\"]}')
        print(f'   NVDA bars: {report[\"total_bars\"]:,}')
        print(f'   Flash crash data: {\"Available\" if report[\"flash_crash_data_available\"] else \"Missing\"}')
        exit(0)
    else:
        print(f'❌ Database validation failed')
        if 'error' in report:
            print(f'   Error: {report[\"error\"]}')
        exit(1)
        
except Exception as e:
    print(f'❌ Database validation error: {e}')
    exit(1)
" 2>/dev/null; then
        print_status "OK" "Historical data validation passed"
    else
        print_status "FAIL" "Historical data validation failed"
        echo "   Check database connection and data integrity"
        return 1
    fi
    
    return 0
}

# Function to download data files
download_data_files() {
    for data_file in "${REQUIRED_DATA_FILES[@]}"; do
        local file_path="$DATA_DIR/$data_file"
        
        if [[ ! -f "$file_path" ]]; then
            echo "📥 Downloading $data_file..."
            
            local downloaded=false
            for source in "${BACKUP_SOURCES[@]}"; do
                local url="$source"
                if [[ "$source" == *"$data_file" ]]; then
                    url="$source"
                else
                    url="$source/$data_file"
                fi
                
                echo "   Trying: $url"
                if curl -f -L --connect-timeout 10 --max-time 300 -o "$file_path" "$url" 2>/dev/null; then
                    print_status "OK" "Downloaded from $url"
                    downloaded=true
                    break
                else
                    print_status "WARN" "Failed to download from $url"
                fi
            done
            
            if [[ "$downloaded" = false ]]; then
                print_status "FAIL" "Could not download $data_file from any source"
                echo "❌ CI pipeline cannot proceed without historical data"
                exit 1
            fi
        fi
    done
}

# Function to check system resources
check_system_resources() {
    echo "💻 Checking system resources..."
    
    # Check available memory (need at least 2GB)
    if command -v free >/dev/null 2>&1; then
        local available_mb=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [[ $available_mb -gt 2048 ]]; then
            print_status "OK" "Available memory: ${available_mb}MB"
        else
            print_status "WARN" "Low memory: ${available_mb}MB (recommended: >2GB)"
        fi
    fi
    
    # Check available disk space (need at least 1GB)
    local available_gb=$(df "$PROJECT_ROOT" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
    if (( $(echo "$available_gb > 1.0" | bc -l) )); then
        print_status "OK" "Available disk space: ${available_gb}GB"
    else
        print_status "WARN" "Low disk space: ${available_gb}GB (recommended: >1GB)"
    fi
    
    return 0
}

# Function to check ports availability
check_ports() {
    echo "🔌 Checking port availability..."
    
    local ports=(8000 3000)  # Prometheus, Grafana
    
    for port in "${ports[@]}"; do
        if lsof -i ":$port" >/dev/null 2>&1; then
            print_status "WARN" "Port $port is in use (may conflict with monitoring)"
        else
            print_status "OK" "Port $port available"
        fi
    done
    
    return 0
}

# Function to validate configuration
validate_config() {
    echo "⚙️  Validating configuration..."
    
    # Check if stress testing config can be loaded
    if python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from stress_testing.core.config import StressTestConfig
config = StressTestConfig()
print(f'Config loaded: {len(config.get_test_scenarios())} scenarios')
" 2>/dev/null; then
        print_status "OK" "Configuration validation passed"
    else
        print_status "FAIL" "Configuration validation failed"
        return 1
    fi
    
    return 0
}

# Function to run pre-flight checks
run_preflight_checks() {
    echo "🚀 Running pre-flight checks..."
    
    # Quick smoke test of core components
    if python -c "
import sys
sys.path.append('$PROJECT_ROOT')
from stress_testing.core.metrics import get_metrics
from stress_testing.core.config import get_config
metrics = get_metrics()
config = get_config()
print('✅ Core components loaded successfully')
" 2>/dev/null; then
        print_status "OK" "Core components smoke test passed"
    else
        print_status "FAIL" "Core components smoke test failed"
        return 1
    fi
    
    return 0
}

# Function to validate Prometheus metrics
validate_prometheus_metrics() {
    echo "🔍 Validating Prometheus metrics availability..."
    
    # Check if Prometheus endpoint is accessible
    if curl -s http://localhost:8000/metrics >/dev/null 2>&1; then
        print_status "OK" "Prometheus endpoint accessible"
    else
        print_status "FAIL" "Prometheus endpoint not accessible on port 8000"
        echo "   Start metrics server or check port configuration"
        return 1
    fi
    
    # Check for required metric series
    local required_metrics=(
        "risk_governor_decision_latency_seconds"
        "risk_governor_decisions_total"
        "risk_governor_recovery_time_seconds"
    )
    
    local missing_metrics=()
    
    for metric in "${required_metrics[@]}"; do
        if curl -s http://localhost:8000/metrics | grep -q "$metric"; then
            print_status "OK" "Metric series available: $metric"
        else
            print_status "WARN" "Missing Prometheus series: $metric"
            missing_metrics+=("$metric")
        fi
    done
    
    # Only fail if all metrics are missing (indicates config issue)
    if [[ ${#missing_metrics[@]} -eq ${#required_metrics[@]} ]]; then
        print_status "FAIL" "All Prometheus series missing - check scrape config"
        echo "   This indicates a silent scrape configuration typo"
        return 1
    elif [[ ${#missing_metrics[@]} -gt 0 ]]; then
        print_status "WARN" "${#missing_metrics[@]} metrics missing but some available"
        echo "   Missing: ${missing_metrics[*]}"
    fi
    
    return 0
}

# Main execution
main() {
    local exit_code=0
    
    echo "Starting CI guards at $(date)"
    echo "Project root: $PROJECT_ROOT"
    echo ""
    
    # Run all checks
    check_python_env || exit_code=1
    install_dependencies || exit_code=1
    check_data_availability || exit_code=1
    check_system_resources || exit_code=1
    check_ports || exit_code=1
    validate_config || exit_code=1
    run_preflight_checks || exit_code=1
    validate_prometheus_metrics || exit_code=1
    
    echo ""
    echo "=================================================="
    
    if [[ $exit_code -eq 0 ]]; then
        print_status "OK" "All CI guards passed - ready for stress testing"
        echo -e "${GREEN}🎯 You can now run: python stress_testing/run_full_suite.py${NC}"
    else
        print_status "FAIL" "CI guards failed - please fix issues before proceeding"
        echo -e "${RED}❌ CI pipeline cannot proceed${NC}"
    fi
    
    echo "Completed at $(date)"
    exit $exit_code
}

# Run main function
main "$@"