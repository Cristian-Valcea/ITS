#!/bin/bash

# Real-Time Monitoring Script
PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_metric() {
    local name=$1
    local value=$2
    local threshold=$3
    local unit=$4
    
    if (( $(echo "$value > $threshold" | bc -l) )); then
        echo -e "${RED}❌ $name: $value$unit (> $threshold$unit)${NC}"
    else
        echo -e "${GREEN}✅ $name: $value$unit${NC}"
    fi
}

show_help() {
    echo "🔍 Stress Testing Platform Monitor"
    echo "=================================="
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status     - Show current platform status"
    echo "  metrics    - Show key performance metrics"
    echo "  alerts     - Show active alerts"
    echo "  latency    - Monitor decision latency in real-time"
    echo "  errors     - Show recent errors"
    echo "  health     - Comprehensive health check"
    echo "  watch      - Continuous monitoring (Ctrl+C to stop)"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 watch"
    echo "  $0 latency"
}

show_status() {
    echo "🎯 Platform Status"
    echo "=================="
    echo "Timestamp: $(date)"
    echo ""
    
    # Service status
    echo "📊 Services:"
    services=("TimescaleDB:5432" "Redis:6379" "Prometheus:9090" "Grafana:3000" "Metrics:8000")
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if nc -z localhost $port > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ $name${NC}"
        else
            echo -e "  ${RED}❌ $name${NC}"
        fi
    done
    echo ""
    
    # Quick metrics
    if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
        echo "📈 Quick Metrics:"
        DECISIONS=$(curl -s http://localhost:8000/metrics | grep "decisions_total" | tail -1 | awk '{print $2}' || echo "0")
        echo "  Total Decisions: $DECISIONS"
        
        # Get latest certification status
        if [ -f "stress_testing/results/certification_report.json" ]; then
            CERT_STATUS=$(cat stress_testing/results/certification_report.json | jq -r '.certified')
            CERT_TIME=$(cat stress_testing/results/certification_report.json | jq -r '.certification_timestamp')
            CERT_DATE=$(date -d @$CERT_TIME 2>/dev/null || echo "unknown")
            echo "  Last Certification: $CERT_STATUS ($CERT_DATE)"
        fi
    else
        echo -e "${RED}❌ Metrics endpoint not accessible${NC}"
    fi
}

show_metrics() {
    echo "📊 Key Performance Metrics"
    echo "=========================="
    echo "Timestamp: $(date)"
    echo ""
    
    if ! curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
        echo -e "${RED}❌ Metrics endpoint not accessible${NC}"
        return 1
    fi
    
    # Get Prometheus metrics
    METRICS=$(curl -s http://localhost:9090/api/v1/query)
    
    echo "🎯 Critical Metrics:"
    
    # Decision latency P99
    P99_QUERY='histogram_quantile(0.99, rate(risk_governor_decision_latency_seconds_bucket[1m])) * 1000'
    P99_VALUE=$(curl -s "http://localhost:9090/api/v1/query?query=$P99_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    print_metric "Decision Latency P99" "$P99_VALUE" "15" "ms"
    
    # Pipeline latency P99
    PIPELINE_QUERY='histogram_quantile(0.99, rate(pipeline_latency_seconds_bucket[1m])) * 1000'
    PIPELINE_VALUE=$(curl -s "http://localhost:9090/api/v1/query?query=$PIPELINE_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    print_metric "Pipeline Latency P99" "$PIPELINE_VALUE" "20" "ms"
    
    # Error rate
    ERROR_QUERY='rate(risk_governor_decision_errors_total[5m]) / rate(risk_governor_decisions_total[5m]) * 100'
    ERROR_VALUE=$(curl -s "http://localhost:9090/api/v1/query?query=$ERROR_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    print_metric "Error Rate" "$ERROR_VALUE" "0.1" "%"
    
    # Position delta
    POSITION_QUERY='abs(risk_governor_position_delta_usd)'
    POSITION_VALUE=$(curl -s "http://localhost:9090/api/v1/query?query=$POSITION_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    print_metric "Position Delta" "$POSITION_VALUE" "1000" "$"
    
    echo ""
    echo "📈 System Metrics:"
    
    # Memory usage
    MEMORY_MB=$(ps aux | grep -E "(prometheus|grafana|python.*stress)" | awk '{sum += $6} END {print sum/1024}' || echo "0")
    print_metric "Memory Usage" "$MEMORY_MB" "1000" "MB"
    
    # Disk usage
    DISK_USAGE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    print_metric "Disk Usage" "$DISK_USAGE" "80" "%"
}

show_alerts() {
    echo "🚨 Active Alerts"
    echo "================"
    echo "Timestamp: $(date)"
    echo ""
    
    if curl -s http://localhost:9090/api/v1/alerts > /dev/null 2>&1; then
        ALERTS=$(curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state == "firing")')
        
        if [ -z "$ALERTS" ]; then
            echo -e "${GREEN}✅ No active alerts${NC}"
        else
            echo -e "${RED}⚠️  Active alerts found:${NC}"
            echo "$ALERTS" | jq -r '.labels.alertname + " - " + .annotations.summary'
        fi
    else
        echo -e "${RED}❌ Cannot access Prometheus alerts endpoint${NC}"
    fi
}

monitor_latency() {
    echo "⏱️  Real-Time Latency Monitor"
    echo "============================"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        if curl -s http://localhost:9090/api/v1/query > /dev/null 2>&1; then
            P99_QUERY='histogram_quantile(0.99, rate(risk_governor_decision_latency_seconds_bucket[1m])) * 1000'
            P95_QUERY='histogram_quantile(0.95, rate(risk_governor_decision_latency_seconds_bucket[1m])) * 1000'
            P50_QUERY='histogram_quantile(0.50, rate(risk_governor_decision_latency_seconds_bucket[1m])) * 1000'
            
            P99=$(curl -s "http://localhost:9090/api/v1/query?query=$P99_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
            P95=$(curl -s "http://localhost:9090/api/v1/query?query=$P95_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
            P50=$(curl -s "http://localhost:9090/api/v1/query?query=$P50_QUERY" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
            
            printf "\r$(date +%H:%M:%S) - P99: %6.1fms | P95: %6.1fms | P50: %6.1fms" "$P99" "$P95" "$P50"
        else
            printf "\r$(date +%H:%M:%S) - Metrics unavailable"
        fi
        
        sleep 1
    done
}

show_errors() {
    echo "❌ Recent Errors"
    echo "================"
    echo "Timestamp: $(date)"
    echo ""
    
    # Check logs for errors
    if [ -f "logs/stress_testing/metrics_server.log" ]; then
        echo "📋 Metrics Server Errors (last 10):"
        tail -20 logs/stress_testing/metrics_server.log | grep -i error | tail -10 || echo "No recent errors"
        echo ""
    fi
    
    if [ -f "logs/stress_testing/prometheus.log" ]; then
        echo "📋 Prometheus Errors (last 10):"
        tail -20 logs/stress_testing/prometheus.log | grep -i error | tail -10 || echo "No recent errors"
        echo ""
    fi
    
    # Check for failed stress tests
    echo "📋 Recent Test Failures:"
    find logs/stress_testing -name "*.log" -mtime -1 -exec grep -l "FAIL\|ERROR" {} \; | head -5 || echo "No recent test failures"
}

health_check() {
    echo "🏥 Comprehensive Health Check"
    echo "============================="
    echo "Timestamp: $(date)"
    echo ""
    
    # Run all checks
    show_status
    echo ""
    show_metrics
    echo ""
    show_alerts
    echo ""
    
    # Additional health indicators
    echo "🔍 Additional Health Indicators:"
    
    # Check if certification is recent
    if [ -f "stress_testing/results/certification_report.json" ]; then
        CERT_TIME=$(cat stress_testing/results/certification_report.json | jq -r '.certification_timestamp')
        CURRENT_TIME=$(date +%s)
        HOURS_SINCE=$(( (CURRENT_TIME - CERT_TIME) / 3600 ))
        
        if [ $HOURS_SINCE -lt 24 ]; then
            echo -e "  ${GREEN}✅ Certification recent ($HOURS_SINCE hours ago)${NC}"
        else
            echo -e "  ${RED}❌ Certification stale ($HOURS_SINCE hours ago)${NC}"
        fi
    fi
    
    # Check log file sizes
    LOG_SIZE=$(du -sh logs/stress_testing/ 2>/dev/null | cut -f1 || echo "0")
    echo "  📁 Log directory size: $LOG_SIZE"
    
    # Check database connectivity
    if docker exec timescaledb_primary pg_isready -U postgres -d trading_data > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Database connectivity${NC}"
    else
        echo -e "  ${RED}❌ Database connectivity${NC}"
    fi
}

continuous_watch() {
    echo "👁️  Continuous Monitoring"
    echo "========================"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        clear
        show_status
        echo ""
        show_metrics
        echo ""
        echo "Next update in 10 seconds..."
        sleep 10
    done
}

# Main command handling
case "${1:-help}" in
    "status")
        show_status
        ;;
    "metrics")
        show_metrics
        ;;
    "alerts")
        show_alerts
        ;;
    "latency")
        monitor_latency
        ;;
    "errors")
        show_errors
        ;;
    "health")
        health_check
        ;;
    "watch")
        continuous_watch
        ;;
    "help"|*)
        show_help
        ;;
esac