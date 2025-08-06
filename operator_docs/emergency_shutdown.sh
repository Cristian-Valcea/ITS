#!/bin/bash
# Emergency Shutdown Script for Production Risk Governor
# Use this when everything goes wrong and you need to stop immediately

set -e  # Exit on any error

echo "🚨 EMERGENCY SHUTDOWN INITIATED - $(date)"
echo "====================================================="

# Create logs directory if it doesn't exist
mkdir -p logs emergency_logs

# 1. Stop all trading processes
echo "⏹️ Stopping all trading processes..."
pkill -f stairways_integration 2>/dev/null || echo "   No stairways processes found"
pkill -f risk_governor 2>/dev/null || echo "   No risk_governor processes found"
pkill -f broker_adapter 2>/dev/null || echo "   No broker_adapter processes found"
pkill -f prometheus_monitoring 2>/dev/null || echo "   No prometheus processes found"
pkill -f start_monitoring 2>/dev/null || echo "   No monitoring processes found"

echo "✅ All trading processes stopped"

# 2. Activate Python environment and flatten positions
echo "📉 Attempting to flatten all positions..."
source venv/bin/activate 2>/dev/null || echo "   Warning: Could not activate venv"

python3 -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from risk_governor.broker_adapter import BrokerExecutionManager
    mgr = BrokerExecutionManager()
    reports = mgr.broker.flatten_all_positions()
    print(f'✅ Attempted to flatten {len(reports)} positions')
    
    # Also cancel any pending orders
    open_orders = mgr.broker.get_open_orders()
    for order in open_orders:
        mgr.broker.cancel_order(order.order_id)
        print(f'❌ Cancelled order: {order.order_id}')
    
    if len(open_orders) == 0:
        print('✅ No pending orders to cancel')
    
except Exception as e:
    print(f'❌ Position flattening failed: {e}')
    print('⚠️ MANUAL INTERVENTION REQUIRED')
" 2>/dev/null

# 3. Save current system state
echo "💾 Saving system state..."

# Create emergency log with timestamp
EMERGENCY_LOG="emergency_logs/shutdown_$(date +%Y%m%d_%H%M%S).log"

# Save Redis state if available
redis-cli ping >/dev/null 2>&1 && {
    echo "   Saving Redis state..."
    redis-cli bgsave >/dev/null 2>&1 || echo "   Warning: Redis backup failed"
} || echo "   Redis not available for backup"

# Save current logs
if [ -f "logs/risk_governor.log" ]; then
    cp "logs/risk_governor.log" "$EMERGENCY_LOG"
    echo "   System logs saved to: $EMERGENCY_LOG"
else
    echo "   No system logs found to save"
fi

# Save process list
ps aux | grep -E "(python|redis)" >> "$EMERGENCY_LOG" 2>/dev/null || true

# 4. Document the shutdown
echo "📝 Documenting emergency shutdown..."
{
    echo "$(date): EMERGENCY SHUTDOWN EXECUTED"
    echo "Reason: Manual emergency shutdown script invoked"
    echo "Operator: $(whoami)"
    echo "System: $(hostname)"
    echo "All trading processes terminated"
    echo "Position flattening attempted"
    echo "System state saved to: $EMERGENCY_LOG"
    echo "---"
} >> incident_log.txt

# 5. Final status check
echo "🔍 Final system status check..."
TRADING_PROCESSES=$(ps aux | grep -E "(stairways|risk_governor|broker)" | grep -v grep | wc -l)

if [ "$TRADING_PROCESSES" -eq 0 ]; then
    echo "✅ All trading processes confirmed stopped"
else
    echo "⚠️ Warning: $TRADING_PROCESSES trading processes still running"
    ps aux | grep -E "(stairways|risk_governor|broker)" | grep -v grep
fi

# 6. Display emergency contacts
echo ""
echo "📞 EMERGENCY CONTACTS:"
echo "====================================================="
echo "Senior Developer: [INSERT PHONE NUMBER]"
echo "Risk Manager: [INSERT PHONE NUMBER]"  
echo "CTO: [INSERT PHONE NUMBER]"
echo "Emergency Slack: #trading-alerts"
echo ""
echo "📧 REQUIRED ACTIONS:"
echo "1. Call senior developer IMMEDIATELY"
echo "2. Send system logs: $EMERGENCY_LOG"
echo "3. Report incident details"
echo "4. Wait for instructions before restarting"
echo ""
echo "🚨 EMERGENCY SHUTDOWN COMPLETE - $(date)"
echo "====================================================="

# Exit with status code to indicate emergency shutdown
exit 42  # Special exit code for emergency shutdown