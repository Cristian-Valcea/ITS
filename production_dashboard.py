#!/usr/bin/env python3
"""
📊 PRODUCTION MONITORING DASHBOARD
Real-time monitoring of IntradayJules production deployment
"""

import os
import sys
import time
import json
import redis
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def get_system_status():
    """Get comprehensive system status"""
    
    status = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'system_health': 'UNKNOWN',
        'components': {},
        'alerts': [],
        'metrics': {}
    }
    
    try:
        # Redis connection for state management
        r = redis.Redis(decode_responses=True)
        r.ping()
        status['components']['redis'] = '✅ Connected'
        
        # Risk Governor state
        try:
            governor_state = r.get('risk_governor:state')
            if governor_state:
                state_data = json.loads(governor_state)
                current_state = state_data.get('state', 'UNKNOWN')
                status['components']['risk_governor'] = f"✅ {current_state}"
                
                if current_state != 'RUNNING':
                    status['alerts'].append(f"⚠️ Risk Governor not in RUNNING state: {current_state}")
            else:
                status['components']['risk_governor'] = '❌ State not found'
                status['alerts'].append('🚨 Risk Governor state missing')
        except Exception as e:
            status['components']['risk_governor'] = f'❌ Error: {str(e)}'
            status['alerts'].append(f'🚨 Risk Governor error: {str(e)}')
        
        # IBKR Connection status (check for recent activity)
        try:
            # Check if there are recent logs indicating IBKR activity
            log_files = list(Path('logs').glob('production/*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                age_minutes = (time.time() - latest_log.stat().st_mtime) / 60
                
                if age_minutes < 10:  # Active within last 10 minutes
                    status['components']['ibkr_connection'] = '✅ Active'
                else:
                    status['components']['ibkr_connection'] = f'⚠️ Idle ({age_minutes:.1f}m ago)'
            else:
                status['components']['ibkr_connection'] = '❌ No recent activity'
        except Exception as e:
            status['components']['ibkr_connection'] = f'❌ Error: {str(e)}'
        
        # Model availability
        model_paths = [
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
        ]
        
        model_found = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                status['components']['trading_model'] = f'✅ {os.path.basename(model_path)}'
                model_found = True
                break
        
        if not model_found:
            status['components']['trading_model'] = '❌ No model found'
            status['alerts'].append('🚨 Trading model missing')
        
        # Enhanced Safety Components
        safety_components = [
            'src/brokers/enhanced_safe_wrapper.py',
            'src/brokers/event_order_monitor.py',
            'src/brokers/connection_validator.py'
        ]
        
        safety_status = []
        for component in safety_components:
            if os.path.exists(component):
                safety_status.append('✅')
            else:
                safety_status.append('❌')
                status['alerts'].append(f'🚨 Missing safety component: {component}')
        
        safety_pct = (safety_status.count('✅') / len(safety_status)) * 100
        status['components']['enhanced_safety'] = f"{safety_pct:.0f}% ({''.join(safety_status)})"
        
        # Determine overall system health
        critical_components = ['redis', 'risk_governor', 'trading_model']
        critical_ok = all(
            '✅' in status['components'].get(comp, '❌') 
            for comp in critical_components
        )
        
        if critical_ok and len(status['alerts']) == 0:
            status['system_health'] = '🟢 HEALTHY'
        elif critical_ok:
            status['system_health'] = '🟡 WARNING'
        else:
            status['system_health'] = '🔴 CRITICAL'
        
        # Add metrics
        status['metrics'] = {
            'critical_components_ok': sum(1 for comp in critical_components if '✅' in status['components'].get(comp, '❌')),
            'total_critical_components': len(critical_components),
            'alert_count': len(status['alerts']),
            'safety_component_health': safety_pct
        }
        
    except Exception as e:
        status['system_health'] = '🔴 CRITICAL'
        status['alerts'].append(f'🚨 System status check failed: {str(e)}')
    
    return status

def display_dashboard():
    """Display real-time production dashboard"""
    
    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Get current system status
        status = get_system_status()
        
        # Header
        print("🚀 INTRADAYJULES PRODUCTION DASHBOARD")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"System Health: {status['system_health']}")
        print()
        
        # Component Status
        print("📊 COMPONENT STATUS:")
        print("-" * 30)
        for component, component_status in status['components'].items():
            print(f"  {component.replace('_', ' ').title():<20} {component_status}")
        print()
        
        # Metrics
        metrics = status['metrics']
        print("📈 SYSTEM METRICS:")
        print("-" * 30)
        print(f"  Critical Components: {metrics['critical_components_ok']}/{metrics['total_critical_components']}")
        print(f"  Safety System Health: {metrics['safety_component_health']:.0f}%")
        print(f"  Active Alerts: {metrics['alert_count']}")
        print()
        
        # Alerts
        if status['alerts']:
            print("🚨 ACTIVE ALERTS:")
            print("-" * 30)
            for alert in status['alerts']:
                print(f"  {alert}")
            print()
        else:
            print("✅ NO ACTIVE ALERTS")
            print()
        
        # Trading Status
        print("💹 TRADING STATUS:")
        print("-" * 30)
        
        # Check if production deployment is running
        try:
            log_files = list(Path('logs/production').glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                age_minutes = (time.time() - latest_log.stat().st_mtime) / 60
                
                if age_minutes < 5:
                    print("  🟢 Production Trading: ACTIVE")
                    print(f"  📝 Latest Activity: {age_minutes:.1f} minutes ago")
                else:
                    print("  🟡 Production Trading: IDLE")
                    print(f"  📝 Last Activity: {age_minutes:.1f} minutes ago")
            else:
                print("  ⚪ Production Trading: NOT STARTED")
        except:
            print("  ❓ Production Trading: STATUS UNKNOWN")
        
        print()
        
        # Quick Actions
        print("🛠️ QUICK ACTIONS:")
        print("-" * 30)
        print("  Ctrl+C: Exit dashboard")
        print("  Check logs: tail -f logs/production/*.log")
        print("  Governor status: python operator_docs/governor_state_manager.py --status")
        print("  Emergency stop: python operator_docs/governor_state_manager.py --emergency-stop")
        print()
        
        # Footer
        print("=" * 60)
        print("Dashboard updates every 30 seconds | Enhanced IBKR Integration Active")
        
        # Wait before next update
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n🛑 Dashboard stopped by operator")
            break

def quick_status():
    """Display quick status summary"""
    
    status = get_system_status()
    
    print("🚀 INTRADAYJULES QUICK STATUS")
    print("=" * 40)
    print(f"Overall Health: {status['system_health']}")
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    print("Key Components:")
    critical_components = ['risk_governor', 'trading_model', 'enhanced_safety']
    for comp in critical_components:
        comp_status = status['components'].get(comp, '❓ Unknown')
        print(f"  {comp.replace('_', ' ').title()}: {comp_status}")
    
    if status['alerts']:
        print(f"\nAlerts: {len(status['alerts'])} active")
        for alert in status['alerts'][:3]:  # Show first 3 alerts
            print(f"  {alert}")
        if len(status['alerts']) > 3:
            print(f"  ... and {len(status['alerts']) - 3} more")
    else:
        print("\n✅ No active alerts")

def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Monitoring Dashboard")
    parser.add_argument('--quick', action='store_true', help='Show quick status and exit')
    parser.add_argument('--json', action='store_true', help='Output status as JSON')
    
    args = parser.parse_args()
    
    if args.quick:
        if args.json:
            status = get_system_status()
            print(json.dumps(status, indent=2))
        else:
            quick_status()
    else:
        print("Starting production dashboard...")
        print("Press Ctrl+C to exit")
        time.sleep(2)
        display_dashboard()

if __name__ == "__main__":
    main()