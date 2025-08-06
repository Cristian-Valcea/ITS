#!/usr/bin/env python3
"""
System Health Check Script
Quick health status for all system components
"""

import os
import sys
import time
import json
from datetime import datetime

# Add src to path
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def check_redis():
    """Check Redis connection and data"""
    try:
        import redis
        r = redis.Redis(decode_responses=True)
        
        # Test connection
        r.ping()
        
        # Check data
        info = r.info()
        memory_usage = info.get('used_memory_human', 'unknown')
        uptime = info.get('uptime_in_seconds', 0)
        
        return {
            "status": "✅ OK",
            "memory": memory_usage,
            "uptime_hours": round(uptime / 3600, 1),
            "connected_clients": info.get('connected_clients', 0)
        }
    except Exception as e:
        return {
            "status": "❌ FAIL",
            "error": str(e)
        }

def check_prometheus():
    """Check Prometheus metrics endpoint"""
    try:
        import requests
        
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        if response.status_code == 200:
            # Count metrics
            metrics_count = len([line for line in response.text.split('\n') 
                               if line and not line.startswith('#')])
            
            return {
                "status": "✅ OK",
                "metrics_count": metrics_count,
                "endpoint": "http://localhost:8000/metrics"
            }
        else:
            return {
                "status": "❌ FAIL",
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "❌ FAIL", 
            "error": str(e)
        }

def check_broker_system():
    """Check broker and risk governor components"""
    try:
        from risk_governor.broker_adapter import BrokerExecutionManager
        from risk_governor.core_governor import ProductionRiskGovernor
        
        # Test broker manager
        mgr = BrokerExecutionManager()
        daily_stats = mgr.get_daily_stats()
        
        # Test risk governor
        governor = ProductionRiskGovernor("MSFT")
        atr_status = governor.position_governor.get_atr_status()
        
        return {
            "status": "✅ OK",
            "daily_orders": f"{daily_stats['daily_order_count']}/{daily_stats['max_daily_orders']}",
            "daily_cost": f"${daily_stats['daily_effective_cost']:.2f}/${daily_stats['max_daily_effective_cost']:.2f}",
            "atr_mode": "intraday" if atr_status['using_intraday'] else "regular",
            "atr_value": f"{atr_status['current_atr']:.3f}"
        }
    except Exception as e:
        return {
            "status": "❌ FAIL",
            "error": str(e)
        }

def check_system_resources():
    """Check system resources"""
    try:
        import shutil
        import psutil
        
        # Disk space
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        total_gb = disk_usage.total / (1024**3)
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        # Memory
        memory = psutil.virtual_memory()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status = "✅ OK"
        if free_gb < 1.0 or memory.percent > 90 or cpu_percent > 90:
            status = "⚠️ WARNING"
        if free_gb < 0.5 or memory.percent > 95 or cpu_percent > 95:
            status = "❌ CRITICAL"
        
        return {
            "status": status,
            "disk_free_gb": round(free_gb, 1),
            "disk_used_percent": round(disk_percent, 1),
            "memory_used_percent": round(memory.percent, 1),
            "cpu_percent": round(cpu_percent, 1)
        }
    except Exception as e:
        return {
            "status": "❌ FAIL",
            "error": str(e)
        }

def check_log_files():
    """Check log files and recent errors"""
    try:
        logs_status = {}
        
        # Check main log file
        main_log = "logs/risk_governor.log"
        if os.path.exists(main_log):
            # Get file size and last modified
            stat = os.stat(main_log)
            size_mb = stat.st_size / (1024**2)
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Count recent errors (last 100 lines)
            try:
                with open(main_log, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    error_count = sum(1 for line in recent_lines if 'ERROR' in line)
                    warning_count = sum(1 for line in recent_lines if 'WARNING' in line)
            except:
                error_count = 0
                warning_count = 0
            
            logs_status["main_log"] = {
                "exists": True,
                "size_mb": round(size_mb, 2),
                "last_modified": last_modified.strftime("%H:%M:%S"),
                "recent_errors": error_count,
                "recent_warnings": warning_count
            }
        else:
            logs_status["main_log"] = {"exists": False}
        
        # Check for incident log
        incident_log = "incident_log.txt"
        if os.path.exists(incident_log):
            with open(incident_log, 'r') as f:
                lines = f.readlines()
                logs_status["incidents"] = {
                    "count": len(lines),
                    "latest": lines[-1].strip() if lines else "None"
                }
        else:
            logs_status["incidents"] = {"count": 0}
        
        # Determine overall status
        status = "✅ OK"
        if logs_status["main_log"].get("recent_errors", 0) > 5:
            status = "⚠️ WARNING"
        if logs_status["main_log"].get("recent_errors", 0) > 20:
            status = "❌ CRITICAL"
        
        return {
            "status": status,
            **logs_status
        }
    except Exception as e:
        return {
            "status": "❌ FAIL",
            "error": str(e)
        }

def check_market_hours():
    """Check if we're in market hours"""
    try:
        # Use Eastern Time for market hours
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        current_time = now_et.strftime("%H:%M")
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = "09:30"
        market_close = "16:00"
        
        is_market_hours = market_open <= current_time <= market_close
        is_weekday = now_et.weekday() < 5  # Monday = 0, Friday = 4
        
        if is_market_hours and is_weekday:
            market_status = "🟢 OPEN"
        elif is_weekday:
            market_status = "🟡 CLOSED (weekday)"
        else:
            market_status = "🔴 CLOSED (weekend)"
        
        return {
            "status": market_status,
            "current_time": current_time,
            "is_trading_hours": is_market_hours and is_weekday
        }
    except Exception as e:
        return {
            "status": "❌ FAIL",
            "error": str(e)
        }

def main():
    """Run complete system health check"""
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 40)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Run all health checks
    checks = {
        "Redis": check_redis(),
        "Prometheus": check_prometheus(),
        "Broker System": check_broker_system(),
        "System Resources": check_system_resources(),
        "Log Files": check_log_files(),
        "Market Hours": check_market_hours()
    }
    
    # Display results
    overall_status = "✅ HEALTHY"
    critical_issues = 0
    warnings = 0
    
    for component, result in checks.items():
        status = result.get("status", "❌ UNKNOWN")
        print(f"{component:15} {status}")
        
        # Show details for each component
        if "error" in result:
            print(f"                Error: {result['error']}")
        else:
            # Show relevant details
            if component == "Redis" and "memory" in result:
                print(f"                Memory: {result['memory']}, Uptime: {result['uptime_hours']}h")
            elif component == "Prometheus" and "metrics_count" in result:
                print(f"                Metrics: {result['metrics_count']} active")
            elif component == "Broker System" and "daily_orders" in result:
                print(f"                Orders: {result['daily_orders']}, Cost: {result['daily_cost']}")
                print(f"                ATR: {result['atr_value']} ({result['atr_mode']} mode)")
            elif component == "System Resources":
                print(f"                Disk: {result.get('disk_free_gb', 0)}GB free ({result.get('disk_used_percent', 0)}% used)")
                print(f"                Memory: {result.get('memory_used_percent', 0)}% used, CPU: {result.get('cpu_percent', 0)}%")
            elif component == "Log Files" and "main_log" in result:
                log_info = result["main_log"]
                if log_info.get("exists"):
                    print(f"                Size: {log_info['size_mb']}MB, Modified: {log_info['last_modified']}")
                    print(f"                Recent errors: {log_info['recent_errors']}, warnings: {log_info['recent_warnings']}")
                print(f"                Incidents: {result['incidents']['count']}")
            elif component == "Market Hours":
                print(f"                Current: {result['current_time']}, Trading: {result['is_trading_hours']}")
        
        print()
        
        # Count issues
        if "❌" in status or "CRITICAL" in status:
            critical_issues += 1
            overall_status = "❌ CRITICAL ISSUES"
        elif "⚠️" in status or "WARNING" in status:
            warnings += 1
            if overall_status == "✅ HEALTHY":
                overall_status = "⚠️ WARNINGS"
    
    # Overall status
    print("=" * 40)
    print(f"OVERALL STATUS: {overall_status}")
    
    if critical_issues > 0:
        print(f"🚨 {critical_issues} critical issue(s) found")
        print("   Action required: Fix critical issues before trading")
    elif warnings > 0:
        print(f"⚠️ {warnings} warning(s) found")
        print("   Action recommended: Monitor closely")
    else:
        print("✅ All systems operational")
    
    print("")
    print("📞 If critical issues persist, contact senior developer")
    print("📝 For details, check logs/risk_governor.log")
    
    # Return appropriate exit code
    if critical_issues > 0:
        return 2  # Critical issues
    elif warnings > 0:
        return 1  # Warnings
    else:
        return 0  # All good

if __name__ == "__main__":
    sys.exit(main())