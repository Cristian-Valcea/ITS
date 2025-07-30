#!/usr/bin/env python3
"""
SLO compliance test script for GitHub Actions
"""
import asyncio
import time
import numpy as np
import sys
import os
sys.path.insert(0, '.')

from src.risk import DrawdownCalculator, TurnoverCalculator, RiskEventBus, RiskEvent, EventType, EventPriority

async def test_slo_compliance():
    """Test SLO compliance for critical components."""
    print("🎯 Testing SLO Compliance")
    
    # Test DrawdownCalculator SLO (target: <150µs P95)
    calc = DrawdownCalculator({'lookback_periods': [1, 5, 20]})
    portfolio_values = np.random.normal(100000, 5000, 100)
    
    latencies = []
    for _ in range(100):
        start = time.time_ns()
        result = calc.calculate_safe({
            'portfolio_values': portfolio_values,
            'start_of_day_value': portfolio_values[0]
        })
        end = time.time_ns()
        latencies.append((end - start) / 1000.0)
    
    p95 = np.percentile(latencies, 95)
    print(f"DrawdownCalculator P95: {p95:.2f}µs (SLO: <150µs)")
    
    if p95 > 150.0:
        print(f"❌ SLO VIOLATION: DrawdownCalculator P95 {p95:.2f}µs > 150µs")
        return False
    
    # Test TurnoverCalculator SLO (target: <100µs P95)
    turnover_calc = TurnoverCalculator({'hourly_window_minutes': 60})
    from datetime import datetime, timedelta
    
    trade_values = [10000] * 50
    trade_timestamps = [datetime.now() + timedelta(minutes=i) for i in range(50)]
    
    latencies = []
    for _ in range(100):
        start = time.time_ns()
        result = turnover_calc.calculate_safe({
            'trade_values': trade_values,
            'trade_timestamps': trade_timestamps,
            'capital_base': 1000000
        })
        end = time.time_ns()
        latencies.append((end - start) / 1000.0)
    
    p95 = np.percentile(latencies, 95)
    print(f"TurnoverCalculator P95: {p95:.2f}µs (SLO: <100µs)")
    
    if p95 > 100.0:
        print(f"❌ SLO VIOLATION: TurnoverCalculator P95 {p95:.2f}µs > 100µs")
        return False
    
    print("✅ All SLOs met")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_slo_compliance())
    sys.exit(0 if result else 1)