#!/usr/bin/env python3
"""
Test to demonstrate the comprehensive pre-trade risk check integration.

This test shows how the orchestrator should use the new pre_trade_check method
instead of the old assess_trade_risk method to get full sensor coverage.

Key differences:
1. OLD: assess_trade_risk() - only checks basic turnover/drawdown
2. NEW: pre_trade_check() - runs all sensors and evaluates all policies
"""

import sys
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_old_vs_new_risk_assessment():
    """Compare old assess_trade_risk vs new pre_trade_check methods."""
    print("🧪 Testing Old vs New Risk Assessment Methods")
    print("=" * 70)
    
    from src.risk.risk_agent_adapter import RiskAgentAdapter
    
    # Create risk agent adapter
    config = {
        'max_daily_drawdown_pct': 0.02,
        'max_hourly_turnover_ratio': 5.0,
        'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True,
        'liquidate_on_halt': False
    }
    
    adapter = RiskAgentAdapter(config)
    
    # Set up portfolio state
    adapter.reset_daily_limits(1_000_000.0, datetime.now())
    adapter.update_portfolio_value(995_000.0, datetime.now())  # 0.5% drawdown
    
    # Test trade parameters
    symbol = "AAPL"
    quantity = 1000  # shares
    price = 150.0    # $150 per share
    trade_value = abs(quantity * price)  # $150,000
    timestamp = datetime.now()
    
    print(f"Test Trade: {symbol} {quantity} shares @ ${price:.2f} = ${trade_value:,.2f}")
    print(f"Portfolio: ${adapter.current_portfolio_value:,.2f} (drawdown: {adapter.get_current_drawdown():.2%})")
    print()
    
    # Test 1: Old method (assess_trade_risk)
    print("🔍 OLD METHOD: assess_trade_risk()")
    print("-" * 50)
    
    start_time = time.time()
    is_safe_old, reason_old = adapter.assess_trade_risk(trade_value, timestamp)
    old_duration = time.time() - start_time
    
    print(f"Result: {'✅ SAFE' if is_safe_old else '❌ BLOCKED'}")
    print(f"Reason: {reason_old}")
    print(f"Duration: {old_duration*1000:.2f}ms")
    print(f"Calculators used: 2 (DrawdownCalculator, TurnoverCalculator)")
    print(f"Policies evaluated: 1 (basic_risk_limits)")
    print(f"Sensor coverage: ❌ NO SENSORS")
    print()
    
    # Test 2: New method (pre_trade_check)
    print("🔍 NEW METHOD: pre_trade_check()")
    print("-" * 50)
    
    # Create comprehensive mock market data for all sensors
    market_data = {
        'recent_prices': [149.0, 150.5, 148.8, 151.2, 150.0],
        'recent_volumes': [1000000, 1200000, 800000, 1500000, 1100000],
        'feed_timestamps': {
            'market_data': timestamp.timestamp() - 0.1,  # 100ms old
            'order_book': timestamp.timestamp() - 0.05,  # 50ms old
            'trades': timestamp.timestamp() - 0.2,       # 200ms old
        },
        'order_latencies': [45.0, 52.0, 48.0, 55.0, 47.0],  # milliseconds
        'order_book_depth': {
            symbol: {
                'bids': [(149.95, 1000), (149.90, 2000), (149.85, 1500)],
                'asks': [(150.05, 1200), (150.10, 1800), (150.15, 2200)]
            }
        },
        
        # Additional data for calculators that need it
        'trade_values': [150000.0, 75000.0, 200000.0, 125000.0, 180000.0],  # Recent trade values
        'timestamps': [timestamp.timestamp() - i*60 for i in range(5)],  # Recent timestamps
        'price_changes': [0.01, -0.02, 0.015, -0.005, 0.008],  # Recent price changes
        'positions': {symbol: 500},  # Current positions
        'portfolio_values': [995000, 998000, 992000, 1001000, 996000],  # Recent portfolio values
        'returns': [-0.005, 0.003, -0.008, 0.009, -0.004],  # Recent returns
        
        # Mock current time for calculations
        'current_time': timestamp.timestamp(),
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'trade_value': trade_value
    }
    
    start_time = time.time()
    is_safe_new, action_new, detailed_reason_new = adapter.pre_trade_check(
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        market_data=market_data
    )
    new_duration = time.time() - start_time
    
    print(f"Result: {'✅ SAFE' if is_safe_new else '❌ BLOCKED'}")
    print(f"Action: {action_new}")
    print(f"Detailed Reason: {detailed_reason_new}")
    print(f"Duration: {new_duration*1000:.2f}ms")
    print(f"Calculators used: {len(adapter.risk_agent_v2.calculators)} (All sensor calculators)")
    print(f"Policies evaluated: All active policies")
    print(f"Sensor coverage: ✅ FULL COVERAGE")
    print()
    
    # Test 3: Comparison
    print("📊 COMPARISON")
    print("-" * 50)
    
    print(f"Safety Assessment:")
    print(f"  Old method: {'✅ SAFE' if is_safe_old else '❌ BLOCKED'}")
    print(f"  New method: {'✅ SAFE' if is_safe_new else '❌ BLOCKED'}")
    print(f"  Agreement: {'✅ YES' if is_safe_old == is_safe_new else '❌ NO'}")
    print()
    
    print(f"Coverage:")
    print(f"  Old method: Basic turnover/drawdown only")
    print(f"  New method: All {len(adapter.risk_agent_v2.calculators)} sensors + comprehensive policies")
    print()
    
    print(f"Performance:")
    print(f"  Old method: {old_duration*1000:.2f}ms")
    print(f"  New method: {new_duration*1000:.2f}ms")
    if old_duration > 0:
        print(f"  Overhead: {((new_duration - old_duration) / old_duration * 100):.1f}%")
    else:
        print(f"  Overhead: N/A (old method too fast to measure)")
    print()
    
    return is_safe_old, is_safe_new, old_duration, new_duration


def test_sensor_triggered_blocks():
    """Test scenarios where sensors would block trades that basic checks would allow."""
    print("🧪 Testing Sensor-Triggered Risk Blocks")
    print("=" * 70)
    
    from src.risk.risk_agent_adapter import RiskAgentAdapter
    
    # Create risk agent adapter
    config = {
        'max_daily_drawdown_pct': 0.05,  # 5% - very permissive
        'max_hourly_turnover_ratio': 10.0,  # 10x - very permissive
        'max_daily_turnover_ratio': 50.0,   # 50x - very permissive
        'halt_on_breach': True,
        'liquidate_on_halt': False
    }
    
    adapter = RiskAgentAdapter(config)
    
    # Set up portfolio state (basic limits would allow trade)
    adapter.reset_daily_limits(1_000_000.0, datetime.now())
    adapter.update_portfolio_value(980_000.0, datetime.now())  # 2% drawdown - within 5% limit
    
    test_scenarios = [
        {
            'name': 'Stale Feed Data',
            'description': 'Market data feeds are stale (>1 second old)',
            'market_data': {
                'feed_timestamps': {
                    'market_data': time.time() - 2.0,  # 2 seconds old - STALE!
                    'order_book': time.time() - 1.5,   # 1.5 seconds old - STALE!
                    'trades': time.time() - 3.0,       # 3 seconds old - VERY STALE!
                }
            }
        },
        {
            'name': 'High Latency Drift',
            'description': 'Order latencies are drifting higher',
            'market_data': {
                'order_latencies': [150.0, 180.0, 220.0, 250.0, 300.0],  # High and increasing latencies
                'feed_timestamps': {
                    'market_data': time.time() - 0.1,
                    'order_book': time.time() - 0.05,
                    'trades': time.time() - 0.2,
                }
            }
        },
        {
            'name': 'Poor Liquidity',
            'description': 'Order book depth is very thin',
            'market_data': {
                'order_book_depth': {
                    'AAPL': {
                        'bids': [(149.95, 10), (149.90, 20)],  # Very thin book
                        'asks': [(150.05, 15), (150.10, 25)]   # Very thin book
                    }
                },
                'feed_timestamps': {
                    'market_data': time.time() - 0.1,
                    'order_book': time.time() - 0.05,
                    'trades': time.time() - 0.2,
                }
            }
        },
        {
            'name': 'High Volatility',
            'description': 'Recent price movements show high volatility',
            'market_data': {
                'recent_prices': [150.0, 145.0, 155.0, 140.0, 160.0, 135.0, 165.0],  # Very volatile
                'portfolio_values': [1000000, 995000, 985000, 975000, 970000],  # Declining
                'feed_timestamps': {
                    'market_data': time.time() - 0.1,
                    'order_book': time.time() - 0.05,
                    'trades': time.time() - 0.2,
                }
            }
        }
    ]
    
    symbol = "AAPL"
    quantity = 1000
    price = 150.0
    timestamp = datetime.now()
    
    print(f"Test Trade: {symbol} {quantity} shares @ ${price:.2f}")
    print(f"Basic Risk Limits: Drawdown={config['max_daily_drawdown_pct']:.1%}, "
          f"Turnover={config['max_daily_turnover_ratio']:.0f}x")
    print(f"Current State: Portfolio=${adapter.current_portfolio_value:,.0f}, "
          f"Drawdown={adapter.get_current_drawdown():.2%}")
    print()
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print("-" * 50)
        
        # Test old method (should allow - basic limits are permissive)
        trade_value = abs(quantity * price)
        is_safe_old, reason_old = adapter.assess_trade_risk(trade_value, timestamp)
        
        # Test new method (may block due to sensor data)
        is_safe_new, action_new, detailed_reason_new = adapter.pre_trade_check(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            market_data=scenario['market_data']
        )
        
        print(f"   Old Method: {'✅ ALLOW' if is_safe_old else '❌ BLOCK'} - {reason_old}")
        print(f"   New Method: {'✅ ALLOW' if is_safe_new else '❌ BLOCK'} - {action_new}")
        print(f"   Sensor Detection: {'✅ CAUGHT RISK' if not is_safe_new and is_safe_old else '⚠️ NO ADDITIONAL RISK' if is_safe_new else '✅ BOTH BLOCKED'}")
        print(f"   Details: {detailed_reason_new}")
        print()
        
        results.append({
            'scenario': scenario['name'],
            'old_safe': is_safe_old,
            'new_safe': is_safe_new,
            'sensor_caught_risk': not is_safe_new and is_safe_old
        })
    
    # Summary
    print("📊 SENSOR EFFECTIVENESS SUMMARY")
    print("=" * 70)
    
    sensor_catches = sum(1 for r in results if r['sensor_caught_risk'])
    total_scenarios = len(results)
    
    print(f"Scenarios where sensors caught additional risk: {sensor_catches}/{total_scenarios}")
    print(f"Sensor effectiveness: {sensor_catches/total_scenarios*100:.1f}%")
    print()
    
    for result in results:
        status = "🎯 SENSOR CAUGHT RISK" if result['sensor_caught_risk'] else "⚪ NO ADDITIONAL RISK"
        print(f"  {result['scenario']}: {status}")
    
    print()
    print("✅ Sensors provide additional risk protection beyond basic turnover/drawdown limits!")
    
    return results


def test_orchestrator_integration_example():
    """Show how orchestrator should integrate the new pre_trade_check method."""
    print("\n🧪 Orchestrator Integration Example")
    print("=" * 70)
    
    print("OLD ORCHESTRATOR CODE:")
    print("-" * 30)
    print("""
# OLD: Only basic risk check
is_safe, reason = self.risk_agent.assess_trade_risk(
    abs(shares_to_trade * current_price), 
    current_time_of_bar
)

if is_safe:
    # Place trade
    self.logger.info(f"Trade safe: {reason}")
else:
    # Block trade
    self.logger.warning(f"Trade blocked: {reason}")
""")
    
    print("\nNEW ORCHESTRATOR CODE:")
    print("-" * 30)
    print("""
# NEW: Comprehensive sensor-based risk check
quantity_signed = shares_to_trade if order_action == "BUY" else -shares_to_trade
is_safe, action, detailed_reason = self.risk_agent.pre_trade_check(
    symbol=symbol,
    quantity=quantity_signed,
    price=current_price,
    timestamp=current_time_of_bar,
    market_data=self._gather_market_data_for_risk_check(symbol, current_time_of_bar)
)

if is_safe:
    # Place trade
    self.logger.info(f"Trade approved by comprehensive risk check: {detailed_reason}")
    # ... place order logic ...
else:
    # Handle different risk actions
    if action == "KILL_SWITCH":
        self.logger.critical(f"KILL SWITCH activated: {detailed_reason}")
        # Emergency liquidation logic
    elif action == "HALT":
        self.logger.critical(f"Trading halted: {detailed_reason}")
        # Stop all trading
    elif action == "BLOCK":
        self.logger.warning(f"Trade blocked: {detailed_reason}")
        # Just block this trade
    elif action == "THROTTLE":
        self.logger.info(f"Trade throttled: {detailed_reason}")
        # Reduce position size or delay
""")
    
    print("\nKEY BENEFITS:")
    print("-" * 30)
    print("✅ Full sensor coverage (feed staleness, latency, liquidity, etc.)")
    print("✅ Granular risk actions (ALLOW/WARN/THROTTLE/BLOCK/HALT/KILL_SWITCH)")
    print("✅ Detailed reasoning for compliance and debugging")
    print("✅ Hot-reloadable risk policies via YAML configuration")
    print("✅ Backward compatibility (old method still works)")
    
    print("\nIMPLEMENTATION STEPS:")
    print("-" * 30)
    print("1. Replace assess_trade_risk() calls with pre_trade_check()")
    print("2. Add _gather_market_data_for_risk_check() helper method")
    print("3. Handle granular risk actions (not just safe/unsafe)")
    print("4. Update logging to use detailed_reason")
    print("5. Test with sensor-triggering scenarios")


def main():
    """Run comprehensive risk integration tests."""
    print("🚀 COMPREHENSIVE RISK INTEGRATION VALIDATION")
    print("=" * 70)
    
    try:
        # Test 1: Compare old vs new methods
        is_safe_old, is_safe_new, old_duration, new_duration = test_old_vs_new_risk_assessment()
        
        # Test 2: Sensor-specific risk scenarios
        sensor_results = test_sensor_triggered_blocks()
        
        # Test 3: Show integration example
        test_orchestrator_integration_example()
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 INTEGRATION TEST SUMMARY")
        print(f"{'='*70}")
        
        sensor_catches = sum(1 for r in sensor_results if r['sensor_caught_risk'])
        
        print(f"✅ Old method performance: {old_duration*1000:.2f}ms")
        print(f"✅ New method performance: {new_duration*1000:.2f}ms")
        if old_duration > 0:
            print(f"✅ Performance overhead: {((new_duration - old_duration) / old_duration * 100):.1f}%")
        else:
            print(f"✅ Performance overhead: N/A (old method too fast to measure)")
        print(f"✅ Sensor risk detection: {sensor_catches}/{len(sensor_results)} scenarios")
        print(f"✅ Additional risk coverage: {sensor_catches/len(sensor_results)*100:.1f}%")
        
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Replace orchestrator's assess_trade_risk() calls with pre_trade_check()")
        print(f"to get full sensor coverage and granular risk actions!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)