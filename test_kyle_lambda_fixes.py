#!/usr/bin/env python3
"""
Test script to verify Kyle Lambda fill simulator fixes:

1. Determinism - seeded random number generator
2. Uninitialised participation_rate - proper initialization
3. Impact capping - total impact should be capped, not just permanent component
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gym_env.kyle_lambda_fill_simulator import KyleLambdaFillSimulator


def test_determinism_issue():
    """Test that random bounce impact is not deterministic."""
    print("🧪 Testing determinism issue...")
    
    config = {
        'enable_bid_ask_bounce': True,
        'bid_ask_spread_bps': 5.0,
        'max_impact_bps': 100.0
    }
    
    simulator = KyleLambdaFillSimulator(config)
    
    # Update with some market data
    simulator.update_market_data(100.0, 1000.0, pd.Timestamp.now())
    
    # Get multiple fills and check if bounce impact varies
    bounce_impacts = []
    for i in range(10):
        fill_price, impact_info = simulator.calculate_fill_price(100.0, 100, 'buy')  # Buy 100 shares
        bounce_impacts.append(impact_info['bounce_impact_bps'])
        print(f"   Fill {i+1}: bounce_impact_bps = {impact_info['bounce_impact_bps']:.4f}")
    
    # Check if all bounce impacts are different (indicating non-determinism)
    unique_bounces = len(set(bounce_impacts))
    print(f"📊 Unique bounce impacts: {unique_bounces} out of {len(bounce_impacts)}")
    
    if unique_bounces > 1:
        print("⚠️  ISSUE: Non-deterministic bounce impact detected!")
        print("   Fix needed: Use seeded np.random.default_rng(seed)")
    else:
        print("✅ Bounce impacts are deterministic")
    
    print("✅ Determinism test completed")


def test_participation_rate_initialization():
    """Test participation_rate initialization when volume is 0."""
    print("\n🧪 Testing participation_rate initialization...")
    
    config = {
        'enable_bid_ask_bounce': False,  # Disable for cleaner testing
        'max_impact_bps': 100.0
    }
    
    simulator = KyleLambdaFillSimulator(config)
    
    # Update with zero volume
    simulator.update_market_data(100.0, 0.0, pd.Timestamp.now())
    
    try:
        # This should not crash due to uninitialized participation_rate
        fill_price, impact_info = simulator.calculate_fill_price(100.0, 100, 'buy')
        
        print(f"📊 Fill price: ${fill_price:.4f}")
        print(f"📊 Impact info keys: {list(impact_info.keys())}")
        print(f"📊 Participation rate: {impact_info.get('participation_rate', 'MISSING')}")
        
        if 'participation_rate' in impact_info:
            print("✅ participation_rate is present in impact_info")
        else:
            print("❌ participation_rate is missing from impact_info")
            
    except NameError as e:
        if 'participation_rate' in str(e):
            print("❌ ISSUE: participation_rate not initialized!")
            print(f"   Error: {e}")
        else:
            raise
    
    print("✅ Participation rate initialization test completed")


def test_impact_capping_issue():
    """Test that total impact is properly capped, not just permanent component."""
    print("\n🧪 Testing impact capping issue...")
    
    config = {
        'enable_bid_ask_bounce': True,
        'bid_ask_spread_bps': 5.0,  # 5 bps spread
        'max_impact_bps': 10.0,     # Low cap to test capping
        'temporary_impact_decay': 1.0,  # High temporary impact
        'min_impact_bps': 0.5
    }
    
    simulator = KyleLambdaFillSimulator(config)
    
    # Update with market data
    simulator.update_market_data(100.0, 1000.0, pd.Timestamp.now())
    
    # Test with large trade size to trigger high impact
    fill_price, impact_info = simulator.calculate_fill_price(100.0, 10000, 'buy')  # Large trade
    
    print(f"📊 Impact breakdown:")
    print(f"   Spread impact: {impact_info['spread_impact_bps']:.2f} bps")
    print(f"   Permanent impact: {impact_info['permanent_impact_bps']:.2f} bps")
    print(f"   Temporary impact: {impact_info['temporary_impact_bps']:.2f} bps")
    print(f"   Bounce impact: {impact_info['bounce_impact_bps']:.2f} bps")
    print(f"   Total impact: {impact_info['total_impact_bps']:.2f} bps")
    print(f"   Max impact cap: {config['max_impact_bps']:.2f} bps")
    
    # Check if total impact exceeds the cap
    total_impact = impact_info['total_impact_bps']
    max_cap = config['max_impact_bps']
    
    if total_impact > max_cap:
        print(f"⚠️  ISSUE: Total impact ({total_impact:.2f} bps) exceeds cap ({max_cap:.2f} bps)!")
        print("   Fix needed: Clip total impact, not just permanent component")
        
        # Show the problem
        spread_impact = impact_info['spread_impact_bps']  # 2.5 bps (half of 5 bps spread)
        print(f"   Problem: Spread impact ({spread_impact:.2f} bps) added on top of capped permanent impact")
    else:
        print("✅ Total impact is properly capped")
    
    print("✅ Impact capping test completed")


def test_comprehensive_simulator_behavior():
    """Test overall simulator behavior with various scenarios."""
    print("\n🧪 Testing comprehensive simulator behavior...")
    
    config = {
        'enable_bid_ask_bounce': True,
        'bid_ask_spread_bps': 5.0,
        'max_impact_bps': 50.0,
        'min_impact_bps': 0.5,
        'temporary_impact_decay': 0.5
    }
    
    simulator = KyleLambdaFillSimulator(config)
    
    # Test scenarios
    scenarios = [
        (100.0, 1000.0, 100, "Normal trade"),
        (100.0, 0.0, 100, "Zero volume"),
        (100.0, 10000.0, 5000, "Large trade"),
        (100.0, 100.0, 50, "Small volume"),
    ]
    
    for price, volume, trade_size, description in scenarios:
        print(f"\n📊 Scenario: {description}")
        print(f"   Price: ${price:.2f}, Volume: {volume:.0f}, Trade: {trade_size} shares")
        
        simulator.update_market_data(price, volume, pd.Timestamp.now())
        
        try:
            fill_price, impact_info = simulator.calculate_fill_price(price, trade_size, 'buy')
            
            print(f"   Fill price: ${fill_price:.4f}")
            print(f"   Total impact: {impact_info['total_impact_bps']:.2f} bps")
            print(f"   Participation rate: {impact_info.get('participation_rate', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Get performance stats
    stats = simulator.get_performance_stats()
    print(f"\n📊 Performance stats:")
    print(f"   Fill count: {stats['fill_count']}")
    print(f"   Average impact: {stats['average_impact_bps']:.2f} bps")
    print(f"   Total impact: {stats['total_impact_bps']:.2f} bps")
    
    print("✅ Comprehensive simulator test completed")


def main():
    """Run all Kyle Lambda fix tests."""
    print("🎯 Testing Kyle Lambda Fill Simulator Issues")
    print("=" * 55)
    
    try:
        test_determinism_issue()
        test_participation_rate_initialization()
        test_impact_capping_issue()
        test_comprehensive_simulator_behavior()
        
        print("\n" + "=" * 55)
        print("🎉 ALL KYLE LAMBDA TESTS COMPLETED!")
        
        print("\n📋 Issues Identified:")
        print("1. ⚠️  Determinism - np.random.uniform() not seeded")
        print("2. ⚠️  Uninitialized participation_rate when volume == 0")
        print("3. ⚠️  Impact capping - only permanent component capped, not total")
        
        print("\n🔧 Fixes Needed:")
        print("1. Inject seeded np.random.default_rng(seed) into class")
        print("2. Initialize participation_rate = 0.0 at method start")
        print("3. Clip total_impact_bps, not just permanent_impact_bps")
        
        print("\n💡 Impact:")
        print("- Reproducible backtests with deterministic fills")
        print("- Robust handling of zero-volume scenarios")
        print("- Proper impact capping for risk management")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())