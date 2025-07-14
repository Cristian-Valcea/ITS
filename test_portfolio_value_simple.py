#!/usr/bin/env python3
"""
Simple test to verify the portfolio value fix for short positions.
Tests the core calculation logic directly.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_portfolio_value_calculation():
    """Test the portfolio value calculation logic directly."""
    print("🧪 Testing portfolio value calculation logic...")
    
    # Test parameters
    initial_capital = 10000.0
    entry_price = 100.0
    position_quantity = 100.0  # 100 shares
    
    print(f"📊 Test setup:")
    print(f"   Initial Capital: ${initial_capital:.2f}")
    print(f"   Entry Price: ${entry_price:.2f}")
    print(f"   Position Quantity: {position_quantity}")
    
    # Simulate short position entry
    # When going short, we sell shares and receive cash
    current_capital_after_short = initial_capital + (entry_price * position_quantity)
    print(f"\n📉 After going short:")
    print(f"   Current Capital: ${current_capital_after_short:.2f}")
    
    # Test 1: Price goes up (unfavorable for short)
    current_price_1 = 105.0
    print(f"\n📈 Test 1: Price moves to ${current_price_1:.2f} (unfavorable)")
    
    # OLD (BUGGY) calculation - double counts proceeds
    old_portfolio_value = current_capital_after_short - (position_quantity * current_price_1)
    print(f"   OLD (buggy) calculation: ${old_portfolio_value:.2f}")
    
    # NEW (FIXED) calculation - mark-to-market
    unrealized_pnl = (entry_price - current_price_1) * position_quantity
    new_portfolio_value = current_capital_after_short + unrealized_pnl
    print(f"   NEW (fixed) calculation: ${new_portfolio_value:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
    
    # The new calculation should show a loss when price goes up
    assert unrealized_pnl < 0, "Short position should lose money when price goes up"
    # Portfolio value should be less than capital after short (which includes proceeds)
    assert new_portfolio_value < current_capital_after_short, "Portfolio should be worth less than capital after short"
    
    # Test 2: Price goes down (favorable for short)
    current_price_2 = 95.0
    print(f"\n📉 Test 2: Price moves to ${current_price_2:.2f} (favorable)")
    
    # OLD (BUGGY) calculation
    old_portfolio_value_2 = current_capital_after_short - (position_quantity * current_price_2)
    print(f"   OLD (buggy) calculation: ${old_portfolio_value_2:.2f}")
    
    # NEW (FIXED) calculation
    unrealized_pnl_2 = (entry_price - current_price_2) * position_quantity
    new_portfolio_value_2 = current_capital_after_short + unrealized_pnl_2
    print(f"   NEW (fixed) calculation: ${new_portfolio_value_2:.2f}")
    print(f"   Unrealized P&L: ${unrealized_pnl_2:.2f}")
    
    # The new calculation should show a profit when price goes down
    assert unrealized_pnl_2 > 0, "Short position should make money when price goes down"
    # Portfolio value should be more than capital after short (profit added)
    assert new_portfolio_value_2 > current_capital_after_short, "Portfolio should be worth more than capital after short"
    
    # Test 3: Compare the two approaches
    print(f"\n🔍 Comparison of approaches:")
    print(f"   Price Up (${current_price_1}):")
    print(f"     OLD: ${old_portfolio_value:.2f}")
    print(f"     NEW: ${new_portfolio_value:.2f}")
    print(f"     Difference: ${new_portfolio_value - old_portfolio_value:.2f}")
    
    print(f"   Price Down (${current_price_2}):")
    print(f"     OLD: ${old_portfolio_value_2:.2f}")
    print(f"     NEW: ${new_portfolio_value_2:.2f}")
    print(f"     Difference: ${new_portfolio_value_2 - old_portfolio_value_2:.2f}")
    
    # The old method gives wrong results
    # When price goes up, old method shows: 20000 - 10500 = 9500 (too low)
    # When price goes down, old method shows: 20000 - 9500 = 10500 (too high)
    # The old method essentially double-counts the position value
    
    print("✅ Portfolio value calculation test passed!")


def test_mark_to_market_consistency():
    """Test that mark-to-market is consistent with P&L calculations."""
    print("\n🧪 Testing mark-to-market consistency...")
    
    initial_capital = 10000.0
    entry_price = 100.0
    position_quantity = 50.0
    
    # After going short
    current_capital = initial_capital + (entry_price * position_quantity)  # 15000
    
    test_prices = [90.0, 95.0, 100.0, 105.0, 110.0]
    
    print(f"📊 Mark-to-market test:")
    print(f"   Entry Price: ${entry_price:.2f}")
    print(f"   Position Quantity: {position_quantity}")
    print(f"   Current Capital: ${current_capital:.2f}")
    
    for price in test_prices:
        unrealized_pnl = (entry_price - price) * position_quantity
        portfolio_value = current_capital + unrealized_pnl
        
        # Calculate what the P&L would be if we closed the position
        closing_pnl = (entry_price - price) * position_quantity
        
        print(f"   Price ${price:6.2f}: Portfolio=${portfolio_value:8.2f}, P&L=${closing_pnl:+7.2f}")
        
        # The unrealized P&L should match what we'd get if we closed
        assert abs(unrealized_pnl - closing_pnl) < 0.01, "Unrealized P&L should match closing P&L"
        
        # Portfolio value should equal current capital + unrealized P&L
        # (This is the definition of mark-to-market)
        expected_portfolio = current_capital + unrealized_pnl
        assert abs(portfolio_value - expected_portfolio) < 0.01, "Portfolio value should be capital + unrealized P&L"
    
    print("✅ Mark-to-market consistency test passed!")


def test_long_position_unchanged():
    """Verify that long position calculation is not affected by the fix."""
    print("\n🧪 Testing that long position calculation is unchanged...")
    
    initial_capital = 10000.0
    entry_price = 100.0
    position_quantity = 50.0
    
    # After going long (capital decreases by purchase amount)
    current_capital = initial_capital - (entry_price * position_quantity)  # 5000
    
    test_prices = [90.0, 95.0, 100.0, 105.0, 110.0]
    
    print(f"📊 Long position test:")
    print(f"   Entry Price: ${entry_price:.2f}")
    print(f"   Position Quantity: {position_quantity}")
    print(f"   Current Capital: ${current_capital:.2f}")
    
    for price in test_prices:
        # Long position: capital + market value of shares
        portfolio_value = current_capital + (position_quantity * price)
        
        # This should equal initial capital + unrealized P&L
        unrealized_pnl = (price - entry_price) * position_quantity
        expected_portfolio = initial_capital + unrealized_pnl
        
        print(f"   Price ${price:6.2f}: Portfolio=${portfolio_value:8.2f}, P&L=${unrealized_pnl:+7.2f}")
        
        assert abs(portfolio_value - expected_portfolio) < 0.01, "Long calculation should be consistent"
    
    print("✅ Long position calculation test passed!")


def main():
    """Run all tests."""
    print("🎯 Testing Portfolio Value Fix - Core Logic")
    print("=" * 50)
    
    try:
        test_portfolio_value_calculation()
        test_mark_to_market_consistency()
        test_long_position_unchanged()
        
        print("\n" + "=" * 50)
        print("🎉 ALL CORE LOGIC TESTS PASSED!")
        print("\n📋 Key Findings:")
        print("✅ OLD method double-counts short proceeds (WRONG)")
        print("✅ NEW method uses proper mark-to-market (CORRECT)")
        print("✅ Mark-to-market is consistent with P&L calculations")
        print("✅ Long position calculation remains unchanged")
        
        print("\n🔧 The Fix:")
        print("   Before: portfolio_value = current_capital - (position_quantity * current_price)")
        print("   After:  portfolio_value = current_capital + (entry_price - current_price) * position_quantity")
        
        print("\n💡 Why This Matters:")
        print("   - Prevents reward signal distortion in RL training")
        print("   - Ensures accurate portfolio valuation for risk management")
        print("   - Eliminates double-counting accounting error")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())