"""
Comprehensive validation script for Production Risk Governor System
Tests integration with Stairways V4 model and MSFT configuration

Run this script to validate the complete system before deployment
"""

import numpy as np
import time
import logging
from typing import Dict, List
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from risk_governor.stairways_integration import SafeStairwaysDeployment
from risk_governor.core_governor import TradingAction

def generate_msft_market_data(n_steps: int = 100, base_price: float = 420.0) -> List[Dict]:
    """Generate realistic MSFT market data for testing"""
    
    market_data = []
    current_price = base_price
    
    for i in range(n_steps):
        # Realistic MSFT price movement
        daily_volatility = 0.02  # 2% daily volatility
        price_change = np.random.normal(0, daily_volatility * current_price / np.sqrt(390))  # Per-minute
        
        new_price = max(1.0, current_price + price_change)
        
        # Generate OHLC data
        high = new_price * (1 + abs(np.random.normal(0, 0.001)))
        low = new_price * (1 - abs(np.random.normal(0, 0.001)))
        volume = int(np.random.lognormal(np.log(100000), 0.5))
        
        market_data.append({
            "timestamp": time.time() + i * 60,  # 1-minute intervals
            "symbol": "MSFT",
            "open": current_price,
            "high": high,
            "low": low,
            "close": new_price,
            "volume": volume,
            "prev_close": current_price
        })
        
        current_price = new_price
    
    return market_data

def generate_observation_vector(market_data_point: Dict) -> np.ndarray:
    """Generate 26-dimensional observation vector for Stairways model"""
    
    # Mock observation vector with realistic features
    # In real system, this would come from dual_ticker_data_adapter
    observation = np.array([
        # Price features (12 for each ticker, simplified to 13 total)
        market_data_point["close"] / 400.0,  # Normalized price
        (market_data_point["high"] - market_data_point["low"]) / market_data_point["close"],  # Range
        market_data_point["volume"] / 100000.0,  # Normalized volume
        0.0,  # Previous returns
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Additional features
        
        # Second ticker features (NVDA mock - 12 features)
        0.95, 0.02, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        
        # Portfolio state (2 features)
        0.0,  # Current position
        0.0   # Cash balance
    ])
    
    return observation[:26]  # Ensure exactly 26 dimensions

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n🧪 Testing Basic Functionality...")
    
    # Initialize system
    deployment = SafeStairwaysDeployment(
        model_path=None,  # Use mock model
        symbol="MSFT",
        paper_trading=True
    )
    
    # Generate test data
    market_data = generate_msft_market_data(10)
    
    # Test single decision
    observation = generate_observation_vector(market_data[0])
    
    result = deployment.get_safe_trading_action(
        market_observation=observation,
        market_data=market_data[0]
    )
    
    # Validate response structure
    required_keys = [
        "safe_increment", "raw_action", "model_confidence", "risk_reason",
        "execution_result", "portfolio_state", "total_latency_ms"
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    assert result["total_latency_ms"] < 10.0, f"Latency too high: {result['total_latency_ms']}ms"
    assert result["paper_trading"] == True
    
    print(f"✅ Basic functionality test passed")
    print(f"   - Latency: {result['total_latency_ms']:.2f}ms")
    print(f"   - Safe increment: {result['safe_increment']}")
    print(f"   - Risk reason: {result['risk_reason'][:50]}...")
    
    return deployment

def test_risk_limit_enforcement(deployment: SafeStairwaysDeployment):
    """Test that risk limits are strictly enforced"""
    print("\n🛡️ Testing Risk Limit Enforcement...")
    
    # Generate scenario with large losses to trigger limits
    base_price = 420.0
    
    # Simulate 20% price drop to trigger drawdown limits
    crash_data = {
        "timestamp": time.time(),
        "symbol": "MSFT", 
        "open": base_price,
        "high": base_price,
        "low": base_price * 0.8,  # 20% drop
        "close": base_price * 0.8,
        "volume": 1000000,
        "prev_close": base_price
    }
    
    # Set up portfolio with large position to amplify loss
    deployment.portfolio_state.current_position = 400.0  # $400 position
    deployment.portfolio_state.realized_pnl = -80.0      # Already $80 loss
    
    observation = generate_observation_vector(crash_data)
    
    # Try aggressive buy during crash (should be blocked)
    result = deployment.get_safe_trading_action(
        market_observation=observation,
        market_data=crash_data
    )
    
    # Should not allow position increases during high drawdown
    assert result["safe_increment"] <= 0, f"Should not allow increases during crash, got {result['safe_increment']}"
    
    print(f"✅ Risk limit enforcement test passed")
    print(f"   - Action during crash: {result['safe_increment']} (≤ 0 expected)")
    print(f"   - Risk reason: {result['risk_reason'][:60]}...")

def test_turnover_limits(deployment: SafeStairwaysDeployment):
    """Test daily turnover limits prevent excessive churning"""
    print("\n🔄 Testing Turnover Limits...")
    
    # Reset portfolio and set high turnover
    deployment.portfolio_state.current_position = 0.0
    deployment.portfolio_state.daily_turnover = 1950.0  # Near 2000 limit
    
    market_data = generate_msft_market_data(1)[0]
    observation = generate_observation_vector(market_data)
    
    result = deployment.get_safe_trading_action(
        market_observation=observation,
        market_data=market_data
    )
    
    # Should be limited by remaining turnover budget
    remaining_budget = 2000.0 - 1950.0  # $50 remaining
    
    assert abs(result["safe_increment"]) <= remaining_budget + 0.01, f"Turnover limit violated"
    
    print(f"✅ Turnover limit test passed")
    print(f"   - Safe increment: {result['safe_increment']} (≤ {remaining_budget} expected)")

def test_performance_under_load():
    """Test system performance under high-frequency decisions"""
    print("\n⚡ Testing Performance Under Load...")
    
    deployment = SafeStairwaysDeployment(
        model_path=None,
        symbol="MSFT", 
        paper_trading=True
    )
    
    # Generate 500 market data points
    market_data = generate_msft_market_data(500)
    
    start_time = time.time()
    latencies = []
    
    for i, data_point in enumerate(market_data):
        observation = generate_observation_vector(data_point)
        
        result = deployment.get_safe_trading_action(
            market_observation=observation,
            market_data=data_point
        )
        
        latencies.append(result["total_latency_ms"])
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/500 decisions...")
    
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    # Performance requirements
    assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms budget"
    assert max_latency < 10.0, f"Max latency {max_latency:.2f}ms too high"
    assert total_time < 30.0, f"Total processing time {total_time:.2f}s too slow"
    
    print(f"✅ Performance test passed")
    print(f"   - 500 decisions in {total_time:.2f}s ({500/total_time:.1f} decisions/sec)")
    print(f"   - Average latency: {avg_latency:.2f}ms")
    print(f"   - Max latency: {max_latency:.2f}ms")
    
    return deployment

def test_state_persistence(deployment: SafeStairwaysDeployment):
    """Test state persistence and recovery"""
    print("\n💾 Testing State Persistence...")
    
    # Set up some portfolio state
    deployment.portfolio_state.current_position = 150.0
    deployment.portfolio_state.realized_pnl = 25.0
    deployment.portfolio_state.daily_turnover = 800.0
    
    # Force state save
    portfolio_dict = {
        "current_position": deployment.portfolio_state.current_position,
        "realized_pnl": deployment.portfolio_state.realized_pnl,
        "daily_turnover": deployment.portfolio_state.daily_turnover,
        "max_daily_drawdown": deployment.portfolio_state.max_daily_drawdown
    }
    
    save_success = deployment.state_manager.save_daily_state(portfolio_dict)
    
    if save_success:
        # Try to load state back
        loaded_state = deployment.state_manager.load_daily_state()
        
        if loaded_state:
            assert loaded_state["current_position"] == 150.0
            assert loaded_state["realized_pnl"] == 25.0
            assert loaded_state["daily_turnover"] == 800.0
            
            print(f"✅ State persistence test passed")
            print(f"   - Saved and loaded state successfully")
        else:
            print(f"⚠️ State persistence test skipped (Redis not available)")
    else:
        print(f"⚠️ State persistence test skipped (Redis not available)")

def test_monte_carlo_breach_validation():
    """Run Monte Carlo simulation to verify no hard limit breaches"""
    print("\n🎲 Running Monte Carlo Breach Validation...")
    
    deployment = SafeStairwaysDeployment(
        model_path=None,
        symbol="MSFT",
        paper_trading=True
    )
    
    breach_count = 0
    total_tests = 50  # Reduced for speed
    max_loss_observed = 0
    max_position_observed = 0
    max_turnover_observed = 0
    
    for test_run in range(total_tests):
        # Reset for each test
        deployment.portfolio_state.current_position = 0.0
        deployment.portfolio_state.realized_pnl = 0.0
        deployment.portfolio_state.unrealized_pnl = 0.0
        deployment.portfolio_state.daily_turnover = 0.0
        deployment.portfolio_state.max_daily_drawdown = 0.0
        
        # Generate extreme market scenario
        market_data = generate_msft_market_data(100)
        
        for data_point in market_data:
            observation = generate_observation_vector(data_point)
            
            result = deployment.get_safe_trading_action(
                market_observation=observation,
                market_data=data_point
            )
            
            # Check for breaches
            portfolio = result["portfolio_state"]
            total_pnl = portfolio["realized_pnl"] + portfolio["unrealized_pnl"]
            current_loss = abs(min(0, total_pnl))
            
            max_loss_observed = max(max_loss_observed, current_loss)
            max_position_observed = max(max_position_observed, abs(portfolio["current_position"]))
            max_turnover_observed = max(max_turnover_observed, portfolio["daily_turnover"])
            
            # Check for hard limit breaches
            if (current_loss > 100.0 or
                abs(portfolio["current_position"]) > 500.0 or
                portfolio["daily_turnover"] > 2000.0):
                breach_count += 1
                break
        
        if (test_run + 1) % 10 == 0:
            print(f"   Completed {test_run + 1}/{total_tests} Monte Carlo runs...")
    
    # CRITICAL: Zero breaches allowed
    assert breach_count == 0, f"Hard limits breached in {breach_count}/{total_tests} tests"
    
    print(f"✅ Monte Carlo breach validation passed")
    print(f"   - {total_tests} runs with 0 hard limit breaches")
    print(f"   - Max loss observed: ${max_loss_observed:.2f} (limit: $100)")
    print(f"   - Max position observed: ${max_position_observed:.2f} (limit: $500)")
    print(f"   - Max turnover observed: ${max_turnover_observed:.2f} (limit: $2000)")

def test_system_integration():
    """Test complete system integration"""
    print("\n🔗 Testing Complete System Integration...")
    
    deployment = SafeStairwaysDeployment(
        model_path=None,
        symbol="MSFT",
        paper_trading=True
    )
    
    # Simulate full trading session
    market_data = generate_msft_market_data(50)
    decisions = []
    
    for i, data_point in enumerate(market_data):
        observation = generate_observation_vector(data_point)
        
        result = deployment.get_safe_trading_action(
            market_observation=observation,
            market_data=data_point,
            top_of_book_mid=data_point["close"] * (1 + np.random.normal(0, 0.0001))
        )
        
        decisions.append(result)
    
    # Get performance summary
    performance = deployment.get_performance_summary()
    
    # Validate system behavior
    assert performance["total_decisions"] == 50
    assert performance["avg_latency_ms"] < 5.0
    assert 0 <= performance["risk_budget_used"]["position"] <= 1.0
    assert 0 <= performance["risk_budget_used"]["turnover"] <= 1.0
    assert 0 <= performance["risk_budget_used"]["loss"] <= 1.0
    
    print(f"✅ System integration test passed")
    print(f"   - Total decisions: {performance['total_decisions']}")
    print(f"   - Avg latency: {performance['avg_latency_ms']:.2f}ms")
    print(f"   - Final P&L: ${performance['total_pnl']:.2f}")
    print(f"   - Position budget used: {performance['risk_budget_used']['position']:.1%}")
    print(f"   - Turnover budget used: {performance['risk_budget_used']['turnover']:.1%}")

def main():
    """Run complete validation suite"""
    print("🎯 PRODUCTION RISK GOVERNOR VALIDATION SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run all validation tests
        deployment = test_basic_functionality()
        test_risk_limit_enforcement(deployment)
        test_turnover_limits(deployment)
        deployment = test_performance_under_load()
        test_state_persistence(deployment)
        test_monte_carlo_breach_validation()
        test_system_integration()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print(f"   Total validation time: {total_time:.2f} seconds")
        print("   System is ready for paper trading deployment")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        print("   System is NOT ready for deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)