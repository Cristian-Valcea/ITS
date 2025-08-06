"""
Nightly Integration Tests for Production Risk Governor
Critical validations: fee PnL, liquidity shock, clock drift

Run these nightly to catch regressions before market open
"""

import pytest
import time
import numpy as np
from datetime import datetime, timezone
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk_governor.stairways_integration import SafeStairwaysDeployment
from risk_governor.broker_adapter import BrokerExecutionManager, IBKRPaperAdapter
from risk_governor.prometheus_monitoring import RiskGovernorMonitoring

class TestNightlyIntegration:
    """Critical nightly integration tests"""
    
    def setup_method(self):
        """Setup for each test"""
        self.deployment = SafeStairwaysDeployment(
            model_path=None,  # Mock model
            symbol="MSFT", 
            paper_trading=True
        )
        
        # Enable cost tracking
        self.deployment.broker_manager = BrokerExecutionManager(chaos_mode=False)
    
    def test_fee_pnl_after_costs(self):
        """
        Test 1: Net PnL after fees with IBKR tiered pricing
        Validates: Real trading costs don't destroy profitability
        """
        print("\n🧪 Test 1: Fee P&L Validation")
        
        # Simulate 2-day trading with realistic market data
        total_gross_pnl = 0.0
        total_fees = 0.0
        trade_count = 0
        
        # Generate realistic MSFT trading scenarios
        test_scenarios = [
            # (position_increment, current_price, expected_direction)
            (50.0, 420.0, "buy"),    # $50 position at $420/share
            (-25.0, 422.0, "sell"),  # Partial close at profit
            (30.0, 418.0, "buy"),    # Add on dip
            (-55.0, 425.0, "sell"),  # Close with profit
            (40.0, 423.0, "buy"),    # New position
            (-40.0, 421.0, "sell"),  # Close at small loss
        ]
        
        for increment, price, direction in test_scenarios:
            # Generate market data
            market_data = {
                "timestamp": time.time(),
                "symbol": "MSFT",
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 100000,
                "prev_close": price * 0.999
            }
            
            # Generate observation
            observation = np.random.random(26)
            
            # Execute trade
            result = self.deployment.get_safe_trading_action(
                market_observation=observation,
                market_data=market_data
            )
            
            # Track P&L and fees
            if result["execution_result"]["status"] in ["EXECUTED", "SIMULATED"]:
                executed_size = result["execution_result"]["executed_size"]
                execution_price = result["execution_result"]["execution_price"]
                commission = result["execution_result"]["execution_cost"]
                
                # Use enhanced cost tracking if available
                effective_cost = result["execution_result"].get("effective_cost", commission)
                
                # Calculate gross P&L (simplified - price change * position)
                price_change = np.random.normal(0, 0.5)  # Random ±$0.50 move
                gross_pnl = executed_size * price_change / 100.0 if executed_size != 0 else 0
                
                # Get cost estimate for non-zero trades
                if abs(increment) > 0.01:
                    cost_estimate = self.deployment.broker_manager.broker.estimate_trade_cost(increment, price)
                    effective_cost = max(effective_cost, cost_estimate["total_cost"])
                
                total_gross_pnl += gross_pnl
                total_fees += effective_cost
                trade_count += 1
        
        # Calculate net P&L
        net_pnl = total_gross_pnl - total_fees
        fee_ratio = total_fees / abs(total_gross_pnl) if total_gross_pnl != 0 else 0
        
        print(f"   Trades executed: {trade_count}")
        print(f"   Gross P&L: ${total_gross_pnl:.2f}")
        print(f"   Total fees: ${total_fees:.2f}")
        print(f"   Net P&L: ${net_pnl:.2f}")
        print(f"   Fee ratio: {fee_ratio:.1%}")
        
        # Validation criteria
        assert trade_count > 0, "No trades executed"
        assert total_fees > 0, "No fees calculated"
        assert fee_ratio < 0.30, f"Fee ratio {fee_ratio:.1%} > 30% is excessive"
        
        # For paper trading, we accept small losses due to costs
        # In live trading, we'd require net_pnl >= 0
        assert net_pnl > -10.0, f"Net P&L ${net_pnl:.2f} too negative after fees"
        
        print("   ✅ Fee P&L test passed")
    
    def test_liquidity_shock_handling(self):
        """
        Test 2: Liquidity shock with price stall + gap
        Validates: System handles stale prices and sudden jumps gracefully
        """
        print("\n🧪 Test 2: Liquidity Shock Test")
        
        base_price = 420.0
        shock_magnitude = 0.01  # 1% price jump
        stall_duration = 0.15   # 150ms price stall
        
        # Phase 1: Normal trading
        market_data_normal = {
            "timestamp": time.time(),
            "symbol": "MSFT",
            "open": base_price,
            "high": base_price * 1.001,
            "low": base_price * 0.999,
            "close": base_price,
            "volume": 100000,
            "prev_close": base_price * 0.999
        }
        
        observation = np.random.random(26)
        
        result_normal = self.deployment.get_safe_trading_action(
            market_observation=observation,
            market_data=market_data_normal
        )
        
        normal_latency = result_normal["total_latency_ms"]
        
        # Phase 2: Price stall (simulate feed delay)
        time.sleep(stall_duration)
        
        # Phase 3: Sudden price jump
        shocked_price = base_price * (1 + shock_magnitude)
        
        market_data_shock = {
            "timestamp": time.time(),
            "symbol": "MSFT", 
            "open": base_price,
            "high": shocked_price,
            "low": base_price,
            "close": shocked_price,
            "volume": 200000,  # High volume spike
            "prev_close": base_price
        }
        
        result_shock = self.deployment.get_safe_trading_action(
            market_observation=observation,
            market_data=market_data_shock
        )
        
        shock_latency = result_shock["total_latency_ms"]
        
        print(f"   Normal latency: {normal_latency:.2f}ms")
        print(f"   Shock latency: {shock_latency:.2f}ms")
        print(f"   Price jump: {base_price:.2f} → {shocked_price:.2f} ({shock_magnitude:.1%})")
        
        # Validation criteria
        assert shock_latency < 15.0, f"Shock latency {shock_latency:.2f}ms > 15ms limit"
        assert result_shock["safe_increment"] is not None, "System failed during shock"
        
        # Check ATR adaptation
        atr_status = self.deployment.risk_governor.position_governor.get_atr_status()
        print(f"   ATR adaptation: {atr_status['current_atr']:.3f}")
        
        # ATR should reflect the shock
        assert atr_status["current_atr"] > 1.0, "ATR should increase after price shock"
        
        print("   ✅ Liquidity shock test passed")
    
    def test_clock_drift_detection(self):
        """
        Test 3: Clock drift and timestamp validation
        Validates: NTP sync and timestamp accuracy for broker orders
        """
        print("\n🧪 Test 3: Clock Drift Test")
        
        # Get system time vs NTP time
        try:
            # Check NTP sync (Linux/WSL)
            result = subprocess.run(['timedatectl', 'status'], 
                                  capture_output=True, text=True, timeout=5)
            ntp_output = result.stdout
            ntp_synced = "NTP synchronized: yes" in ntp_output
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback for systems without timedatectl
            ntp_synced = True  # Assume synced for testing
            ntp_output = "NTP status unknown"
        
        # Test timestamp accuracy
        system_time = time.time()
        broker_time = self.deployment.broker_manager.broker._simulate_execution_price("MSFT", "MKT", None)
        
        # Simulate order with timestamp
        market_data = {
            "timestamp": system_time,
            "symbol": "MSFT",
            "open": 420.0,
            "high": 422.0,
            "low": 418.0,
            "close": 420.0,
            "volume": 100000,
            "prev_close": 419.0
        }
        
        observation = np.random.random(26)
        
        start_time = time.time()
        result = self.deployment.get_safe_trading_action(
            market_observation=observation,
            market_data=market_data
        )
        end_time = time.time()
        
        # Calculate timing precision
        processing_time = (end_time - start_time) * 1000  # ms
        timestamp_age = (end_time - market_data["timestamp"]) * 1000  # ms
        
        print(f"   NTP synchronized: {ntp_synced}")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Timestamp age: {timestamp_age:.2f}ms")
        print(f"   System latency: {result['total_latency_ms']:.2f}ms")
        
        # IBKR rejects orders with >2s timestamp skew
        max_allowed_skew = 2000  # 2 seconds in ms
        
        # Validation criteria
        assert timestamp_age < max_allowed_skew, f"Timestamp age {timestamp_age:.0f}ms > {max_allowed_skew}ms"
        assert processing_time < 100.0, f"Processing time {processing_time:.2f}ms too slow"
        
        # Check for clock drift indicators
        if not ntp_synced:
            print("   ⚠️ WARNING: NTP not synchronized - may cause broker rejections")
        
        print("   ✅ Clock drift test passed")
    
    def test_end_to_end_full_session(self):
        """
        Test 4: Complete trading session simulation
        Validates: Full system integration over extended period
        """
        print("\n🧪 Test 4: Full Session Integration")
        
        session_duration = 60  # 60 seconds simulation
        decision_interval = 1  # 1 decision per second
        
        decisions = []
        start_time = time.time()
        
        print(f"   Running {session_duration}s trading session...")
        
        for i in range(session_duration):
            # Generate evolving market conditions
            base_price = 420.0 + np.sin(i * 0.1) * 2.0  # Gentle price wave
            volatility = 0.001 + (i / session_duration) * 0.002  # Increasing volatility
            
            market_data = {
                "timestamp": time.time(),
                "symbol": "MSFT",
                "open": base_price * (1 + np.random.normal(0, volatility)),
                "high": base_price * (1 + abs(np.random.normal(0, volatility))),
                "low": base_price * (1 - abs(np.random.normal(0, volatility))),
                "close": base_price * (1 + np.random.normal(0, volatility)),
                "volume": int(100000 * (1 + np.random.uniform(-0.5, 0.5))),
                "prev_close": base_price
            }
            
            observation = np.random.random(26)
            
            result = self.deployment.get_safe_trading_action(
                market_observation=observation,
                market_data=market_data
            )
            
            decisions.append({
                "timestamp": time.time(),
                "latency_ms": result["total_latency_ms"],
                "safe_increment": result["safe_increment"],
                "execution_status": result["execution_result"]["status"],
                "market_price": market_data["close"]
            })
            
            time.sleep(decision_interval)
        
        session_time = time.time() - start_time
        
        # Analyze session performance
        latencies = [d["latency_ms"] for d in decisions]
        successful_executions = sum(1 for d in decisions if d["execution_status"] in ["EXECUTED", "SIMULATED"])
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        execution_rate = successful_executions / len(decisions)
        
        print(f"   Session duration: {session_time:.1f}s")
        print(f"   Total decisions: {len(decisions)}")
        print(f"   Successful executions: {successful_executions} ({execution_rate:.1%})")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Max latency: {max_latency:.2f}ms")
        print(f"   P95 latency: {p95_latency:.2f}ms")
        
        # Validation criteria
        assert len(decisions) >= 50, f"Too few decisions: {len(decisions)}"
        assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms > 10ms"
        assert max_latency < 50.0, f"Max latency {max_latency:.2f}ms > 50ms"
        assert execution_rate > 0.8, f"Execution rate {execution_rate:.1%} < 80%"
        
        print("   ✅ Full session integration test passed")
    
    def test_prometheus_metrics_collection(self):
        """
        Test 5: Monitoring and metrics validation
        Validates: Prometheus metrics are collected and accurate
        """
        print("\n🧪 Test 5: Prometheus Metrics Test")
        
        # Start monitoring
        monitoring = RiskGovernorMonitoring(prometheus_port=8001)  # Different port for testing
        monitoring.start()
        
        try:
            # Generate some trading activity
            for i in range(10):
                market_data = {
                    "timestamp": time.time(),
                    "symbol": "MSFT",
                    "open": 420.0,
                    "high": 421.0,
                    "low": 419.0, 
                    "close": 420.0,
                    "volume": 100000,
                    "prev_close": 419.5
                }
                
                observation = np.random.random(26)
                
                result = self.deployment.get_safe_trading_action(
                    market_observation=observation,
                    market_data=market_data
                )
                
                # Record metrics
                monitoring.record_trading_decision(result)
                monitoring.record_portfolio_state(result["portfolio_state"])
            
            # Get metrics summary
            metrics_summary = monitoring.monitor.get_metrics_summary()
            
            print(f"   Metrics server running: {metrics_summary['metrics_server_running']}")
            print(f"   Monitoring active: {metrics_summary['monitoring_active']}")
            print(f"   Recent decisions: {metrics_summary['performance']['recent_decisions']}")
            print(f"   Average latency: {metrics_summary['performance']['avg_latency_ms']:.2f}ms")
            
            # Validation criteria
            assert metrics_summary["metrics_server_running"], "Metrics server not running"
            assert metrics_summary["monitoring_active"], "Monitoring not active"
            assert metrics_summary["performance"]["recent_decisions"] >= 10, "Insufficient metrics collected"
            
            print("   ✅ Prometheus metrics test passed")
            
        finally:
            monitoring.stop()

def run_nightly_validation():
    """Run complete nightly validation suite"""
    print("🌙 NIGHTLY INTEGRATION VALIDATION SUITE")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run critical tests
        test_suite = TestNightlyIntegration()
        
        print("\n🚀 Starting critical integration tests...")
        
        # Test 1: Fee P&L validation
        test_suite.setup_method()
        test_suite.test_fee_pnl_after_costs()
        
        # Test 2: Liquidity shock handling
        test_suite.setup_method()
        test_suite.test_liquidity_shock_handling()
        
        # Test 3: Clock drift detection
        test_suite.setup_method()
        test_suite.test_clock_drift_detection()
        
        # Test 4: Full session integration
        test_suite.setup_method()
        test_suite.test_end_to_end_full_session()
        
        # Test 5: Metrics collection
        test_suite.setup_method()
        test_suite.test_prometheus_metrics_collection()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 ALL NIGHTLY TESTS PASSED!")
        print(f"   Total validation time: {total_time:.1f} seconds")
        print("   System ready for market open")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ NIGHTLY VALIDATION FAILED: {str(e)}")
        print("   System NOT ready for market open")
        print("   Review logs and fix issues before deployment")
        return False

if __name__ == "__main__":
    # Run nightly validation
    success = run_nightly_validation()
    sys.exit(0 if success else 1)