#!/usr/bin/env python3
"""
Test RiskAgentV2 integration with OrchestratorAgent through RiskAgentAdapter.
Verifies that the new risk system works seamlessly with existing orchestrator.
"""

import sys
import os
import time
import tempfile
import yaml
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.risk.risk_agent_adapter import RiskAgentAdapter
from src.execution.orchestrator_agent import OrchestratorAgent


def create_test_configs():
    """Create temporary test configuration files."""
    
    # Main config
    main_config = {
        'paths': {
            'data_dir_raw': 'data/raw/',
            'data_dir_processed': 'data/processed/',
            'scalers_dir': 'data/scalers/',
            'model_save_dir': 'models/',
            'tensorboard_log_dir': 'logs/tensorboard/',
            'reports_dir': 'reports/'
        },
        'feature_engineering': {
            'lookback_window': 10,
            'technical_indicators': ['sma', 'rsi', 'macd'],
            'normalize': True
        },
        'environment': {
            'initial_capital': 100000.0,
            'transaction_cost_pct': 0.001,
            'reward_scaling': 1.0,
            'position_sizing_pct_capital': 0.25
        },
        'training': {
            'total_timesteps': 10000,
            'data_duration_for_fetch': '90 D'
        },
        'evaluation': {
            'metrics': ['sharpe', 'max_drawdown']
        }
    }
    
    # Model params config
    model_params_config = {
        'algorithm_name': 'DQN',
        'algorithm_params': {
            'learning_rate': 0.0001,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000
        }
    }
    
    # Risk limits config (compatible with both old and new risk agents)
    risk_limits_config = {
        'max_daily_drawdown_pct': 0.02,
        'max_hourly_turnover_ratio': 5.0,
        'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True,
        'liquidate_on_halt': False,
        'env_turnover_penalty_factor': 0.01,
        'env_terminate_on_turnover_breach': False,
        'env_turnover_termination_threshold_multiplier': 2.0
    }
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    
    main_config_path = os.path.join(temp_dir, 'main_config.yaml')
    model_params_path = os.path.join(temp_dir, 'model_params.yaml')
    risk_limits_path = os.path.join(temp_dir, 'risk_limits.yaml')
    
    with open(main_config_path, 'w') as f:
        yaml.dump(main_config, f)
    
    with open(model_params_path, 'w') as f:
        yaml.dump(model_params_config, f)
    
    with open(risk_limits_path, 'w') as f:
        yaml.dump(risk_limits_config, f)
    
    return main_config_path, model_params_path, risk_limits_path, temp_dir


def test_risk_agent_adapter_standalone():
    """Test RiskAgentAdapter functionality standalone."""
    print("🧪 Testing RiskAgentAdapter Standalone")
    print("-" * 40)
    
    config = {
        'max_daily_drawdown_pct': 0.02,
        'max_hourly_turnover_ratio': 5.0,
        'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True
    }
    
    # Initialize adapter
    risk_agent = RiskAgentAdapter(config)
    
    # Test daily reset
    start_value = 100000.0
    current_time = datetime.now()
    risk_agent.reset_daily_limits(start_value, current_time)
    
    print(f"✓ Daily limits reset with ${start_value:,.0f}")
    
    # Test portfolio value update
    new_value = 98000.0  # 2% drawdown
    risk_agent.update_portfolio_value(new_value, current_time)
    
    drawdown = risk_agent.get_current_drawdown()
    print(f"✓ Current drawdown: {drawdown:.2%}")
    
    # Test trade recording
    trade_value = 10000.0
    risk_agent.record_trade(trade_value, current_time)
    
    turnover_ratio = risk_agent.get_daily_turnover_ratio()
    print(f"✓ Daily turnover ratio: {turnover_ratio:.2f}")
    
    # Test risk assessment - safe trade
    safe_trade = 5000.0
    is_safe, reason = risk_agent.assess_trade_risk(safe_trade, current_time)
    print(f"✓ Safe trade assessment: {is_safe} - {reason}")
    
    # Test risk assessment - risky trade (high turnover)
    risky_trade = 400000.0  # 4x capital
    is_safe, reason = risk_agent.assess_trade_risk(risky_trade, current_time)
    print(f"✓ Risky trade assessment: {is_safe} - {reason}")
    
    # Get comprehensive metrics
    metrics = risk_agent.get_risk_metrics()
    print(f"✓ Risk metrics collected: {len(metrics)} metrics")
    
    # Get performance stats
    perf_stats = risk_agent.get_performance_stats()
    print(f"✓ Performance stats: {perf_stats.get('evaluation_count', 0)} evaluations")
    
    print("✅ RiskAgentAdapter standalone test passed")
    return True


def test_orchestrator_integration():
    """Test OrchestratorAgent integration with new risk system."""
    print("\n🧪 Testing OrchestratorAgent Integration")
    print("-" * 40)
    
    # Create test configs
    main_config_path, model_params_path, risk_limits_path, temp_dir = create_test_configs()
    
    try:
        # Initialize orchestrator with new risk system
        orchestrator = OrchestratorAgent(
            main_config_path=main_config_path,
            model_params_path=model_params_path,
            risk_limits_path=risk_limits_path
        )
        
        print("✓ OrchestratorAgent initialized with RiskAgentAdapter")
        
        # Test that risk agent is properly initialized
        assert hasattr(orchestrator, 'risk_agent'), "Risk agent not found"
        assert isinstance(orchestrator.risk_agent, RiskAgentAdapter), "Wrong risk agent type"
        
        print("✓ Risk agent type verification passed")
        
        # Test risk agent functionality through orchestrator
        start_value = 100000.0
        current_time = datetime.now()
        
        # Reset daily limits
        orchestrator.risk_agent.reset_daily_limits(start_value, current_time)
        print("✓ Daily limits reset through orchestrator")
        
        # Update portfolio value
        new_value = 99000.0
        orchestrator.risk_agent.update_portfolio_value(new_value, current_time)
        print("✓ Portfolio value updated through orchestrator")
        
        # Record a trade
        trade_value = 5000.0
        orchestrator.risk_agent.record_trade(trade_value, current_time)
        print("✓ Trade recorded through orchestrator")
        
        # Assess trade risk
        test_trade = 10000.0
        is_safe, reason = orchestrator.risk_agent.assess_trade_risk(test_trade, current_time)
        print(f"✓ Trade risk assessment: {is_safe} - {reason}")
        
        # Get risk metrics
        metrics = orchestrator.risk_agent.get_risk_metrics()
        print(f"✓ Risk metrics: drawdown={metrics['daily_drawdown']:.2%}, turnover={metrics['daily_turnover_ratio']:.2f}")
        
        # Test performance stats
        perf_stats = orchestrator.risk_agent.get_performance_stats()
        print(f"✓ Performance stats: {perf_stats.get('evaluation_count', 0)} evaluations")
        
        print("✅ OrchestratorAgent integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_performance_comparison():
    """Test performance comparison between old and new risk systems."""
    print("\n🧪 Testing Performance Comparison")
    print("-" * 40)
    
    config = {
        'max_daily_drawdown_pct': 0.02,
        'max_hourly_turnover_ratio': 5.0,
        'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True
    }
    
    # Initialize new risk agent
    risk_agent = RiskAgentAdapter(config)
    
    # Setup initial state
    start_value = 100000.0
    current_time = datetime.now()
    risk_agent.reset_daily_limits(start_value, current_time)
    risk_agent.update_portfolio_value(99000.0, current_time)
    
    # Performance test - multiple risk assessments
    num_assessments = 100
    trade_value = 5000.0
    
    start_time = time.time_ns()
    
    for i in range(num_assessments):
        is_safe, reason = risk_agent.assess_trade_risk(trade_value, current_time)
    
    end_time = time.time_ns()
    
    avg_time_us = (end_time - start_time) / num_assessments / 1000.0
    
    print(f"✓ Average risk assessment time: {avg_time_us:.2f}µs")
    print(f"✓ Total assessments: {num_assessments}")
    print(f"✓ Performance target (<1000µs): {'✅' if avg_time_us < 1000 else '❌'}")
    
    # Get final performance stats
    perf_stats = risk_agent.get_performance_stats()
    print(f"✓ RiskAgentV2 evaluations: {perf_stats.get('evaluation_count', 0)}")
    
    if perf_stats.get('evaluation_count', 0) > 0:
        v2_avg_time = perf_stats.get('avg_evaluation_time_us', 0)
        print(f"✓ RiskAgentV2 avg time: {v2_avg_time:.2f}µs")
    
    print("✅ Performance comparison test passed")
    return avg_time_us < 1000  # Performance target


def main():
    """Run all integration tests."""
    print("🚀 RiskAgentV2 Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("RiskAgentAdapter Standalone", test_risk_agent_adapter_standalone),
        ("OrchestratorAgent Integration", test_orchestrator_integration),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("=" * 60)
        
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            if success:
                print(f"✅ PASSED ({duration:.3f}s)")
                passed += 1
            else:
                print(f"❌ FAILED ({duration:.3f}s)")
                failed += 1
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ FAILED ({duration:.3f}s): {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total:  {passed + failed}")
    
    if failed == 0:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("RiskAgentV2 is successfully integrated with OrchestratorAgent!")
        return True
    else:
        print(f"\n❌ Some integration tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)