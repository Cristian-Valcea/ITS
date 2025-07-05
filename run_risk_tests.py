#!/usr/bin/env python3
"""
Local test runner for risk system tests with latency benchmarks.
Provides comprehensive testing and performance validation.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED ({duration:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print(f"❌ {description} FAILED ({duration:.2f}s)")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ {description} ERROR: {e}")
        return False

def main():
    """Run all risk system tests."""
    print("🚀 Risk System Test Suite")
    print("Testing enterprise-grade risk management system with latency benchmarks")
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Set Python path
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # Test commands
    tests = [
        {
            'cmd': 'python -m pytest tests/test_risk_calculators.py::TestDrawdownCalculator::test_golden_file_simple_decline -v -s',
            'description': 'Golden File Test - Drawdown Simple Decline'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_calculators.py::TestDrawdownCalculator::test_latency_benchmark_small_dataset -v -s',
            'description': 'Latency Benchmark - DrawdownCalculator (Small Dataset)'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_calculators.py::TestTurnoverCalculator::test_golden_file_steady_trading -v -s',
            'description': 'Golden File Test - Turnover Steady Trading'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_calculators.py::TestTurnoverCalculator::test_latency_benchmark_small_dataset -v -s',
            'description': 'Latency Benchmark - TurnoverCalculator (Small Dataset)'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_integration.py::TestEventBusRulesEngineIntegration::test_end_to_end_risk_pipeline -v -s',
            'description': 'Integration Test - End-to-End Risk Pipeline'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_integration.py::TestEventBusRulesEngineIntegration::test_policy_evaluation_latency -v -s',
            'description': 'Latency Benchmark - Policy Evaluation'
        },
        {
            'cmd': 'python -m pytest tests/test_risk_integration.py::TestRiskSystemGoldenFiles::test_golden_scenario_market_crash -v -s',
            'description': 'Golden File Test - Market Crash Scenario'
        }
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for test in tests:
        if run_command(test['cmd'], test['description']):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total:  {passed + failed}")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Risk system is ready for production deployment.")
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        print("Please review failures before deployment.")
    
    # Performance summary
    print(f"\n{'='*60}")
    print(f"⚡ PERFORMANCE TARGETS")
    print(f"{'='*60}")
    print("🎯 DrawdownCalculator: P50 < 150µs, P95 < 300µs")
    print("🎯 TurnoverCalculator: P50 < 100µs, P95 < 200µs")
    print("🎯 Policy Evaluation: P50 < 100µs, P95 < 200µs")
    print("🎯 End-to-End Pipeline: < 10ms total latency")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)