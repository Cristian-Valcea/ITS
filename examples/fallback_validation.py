#!/usr/bin/env python3
# examples/fallback_validation.py
"""
Comprehensive validation of fallback policy system for production readiness.

Validates:
1. Fallback activates when policy.pt fails to load
2. Performance meets <50µs P95 requirement for CRITICAL lane
3. Always returns HOLD (action=1) for safety
4. No throttling or kill switches - graceful degradation
"""

import sys
import time
import tempfile
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_fallback_activation():
    """Validate fallback activates in all failure scenarios."""
    print("🎯 Validating Fallback Activation")
    print("=" * 50)
    
    from execution.execution_agent_stub import ExecutionAgentStub
    
    test_cases = [
        ("Non-existent bundle", Path("/non/existent/bundle")),
        ("Empty directory", None),  # Will create empty dir
        ("Missing policy.pt", "missing_policy"),
        ("Corrupted policy.pt", "corrupted_policy"),
        ("Invalid metadata", "invalid_metadata"),
    ]
    
    results = []
    
    for case_name, bundle_path in test_cases:
        print(f"\n📋 {case_name}")
        
        try:
            if case_name == "Empty directory":
                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_path = Path(temp_dir) / "empty_bundle"
                    bundle_path.mkdir()
                    agent = ExecutionAgentStub(bundle_path, enable_soft_deadline=False)
            
            elif case_name == "Missing policy.pt":
                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_path = Path(temp_dir) / "missing_policy"
                    bundle_path.mkdir()
                    # Create metadata but no policy.pt
                    with open(bundle_path / "metadata.json", "w") as f:
                        json.dump({"policy_id": "test", "version": "1.0"}, f)
                    agent = ExecutionAgentStub(bundle_path, enable_soft_deadline=False)
            
            elif case_name == "Corrupted policy.pt":
                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_path = Path(temp_dir) / "corrupted_policy"
                    bundle_path.mkdir()
                    with open(bundle_path / "metadata.json", "w") as f:
                        json.dump({"policy_id": "test", "version": "1.0"}, f)
                    with open(bundle_path / "policy.pt", "w") as f:
                        f.write("corrupted content")
                    agent = ExecutionAgentStub(bundle_path, enable_soft_deadline=False)
            
            elif case_name == "Invalid metadata":
                with tempfile.TemporaryDirectory() as temp_dir:
                    bundle_path = Path(temp_dir) / "invalid_metadata"
                    bundle_path.mkdir()
                    with open(bundle_path / "metadata.json", "w") as f:
                        f.write("invalid json")
                    (bundle_path / "policy.pt").touch()
                    agent = ExecutionAgentStub(bundle_path, enable_soft_deadline=False)
            
            else:
                agent = ExecutionAgentStub(bundle_path, enable_soft_deadline=False)
            
            # Test prediction
            obs = np.random.randn(10).astype(np.float32)
            action, info = agent.predict(obs)
            
            is_fallback = getattr(agent.policy, 'policy_id', '') == 'fallback_hold_cash'
            
            result = {
                'case': case_name,
                'fallback_activated': is_fallback,
                'action': action,
                'is_hold': action == 1,
                'success': is_fallback and action == 1
            }
            
            results.append(result)
            
            if result['success']:
                print(f"   ✅ Fallback activated, returns HOLD")
            else:
                print(f"   ❌ Failed - fallback: {is_fallback}, action: {action}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'case': case_name,
                'fallback_activated': False,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 Fallback Activation: {successful}/{len(results)} cases passed")
    
    return all(r.get('success', False) for r in results)


def validate_performance_requirements():
    """Validate fallback policy meets performance requirements."""
    print("\n🎯 Validating Performance Requirements")
    print("=" * 50)
    
    from execution.execution_agent_stub import ExecutionAgentStub
    
    # Create agent with fallback
    agent = ExecutionAgentStub(Path("/non/existent"), enable_soft_deadline=False)
    obs = np.random.randn(10).astype(np.float32)
    
    # Warm-up
    for _ in range(100):
        agent.predict(obs)
    
    # Performance measurement
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        action, info = agent.predict(obs)
        lat_us = (time.perf_counter_ns() - start) / 1_000
        latencies.append(lat_us)
        
        assert action == 1, f"Expected HOLD (1), got {action}"
    
    # Calculate statistics
    latencies = np.array(latencies)
    stats = {
        'mean_us': np.mean(latencies),
        'median_us': np.median(latencies),
        'p95_us': np.percentile(latencies, 95),
        'p99_us': np.percentile(latencies, 99),
        'max_us': np.max(latencies),
        'min_us': np.min(latencies)
    }
    
    print("📊 Performance Statistics:")
    for metric, value in stats.items():
        print(f"   {metric}: {value:.2f}µs")
    
    # Validate requirements
    requirements = [
        ('P95 < 50µs (CRITICAL lane)', stats['p95_us'] < 50),
        ('Mean < 10µs (ultra-fast)', stats['mean_us'] < 10),
        ('Max < 100µs (no outliers)', stats['max_us'] < 100),
    ]
    
    print("\n📋 Requirements Check:")
    all_passed = True
    for req_name, passed in requirements:
        status = "✅" if passed else "❌"
        print(f"   {status} {req_name}")
        all_passed = all_passed and passed
    
    return all_passed


def validate_safety_behavior():
    """Validate fallback policy always returns safe HOLD action."""
    print("\n🎯 Validating Safety Behavior")
    print("=" * 50)
    
    from execution.execution_agent_stub import ExecutionAgentStub
    
    agent = ExecutionAgentStub(Path("/non/existent"), enable_soft_deadline=False)
    
    # Test with various observation patterns
    test_observations = [
        ("zeros", np.zeros(10).astype(np.float32)),
        ("ones", np.ones(10).astype(np.float32)),
        ("random_normal", np.random.randn(10).astype(np.float32)),
        ("extreme_positive", np.ones(10).astype(np.float32) * 1000),
        ("extreme_negative", np.ones(10).astype(np.float32) * -1000),
        ("high_volatility", np.random.randn(10).astype(np.float32) * 100),
        ("nan_values", np.full(10, np.nan).astype(np.float32)),
        ("inf_values", np.full(10, np.inf).astype(np.float32)),
    ]
    
    all_hold = True
    
    for obs_name, obs in test_observations:
        try:
            action, info = agent.predict(obs)
            is_hold = action == 1
            
            print(f"   {obs_name}: action={action} {'✅' if is_hold else '❌'}")
            
            if not is_hold:
                all_hold = False
                
        except Exception as e:
            print(f"   {obs_name}: ERROR - {e} ❌")
            all_hold = False
    
    print(f"\n📊 Safety Check: {'✅ All HOLD' if all_hold else '❌ Non-HOLD detected'}")
    return all_hold


def validate_no_throttling():
    """Validate system doesn't throttle or kill switch on fallback."""
    print("\n🎯 Validating No Throttling/Kill Switch")
    print("=" * 50)
    
    from execution.execution_agent_stub import ExecutionAgentStub
    
    # Create multiple agents with failed policy loading
    agents = []
    for i in range(5):
        try:
            agent = ExecutionAgentStub(Path(f"/non/existent/{i}"), enable_soft_deadline=False)
            agents.append(agent)
        except Exception as e:
            print(f"   ❌ Agent {i} creation failed: {e}")
            return False
    
    print(f"   ✅ Created {len(agents)} agents with fallback policies")
    
    # Test continuous operation
    obs = np.random.randn(10).astype(np.float32)
    total_predictions = 0
    
    for i, agent in enumerate(agents):
        for _ in range(100):  # 100 predictions per agent
            try:
                action, info = agent.predict(obs)
                assert action == 1, f"Agent {i} returned non-HOLD action: {action}"
                total_predictions += 1
            except Exception as e:
                print(f"   ❌ Agent {i} prediction failed: {e}")
                return False
    
    print(f"   ✅ Completed {total_predictions} predictions without throttling")
    print("   ✅ No kill switch activated")
    print("   ✅ Graceful degradation working")
    
    return True


def main():
    """Run comprehensive fallback policy validation."""
    print("🚀 Fallback Policy System Validation")
    print("=" * 80)
    
    tests = [
        ("Fallback Activation", validate_fallback_activation),
        ("Performance Requirements", validate_performance_requirements),
        ("Safety Behavior", validate_safety_behavior),
        ("No Throttling", validate_no_throttling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n🎉 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎯 PRODUCTION READY")
        print("✅ ExecutionAgent will never fail due to policy.pt loading issues")
        print("✅ Fallback policy meets <50µs P95 requirement for CRITICAL lane")
        print("✅ Always returns safe HOLD action")
        print("✅ No throttling or kill switches - graceful degradation")
        print("✅ System continues trading with fallback policy")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed - needs fixes before production")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)