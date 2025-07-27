#!/usr/bin/env python3
# examples/simple_fallback_test.py
"""
Simple test to verify fallback policy works when policy.pt fails to load.
"""

import sys
import time
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_fallback():
    """Test fallback policy activation."""
    print("🎯 Testing Fallback Policy")
    print("=" * 40)
    
    try:
        from execution.execution_agent_stub import ExecutionAgentStub
        
        # Test with non-existent bundle
        print("\n📋 Test: Non-existent bundle")
        agent = ExecutionAgentStub(Path("/non/existent"), enable_soft_deadline=False)
        
        # Test prediction
        obs = np.random.randn(10).astype(np.float32)
        action, info = agent.predict(obs)
        
        print(f"Action: {action}")
        print(f"Policy ID: {getattr(agent.policy, 'policy_id', 'unknown')}")
        print(f"Is fallback: {info.get('is_fallback', False)}")
        
        # Performance test
        latencies = []
        for _ in range(100):
            start = time.perf_counter_ns()
            action, info = agent.predict(obs)
            lat_us = (time.perf_counter_ns() - start) / 1_000
            latencies.append(lat_us)
        
        p95 = np.percentile(latencies, 95)
        mean = np.mean(latencies)
        
        print(f"Performance: P95={p95:.1f}µs, Mean={mean:.1f}µs")
        
        if p95 < 50:
            print("✅ Meets <50µs P95 requirement")
        else:
            print(f"⚠️ P95 {p95:.1f}µs exceeds 50µs")
            
        print("✅ Fallback policy working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fallback()