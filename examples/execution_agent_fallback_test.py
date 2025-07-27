#!/usr/bin/env python3
# examples/execution_agent_fallback_test.py
"""
Test ExecutionAgent factory function with fallback policy.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_execution_agent_factory():
    """Test the create_execution_agent_stub factory with fallback."""
    print("🎯 Testing ExecutionAgent Factory with Fallback")
    print("=" * 60)
    
    try:
        from execution.execution_agent_stub import create_execution_agent_stub
        
        # Test with non-existent bundle (should use fallback)
        print("\n📋 Testing factory with non-existent bundle...")
        
        try:
            agent = create_execution_agent_stub(
                policy_bundle_path=Path("/non/existent/bundle"),
                enable_soft_deadline=False
            )
            print("❌ Expected FileNotFoundError from factory validation")
        except FileNotFoundError as e:
            print(f"✅ Factory correctly validates bundle existence: {e}")
        
        # The factory validates bundle structure, so fallback happens in ExecutionAgentStub
        # Let's test by creating a minimal bundle structure
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "test_bundle"
            bundle_path.mkdir()
            
            # Create required files for factory validation
            metadata = {"policy_id": "test", "version": "1.0"}
            with open(bundle_path / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # Create corrupted policy.pt (will fail TorchScript loading)
            with open(bundle_path / "policy.pt", "w") as f:
                f.write("corrupted")
            
            print(f"\n📋 Testing factory with corrupted policy.pt...")
            
            agent = create_execution_agent_stub(
                policy_bundle_path=bundle_path,
                enable_soft_deadline=False
            )
            
            # Test prediction
            obs = np.random.randn(10).astype(np.float32)
            action, info = agent.predict(obs)
            
            print(f"Action: {action}")
            print(f"Policy ID: {getattr(agent.policy, 'policy_id', 'unknown')}")
            print(f"Is fallback: {info.get('is_fallback', False)}")
            
            if action == 1 and info.get('is_fallback', False):
                print("✅ Factory + ExecutionAgent fallback working correctly")
                
                # Performance test
                latencies = []
                for _ in range(100):
                    start = time.perf_counter_ns()
                    action, info = agent.predict(obs)
                    lat_us = (time.perf_counter_ns() - start) / 1_000
                    latencies.append(lat_us)
                
                p95 = np.percentile(latencies, 95)
                print(f"Performance: P95={p95:.1f}µs")
                
                if p95 < 50:
                    print("✅ Meets CRITICAL lane <50µs P95 requirement")
                else:
                    print(f"⚠️ P95 {p95:.1f}µs exceeds 50µs")
                    
                return True
            else:
                print(f"❌ Fallback not working - action: {action}, fallback: {info.get('is_fallback')}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_execution_agent_factory()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)