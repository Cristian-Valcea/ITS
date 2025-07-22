#!/usr/bin/env python3
"""
Quick runner for reward bounds regression test.
"""

import subprocess
import sys
import time

def run_reward_bounds_test():
    """Run the reward bounds test and report results."""
    
    print("🧪 RUNNING REWARD BOUNDS REGRESSION TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the specific reward bounds test
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_reward_bounds.py::TestRewardBounds::test_reward_bounds_random_actions",
            "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=".")
        
        duration = time.time() - start_time
        
        print(f"⏱️ Test completed in {duration:.1f} seconds")
        print()
        
        if result.returncode == 0:
            print("✅ REWARD BOUNDS TEST PASSED!")
            print("   All 1000 random rewards were within bounds [-150, 150]")
            
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Reward range:" in line:
                    print(f"   {line.strip()}")
                elif "Mean reward:" in line:
                    print(f"   {line.strip()}")
                elif "Violations:" in line:
                    print(f"   {line.strip()}")
            
            print()
            print("🛡️ REGRESSION PROTECTION ACTIVE")
            print("   Future reward system changes will be caught immediately")
            
        else:
            print("❌ REWARD BOUNDS TEST FAILED!")
            print()
            print("STDOUT:")
            print(result.stdout)
            print()
            print("STDERR:")
            print(result.stderr)
            
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_reward_bounds_test()
    sys.exit(0 if success else 1)