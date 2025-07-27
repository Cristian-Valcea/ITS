#!/usr/bin/env python3
"""
Test Runner for IntradayJules Test Suite
Runs key tests and provides summary results.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

def run_test(test_file: str) -> Tuple[bool, str]:
    """Run a single test file and return success status and output."""
    try:
        # Set environment to handle Unicode properly
        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"❌ Test {test_file} timed out after 5 minutes"
    except Exception as e:
        return False, f"❌ Test {test_file} failed with exception: {e}"

def main():
    """Run the test suite."""
    print("🧪 INTRADAYJULES TEST SUITE")
    print("=" * 60)
    
    # Key tests to run
    key_tests = [
        "test_policy_latency.py",
        "test_risk_penalty.py",
    ]
    
    # Optional tests (may require specific config files)
    optional_tests = [
        "test_all_risk_calculators.py",
        "test_comprehensive_risk_integration.py",
    ]
    
    results = []
    
    print("\n🎯 Running Key Tests:")
    print("-" * 30)
    
    for test in key_tests:
        print(f"\n🔍 Running {test}...")
        success, output = run_test(test)
        results.append((test, success, "KEY"))
        
        if success:
            print(f"✅ {test} PASSED")
        else:
            print(f"❌ {test} FAILED")
            # Show last few lines of output for debugging
            lines = output.split('\n')
            print("Last few lines of output:")
            for line in lines[-5:]:
                if line.strip():
                    print(f"   {line}")
    
    print(f"\n🔧 Running Optional Tests:")
    print("-" * 30)
    
    for test in optional_tests:
        if Path(test).exists():
            print(f"\n🔍 Running {test}...")
            success, output = run_test(test)
            results.append((test, success, "OPTIONAL"))
            
            if success:
                print(f"✅ {test} PASSED")
            else:
                print(f"⚠️  {test} FAILED (optional)")
        else:
            print(f"⏭️  Skipping {test} (not found)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    key_passed = sum(1 for test, success, category in results if category == "KEY" and success)
    key_total = sum(1 for test, success, category in results if category == "KEY")
    optional_passed = sum(1 for test, success, category in results if category == "OPTIONAL" and success)
    optional_total = sum(1 for test, success, category in results if category == "OPTIONAL")
    
    print(f"🎯 Key Tests: {key_passed}/{key_total} passed")
    print(f"🔧 Optional Tests: {optional_passed}/{optional_total} passed")
    
    if key_passed == key_total:
        print("\n🎉 ALL KEY TESTS PASSED!")
        print("✅ Core functionality is working correctly")
        return 0
    else:
        print(f"\n❌ {key_total - key_passed} key tests failed")
        print("⚠️  Core functionality may have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())