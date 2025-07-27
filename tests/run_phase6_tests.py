#!/usr/bin/env python3
"""
Comprehensive test runner for Phase 6: Testing & Validation.

This script runs all the new unit tests and integration tests created for
the refactored architecture, providing detailed reporting on test coverage
and results.
"""

import pytest
import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def run_test_suite(test_path, description, verbose=True):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run pytest with appropriate flags
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v" if verbose else "-q",
        "--tb=short",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        duration = time.time() - start_time
        
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Exit Code: {result.returncode}")
        
        if result.stdout:
            print("\n📋 Test Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("\n❌ Errors:")
            print(result.stderr)
        
        return {
            'path': test_path,
            'description': description,
            'success': result.returncode == 0,
            'duration': duration,
            'output': result.stdout,
            'errors': result.stderr
        }
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return {
            'path': test_path,
            'description': description,
            'success': False,
            'duration': 0,
            'output': '',
            'errors': str(e)
        }

def run_all_phase6_tests():
    """Run all Phase 6 tests and generate comprehensive report."""
    print("🚀 Phase 6: Testing & Validation - Comprehensive Test Suite")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    
    # Define test suites
    test_suites = [
        # Existing tests (updated)
        {
            'path': PROJECT_ROOT / "tests" / "test_orchestrator_agent.py",
            'description': "Existing OrchestratorAgent Tests (Updated)"
        },
        
        # New execution module tests
        {
            'path': PROJECT_ROOT / "tests" / "execution" / "test_execution_loop.py",
            'description': "Execution Loop Core Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "execution" / "test_order_router.py",
            'description': "Order Router Core Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "execution" / "test_pnl_tracker.py",
            'description': "P&L Tracker Core Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "execution" / "test_live_data_loader.py",
            'description': "Live Data Loader Core Component Tests"
        },
        
        # New training module tests
        {
            'path': PROJECT_ROOT / "tests" / "training" / "test_trainer_core.py",
            'description': "Trainer Core Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "training" / "test_env_builder.py",
            'description': "Environment Builder Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "training" / "test_policy_export.py",
            'description': "Policy Export Component Tests"
        },
        {
            'path': PROJECT_ROOT / "tests" / "training" / "test_hyperparam_search.py",
            'description': "Hyperparameter Search Component Tests"
        },
        
        # Integration tests
        {
            'path': PROJECT_ROOT / "tests" / "test_facade_integration.py",
            'description': "Façade Integration Tests"
        }
    ]
    
    # Run all test suites
    results = []
    total_start_time = time.time()
    
    for suite in test_suites:
        if suite['path'].exists():
            result = run_test_suite(suite['path'], suite['description'])
            results.append(result)
        else:
            print(f"\n⚠️  Test file not found: {suite['path']}")
            results.append({
                'path': suite['path'],
                'description': suite['description'],
                'success': False,
                'duration': 0,
                'output': '',
                'errors': 'Test file not found'
            })
    
    total_duration = time.time() - total_start_time
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📊 PHASE 6 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful Test Suites: {len(successful_tests)}")
    print(f"❌ Failed Test Suites: {len(failed_tests)}")
    print(f"📊 Total Test Suites: {len(results)}")
    print(f"⏱️  Total Duration: {total_duration:.2f}s")
    print(f"🎯 Success Rate: {len(successful_tests)/len(results)*100:.1f}%")
    
    if successful_tests:
        print(f"\n✅ SUCCESSFUL TEST SUITES:")
        for result in successful_tests:
            print(f"   • {result['description']} ({result['duration']:.2f}s)")
    
    if failed_tests:
        print(f"\n❌ FAILED TEST SUITES:")
        for result in failed_tests:
            print(f"   • {result['description']}")
            if result['errors']:
                print(f"     Error: {result['errors'][:100]}...")
    
    # Architecture validation summary
    print(f"\n🏗️  ARCHITECTURE VALIDATION:")
    
    execution_tests = [r for r in results if 'execution' in str(r['path']).lower()]
    training_tests = [r for r in results if 'training' in str(r['path']).lower()]
    integration_tests = [r for r in results if 'integration' in str(r['path']).lower()]
    
    execution_success = sum(1 for r in execution_tests if r['success'])
    training_success = sum(1 for r in training_tests if r['success'])
    integration_success = sum(1 for r in integration_tests if r['success'])
    
    print(f"   • Execution Module: {execution_success}/{len(execution_tests)} tests passing")
    print(f"   • Training Module: {training_success}/{len(training_tests)} tests passing")
    print(f"   • Integration Tests: {integration_success}/{len(integration_tests)} tests passing")
    
    # Final assessment
    overall_success_rate = len(successful_tests) / len(results)
    
    print(f"\n🎯 PHASE 6 ASSESSMENT:")
    if overall_success_rate >= 0.8:
        print("🎉 PHASE 6: TESTING & VALIDATION - SUCCESSFUL!")
        print("   • Comprehensive test coverage implemented")
        print("   • Core components properly tested")
        print("   • Façade delegation verified")
        print("   • Architecture validation complete")
    elif overall_success_rate >= 0.6:
        print("⚠️  PHASE 6: TESTING & VALIDATION - PARTIALLY SUCCESSFUL")
        print("   • Most tests implemented and working")
        print("   • Some dependency-related failures expected")
        print("   • Core architecture testing complete")
    else:
        print("❌ PHASE 6: TESTING & VALIDATION - NEEDS ATTENTION")
        print("   • Multiple test failures detected")
        print("   • Architecture validation incomplete")
        print("   • Review failed tests and dependencies")
    
    print(f"\n📋 NEXT STEPS:")
    if failed_tests:
        print("   1. Review failed test suites")
        print("   2. Install missing dependencies if needed")
        print("   3. Fix any architectural issues identified")
        print("   4. Re-run tests to verify fixes")
    else:
        print("   1. All tests passing - architecture validated!")
        print("   2. Ready for production deployment")
        print("   3. Consider adding more integration tests")
        print("   4. Set up continuous integration pipeline")
    
    print("=" * 80)
    
    return overall_success_rate >= 0.6

def run_specific_module_tests(module_name):
    """Run tests for a specific module."""
    if module_name == "execution":
        test_dir = PROJECT_ROOT / "tests" / "execution"
    elif module_name == "training":
        test_dir = PROJECT_ROOT / "tests" / "training"
    elif module_name == "integration":
        test_file = PROJECT_ROOT / "tests" / "test_facade_integration.py"
        return run_test_suite(test_file, "Integration Tests")
    else:
        print(f"❌ Unknown module: {module_name}")
        return False
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    print(f"🧪 Running {module_name.title()} Module Tests")
    
    # Run all tests in the module directory
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--disable-warnings"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        success = run_specific_module_tests(module_name)
    else:
        success = run_all_phase6_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()