#!/usr/bin/env python3
"""
Day 1 Setup Verification Script
Verifies all components are working correctly for dual-ticker system
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    print(f"🔍 {description}...")
    if Path(filepath).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False

def main():
    print("🚀 DAY 1 DUAL-TICKER SETUP VERIFICATION")
    print("=" * 50)
    
    results = []
    
    # 1. Check Docker Compose files
    results.append(check_file_exists("docker-compose.yml", "Docker Compose configuration"))
    results.append(check_file_exists("sql/docker-entrypoint-initdb.d/01_schema.sql", "TimescaleDB schema"))
    
    # 2. Check test fixtures
    results.append(check_file_exists("tests/fixtures/dual_ticker_sample.parquet", "Dual-ticker test fixtures"))
    results.append(check_file_exists("tests/fixtures/nvda_sample.parquet", "NVDA test data"))
    results.append(check_file_exists("tests/fixtures/msft_sample.parquet", "MSFT test data"))
    
    # 3. Check CI pipeline
    results.append(check_file_exists(".github/workflows/dual_ticker_ci.yml", "GitHub Actions CI pipeline"))
    
    # 4. Check smoke tests
    results.append(check_file_exists("tests/dual_ticker/test_smoke.py", "Smoke test suite"))
    
    # 5. Test fixture generation
    results.append(run_command(
        "python3 tests/fixtures/generate_test_data.py",
        "Test fixture generation"
    ))
    
    # 6. Run smoke tests (without database)
    results.append(run_command(
        "python3 -m pytest tests/dual_ticker/test_smoke.py::test_dual_ticker_fixtures_exist -v",
        "Fixture validation test"
    ))
    
    results.append(run_command(
        "python3 -m pytest tests/dual_ticker/test_smoke.py::test_dual_ticker_data_quality -v",
        "Data quality test"
    ))
    
    results.append(run_command(
        "python3 -m pytest tests/dual_ticker/test_smoke.py::test_performance_benchmark -v",
        "Performance benchmark test"
    ))
    
    # 7. Check Docker services
    results.append(run_command(
        "docker-compose ps timescaledb",
        "TimescaleDB container status"
    ))
    
    # 8. Test database connection (if running)
    results.append(run_command(
        "TEST_DB_HOST=localhost python3 -m pytest tests/dual_ticker/test_smoke.py::test_database_connection -v",
        "Database connection test"
    ))
    
    # 9. Verify NVDA vs AAPL consistency
    print("🔍 Checking NVDA vs AAPL consistency...")
    aapl_count = 0
    nvda_count = 0
    
    # Check key files for symbol references
    key_files = [
        "sql/docker-entrypoint-initdb.d/01_schema.sql",
        "tests/fixtures/generate_test_data.py",
        "tests/dual_ticker/test_smoke.py",
        "DUAL_TICKER_DEVELOPMENT_PLAN_REVISED.md"
    ]
    
    for filepath in key_files:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                content = f.read()
                aapl_count += content.upper().count('AAPL')
                nvda_count += content.upper().count('NVDA')
    
    if aapl_count == 0 and nvda_count > 0:
        print("✅ Symbol consistency - NVDA used correctly, no AAPL references")
        results.append(True)
    else:
        print(f"❌ Symbol consistency - Found {aapl_count} AAPL refs, {nvda_count} NVDA refs")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"✅ Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED - Day 1 setup complete!")
        print("🚀 Ready for Day 2 implementation!")
        return 0
    else:
        failed = total - passed
        print(f"❌ Failed: {failed}/{total}")
        print("🔧 Please fix the failing components before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())