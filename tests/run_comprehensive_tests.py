# tests/run_comprehensive_tests.py
"""
Comprehensive Test Runner for IntradayJules Trading System.

Runs all test types in sequence:
1. Unit Tests (416 tests)
2. Integration Tests (27 tests) 
3. Latency Tests (4 tests)
4. Chaos Tests (NEW - resilience testing)
5. Property-Based Tests (NEW - mathematical invariants)

Provides detailed reporting and metrics collection.
"""

import pytest
import sys
import time
import json
import subprocess
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class TestResult:
    """Test result data structure."""
    test_type: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage: Optional[float] = None
    memory_peak: Optional[float] = None
    cpu_avg: Optional[float] = None
    errors: List[str] = None


@dataclass
class TestSuite:
    """Complete test suite results."""
    timestamp: str
    total_duration: float
    overall_status: str
    results: List[TestResult]
    system_info: Dict[str, Any]
    summary: Dict[str, Any]


class ComprehensiveTestRunner:
    """Comprehensive test runner with metrics collection."""
    
    def __init__(self, verbose: bool = True, collect_metrics: bool = True):
        self.verbose = verbose
        self.collect_metrics = collect_metrics
        self.logger = self._setup_logging()
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("TestRunner")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('.').percent
        }
    
    def _run_pytest_with_metrics(self, test_path: str, markers: str = None, 
                                 extra_args: List[str] = None) -> TestResult:
        """Run pytest with metrics collection."""
        self.logger.info(f"Running tests: {test_path}")
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', test_path, '-v', '--tb=short']
        
        if markers:
            cmd.extend(['-m', markers])
        
        if extra_args:
            cmd.extend(extra_args)
        
        # Add coverage if requested
        if self.collect_metrics:
            cmd.extend(['--cov=src', '--cov-report=json'])
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        cpu_samples = []
        
        start_time = time.time()
        
        try:
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            duration = time.time() - start_time
            
            # Collect resource metrics
            if self.collect_metrics:
                final_memory = process.memory_info().rss
                memory_peak = (final_memory - initial_memory) / 1024 / 1024  # MB
                cpu_avg = psutil.cpu_percent(interval=0.1)
            else:
                memory_peak = None
                cpu_avg = None
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            
            # Extract test counts from pytest summary
            passed, failed, skipped = self._parse_pytest_output(output_lines)
            total_tests = passed + failed + skipped
            
            # Extract coverage if available
            coverage = self._extract_coverage() if self.collect_metrics else None
            
            # Extract errors
            errors = []
            if result.returncode != 0:
                errors = [line for line in output_lines if 'FAILED' in line or 'ERROR' in line]
            
            return TestResult(
                test_type=Path(test_path).name,
                total_tests=total_tests,
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration=duration,
                coverage=coverage,
                memory_peak=memory_peak,
                cpu_avg=cpu_avg,
                errors=errors[:10]  # Limit to first 10 errors
            )
            
        except Exception as e:
            self.logger.error(f"Failed to run tests {test_path}: {e}")
            return TestResult(
                test_type=Path(test_path).name,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _parse_pytest_output(self, output_lines: List[str]) -> tuple:
        """Parse pytest output to extract test counts."""
        passed = failed = skipped = 0
        
        for line in output_lines:
            if '=====' in line and 'test session starts' in line:
                continue
            elif '=====' in line and ('passed' in line or 'failed' in line):
                # Parse summary line like "===== 10 passed, 2 failed, 1 skipped in 5.23s ====="
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        passed = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        failed = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        skipped = int(parts[i-1])
                break
        
        return passed, failed, skipped
    
    def _extract_coverage(self) -> Optional[float]:
        """Extract coverage percentage from coverage report."""
        try:
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return coverage_data.get('totals', {}).get('percent_covered')
        except Exception as e:
            self.logger.warning(f"Could not extract coverage: {e}")
        return None
    
    def run_unit_tests(self) -> TestResult:
        """Run unit tests."""
        self.logger.info("🧪 Running Unit Tests")
        return self._run_pytest_with_metrics(
            'tests/unit',
            extra_args=['--maxfail=10']
        )
    
    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        self.logger.info("🔗 Running Integration Tests")
        return self._run_pytest_with_metrics(
            'tests/integration',
            markers='integration',
            extra_args=['--maxfail=5']
        )
    
    def run_latency_tests(self) -> TestResult:
        """Run latency tests."""
        self.logger.info("⚡ Running Latency Tests")
        return self._run_pytest_with_metrics(
            'tests/latency',
            markers='latency',
            extra_args=['--maxfail=2']
        )
    
    def run_chaos_tests(self) -> TestResult:
        """Run chaos engineering tests."""
        self.logger.info("💥 Running Chaos Tests")
        return self._run_pytest_with_metrics(
            'tests/chaos',
            markers='chaos',
            extra_args=['--maxfail=3', '--timeout=60']
        )
    
    def run_property_tests(self) -> TestResult:
        """Run property-based tests."""
        self.logger.info("🔍 Running Property-Based Tests")
        return self._run_pytest_with_metrics(
            'tests/property',
            extra_args=['--maxfail=5', '--hypothesis-show-statistics']
        )
    
    def run_all_tests(self) -> TestSuite:
        """Run all test suites."""
        self.logger.info("🚀 Starting Comprehensive Test Suite")
        self.logger.info("=" * 80)
        
        # Run each test suite
        test_suites = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Latency Tests", self.run_latency_tests),
            ("Chaos Tests", self.run_chaos_tests),
            ("Property-Based Tests", self.run_property_tests)
        ]
        
        for suite_name, suite_runner in test_suites:
            try:
                result = suite_runner()
                self.results.append(result)
                self._print_test_result(result)
            except Exception as e:
                self.logger.error(f"Failed to run {suite_name}: {e}")
                error_result = TestResult(
                    test_type=suite_name,
                    total_tests=0,
                    passed=0,
                    failed=1,
                    skipped=0,
                    duration=0,
                    errors=[str(e)]
                )
                self.results.append(error_result)
        
        # Generate final report
        total_duration = time.time() - self.start_time
        return self._generate_final_report(total_duration)
    
    def _print_test_result(self, result: TestResult):
        """Print individual test result."""
        status_emoji = "✅" if result.failed == 0 else "❌"
        
        print(f"\n{status_emoji} {result.test_type}")
        print(f"   Tests: {result.total_tests} total, {result.passed} passed, {result.failed} failed, {result.skipped} skipped")
        print(f"   Duration: {result.duration:.2f}s")
        
        if result.coverage:
            print(f"   Coverage: {result.coverage:.1f}%")
        
        if result.memory_peak:
            print(f"   Memory Peak: {result.memory_peak:.1f} MB")
        
        if result.cpu_avg:
            print(f"   CPU Average: {result.cpu_avg:.1f}%")
        
        if result.errors:
            print(f"   Errors: {len(result.errors)} (showing first few)")
            for error in result.errors[:3]:
                print(f"     - {error}")
    
    def _generate_final_report(self, total_duration: float) -> TestSuite:
        """Generate final test report."""
        # Calculate summary statistics
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        
        overall_status = "PASSED" if total_failed == 0 else "FAILED"
        
        # Calculate average coverage
        coverages = [r.coverage for r in self.results if r.coverage is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else None
        
        summary = {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'skipped': total_skipped,
            'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'average_coverage': avg_coverage,
            'test_suites': len(self.results),
            'failed_suites': sum(1 for r in self.results if r.failed > 0)
        }
        
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=total_duration,
            overall_status=overall_status,
            results=self.results,
            system_info=self._collect_system_info(),
            summary=summary
        )
        
        self._print_final_report(test_suite)
        return test_suite
    
    def _print_final_report(self, test_suite: TestSuite):
        """Print final comprehensive report."""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 80)
        
        # Overall status
        status_emoji = "✅" if test_suite.overall_status == "PASSED" else "❌"
        print(f"\n{status_emoji} Overall Status: {test_suite.overall_status}")
        print(f"⏱️  Total Duration: {test_suite.total_duration:.2f}s")
        
        # Summary statistics
        summary = test_suite.summary
        print(f"\n📊 Summary Statistics:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"   Failed: {summary['failed']}")
        print(f"   Skipped: {summary['skipped']}")
        print(f"   Test Suites: {summary['test_suites']}")
        print(f"   Failed Suites: {summary['failed_suites']}")
        
        if summary['average_coverage']:
            print(f"   Average Coverage: {summary['average_coverage']:.1f}%")
        
        # Individual suite results
        print(f"\n📋 Individual Suite Results:")
        for result in test_suite.results:
            status = "✅ PASS" if result.failed == 0 else "❌ FAIL"
            print(f"   {result.test_type}: {status} ({result.passed}/{result.total_tests} passed)")
        
        # System information
        print(f"\n💻 System Information:")
        sys_info = test_suite.system_info
        print(f"   Platform: {sys_info['platform']}")
        print(f"   CPU Cores: {sys_info['cpu_count']}")
        print(f"   Memory: {sys_info['memory_total'] / 1024**3:.1f} GB total, {sys_info['memory_available'] / 1024**3:.1f} GB available")
        print(f"   Disk Usage: {sys_info['disk_usage']:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if summary['failed'] > 0:
            print("   ⚠️  Fix failing tests before deployment")
        if summary['average_coverage'] and summary['average_coverage'] < 80:
            print("   📈 Increase test coverage (target: 80%+)")
        if summary['failed_suites'] > 0:
            print("   🔧 Address failing test suites")
        if summary['failed'] == 0:
            print("   🎉 All tests passing - ready for deployment!")
        
        print("=" * 80)
    
    def save_report(self, test_suite: TestSuite, filename: str = None):
        """Save test report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = Path("test_reports") / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(asdict(test_suite), f, indent=2, default=str)
        
        self.logger.info(f"Test report saved to: {report_path}")
        return report_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument('--suite', choices=['unit', 'integration', 'latency', 'chaos', 'property', 'all'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics collection')
    parser.add_argument('--save-report', action='store_true', help='Save report to JSON file')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(
        verbose=args.verbose,
        collect_metrics=not args.no_metrics
    )
    
    # Run selected test suite
    if args.suite == 'all':
        test_suite = runner.run_all_tests()
    elif args.suite == 'unit':
        result = runner.run_unit_tests()
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=result.duration,
            overall_status="PASSED" if result.failed == 0 else "FAILED",
            results=[result],
            system_info=runner._collect_system_info(),
            summary={'total_tests': result.total_tests, 'passed': result.passed, 'failed': result.failed}
        )
    elif args.suite == 'integration':
        result = runner.run_integration_tests()
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=result.duration,
            overall_status="PASSED" if result.failed == 0 else "FAILED",
            results=[result],
            system_info=runner._collect_system_info(),
            summary={'total_tests': result.total_tests, 'passed': result.passed, 'failed': result.failed}
        )
    elif args.suite == 'latency':
        result = runner.run_latency_tests()
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=result.duration,
            overall_status="PASSED" if result.failed == 0 else "FAILED",
            results=[result],
            system_info=runner._collect_system_info(),
            summary={'total_tests': result.total_tests, 'passed': result.passed, 'failed': result.failed}
        )
    elif args.suite == 'chaos':
        result = runner.run_chaos_tests()
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=result.duration,
            overall_status="PASSED" if result.failed == 0 else "FAILED",
            results=[result],
            system_info=runner._collect_system_info(),
            summary={'total_tests': result.total_tests, 'passed': result.passed, 'failed': result.failed}
        )
    elif args.suite == 'property':
        result = runner.run_property_tests()
        test_suite = TestSuite(
            timestamp=datetime.now().isoformat(),
            total_duration=result.duration,
            overall_status="PASSED" if result.failed == 0 else "FAILED",
            results=[result],
            system_info=runner._collect_system_info(),
            summary={'total_tests': result.total_tests, 'passed': result.passed, 'failed': result.failed}
        )
    
    # Save report if requested
    if args.save_report:
        runner.save_report(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if test_suite.overall_status == "PASSED" else 1)


if __name__ == "__main__":
    main()