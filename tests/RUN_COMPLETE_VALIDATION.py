#!/usr/bin/env python3
"""
🔍 COMPLETE SECRETS MANAGEMENT VALIDATION RUNNER
==============================================

This script runs the complete validation suite for the secrets management system,
including both core functionality and trading system integration.

⚠️ TRUST BUT VERIFY ⚠️
This is the master validation script that will tell you definitively whether
the programmer actually implemented everything they claimed.

Validation Scope:
- Core secrets management functionality (Phase 1-3)
- Trading system integration (Phase 4)
- Performance and security validation
- Production readiness assessment

Usage:
    python tests/RUN_COMPLETE_VALIDATION.py
    
Exit Codes:
    0: All tests passed - Deploy to production
    1: Minor issues - Deploy with monitoring  
    2: Significant issues - Staging only
    3: Major failures - Do not deploy
    4: Critical system error
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List

class CompleteValidationRunner:
    """
    Master validation runner that executes all test suites and provides
    comprehensive assessment of the secrets management system.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        self.start_time = time.time()
        
    def run_validation_suite(self, suite_name: str, script_path: str) -> Dict[str, Any]:
        """Run a validation suite and capture results."""
        print(f"\n🔍 Running {suite_name}...")
        print("=" * 80)
        
        try:
            # Run the validation script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse results
            suite_result = {
                'suite_name': suite_name,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'timestamp': time.time()
            }
            
            # Try to load detailed JSON report if available
            json_report_paths = [
                self.project_root / f"{suite_name.upper()}_REPORT.json",
                self.project_root / "EXHAUSTIVE_VALIDATION_REPORT.json",
                self.project_root / "TRADING_INTEGRATION_REPORT.json"
            ]
            
            for report_path in json_report_paths:
                if report_path.exists():
                    try:
                        with open(report_path, 'r') as f:
                            detailed_report = json.load(f)
                        suite_result['detailed_results'] = detailed_report
                        break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load detailed report: {e}")
            
            return suite_result
            
        except Exception as e:
            return {
                'suite_name': suite_name,
                'exit_code': 99,
                'error': str(e),
                'success': False,
                'timestamp': time.time()
            }
    
    def analyze_comprehensive_results(self) -> Dict[str, Any]:
        """Analyze results from all validation suites."""
        
        # Extract key metrics
        total_suites = len(self.results)
        successful_suites = sum(1 for r in self.results.values() if r['success'])
        
        # Aggregate test counts from detailed results
        total_tests = 0
        passed_tests = 0
        
        for suite_result in self.results.values():
            if 'detailed_results' in suite_result:
                detailed = suite_result['detailed_results']
                if isinstance(detailed, dict):
                    total_tests += detailed.get('total_tests', 0)
                    passed_tests += detailed.get('passed', 0)
        
        # Calculate overall metrics
        suite_success_rate = (successful_suites / total_suites) * 100 if total_suites > 0 else 0
        test_success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine readiness level
        readiness_level = self._determine_readiness_level(suite_success_rate, test_success_rate)
        
        return {
            'total_suites': total_suites,
            'successful_suites': successful_suites,
            'suite_success_rate': suite_success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'test_success_rate': test_success_rate,
            'readiness_level': readiness_level,
            'execution_time': time.time() - self.start_time
        }
    
    def _determine_readiness_level(self, suite_rate: float, test_rate: float) -> Dict[str, Any]:
        """Determine production readiness level."""
        
        # Critical thresholds
        if suite_rate == 100 and test_rate >= 95:
            return {
                'level': 'PRODUCTION_READY',
                'recommendation': 'DEPLOY TO PRODUCTION',
                'confidence': 'HIGH',
                'exit_code': 0
            }
        elif suite_rate >= 100 and test_rate >= 85:
            return {
                'level': 'MOSTLY_READY',
                'recommendation': 'DEPLOY WITH MONITORING',
                'confidence': 'MEDIUM-HIGH',
                'exit_code': 1
            }
        elif suite_rate >= 50 and test_rate >= 70:
            return {
                'level': 'STAGING_READY',
                'recommendation': 'DEPLOY TO STAGING FIRST',
                'confidence': 'MEDIUM',
                'exit_code': 2
            }
        else:
            return {
                'level': 'NOT_READY',
                'recommendation': 'DO NOT DEPLOY',
                'confidence': 'LOW',
                'exit_code': 3
            }
    
    def generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate executive summary report."""
        
        readiness = analysis['readiness_level']
        
        summary = f"""
🎯 SECRETS MANAGEMENT SYSTEM - EXECUTIVE VALIDATION REPORT
=========================================================

📊 OVERALL RESULTS:
   • Test Suites: {analysis['successful_suites']}/{analysis['total_suites']} passed ({analysis['suite_success_rate']:.1f}%)
   • Individual Tests: {analysis['passed_tests']}/{analysis['total_tests']} passed ({analysis['test_success_rate']:.1f}%)
   • Execution Time: {analysis['execution_time']:.1f} seconds

🎖️ READINESS ASSESSMENT: {readiness['level']}
   • Recommendation: {readiness['recommendation']}
   • Confidence Level: {readiness['confidence']}

📋 DETAILED SUITE RESULTS:
"""
        
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            summary += f"   {status} {suite_name}\n"
            
            if not result['success']:
                summary += f"      Exit Code: {result['exit_code']}\n"
                if 'error' in result:
                    summary += f"      Error: {result['error']}\n"
        
        # Add specific findings
        summary += f"\n🔍 KEY FINDINGS:\n"
        
        # Phase-specific analysis
        for suite_name, result in self.results.items():
            if 'detailed_results' in result:
                detailed = result['detailed_results']
                if isinstance(detailed, dict) and 'results' in detailed:
                    failed_tests = [name for name, res in detailed['results'].items() 
                                   if not res.get('passed', True)]
                    if failed_tests:
                        summary += f"   • {suite_name} failures: {', '.join(failed_tests[:3])}"
                        if len(failed_tests) > 3:
                            summary += f" (and {len(failed_tests)-3} more)"
                        summary += "\n"
        
        # Trust assessment
        if analysis['test_success_rate'] >= 95:
            trust_level = "TRUSTWORTHY - All claimed features verified"
        elif analysis['test_success_rate'] >= 85:
            trust_level = "MOSTLY TRUSTWORTHY - Minor issues found"
        elif analysis['test_success_rate'] >= 70:
            trust_level = "QUESTIONABLE - Multiple missing features"
        else:
            trust_level = "NOT TRUSTWORTHY - Many features missing/broken"
        
        summary += f"\n🕵️ PROGRAMMER TRUST ASSESSMENT: {trust_level}\n"
        
        # Final recommendation
        summary += f"\n🚀 FINAL RECOMMENDATION:\n"
        summary += f"   {readiness['recommendation']}\n"
        
        if readiness['level'] == 'PRODUCTION_READY':
            summary += "   • System meets all production requirements\n"
            summary += "   • All claimed features are implemented and working\n"
            summary += "   • Trading system integration is complete\n"
        elif readiness['level'] == 'MOSTLY_READY':
            summary += "   • System is functional with minor issues\n"
            summary += "   • Monitor closely during initial deployment\n"
            summary += "   • Address remaining issues in next iteration\n"
        elif readiness['level'] == 'STAGING_READY':
            summary += "   • System needs significant work before production\n"
            summary += "   • Test thoroughly in staging environment\n"
            summary += "   • Address critical failures before proceeding\n"
        else:
            summary += "   • System is not ready for any deployment\n"
            summary += "   • Major architectural or implementation issues\n"
            summary += "   • Consider alternative approach or complete rebuild\n"
        
        return summary
    
    def run_complete_validation(self):
        """Run complete validation suite."""
        print("🔍 COMPLETE SECRETS MANAGEMENT VALIDATION")
        print("=" * 80)
        print("⚠️  COMPREHENSIVE TRUST VERIFICATION IN PROGRESS")
        print("=" * 80)
        
        # Define validation suites to run
        validation_suites = [
            ('Core_Secrets_Management', 'tests/EXHAUSTIVE_SECRETS_VALIDATION.py'),
            ('Trading_System_Integration', 'tests/TRADING_SYSTEM_INTEGRATION_VALIDATION.py')
        ]
        
        # Run each validation suite
        for suite_name, script_path in validation_suites:
            full_script_path = self.project_root / script_path
            
            if full_script_path.exists():
                self.results[suite_name] = self.run_validation_suite(suite_name, str(full_script_path))
            else:
                print(f"❌ Warning: Validation suite not found: {script_path}")
                self.results[suite_name] = {
                    'suite_name': suite_name,
                    'exit_code': 404,
                    'error': f"Suite not found: {script_path}",
                    'success': False,
                    'timestamp': time.time()
                }
        
        # Analyze comprehensive results
        analysis = self.analyze_comprehensive_results()
        
        # Generate and display executive summary
        summary = self.generate_executive_summary(analysis)
        print(summary)
        
        # Save comprehensive report
        comprehensive_report = {
            'analysis': analysis,
            'suite_results': self.results,
            'executive_summary': summary,
            'generated_at': time.time()
        }
        
        report_path = self.project_root / "COMPREHENSIVE_VALIDATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive report saved to: {report_path}")
        
        # Return exit code based on readiness
        return analysis['readiness_level']['exit_code']


def main():
    """Main execution function."""
    try:
        runner = CompleteValidationRunner()
        exit_code = runner.run_complete_validation()
        
        print(f"\n🏁 Validation completed with exit code: {exit_code}")
        
        # Exit with appropriate code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 CRITICAL VALIDATION ERROR: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()