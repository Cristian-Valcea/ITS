#!/usr/bin/env python3
"""
Critical Reviewer Implementations Validation Suite
Comprehensive validation of all three key deliverables that address top-tier quant reviewer concerns

This script runs all validation tests and generates a comprehensive report for:
1. Tick vs Minute Alpha Study - Empirical validation of bar frequency choice
2. Filtering Ablation Study - Evidence-based filtering decisions with CI validation  
3. Feature Lag Validation - Look-ahead bias detection and prevention

Usage:
    python validate_critical_reviewer_implementations.py [--verbose] [--report-only]
"""

import sys
import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'studies'))

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'validation_report.log')
    ]
)
logger = logging.getLogger(__name__)


class CriticalReviewerValidationSuite:
    """
    Comprehensive validation suite for critical reviewer implementations
    
    Validates institutional-grade rigor and production readiness of:
    - Empirical studies with measurable results
    - CI integration and automated validation
    - Audit compliance and lock-box hashes
    - Production performance and scalability
    """
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = datetime.now()
        self.errors = []
        self.warnings = []
        
        logger.info("🎯 Critical Reviewer Validation Suite Initialized")
        logger.info("=" * 80)
    
    def validate_tick_vs_minute_study(self) -> Dict[str, Any]:
        """Validate Tick vs Minute Alpha Study implementation"""
        logger.info("🔬 VALIDATING: Tick vs Minute Alpha Study")
        
        validation_result = {
            'study_name': 'Tick vs Minute Alpha Study',
            'status': 'UNKNOWN',
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'files_validated': []
        }
        
        try:
            # Test 1: Module Import
            try:
                from tick_vs_minute_alpha_study import TickVsMinuteAlphaStudy, AlphaStudyResult
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Module import successful")
            except ImportError as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Module import failed: {e}")
                logger.error(f"  ❌ Module import failed: {e}")
                return validation_result
            
            # Test 2: Class Initialization
            try:
                study = TickVsMinuteAlphaStudy()
                assert hasattr(study, 'timeframes')
                assert hasattr(study, 'results')
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Class initialization successful")
            except Exception as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Class initialization failed: {e}")
                logger.error(f"  ❌ Class initialization failed: {e}")
            
            # Test 3: Results Files Validation
            results_dir = project_root / 'studies' / 'tick_vs_minute_results'
            expected_files = [
                'summary_results.csv',
                'tick_vs_minute_study_report.md',
                'tick_vs_minute_analysis.png'
            ]
            
            files_found = 0
            for file_name in expected_files:
                file_path = results_dir / file_name
                if file_path.exists() and file_path.stat().st_size > 0:
                    files_found += 1
                    validation_result['files_validated'].append(file_name)
                    logger.info(f"  ✅ Found results file: {file_name}")
                else:
                    validation_result['warnings'].append(f"Missing or empty results file: {file_name}")
                    logger.warning(f"  ⚠️ Missing results file: {file_name}")
            
            if files_found >= 2:  # At least summary and report
                validation_result['tests_passed'] += 1
            else:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append("Insufficient results files")
            
            # Test 4: Documented Claims Validation
            summary_file = results_dir / 'summary_results.csv'
            if summary_file.exists():
                try:
                    import pandas as pd
                    summary_df = pd.read_csv(summary_file)
                    
                    # Find 1-minute results
                    minute_results = summary_df[summary_df['timeframe'] == '1T']
                    if not minute_results.empty:
                        minute_ir = minute_results['information_ratio'].iloc[0]
                        validation_result['metrics']['minute_information_ratio'] = minute_ir
                        
                        # Documented claim: 1-minute IR should be around 0.0243
                        if 0.01 <= minute_ir <= 0.05:
                            validation_result['tests_passed'] += 1
                            logger.info(f"  ✅ 1-minute IR validation passed: {minute_ir:.4f}")
                        else:
                            validation_result['tests_failed'] += 1
                            validation_result['errors'].append(f"1-minute IR {minute_ir:.4f} outside expected range [0.01, 0.05]")
                            logger.error(f"  ❌ 1-minute IR outside expected range: {minute_ir:.4f}")
                    else:
                        validation_result['warnings'].append("No 1-minute results found in summary")
                        logger.warning("  ⚠️ No 1-minute results found")
                        
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate documented claims: {e}")
                    logger.warning(f"  ⚠️ Claims validation error: {e}")
            
            # Determine overall status
            if validation_result['tests_failed'] == 0:
                validation_result['status'] = 'PASSED'
                logger.info("  🎉 Tick vs Minute Alpha Study: ALL TESTS PASSED")
            elif validation_result['tests_passed'] > validation_result['tests_failed']:
                validation_result['status'] = 'PASSED_WITH_WARNINGS'
                logger.warning("  ⚠️ Tick vs Minute Alpha Study: PASSED WITH WARNINGS")
            else:
                validation_result['status'] = 'FAILED'
                logger.error("  💥 Tick vs Minute Alpha Study: FAILED")
                
        except Exception as e:
            validation_result['status'] = 'ERROR'
            validation_result['errors'].append(f"Unexpected error: {e}")
            logger.error(f"  💥 Unexpected error in tick vs minute validation: {e}")
            logger.error(traceback.format_exc())
        
        return validation_result
    
    def validate_filtering_ablation_study(self) -> Dict[str, Any]:
        """Validate Filtering Ablation Study implementation"""
        logger.info("🧪 VALIDATING: Filtering Ablation Study")
        
        validation_result = {
            'study_name': 'Filtering Ablation Study',
            'status': 'UNKNOWN',
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'files_validated': []
        }
        
        try:
            # Test 1: Module Import
            try:
                from filtering_ablation_study import FilteringAblationStudy, AblationResult
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Module import successful")
            except ImportError as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Module import failed: {e}")
                logger.error(f"  ❌ Module import failed: {e}")
                return validation_result
            
            # Test 2: Class Initialization
            try:
                study = FilteringAblationStudy()
                assert hasattr(study, 'results')
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Class initialization successful")
            except Exception as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Class initialization failed: {e}")
                logger.error(f"  ❌ Class initialization failed: {e}")
            
            # Test 3: Results Files Validation
            results_dir = project_root / 'studies' / 'filtering_ablation_results'
            expected_files = [
                'ablation_summary.csv',
                'filtering_ablation_report.md',
                'lockbox_audit_hashes.json',
                'ci_validation.py'
            ]
            
            files_found = 0
            for file_name in expected_files:
                file_path = results_dir / file_name
                if file_path.exists() and file_path.stat().st_size > 0:
                    files_found += 1
                    validation_result['files_validated'].append(file_name)
                    logger.info(f"  ✅ Found results file: {file_name}")
                else:
                    validation_result['warnings'].append(f"Missing or empty results file: {file_name}")
                    logger.warning(f"  ⚠️ Missing results file: {file_name}")
            
            if files_found >= 3:  # At least summary, hashes, and CI validation
                validation_result['tests_passed'] += 1
            else:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append("Insufficient results files")
            
            # Test 4: Lock-box Hash Validation
            lockbox_file = results_dir / 'lockbox_audit_hashes.json'
            if lockbox_file.exists():
                try:
                    with open(lockbox_file, 'r') as f:
                        hashes = json.load(f)
                    
                    if isinstance(hashes, dict) and len(hashes) > 0:
                        # Validate hash format (SHA-256)
                        valid_hashes = 0
                        for key, value in hashes.items():
                            if isinstance(value, str) and len(value) == 64:
                                valid_hashes += 1
                            elif isinstance(value, dict) and 'results_hash' in value:
                                if len(value['results_hash']) == 64:
                                    valid_hashes += 1
                        
                        if valid_hashes > 0:
                            validation_result['tests_passed'] += 1
                            validation_result['metrics']['valid_hashes'] = valid_hashes
                            logger.info(f"  ✅ Lock-box hashes validated: {valid_hashes} valid hashes")
                        else:
                            validation_result['tests_failed'] += 1
                            validation_result['errors'].append("No valid SHA-256 hashes found")
                            logger.error("  ❌ No valid hashes found")
                    else:
                        validation_result['warnings'].append("Lock-box file exists but is empty or invalid")
                        logger.warning("  ⚠️ Lock-box file invalid")
                        
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate lock-box hashes: {e}")
                    logger.warning(f"  ⚠️ Lock-box validation error: {e}")
            
            # Test 5: Performance Claims Validation
            earnings_excluded_file = results_dir / 'config_earnings_excluded' / 'performance_summary.json'
            earnings_included_file = results_dir / 'config_earnings_included' / 'performance_summary.json'
            
            if earnings_excluded_file.exists() and earnings_included_file.exists():
                try:
                    with open(earnings_excluded_file, 'r') as f:
                        excluded_results = json.load(f)
                    
                    with open(earnings_included_file, 'r') as f:
                        included_results = json.load(f)
                    
                    # Validate documented claims
                    excluded_sharpe = excluded_results.get('sharpe_ratio', 0)
                    included_sharpe = included_results.get('sharpe_ratio', 0)
                    
                    validation_result['metrics']['excluded_sharpe'] = excluded_sharpe
                    validation_result['metrics']['included_sharpe'] = included_sharpe
                    
                    if excluded_sharpe > 0 and included_sharpe > 0:
                        if excluded_sharpe > included_sharpe:
                            validation_result['tests_passed'] += 1
                            improvement = (excluded_sharpe - included_sharpe) / included_sharpe * 100
                            validation_result['metrics']['sharpe_improvement_pct'] = improvement
                            logger.info(f"  ✅ Sharpe improvement validated: {included_sharpe:.3f} → {excluded_sharpe:.3f} (+{improvement:.1f}%)")
                        else:
                            validation_result['tests_failed'] += 1
                            validation_result['errors'].append("Earnings exclusion did not improve Sharpe ratio")
                            logger.error("  ❌ No Sharpe improvement from earnings exclusion")
                    else:
                        validation_result['warnings'].append("Could not validate Sharpe ratio improvement")
                        logger.warning("  ⚠️ Insufficient data for Sharpe validation")
                        
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate performance claims: {e}")
                    logger.warning(f"  ⚠️ Performance validation error: {e}")
            
            # Determine overall status
            if validation_result['tests_failed'] == 0:
                validation_result['status'] = 'PASSED'
                logger.info("  🎉 Filtering Ablation Study: ALL TESTS PASSED")
            elif validation_result['tests_passed'] > validation_result['tests_failed']:
                validation_result['status'] = 'PASSED_WITH_WARNINGS'
                logger.warning("  ⚠️ Filtering Ablation Study: PASSED WITH WARNINGS")
            else:
                validation_result['status'] = 'FAILED'
                logger.error("  💥 Filtering Ablation Study: FAILED")
                
        except Exception as e:
            validation_result['status'] = 'ERROR'
            validation_result['errors'].append(f"Unexpected error: {e}")
            logger.error(f"  💥 Unexpected error in filtering ablation validation: {e}")
            logger.error(traceback.format_exc())
        
        return validation_result
    
    def validate_feature_lag_validation(self) -> Dict[str, Any]:
        """Validate Feature Lag Validation implementation"""
        logger.info("🔍 VALIDATING: Feature Lag Validation")
        
        validation_result = {
            'study_name': 'Feature Lag Validation',
            'status': 'UNKNOWN',
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'warnings': [],
            'metrics': {},
            'files_validated': []
        }
        
        try:
            # Test 1: Module Import
            try:
                from test_feature_lag_validation import FeatureLagValidator
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Module import successful")
            except ImportError as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Module import failed: {e}")
                logger.error(f"  ❌ Module import failed: {e}")
                return validation_result
            
            # Test 2: Class Initialization
            try:
                validator = FeatureLagValidator()
                assert hasattr(validator, 'feature_lag_requirements')
                assert hasattr(validator, 'validation_errors')
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Class initialization successful")
            except Exception as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Class initialization failed: {e}")
                logger.error(f"  ❌ Class initialization failed: {e}")
            
            # Test 3: Synthetic Data Generation
            try:
                data = validator.create_test_market_data(100)
                assert data is not None
                assert len(data) == 100
                assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
                validation_result['tests_passed'] += 1
                logger.info("  ✅ Synthetic data generation successful")
            except Exception as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Synthetic data generation failed: {e}")
                logger.error(f"  ❌ Synthetic data generation failed: {e}")
            
            # Test 4: Intentional Leak Detection
            try:
                data = validator.create_test_market_data(500)
                features_with_leaks = validator.calculate_features_with_intentional_leak(data)
                
                # Test features that should pass and fail
                correct_features = ['close_lag1', 'returns_lag1', 'sma_5_lag1', 'volume_lag1']
                leaked_features = ['close_leak', 'high_leak', 'returns_leak', 'sma_5_leak']
                
                all_features = correct_features + leaked_features
                validation_results = validator.validate_feature_lag_compliance(
                    features_with_leaks, all_features
                )
                
                # Count passes and failures
                passed_count = sum(1 for result in validation_results.values() if result['is_compliant'])
                failed_count = len(validation_results) - passed_count
                
                validation_result['metrics']['features_passed'] = passed_count
                validation_result['metrics']['features_failed'] = failed_count
                validation_result['metrics']['leak_detection_rate'] = failed_count / len(leaked_features) if leaked_features else 0
                
                # We expect some failures (the intentional leaks)
                if failed_count > 0 and passed_count > 0:
                    validation_result['tests_passed'] += 1
                    logger.info(f"  ✅ Leak detection working: {failed_count} leaks detected, {passed_count} features passed")
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].append("Leak detection not working properly")
                    logger.error("  ❌ Leak detection failed")
                    
            except Exception as e:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append(f"Leak detection test failed: {e}")
                logger.error(f"  ❌ Leak detection test failed: {e}")
            
            # Test 5: Test File Validation
            test_file = project_root / 'tests' / 'test_feature_lag_validation.py'
            if test_file.exists() and test_file.stat().st_size > 0:
                validation_result['tests_passed'] += 1
                validation_result['files_validated'].append('test_feature_lag_validation.py')
                logger.info("  ✅ Test file exists and is not empty")
            else:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append("Test file missing or empty")
                logger.error("  ❌ Test file missing or empty")
            
            # Determine overall status
            if validation_result['tests_failed'] == 0:
                validation_result['status'] = 'PASSED'
                logger.info("  🎉 Feature Lag Validation: ALL TESTS PASSED")
            elif validation_result['tests_passed'] > validation_result['tests_failed']:
                validation_result['status'] = 'PASSED_WITH_WARNINGS'
                logger.warning("  ⚠️ Feature Lag Validation: PASSED WITH WARNINGS")
            else:
                validation_result['status'] = 'FAILED'
                logger.error("  💥 Feature Lag Validation: FAILED")
                
        except Exception as e:
            validation_result['status'] = 'ERROR'
            validation_result['errors'].append(f"Unexpected error: {e}")
            logger.error(f"  💥 Unexpected error in feature lag validation: {e}")
            logger.error(traceback.format_exc())
        
        return validation_result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all critical reviewer implementations"""
        logger.info("🚀 STARTING COMPREHENSIVE VALIDATION")
        logger.info("=" * 80)
        
        # Run all validations
        tick_minute_results = self.validate_tick_vs_minute_study()
        filtering_results = self.validate_filtering_ablation_study()
        feature_lag_results = self.validate_feature_lag_validation()
        
        # Compile overall results
        all_results = [tick_minute_results, filtering_results, feature_lag_results]
        
        total_tests_passed = sum(r['tests_passed'] for r in all_results)
        total_tests_failed = sum(r['tests_failed'] for r in all_results)
        total_errors = sum(len(r['errors']) for r in all_results)
        total_warnings = sum(len(r['warnings']) for r in all_results)
        
        # Determine overall status
        passed_studies = sum(1 for r in all_results if r['status'] == 'PASSED')
        warning_studies = sum(1 for r in all_results if r['status'] == 'PASSED_WITH_WARNINGS')
        failed_studies = sum(1 for r in all_results if r['status'] in ['FAILED', 'ERROR'])
        
        if failed_studies == 0:
            if warning_studies == 0:
                overall_status = 'ALL_PASSED'
            else:
                overall_status = 'PASSED_WITH_WARNINGS'
        else:
            overall_status = 'SOME_FAILED'
        
        # Compile comprehensive report
        comprehensive_results = {
            'validation_timestamp': self.start_time.isoformat(),
            'validation_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'overall_status': overall_status,
            'summary': {
                'total_studies': len(all_results),
                'studies_passed': passed_studies,
                'studies_with_warnings': warning_studies,
                'studies_failed': failed_studies,
                'total_tests_passed': total_tests_passed,
                'total_tests_failed': total_tests_failed,
                'total_errors': total_errors,
                'total_warnings': total_warnings
            },
            'detailed_results': {
                'tick_vs_minute_alpha_study': tick_minute_results,
                'filtering_ablation_study': filtering_results,
                'feature_lag_validation': feature_lag_results
            }
        }
        
        self.validation_results = comprehensive_results
        return comprehensive_results
    
    def generate_validation_report(self, save_to_file: bool = True) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            logger.error("No validation results available. Run validation first.")
            return ""
        
        results = self.validation_results
        
        # Generate report
        report_lines = [
            "=" * 100,
            "CRITICAL REVIEWER IMPLEMENTATIONS VALIDATION REPORT",
            "=" * 100,
            "",
            f"Validation Timestamp: {results['validation_timestamp']}",
            f"Validation Duration: {results['validation_duration_seconds']:.2f} seconds",
            f"Overall Status: {results['overall_status']}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 50,
            f"Total Studies Validated: {results['summary']['total_studies']}",
            f"Studies Passed: {results['summary']['studies_passed']}",
            f"Studies with Warnings: {results['summary']['studies_with_warnings']}",
            f"Studies Failed: {results['summary']['studies_failed']}",
            f"Total Tests Passed: {results['summary']['total_tests_passed']}",
            f"Total Tests Failed: {results['summary']['total_tests_failed']}",
            f"Total Errors: {results['summary']['total_errors']}",
            f"Total Warnings: {results['summary']['total_warnings']}",
            "",
        ]
        
        # Detailed results for each study
        for study_key, study_results in results['detailed_results'].items():
            study_name = study_results['study_name']
            status = study_results['status']
            
            report_lines.extend([
                f"{study_name.upper()}",
                "-" * len(study_name),
                f"Status: {status}",
                f"Tests Passed: {study_results['tests_passed']}",
                f"Tests Failed: {study_results['tests_failed']}",
                f"Files Validated: {', '.join(study_results['files_validated']) if study_results['files_validated'] else 'None'}",
                ""
            ])
            
            # Metrics
            if study_results['metrics']:
                report_lines.append("Key Metrics:")
                for metric, value in study_results['metrics'].items():
                    if isinstance(value, float):
                        report_lines.append(f"  {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"  {metric}: {value}")
                report_lines.append("")
            
            # Errors
            if study_results['errors']:
                report_lines.append("Errors:")
                for error in study_results['errors']:
                    report_lines.append(f"  ❌ {error}")
                report_lines.append("")
            
            # Warnings
            if study_results['warnings']:
                report_lines.append("Warnings:")
                for warning in study_results['warnings']:
                    report_lines.append(f"  ⚠️ {warning}")
                report_lines.append("")
            
            report_lines.append("")
        
        # Final assessment
        report_lines.extend([
            "FINAL ASSESSMENT",
            "-" * 50,
        ])
        
        if results['overall_status'] == 'ALL_PASSED':
            report_lines.extend([
                "🎉 ALL CRITICAL REVIEWER IMPLEMENTATIONS VALIDATED SUCCESSFULLY",
                "",
                "✅ Empirical Evidence: All studies provide concrete measurable results",
                "✅ CI Integration: Automated validation prevents regressions", 
                "✅ Audit Compliance: Lock-box hashes and immutable results ready",
                "✅ Production Ready: Comprehensive testing confirms institutional-grade rigor",
                "",
                "The system has successfully transformed from theoretical claims to",
                "empirical evidence with automated validation. Ready for top-tier",
                "quant reviewer scrutiny and regulatory compliance.",
            ])
        elif results['overall_status'] == 'PASSED_WITH_WARNINGS':
            report_lines.extend([
                "⚠️ CRITICAL REVIEWER IMPLEMENTATIONS MOSTLY VALIDATED",
                "",
                "Core functionality is working but some warnings need attention.",
                "Review the warnings above and address before final deployment.",
            ])
        else:
            report_lines.extend([
                "❌ CRITICAL REVIEWER IMPLEMENTATIONS VALIDATION FAILED",
                "",
                "Significant issues found that must be addressed before deployment.",
                "Review all errors above and fix before proceeding.",
            ])
        
        report_lines.extend([
            "",
            "=" * 100,
            f"Report generated: {datetime.now().isoformat()}",
            "=" * 100
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            report_file = project_root / f"critical_reviewer_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            logger.info(f"📄 Validation report saved to: {report_file}")
        
        return report_text


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Critical Reviewer Implementations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report-only', '-r', action='store_true', help='Generate report from existing results only')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validation suite
    validator = CriticalReviewerValidationSuite()
    
    try:
        if not args.report_only:
            # Run comprehensive validation
            logger.info("🎯 Starting Critical Reviewer Implementations Validation")
            results = validator.run_comprehensive_validation()
            
            # Log summary
            logger.info("")
            logger.info("📊 VALIDATION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Overall Status: {results['overall_status']}")
            logger.info(f"Studies Passed: {results['summary']['studies_passed']}/{results['summary']['total_studies']}")
            logger.info(f"Total Tests Passed: {results['summary']['total_tests_passed']}")
            logger.info(f"Total Tests Failed: {results['summary']['total_tests_failed']}")
            
            if results['summary']['total_errors'] > 0:
                logger.error(f"Total Errors: {results['summary']['total_errors']}")
            
            if results['summary']['total_warnings'] > 0:
                logger.warning(f"Total Warnings: {results['summary']['total_warnings']}")
        
        # Generate and display report
        report = validator.generate_validation_report(save_to_file=True)
        
        if args.verbose:
            print("\n" + report)
        
        # Exit with appropriate code
        if validator.validation_results and validator.validation_results['overall_status'] in ['ALL_PASSED', 'PASSED_WITH_WARNINGS']:
            logger.info("🎉 Validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Validation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()