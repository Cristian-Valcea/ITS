#!/usr/bin/env python3
"""
Data Quality Gate Runner
Validates data quality and generates qc_report.json with pass/fail status
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.quality_validator import DataQualityValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityGate:
    """Data quality gate with configurable thresholds and reporting"""
    
    def __init__(self, max_missing: float = 0.05, environment: str = "ci"):
        self.max_missing = max_missing
        self.environment = environment
        self.validator = DataQualityValidator()
        self.raw_data_dir = Path("raw")
        self.output_file = Path("qc_report.json")
        
    def find_latest_data_files(self) -> List[Path]:
        """Find latest dual-ticker data files in raw/ directory"""
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory {self.raw_data_dir} does not exist")
            return []
        
        # Look for CSV files with dual_ticker pattern
        csv_files = list(self.raw_data_dir.glob("dual_ticker_*.csv"))
        
        if not csv_files:
            logger.warning(f"No dual_ticker CSV files found in {self.raw_data_dir}")
            return []
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        logger.info(f"Found {len(csv_files)} data files, using latest: {csv_files[0]}")
        return csv_files
    
    def load_data_file(self, file_path: Path) -> Optional[Any]:
        """Load data file and return DataFrame"""
        try:
            import pandas as pd
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Loaded CSV with {len(df)} rows: {file_path}")
                return df
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded JSON: {file_path}")
                return data
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def validate_data_completeness(self, df) -> Dict[str, Any]:
        """Check data completeness against thresholds"""
        import pandas as pd
        
        results = {
            'missing_data_check': {
                'status': 'UNKNOWN',
                'details': {}
            }
        }
        
        if df is None or len(df) == 0:
            results['missing_data_check']['status'] = 'FAIL'
            results['missing_data_check']['details'] = {
                'error': 'No data available',
                'missing_pct': 1.0
            }
            return results
        
        # Check required columns
        required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['missing_data_check']['status'] = 'FAIL'
            results['missing_data_check']['details'] = {
                'error': f'Missing required columns: {missing_columns}',
                'missing_pct': len(missing_columns) / len(required_columns)
            }
            return results
        
        # Check for missing values
        total_cells = len(df) * len(required_columns)
        missing_cells = df[required_columns].isnull().sum().sum()
        missing_pct = missing_cells / total_cells if total_cells > 0 else 1.0
        
        # Check symbol coverage
        expected_symbols = {'NVDA', 'MSFT'}
        actual_symbols = set(df['symbol'].unique()) if 'symbol' in df.columns else set()
        missing_symbols = expected_symbols - actual_symbols
        
        details = {
            'total_rows': len(df),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells),
            'missing_pct': round(missing_pct, 4),
            'threshold': self.max_missing,
            'expected_symbols': list(expected_symbols),
            'actual_symbols': list(actual_symbols),
            'missing_symbols': list(missing_symbols)
        }
        
        # Determine pass/fail
        completeness_ok = missing_pct <= self.max_missing
        symbols_ok = len(missing_symbols) == 0
        
        if completeness_ok and symbols_ok:
            results['missing_data_check']['status'] = 'PASS'
        else:
            results['missing_data_check']['status'] = 'FAIL'
            if not completeness_ok:
                details['failure_reason'] = f'Missing data {missing_pct:.1%} exceeds threshold {self.max_missing:.1%}'
            if not symbols_ok:
                details['failure_reason'] = f'Missing symbols: {missing_symbols}'
        
        results['missing_data_check']['details'] = details
        return results
    
    def validate_ohlc_relationships(self, df) -> Dict[str, Any]:
        """Validate OHLC price relationships"""
        results = {
            'ohlc_validation': {
                'status': 'UNKNOWN',
                'details': {}
            }
        }
        
        if df is None or len(df) == 0:
            results['ohlc_validation']['status'] = 'FAIL'
            results['ohlc_validation']['details'] = {'error': 'No data for OHLC validation'}
            return results
        
        try:
            # Check OHLC columns exist
            ohlc_columns = ['open', 'high', 'low', 'close']
            missing_ohlc = [col for col in ohlc_columns if col not in df.columns]
            
            if missing_ohlc:
                results['ohlc_validation']['status'] = 'FAIL'
                results['ohlc_validation']['details'] = {
                    'error': f'Missing OHLC columns: {missing_ohlc}'
                }
                return results
            
            # Validate OHLC relationships
            violations = 0
            total_bars = len(df)
            
            # High >= Low
            high_low_violations = (df['high'] < df['low']).sum()
            violations += high_low_violations
            
            # High >= Open
            high_open_violations = (df['high'] < df['open']).sum()
            violations += high_open_violations
            
            # High >= Close
            high_close_violations = (df['high'] < df['close']).sum()
            violations += high_close_violations
            
            # Low <= Open
            low_open_violations = (df['low'] > df['open']).sum()
            violations += low_open_violations
            
            # Low <= Close
            low_close_violations = (df['low'] > df['close']).sum()
            violations += low_close_violations
            
            violation_pct = violations / (total_bars * 5) if total_bars > 0 else 0.0  # 5 checks per bar
            
            details = {
                'total_bars': total_bars,
                'total_violations': int(violations),
                'violation_pct': round(violation_pct, 4),
                'breakdown': {
                    'high_lt_low': int(high_low_violations),
                    'high_lt_open': int(high_open_violations),
                    'high_lt_close': int(high_close_violations),
                    'low_gt_open': int(low_open_violations),
                    'low_gt_close': int(low_close_violations)
                }
            }
            
            # OHLC should have zero violations
            if violations == 0:
                results['ohlc_validation']['status'] = 'PASS'
            else:
                results['ohlc_validation']['status'] = 'FAIL'
                details['failure_reason'] = f'{violations} OHLC violations found'
            
            results['ohlc_validation']['details'] = details
            
        except Exception as e:
            results['ohlc_validation']['status'] = 'ERROR'
            results['ohlc_validation']['details'] = {'error': str(e)}
        
        return results
    
    def validate_dual_ticker_sync(self, df) -> Dict[str, Any]:
        """Validate NVDA/MSFT timestamp synchronization"""
        results = {
            'dual_ticker_sync': {
                'status': 'UNKNOWN',
                'details': {}
            }
        }
        
        if df is None or len(df) == 0:
            results['dual_ticker_sync']['status'] = 'FAIL'
            results['dual_ticker_sync']['details'] = {'error': 'No data for sync validation'}
            return results
        
        try:
            if 'symbol' not in df.columns or 'timestamp' not in df.columns:
                results['dual_ticker_sync']['status'] = 'FAIL'
                results['dual_ticker_sync']['details'] = {
                    'error': 'Missing symbol or timestamp columns'
                }
                return results
            
            # Get timestamps for each symbol
            nvda_timestamps = set(df[df['symbol'] == 'NVDA']['timestamp'].unique())
            msft_timestamps = set(df[df['symbol'] == 'MSFT']['timestamp'].unique())
            
            # Calculate sync metrics
            common_timestamps = nvda_timestamps & msft_timestamps
            total_unique_timestamps = nvda_timestamps | msft_timestamps
            
            sync_pct = len(common_timestamps) / len(total_unique_timestamps) if total_unique_timestamps else 0.0
            
            details = {
                'nvda_timestamps': len(nvda_timestamps),
                'msft_timestamps': len(msft_timestamps),
                'common_timestamps': len(common_timestamps),
                'total_unique_timestamps': len(total_unique_timestamps),
                'sync_pct': round(sync_pct, 4),
                'sync_threshold': 0.8  # 80% sync required
            }
            
            # Check sync threshold (80% minimum)
            if sync_pct >= 0.8:
                results['dual_ticker_sync']['status'] = 'PASS'
            else:
                results['dual_ticker_sync']['status'] = 'FAIL'
                details['failure_reason'] = f'Sync {sync_pct:.1%} below 80% threshold'
            
            results['dual_ticker_sync']['details'] = details
            
        except Exception as e:
            results['dual_ticker_sync']['status'] = 'ERROR'
            results['dual_ticker_sync']['details'] = {'error': str(e)}
        
        return results
    
    def run_quality_gate(self) -> Dict[str, Any]:
        """Run complete data quality gate"""
        logger.info(f"🚀 Starting data quality gate (max_missing={self.max_missing:.1%})")
        
        start_time = datetime.now()
        
        # Initialize report
        report = {
            'timestamp': start_time.isoformat(),
            'environment': self.environment,
            'max_missing_threshold': self.max_missing,
            'status': 'UNKNOWN',
            'checks': {},
            'summary': {},
            'pipeline_action': 'UNKNOWN'
        }
        
        try:
            # Find latest data files
            data_files = self.find_latest_data_files()
            
            if not data_files:
                report['status'] = 'FAIL'
                report['checks']['file_availability'] = {
                    'status': 'FAIL',
                    'details': {'error': 'No data files found in raw/ directory'}
                }
                report['pipeline_action'] = 'BLOCK'
                return report
            
            # Load latest data file
            latest_file = data_files[0]
            df = self.load_data_file(latest_file)
            
            if df is None:
                report['status'] = 'FAIL'
                report['checks']['file_loading'] = {
                    'status': 'FAIL',
                    'details': {'error': f'Failed to load {latest_file}'}
                }
                report['pipeline_action'] = 'BLOCK'
                return report
            
            # Record file info
            report['data_source'] = {
                'file_path': str(latest_file),
                'file_size_bytes': latest_file.stat().st_size,
                'modification_time': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
                'total_rows': len(df) if hasattr(df, '__len__') else 'unknown'
            }
            
            # Run quality checks
            logger.info("🔍 Running data completeness check...")
            completeness_results = self.validate_data_completeness(df)
            report['checks'].update(completeness_results)
            
            logger.info("🔍 Running OHLC validation...")
            ohlc_results = self.validate_ohlc_relationships(df)
            report['checks'].update(ohlc_results)
            
            logger.info("🔍 Running dual-ticker sync validation...")
            sync_results = self.validate_dual_ticker_sync(df)
            report['checks'].update(sync_results)
            
            # Calculate overall status
            all_statuses = [check['status'] for check in report['checks'].values()]
            passed_checks = sum(1 for status in all_statuses if status == 'PASS')
            total_checks = len(all_statuses)
            
            report['summary'] = {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': total_checks - passed_checks,
                'pass_rate': round(passed_checks / total_checks, 4) if total_checks > 0 else 0.0
            }
            
            # Determine overall status and pipeline action
            if all(status == 'PASS' for status in all_statuses):
                report['status'] = 'PASS'
                report['pipeline_action'] = 'CONTINUE'
            elif any(status == 'ERROR' for status in all_statuses):
                report['status'] = 'ERROR'
                report['pipeline_action'] = 'BLOCK'
            else:
                # Some checks failed
                critical_failures = ['missing_data_check', 'ohlc_validation']
                critical_failed = any(
                    report['checks'].get(check, {}).get('status') == 'FAIL'
                    for check in critical_failures
                )
                
                if critical_failed:
                    report['status'] = 'FAIL'
                    report['pipeline_action'] = 'BLOCK'
                else:
                    report['status'] = 'WARN'
                    report['pipeline_action'] = 'WARN'
            
        except Exception as e:
            logger.error(f"💥 Quality gate execution failed: {e}")
            report['status'] = 'ERROR'
            report['checks']['execution_error'] = {
                'status': 'ERROR',
                'details': {'error': str(e)}
            }
            report['pipeline_action'] = 'BLOCK'
        
        finally:
            end_time = datetime.now()
            report['execution_time_seconds'] = (end_time - start_time).total_seconds()
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save QC report to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ QC report saved: {self.output_file}")
            return str(self.output_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to save QC report: {e}")
            raise
    
    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary"""
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'ERROR': '💥',
            'UNKNOWN': '❓'
        }
        
        action_emoji = {
            'CONTINUE': '🚀',
            'WARN': '⚠️',
            'BLOCK': '🚫',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 DATA QUALITY GATE REPORT")
        print("=" * 50)
        print(f"Status: {status_emoji.get(report['status'], '❓')} {report['status']}")
        print(f"Pipeline Action: {action_emoji.get(report['pipeline_action'], '❓')} {report['pipeline_action']}")
        print(f"Environment: {report['environment']}")
        print(f"Max Missing Threshold: {report['max_missing_threshold']:.1%}")
        
        if 'data_source' in report:
            print(f"\nData Source:")
            print(f"  File: {report['data_source']['file_path']}")
            print(f"  Rows: {report['data_source']['total_rows']}")
            print(f"  Size: {report['data_source']['file_size_bytes']:,} bytes")
        
        print(f"\nCheck Results:")
        for check_name, check_result in report['checks'].items():
            status = check_result['status']
            emoji = status_emoji.get(status, '❓')
            print(f"  {emoji} {check_name.replace('_', ' ').title()}: {status}")
            
            if status == 'FAIL' and 'failure_reason' in check_result.get('details', {}):
                print(f"    Reason: {check_result['details']['failure_reason']}")
        
        if 'summary' in report:
            summary = report['summary']
            print(f"\nSummary:")
            print(f"  Total Checks: {summary['total_checks']}")
            print(f"  Passed: {summary['passed_checks']}")
            print(f"  Failed: {summary['failed_checks']}")
            print(f"  Pass Rate: {summary['pass_rate']:.1%}")
        
        print(f"\nExecution Time: {report.get('execution_time_seconds', 0):.2f} seconds")
        print(f"Report saved: {self.output_file}")

def main():
    """CLI interface for data quality gate"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Gate Runner")
    parser.add_argument('--max-missing', type=float, default=0.05,
                       help='Maximum missing data percentage (default: 0.05 = 5%)')
    parser.add_argument('--environment', choices=['ci', 'staging', 'production'], default='ci',
                       help='Environment (affects thresholds)')
    parser.add_argument('--output', default='qc_report.json',
                       help='Output report file (default: qc_report.json)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Adjust thresholds by environment
    environment_thresholds = {
        'ci': 0.05,          # 5% missing allowed in CI
        'staging': 0.02,     # 2% missing allowed in staging
        'production': 0.01   # 1% missing allowed in production
    }
    
    max_missing = args.max_missing
    if args.environment in environment_thresholds:
        max_missing = min(max_missing, environment_thresholds[args.environment])
    
    # Create quality gate
    gate = DataQualityGate(
        max_missing=max_missing,
        environment=args.environment
    )
    gate.output_file = Path(args.output)
    
    try:
        # Run quality gate
        report = gate.run_quality_gate()
        
        # Save report
        gate.save_report(report)
        
        # Print summary
        gate.print_summary(report)
        
        # Exit code based on pipeline action
        exit_codes = {
            'CONTINUE': 0,  # Success - continue pipeline
            'WARN': 0,      # Success with warnings
            'BLOCK': 1,     # Failure - block pipeline
            'UNKNOWN': 2    # Error - unknown status
        }
        
        exit_code = exit_codes.get(report['pipeline_action'], 2)
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"💥 Quality gate failed: {e}")
        
        # Create minimal error report
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'ERROR',
            'pipeline_action': 'BLOCK',
            'error': str(e)
        }
        
        try:
            gate.save_report(error_report)
        except:
            pass  # Ignore save errors in error case
        
        sys.exit(2)

if __name__ == "__main__":
    main()