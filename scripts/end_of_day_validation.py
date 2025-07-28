#!/usr/bin/env python3
"""
End-of-Day Validation Script
Comprehensive system validation after trading day completion
"""

import os
import sys
import json
import logging
import psycopg2
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndOfDayValidator:
    """Comprehensive end-of-day validation for dual-ticker trading system"""
    
    def __init__(self, trading_date: Optional[str] = None):
        self.trading_date = trading_date or datetime.now().strftime("%Y-%m-%d")
        self.trading_date_obj = datetime.strptime(self.trading_date, "%Y-%m-%d").date()
        
        # Expected trading hours (NYSE)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET
        
        # Expected data parameters
        self.expected_symbols = {'NVDA', 'MSFT'}
        self.expected_trading_minutes = 390  # 6.5 hours * 60 minutes
        self.min_data_completeness = 0.95  # 95% minimum data completeness
        
        # Database connection
        self.db_connection = None
        
        # Validation results
        self.validation_results = {
            'trading_date': self.trading_date,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'summary': {},
            'recommendations': []
        }
    
    def connect_database(self) -> bool:
        """Connect to TimescaleDB"""
        try:
            connection_string = self._build_connection_string()
            self.db_connection = psycopg2.connect(connection_string)
            logger.info("✅ Connected to TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def _build_connection_string(self) -> str:
        """Build database connection string"""
        host = os.getenv('TIMESCALEDB_HOST', 'localhost')
        port = os.getenv('TIMESCALEDB_PORT', '5432')
        database = os.getenv('TIMESCALEDB_DATABASE', 'trading_data')
        username = os.getenv('TIMESCALEDB_USERNAME', 'postgres')
        password = os.getenv('TIMESCALEDB_PASSWORD', 'postgres')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def validate_market_data_completeness(self) -> Dict[str, Any]:
        """Validate market data completeness for the trading day"""
        logger.info(f"🔍 Validating market data completeness for {self.trading_date}")
        
        result = {
            'status': 'UNKNOWN',
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if not self.db_connection:
                result['status'] = 'ERROR'
                result['details']['error'] = 'No database connection'
                return result
            
            with self.db_connection.cursor() as cursor:
                # Get data for trading day
                cursor.execute('''
                    SELECT 
                        symbol,
                        COUNT(*) as bar_count,
                        MIN(timestamp) as first_bar,
                        MAX(timestamp) as last_bar,
                        COUNT(DISTINCT DATE_TRUNC('minute', timestamp)) as unique_minutes
                    FROM market_data 
                    WHERE DATE(timestamp) = %s
                    GROUP BY symbol
                    ORDER BY symbol
                ''', (self.trading_date,))
                
                symbol_data = cursor.fetchall()
                
                if not symbol_data:
                    result['status'] = 'FAIL'
                    result['details']['error'] = f'No market data found for {self.trading_date}'
                    return result
                
                # Analyze each symbol
                symbol_analysis = {}
                overall_completeness_ok = True
                
                for row in symbol_data:
                    symbol, bar_count, first_bar, last_bar, unique_minutes = row
                    
                    # Calculate completeness
                    completeness_pct = unique_minutes / self.expected_trading_minutes
                    
                    # Check market hours coverage
                    first_bar_time = first_bar.time() if first_bar else None
                    last_bar_time = last_bar.time() if last_bar else None
                    
                    market_hours_ok = (
                        first_bar_time and last_bar_time and
                        first_bar_time <= time(9, 45) and  # Started by 9:45 AM
                        last_bar_time >= time(15, 45)      # Ended after 3:45 PM
                    )
                    
                    symbol_ok = (
                        completeness_pct >= self.min_data_completeness and
                        market_hours_ok
                    )
                    
                    if not symbol_ok:
                        overall_completeness_ok = False
                    
                    symbol_analysis[symbol] = {
                        'bar_count': bar_count,
                        'unique_minutes': unique_minutes,
                        'completeness_pct': round(completeness_pct, 4),
                        'first_bar': first_bar.isoformat() if first_bar else None,
                        'last_bar': last_bar.isoformat() if last_bar else None,
                        'market_hours_ok': market_hours_ok,
                        'completeness_ok': completeness_pct >= self.min_data_completeness,
                        'overall_ok': symbol_ok
                    }
                
                # Check for missing symbols
                found_symbols = set(symbol_analysis.keys())
                missing_symbols = self.expected_symbols - found_symbols
                
                result['details'] = {
                    'expected_symbols': list(self.expected_symbols),
                    'found_symbols': list(found_symbols),
                    'missing_symbols': list(missing_symbols),
                    'expected_minutes': self.expected_trading_minutes,
                    'min_completeness_threshold': self.min_data_completeness,
                    'symbols': symbol_analysis
                }
                
                # Overall status
                if missing_symbols:
                    result['status'] = 'FAIL'
                    result['details']['failure_reason'] = f'Missing symbols: {missing_symbols}'
                elif not overall_completeness_ok:
                    result['status'] = 'FAIL'
                    result['details']['failure_reason'] = 'Data completeness below threshold'
                else:
                    result['status'] = 'PASS'
                
        except Exception as e:
            logger.error(f"💥 Market data validation failed: {e}")
            result['status'] = 'ERROR'
            result['details']['error'] = str(e)
        
        return result
    
    def validate_data_quality_reports(self) -> Dict[str, Any]:
        """Validate data quality reports for the trading day"""
        logger.info(f"🔍 Validating data quality reports for {self.trading_date}")
        
        result = {
            'status': 'UNKNOWN',
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if not self.db_connection:
                result['status'] = 'ERROR'
                result['details']['error'] = 'No database connection'
                return result
            
            with self.db_connection.cursor() as cursor:
                # Get quality reports for trading day
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_reports,
                        COUNT(*) FILTER (WHERE status = 'PASS') as passed_reports,
                        COUNT(*) FILTER (WHERE status = 'FAIL') as failed_reports,
                        COUNT(*) FILTER (WHERE pipeline_action = 'BLOCK') as blocked_reports,
                        AVG(pass_rate) as avg_pass_rate,
                        MIN(timestamp) as first_report,
                        MAX(timestamp) as last_report
                    FROM data_quality_reports 
                    WHERE DATE(timestamp) = %s
                ''', (self.trading_date,))
                
                report_stats = cursor.fetchone()
                
                if not report_stats or report_stats[0] == 0:
                    result['status'] = 'WARN'
                    result['details'] = {
                        'warning': f'No quality reports found for {self.trading_date}',
                        'total_reports': 0
                    }
                    return result
                
                total_reports, passed_reports, failed_reports, blocked_reports, avg_pass_rate, first_report, last_report = report_stats
                
                # Calculate metrics
                pass_rate = passed_reports / total_reports if total_reports > 0 else 0.0
                block_rate = blocked_reports / total_reports if total_reports > 0 else 0.0
                
                # Get latest report details
                cursor.execute('''
                    SELECT status, environment, pipeline_action, pass_rate, checks
                    FROM data_quality_reports 
                    WHERE DATE(timestamp) = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (self.trading_date,))
                
                latest_report = cursor.fetchone()
                
                result['details'] = {
                    'total_reports': total_reports,
                    'passed_reports': passed_reports,
                    'failed_reports': failed_reports,
                    'blocked_reports': blocked_reports,
                    'pass_rate': round(pass_rate, 4),
                    'block_rate': round(block_rate, 4),
                    'avg_pass_rate': round(float(avg_pass_rate or 0), 4),
                    'first_report': first_report.isoformat() if first_report else None,
                    'last_report': last_report.isoformat() if last_report else None,
                    'latest_report': {
                        'status': latest_report[0] if latest_report else None,
                        'environment': latest_report[1] if latest_report else None,
                        'pipeline_action': latest_report[2] if latest_report else None,
                        'pass_rate': float(latest_report[3]) if latest_report and latest_report[3] else None
                    } if latest_report else None
                }
                
                # Determine status
                if block_rate > 0.1:  # More than 10% blocked
                    result['status'] = 'FAIL'
                    result['details']['failure_reason'] = f'High block rate: {block_rate:.1%}'
                elif pass_rate < 0.8:  # Less than 80% pass rate
                    result['status'] = 'WARN'
                    result['details']['warning'] = f'Low pass rate: {pass_rate:.1%}'
                else:
                    result['status'] = 'PASS'
                
        except Exception as e:
            logger.error(f"💥 Quality reports validation failed: {e}")
            result['status'] = 'ERROR'
            result['details']['error'] = str(e)
        
        return result
    
    def validate_system_performance(self) -> Dict[str, Any]:
        """Validate system performance metrics"""
        logger.info(f"🔍 Validating system performance for {self.trading_date}")
        
        result = {
            'status': 'UNKNOWN',
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check data ingestion latency
            raw_files = list(Path("raw").glob(f"*{self.trading_date.replace('-', '')}*.csv"))
            
            if raw_files:
                latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
                file_age_hours = (datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)).total_seconds() / 3600
                
                data_freshness_ok = file_age_hours < 2  # Data should be less than 2 hours old
            else:
                data_freshness_ok = False
                file_age_hours = 999
            
            # Check log files for errors
            log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
            error_count = 0
            warning_count = 0
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        error_count += content.count('ERROR')
                        warning_count += content.count('WARNING')
                except:
                    pass
            
            # Check disk space
            disk_usage_ok = True
            try:
                import shutil
                total, used, free = shutil.disk_usage(Path.cwd())
                free_gb = free / (1024**3)
                disk_usage_ok = free_gb > 1  # At least 1GB free
            except:
                free_gb = 0
                disk_usage_ok = False
            
            result['details'] = {
                'data_freshness': {
                    'latest_file_age_hours': round(file_age_hours, 2),
                    'freshness_ok': data_freshness_ok,
                    'threshold_hours': 2
                },
                'log_analysis': {
                    'error_count': error_count,
                    'warning_count': warning_count,
                    'log_files_checked': len(log_files)
                },
                'disk_space': {
                    'free_gb': round(free_gb, 2),
                    'disk_usage_ok': disk_usage_ok,
                    'min_free_gb': 1
                }
            }
            
            # Overall status
            if not data_freshness_ok:
                result['status'] = 'WARN'
                result['details']['warning'] = f'Data freshness issue: {file_age_hours:.1f} hours old'
            elif error_count > 10:
                result['status'] = 'WARN'
                result['details']['warning'] = f'High error count: {error_count} errors in logs'
            elif not disk_usage_ok:
                result['status'] = 'WARN'
                result['details']['warning'] = f'Low disk space: {free_gb:.1f} GB free'
            else:
                result['status'] = 'PASS'
                
        except Exception as e:
            logger.error(f"💥 System performance validation failed: {e}")
            result['status'] = 'ERROR'
            result['details']['error'] = str(e)
        
        return result
    
    def validate_file_integrity(self) -> Dict[str, Any]:
        """Validate file integrity and backup status"""
        logger.info(f"🔍 Validating file integrity for {self.trading_date}")
        
        result = {
            'status': 'UNKNOWN',
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check raw data files
            raw_dir = Path("raw")
            raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
            raw_files_today = [f for f in raw_files if self.trading_date.replace('-', '') in f.name]
            
            # Check quality reports
            qc_files = list(raw_dir.glob("qc_report*.json")) if raw_dir.exists() else []
            qc_files_today = [f for f in qc_files if self.trading_date.replace('-', '') in f.name]
            
            # Check model files
            model_dir = Path("models")
            recent_models = []
            if model_dir.exists():
                for model_file in model_dir.rglob("*.zip"):
                    if (datetime.now() - datetime.fromtimestamp(model_file.stat().st_mtime)).days <= 1:
                        recent_models.append(model_file)
            
            # Calculate file sizes
            total_raw_size = sum(f.stat().st_size for f in raw_files_today)
            total_qc_size = sum(f.stat().st_size for f in qc_files_today)
            
            result['details'] = {
                'raw_data_files': {
                    'total_files': len(raw_files),
                    'today_files': len(raw_files_today),
                    'today_size_mb': round(total_raw_size / (1024**2), 2),
                    'files': [f.name for f in raw_files_today]
                },
                'quality_reports': {
                    'total_files': len(qc_files),
                    'today_files': len(qc_files_today),
                    'today_size_kb': round(total_qc_size / 1024, 2),
                    'files': [f.name for f in qc_files_today]
                },
                'model_files': {
                    'recent_models': len(recent_models),
                    'model_files': [f.name for f in recent_models]
                }
            }
            
            # Validation criteria
            if len(raw_files_today) == 0:
                result['status'] = 'FAIL'
                result['details']['failure_reason'] = 'No raw data files found for today'
            elif total_raw_size < 1024:  # Less than 1KB total
                result['status'] = 'WARN'
                result['details']['warning'] = 'Raw data files are very small'
            else:
                result['status'] = 'PASS'
                
        except Exception as e:
            logger.error(f"💥 File integrity validation failed: {e}")
            result['status'] = 'ERROR'
            result['details']['error'] = str(e)
        
        return result
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check each validation result
        for check_name, check_result in self.validation_results['checks'].items():
            status = check_result.get('status', 'UNKNOWN')
            
            if status == 'FAIL':
                if check_name == 'market_data_completeness':
                    if 'missing_symbols' in check_result.get('details', {}):
                        missing = check_result['details']['missing_symbols']
                        recommendations.append(f"🔧 Fix data ingestion for missing symbols: {missing}")
                    else:
                        recommendations.append("🔧 Investigate market data completeness issues")
                
                elif check_name == 'file_integrity':
                    recommendations.append("📁 Check data pipeline and file generation processes")
                
            elif status == 'WARN':
                if check_name == 'data_quality_reports':
                    recommendations.append("📊 Review and improve data quality validation rules")
                
                elif check_name == 'system_performance':
                    if 'data_freshness' in check_result.get('details', {}):
                        recommendations.append("⏰ Investigate data ingestion delays")
                    if 'disk_space' in check_result.get('details', {}):
                        recommendations.append("💾 Clean up old files or increase disk space")
                    if 'log_analysis' in check_result.get('details', {}):
                        recommendations.append("🔍 Review system logs for recurring issues")
            
            elif status == 'ERROR':
                recommendations.append(f"🚨 Fix technical issues in {check_name} validation")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ System is healthy - continue monitoring")
        
        return recommendations
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete end-of-day validation"""
        logger.info(f"🚀 Starting end-of-day validation for {self.trading_date}")
        
        start_time = datetime.now()
        
        # Connect to database
        db_connected = self.connect_database()
        
        # Run all validation checks
        checks = [
            ('market_data_completeness', self.validate_market_data_completeness),
            ('data_quality_reports', self.validate_data_quality_reports),
            ('system_performance', self.validate_system_performance),
            ('file_integrity', self.validate_file_integrity)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"🔍 Running {check_name} validation...")
            try:
                result = check_func()
                self.validation_results['checks'][check_name] = result
                
                status_emoji = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'ERROR': '💥'}.get(result['status'], '❓')
                logger.info(f"{status_emoji} {check_name}: {result['status']}")
                
            except Exception as e:
                logger.error(f"💥 {check_name} validation failed: {e}")
                self.validation_results['checks'][check_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Calculate overall status
        statuses = [check['status'] for check in self.validation_results['checks'].values()]
        
        if 'ERROR' in statuses:
            overall_status = 'ERROR'
        elif 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'
        
        self.validation_results['overall_status'] = overall_status
        
        # Generate summary
        status_counts = {status: statuses.count(status) for status in set(statuses)}
        
        self.validation_results['summary'] = {
            'total_checks': len(statuses),
            'status_counts': status_counts,
            'pass_rate': statuses.count('PASS') / len(statuses) if statuses else 0.0,
            'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
            'database_connected': db_connected
        }
        
        # Generate recommendations
        self.validation_results['recommendations'] = self.generate_recommendations()
        
        # Cleanup
        if self.db_connection:
            self.db_connection.close()
        
        logger.info(f"🎉 End-of-day validation complete: {overall_status}")
        return self.validation_results
    
    def save_report(self, output_file: str = "eod_validation_report.json") -> str:
        """Save validation report to file"""
        try:
            output_path = Path(output_file)
            
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            logger.info(f"✅ End-of-day report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
            raise
    
    def print_summary(self):
        """Print human-readable validation summary"""
        status_emoji = {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌',
            'ERROR': '💥',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 END-OF-DAY VALIDATION REPORT")
        print("=" * 60)
        print(f"Trading Date: {self.validation_results['trading_date']}")
        print(f"Overall Status: {status_emoji.get(self.validation_results['overall_status'], '❓')} {self.validation_results['overall_status']}")
        
        # Summary stats
        if 'summary' in self.validation_results:
            summary = self.validation_results['summary']
            print(f"Total Checks: {summary['total_checks']}")
            print(f"Pass Rate: {summary['pass_rate']:.1%}")
            print(f"Execution Time: {summary['execution_time_seconds']:.2f} seconds")
        
        # Individual check results
        print(f"\n🔍 VALIDATION CHECKS:")
        for check_name, check_result in self.validation_results['checks'].items():
            status = check_result['status']
            emoji = status_emoji.get(status, '❓')
            print(f"{emoji} {check_name.replace('_', ' ').title()}: {status}")
            
            # Show key details for failed/warning checks
            if status in ['FAIL', 'WARN'] and 'details' in check_result:
                if 'failure_reason' in check_result['details']:
                    print(f"   Reason: {check_result['details']['failure_reason']}")
                elif 'warning' in check_result['details']:
                    print(f"   Warning: {check_result['details']['warning']}")
        
        # Recommendations
        if self.validation_results.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(self.validation_results['recommendations'], 1):
                print(f"{i}. {recommendation}")

def main():
    """CLI interface for end-of-day validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="End-of-Day Validation Script")
    parser.add_argument('--date', 
                       help='Trading date to validate (YYYY-MM-DD, default: today)')
    parser.add_argument('--output', default='eod_validation_report.json',
                       help='Output report file (default: eod_validation_report.json)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create validator
        validator = EndOfDayValidator(trading_date=args.date)
        
        # Run validation
        results = validator.run_full_validation()
        
        # Save report
        report_file = validator.save_report(args.output)
        
        # Print summary
        validator.print_summary()
        
        print(f"\n📄 Full report saved: {report_file}")
        
        # Exit code based on overall status
        status_codes = {
            'PASS': 0,
            'WARN': 1,
            'FAIL': 2,
            'ERROR': 3
        }
        
        exit_code = status_codes.get(results['overall_status'], 3)
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"💥 End-of-day validation failed: {e}")
        sys.exit(4)

if __name__ == "__main__":
    main()