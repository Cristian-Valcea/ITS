#!/usr/bin/env python3
"""
TimescaleDB Data Loader
Loads dual-ticker data from raw files into TimescaleDB hypertables
"""

import os
import sys
import json
import logging
import psycopg2
from datetime import datetime, timedelta
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

class TimescaleDBLoader:
    """Load dual-ticker data into TimescaleDB hypertables"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or self._build_connection_string()
        self.raw_data_dir = Path("raw")
        self.connection = None
        
        # Table configurations
        self.tables = {
            'market_data': {
                'schema': '''
                    CREATE TABLE IF NOT EXISTS market_data (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        open DECIMAL(10,4) NOT NULL,
                        high DECIMAL(10,4) NOT NULL,
                        low DECIMAL(10,4) NOT NULL,
                        close DECIMAL(10,4) NOT NULL,
                        volume BIGINT NOT NULL,
                        source VARCHAR(50) DEFAULT 'unknown',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                ''',
                'hypertable': "SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);",
                'indexes': [
                    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_market_data_source ON market_data (source);"
                ]
            },
            'data_quality_reports': {
                'schema': '''
                    CREATE TABLE IF NOT EXISTS data_quality_reports (
                        timestamp TIMESTAMPTZ NOT NULL,
                        report_id VARCHAR(100) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        environment VARCHAR(20) NOT NULL,
                        pipeline_action VARCHAR(20) NOT NULL,
                        total_checks INTEGER NOT NULL,
                        passed_checks INTEGER NOT NULL,
                        failed_checks INTEGER NOT NULL,
                        pass_rate DECIMAL(5,4) NOT NULL,
                        data_source JSONB,
                        checks JSONB,
                        execution_time_seconds DECIMAL(10,4),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                ''',
                'hypertable': "SELECT create_hypertable('data_quality_reports', 'timestamp', if_not_exists => TRUE);",
                'indexes': [
                    "CREATE INDEX IF NOT EXISTS idx_dq_reports_status ON data_quality_reports (status);",
                    "CREATE INDEX IF NOT EXISTS idx_dq_reports_environment ON data_quality_reports (environment);"
                ]
            }
        }
    
    def _build_connection_string(self) -> str:
        """Build connection string from environment variables"""
        host = os.getenv('TIMESCALEDB_HOST', 'localhost')
        port = os.getenv('TIMESCALEDB_PORT', '5432')
        database = os.getenv('TIMESCALEDB_DATABASE', 'trading_data')
        username = os.getenv('TIMESCALEDB_USERNAME', 'postgres')
        password = os.getenv('TIMESCALEDB_PASSWORD', 'postgres')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def connect(self) -> bool:
        """Connect to TimescaleDB"""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = True
            
            # Test connection
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Connected to TimescaleDB: {version}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to TimescaleDB: {e}")
            logger.info(f"Connection string: {self.connection_string.replace(password, '***') if 'password' in locals() else 'N/A'}")
            return False
    
    def disconnect(self):
        """Disconnect from TimescaleDB"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("🔌 Disconnected from TimescaleDB")
    
    def create_tables(self) -> bool:
        """Create tables and hypertables"""
        if not self.connection:
            logger.error("❌ No database connection")
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Enable TimescaleDB extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                logger.info("✅ TimescaleDB extension enabled")
                
                for table_name, config in self.tables.items():
                    logger.info(f"🔧 Creating table: {table_name}")
                    
                    # Create table
                    cursor.execute(config['schema'])
                    logger.info(f"  ✅ Table {table_name} created")
                    
                    # Create hypertable
                    try:
                        cursor.execute(config['hypertable'])
                        logger.info(f"  ✅ Hypertable {table_name} created")
                    except psycopg2.Error as e:
                        if "already a hypertable" in str(e):
                            logger.info(f"  ✅ Hypertable {table_name} already exists")
                        else:
                            logger.warning(f"  ⚠️ Hypertable creation failed: {e}")
                    
                    # Create indexes
                    for index_sql in config['indexes']:
                        try:
                            cursor.execute(index_sql)
                            logger.info(f"  ✅ Index created")
                        except psycopg2.Error as e:
                            logger.warning(f"  ⚠️ Index creation failed: {e}")
            
            logger.info("🎉 Database schema setup complete")
            return True
            
        except Exception as e:
            logger.error(f"💥 Failed to create tables: {e}")
            return False
    
    def find_data_files(self, pattern: str = "dual_ticker_*.csv") -> List[Path]:
        """Find data files matching pattern"""
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory {self.raw_data_dir} does not exist")
            return []
        
        files = list(self.raw_data_dir.glob(pattern))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        logger.info(f"Found {len(files)} files matching {pattern}")
        return files
    
    def load_market_data_from_csv(self, csv_file: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load market data from CSV file"""
        logger.info(f"📊 Loading market data from {csv_file}")
        
        stats = {
            'file_path': str(csv_file),
            'total_rows': 0,
            'loaded_rows': 0,
            'skipped_rows': 0,
            'error_rows': 0,
            'symbols': set(),
            'time_range': {},
            'processing_time_seconds': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            stats['total_rows'] = len(df)
            
            if len(df) == 0:
                logger.warning("📭 Empty CSV file")
                return True, stats
            
            # Validate required columns
            required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return False, stats
            
            # Add source column if missing
            if 'source' not in df.columns:
                df['source'] = 'csv_import'
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for supported symbols
            supported_symbols = {'NVDA', 'MSFT'}
            df_filtered = df[df['symbol'].isin(supported_symbols)].copy()
            
            if len(df_filtered) < len(df):
                skipped = len(df) - len(df_filtered)
                logger.info(f"⏭️ Skipped {skipped} rows with unsupported symbols")
                stats['skipped_rows'] = skipped
            
            stats['symbols'] = set(df_filtered['symbol'].unique())
            stats['loaded_rows'] = len(df_filtered)
            
            if len(df_filtered) == 0:
                logger.warning("📭 No supported symbols found in data")
                return True, stats
            
            # Time range
            stats['time_range'] = {
                'start': df_filtered['timestamp'].min().isoformat(),
                'end': df_filtered['timestamp'].max().isoformat()
            }
            
            # Insert data in batches
            batch_size = 1000
            total_inserted = 0
            
            with self.connection.cursor() as cursor:
                for i in range(0, len(df_filtered), batch_size):
                    batch = df_filtered.iloc[i:i+batch_size]
                    
                    # Prepare batch insert
                    values = []
                    for _, row in batch.iterrows():
                        values.append((
                            row['timestamp'],
                            row['symbol'],
                            float(row['open']),
                            float(row['high']),
                            float(row['low']),
                            float(row['close']),
                            int(row['volume']),
                            row['source']
                        ))
                    
                    # Execute batch insert with conflict handling
                    insert_sql = '''
                        INSERT INTO market_data 
                        (timestamp, symbol, open, high, low, close, volume, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source,
                            created_at = NOW()
                    '''
                    
                    try:
                        cursor.executemany(insert_sql, values)
                        total_inserted += len(values)
                        
                        if i % (batch_size * 5) == 0:  # Log every 5 batches
                            logger.info(f"  📊 Inserted {total_inserted}/{len(df_filtered)} rows...")
                    
                    except Exception as e:
                        logger.error(f"❌ Batch insert failed at row {i}: {e}")
                        stats['error_rows'] += len(values)
            
            stats['loaded_rows'] = total_inserted
            end_time = datetime.now()
            stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Market data loaded: {total_inserted} rows in {stats['processing_time_seconds']:.2f}s")
            return True, stats
            
        except Exception as e:
            logger.error(f"💥 Failed to load market data: {e}")
            stats['processing_time_seconds'] = (datetime.now() - start_time).total_seconds()
            return False, stats
    
    def load_quality_report(self, report_file: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load quality report from JSON file"""
        logger.info(f"📋 Loading quality report from {report_file}")
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Extract key fields
            timestamp = pd.to_datetime(report['timestamp'])
            report_id = f"{report.get('environment', 'unknown')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            summary = report.get('summary', {})
            
            # Insert into database
            with self.connection.cursor() as cursor:
                insert_sql = '''
                    INSERT INTO data_quality_reports 
                    (timestamp, report_id, status, environment, pipeline_action,
                     total_checks, passed_checks, failed_checks, pass_rate,
                     data_source, checks, execution_time_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, report_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        pipeline_action = EXCLUDED.pipeline_action,
                        total_checks = EXCLUDED.total_checks,
                        passed_checks = EXCLUDED.passed_checks,
                        failed_checks = EXCLUDED.failed_checks,
                        pass_rate = EXCLUDED.pass_rate,
                        data_source = EXCLUDED.data_source,
                        checks = EXCLUDED.checks,
                        execution_time_seconds = EXCLUDED.execution_time_seconds,
                        created_at = NOW()
                '''
                
                cursor.execute(insert_sql, (
                    timestamp,
                    report_id,
                    report.get('status', 'UNKNOWN'),
                    report.get('environment', 'unknown'),
                    report.get('pipeline_action', 'UNKNOWN'),
                    summary.get('total_checks', 0),
                    summary.get('passed_checks', 0),
                    summary.get('failed_checks', 0),
                    summary.get('pass_rate', 0.0),
                    json.dumps(report.get('data_source', {})),
                    json.dumps(report.get('checks', {})),
                    report.get('execution_time_seconds', 0.0)
                ))
            
            logger.info(f"✅ Quality report loaded: {report_id}")
            return True, {'report_id': report_id}
            
        except Exception as e:
            logger.error(f"💥 Failed to load quality report: {e}")
            return False, {'error': str(e)}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        if not self.connection:
            return {'error': 'No database connection'}
        
        try:
            summary = {}
            
            with self.connection.cursor() as cursor:
                # Market data summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(timestamp) as earliest_data,
                        MAX(timestamp) as latest_data,
                        COUNT(DISTINCT DATE(timestamp)) as trading_days
                    FROM market_data
                ''')
                
                market_data = cursor.fetchone()
                summary['market_data'] = {
                    'total_rows': market_data[0],
                    'unique_symbols': market_data[1],
                    'earliest_data': market_data[2].isoformat() if market_data[2] else None,
                    'latest_data': market_data[3].isoformat() if market_data[3] else None,
                    'trading_days': market_data[4]
                }
                
                # Symbol breakdown
                cursor.execute('''
                    SELECT symbol, COUNT(*) as row_count, MIN(timestamp) as first_bar, MAX(timestamp) as last_bar
                    FROM market_data
                    GROUP BY symbol
                    ORDER BY symbol
                ''')
                
                symbol_data = cursor.fetchall()
                summary['symbols'] = {}
                for row in symbol_data:
                    summary['symbols'][row[0]] = {
                        'row_count': row[1],
                        'first_bar': row[2].isoformat() if row[2] else None,
                        'last_bar': row[3].isoformat() if row[3] else None
                    }
                
                # Quality reports summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_reports,
                        COUNT(*) FILTER (WHERE status = 'PASS') as passed_reports,
                        COUNT(*) FILTER (WHERE status = 'FAIL') as failed_reports,
                        MAX(timestamp) as latest_report
                    FROM data_quality_reports
                ''')
                
                quality_data = cursor.fetchone()
                summary['quality_reports'] = {
                    'total_reports': quality_data[0],
                    'passed_reports': quality_data[1],
                    'failed_reports': quality_data[2],
                    'latest_report': quality_data[3].isoformat() if quality_data[3] else None
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"💥 Failed to get data summary: {e}")
            return {'error': str(e)}
    
    def run_full_load(self, limit_files: Optional[int] = None) -> Dict[str, Any]:
        """Run complete data loading process"""
        logger.info("🚀 Starting TimescaleDB data loading process")
        
        start_time = datetime.now()
        
        results = {
            'start_time': start_time.isoformat(),
            'status': 'UNKNOWN',
            'connection_successful': False,
            'tables_created': False,
            'files_processed': 0,
            'total_rows_loaded': 0,
            'errors': [],
            'file_results': []
        }
        
        try:
            # Connect to database
            if not self.connect():
                results['status'] = 'FAILED'
                results['errors'].append('Database connection failed')
                return results
            
            results['connection_successful'] = True
            
            # Create tables
            if not self.create_tables():
                results['status'] = 'FAILED'
                results['errors'].append('Table creation failed')
                return results
            
            results['tables_created'] = True
            
            # Find and load data files
            data_files = self.find_data_files("dual_ticker_*.csv")
            
            if limit_files:
                data_files = data_files[:limit_files]
            
            for file_path in data_files:
                logger.info(f"📁 Processing: {file_path}")
                
                success, stats = self.load_market_data_from_csv(file_path)
                stats['success'] = success
                results['file_results'].append(stats)
                
                if success:
                    results['files_processed'] += 1
                    results['total_rows_loaded'] += stats['loaded_rows']
                else:
                    results['errors'].append(f"Failed to load {file_path}")
            
            # Load quality reports
            quality_files = self.find_data_files("qc_report*.json")
            for report_file in quality_files:
                success, stats = self.load_quality_report(report_file)
                if not success:
                    results['errors'].append(f"Failed to load quality report {report_file}")
            
            # Get data summary
            results['data_summary'] = self.get_data_summary()
            
            # Determine overall status
            if results['files_processed'] > 0 and len(results['errors']) == 0:
                results['status'] = 'SUCCESS'
            elif results['files_processed'] > 0:
                results['status'] = 'PARTIAL_SUCCESS'
            else:
                results['status'] = 'FAILED'
            
        except Exception as e:
            logger.error(f"💥 Loading process failed: {e}")
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        finally:
            self.disconnect()
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['execution_time_seconds'] = (end_time - start_time).total_seconds()
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print loading summary"""
        status_emoji = {
            'SUCCESS': '✅',
            'PARTIAL_SUCCESS': '⚠️',
            'FAILED': '❌',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 TIMESCALEDB LOADING SUMMARY")
        print("=" * 50)
        print(f"Status: {status_emoji.get(results['status'], '❓')} {results['status']}")
        print(f"Files Processed: {results['files_processed']}")
        print(f"Total Rows Loaded: {results['total_rows_loaded']:,}")
        print(f"Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds")
        
        if results.get('data_summary'):
            summary = results['data_summary']
            if 'market_data' in summary:
                md = summary['market_data']
                print(f"\nMarket Data:")
                print(f"  Total Rows: {md['total_rows']:,}")
                print(f"  Symbols: {md['unique_symbols']}")
                print(f"  Trading Days: {md['trading_days']}")
                print(f"  Date Range: {md.get('earliest_data', 'N/A')} to {md.get('latest_data', 'N/A')}")
            
            if 'symbols' in summary:
                print(f"\nSymbol Breakdown:")
                for symbol, data in summary['symbols'].items():
                    print(f"  {symbol}: {data['row_count']:,} rows")
        
        if results.get('errors'):
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  ❌ {error}")

def main():
    """CLI interface for TimescaleDB loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TimescaleDB Data Loader")
    parser.add_argument('--connection-string', 
                       help='Database connection string (default: from environment)')
    parser.add_argument('--limit-files', type=int,
                       help='Limit number of files to process')
    parser.add_argument('--create-only', action='store_true',
                       help='Only create tables, don\'t load data')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only show data summary, don\'t load')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create loader
    loader = TimescaleDBLoader(connection_string=args.connection_string)
    
    try:
        if args.summary_only:
            # Just show summary
            if loader.connect():
                summary = loader.get_data_summary()
                loader.disconnect()
                
                print("\n📊 DATABASE SUMMARY")
                print("=" * 30)
                print(json.dumps(summary, indent=2, default=str))
            else:
                print("❌ Failed to connect to database")
                sys.exit(1)
        
        elif args.create_only:
            # Just create tables
            if loader.connect():
                success = loader.create_tables()
                loader.disconnect()
                
                if success:
                    print("✅ Tables created successfully")
                    sys.exit(0)
                else:
                    print("❌ Failed to create tables")
                    sys.exit(1)
            else:
                print("❌ Failed to connect to database")
                sys.exit(1)
        
        else:
            # Run full load
            results = loader.run_full_load(limit_files=args.limit_files)
            loader.print_summary(results)
            
            # Exit code based on status
            if results['status'] == 'SUCCESS':
                sys.exit(0)
            elif results['status'] == 'PARTIAL_SUCCESS':
                sys.exit(1)
            else:
                sys.exit(2)
    
    except Exception as e:
        logger.error(f"💥 Script failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()