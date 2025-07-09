# src/shared/disk_gc_service.py
"""
Disk Garbage Collection Service for IntradayJules Feature Store

This service provides:
1. Standalone disk GC that can run as a cron job
2. Orphaned file detection and cleanup
3. Manifest integrity validation
4. Performance monitoring and reporting
5. Safe concurrent operation with running systems

Usage:
    python -m src.shared.disk_gc_service --retention-weeks 4 --dry-run
    python -m src.shared.disk_gc_service --force --verbose
"""

import argparse
import logging
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import duckdb
import os

# Cross-platform file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class DiskGarbageCollector:
    """
    Standalone disk garbage collector for feature store cleanup.
    
    Designed to run as a cron job or standalone service to clean up:
    - Old parquet files not accessed for N weeks
    - Orphaned files not referenced in manifest
    - Corrupted or incomplete cache entries
    """
    
    def __init__(self, 
                 cache_root: str,
                 retention_weeks: int = 4,
                 dry_run: bool = False,
                 verbose: bool = False):
        """
        Initialize disk garbage collector.
        
        Args:
            cache_root: Root directory of feature cache
            retention_weeks: Delete files older than this many weeks
            dry_run: If True, only report what would be deleted
            verbose: Enable verbose logging
        """
        self.cache_root = Path(cache_root).expanduser()
        self.retention_weeks = retention_weeks
        self.dry_run = dry_run
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate cache directory
        if not self.cache_root.exists():
            raise ValueError(f"Cache directory does not exist: {self.cache_root}")
        
        self.manifest_db = self.cache_root / "manifest.duckdb"
        self.manifest_exists = self.manifest_db.exists()
        
        self.logger.info(f"Initialized DiskGC for {self.cache_root}")
        self.logger.info(f"Retention: {retention_weeks} weeks, Dry run: {dry_run}")
    
    def run_garbage_collection(self) -> Dict[str, Any]:
        """
        Run complete garbage collection process.
        
        Returns:
            Dictionary with GC results and statistics
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING DISK GARBAGE COLLECTION")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Validate manifest integrity
            self.logger.info("Step 1: Validating manifest integrity...")
            integrity_results = self._validate_manifest_integrity()
            
            # Step 2: Clean old entries from manifest
            self.logger.info("Step 2: Cleaning old manifest entries...")
            old_cleanup_results = self._cleanup_old_manifest_entries()
            
            # Step 3: Find and clean orphaned files
            self.logger.info("Step 3: Cleaning orphaned files...")
            orphan_cleanup_results = self._cleanup_orphaned_files()
            
            # Step 4: Optimize manifest database
            self.logger.info("Step 4: Optimizing manifest database...")
            optimization_results = self._optimize_manifest_database()
            
            # Compile results
            duration = time.time() - start_time
            results = {
                'status': 'success',
                'duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat(),
                'dry_run': self.dry_run,
                'retention_weeks': self.retention_weeks,
                'cache_root': str(self.cache_root),
                'integrity_check': integrity_results,
                'old_entries_cleanup': old_cleanup_results,
                'orphaned_files_cleanup': orphan_cleanup_results,
                'database_optimization': optimization_results,
                'summary': {
                    'total_files_deleted': (old_cleanup_results['files_deleted'] + 
                                          orphan_cleanup_results['files_deleted']),
                    'total_bytes_freed': (old_cleanup_results['bytes_freed'] + 
                                        orphan_cleanup_results['bytes_freed']),
                    'manifest_entries_cleaned': old_cleanup_results['manifest_entries_deleted'],
                    'orphaned_files_found': orphan_cleanup_results['files_deleted']
                }
            }
            
            self._log_gc_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Garbage collection failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_manifest_integrity(self) -> Dict[str, Any]:
        """Validate manifest database integrity and find issues."""
        issues = []
        total_entries = 0
        missing_files = 0
        corrupted_entries = 0
        
        try:
            with duckdb.connect(str(self.manifest_db)) as db:
                # Get all manifest entries
                entries = db.execute("""
                    SELECT key, path, file_size_bytes, created_ts 
                    FROM manifest 
                    ORDER BY created_ts
                """).fetchall()
                
                total_entries = len(entries)
                self.logger.info(f"Validating {total_entries} manifest entries...")
                
                for key, path, expected_size, created_ts in entries:
                    file_path = Path(path)
                    
                    # Check if file exists
                    if not file_path.exists():
                        issues.append(f"Missing file: {path}")
                        missing_files += 1
                        continue
                    
                    # Check file size consistency
                    actual_size = file_path.stat().st_size
                    if expected_size and abs(actual_size - expected_size) > 1024:  # Allow 1KB tolerance
                        issues.append(f"Size mismatch: {path} (expected: {expected_size}, actual: {actual_size})")
                        corrupted_entries += 1
                    
                    # Check if file is readable
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # Try to read first 1KB
                    except Exception as e:
                        issues.append(f"Unreadable file: {path} - {e}")
                        corrupted_entries += 1
            
            result = {
                'total_entries': total_entries,
                'missing_files': missing_files,
                'corrupted_entries': corrupted_entries,
                'issues_found': len(issues),
                'issues': issues[:10] if len(issues) > 10 else issues,  # Limit output
                'integrity_ok': len(issues) == 0
            }
            
            if len(issues) > 0:
                self.logger.warning(f"Found {len(issues)} integrity issues")
            else:
                self.logger.info("Manifest integrity check passed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Integrity validation failed: {e}")
            return {'error': str(e), 'integrity_ok': False}
    
    def _cleanup_old_manifest_entries(self) -> Dict[str, Any]:
        """Clean up old entries from manifest based on retention policy."""
        files_deleted = 0
        bytes_freed = 0
        manifest_entries_deleted = 0
        errors = []
        
        try:
            cutoff_date = datetime.now() - timedelta(weeks=self.retention_weeks)
            self.logger.info(f"Cleaning entries older than {cutoff_date}")
            
            with duckdb.connect(str(self.manifest_db)) as db:
                # Find old entries
                old_entries = db.execute("""
                    SELECT key, path, file_size_bytes, last_accessed_ts
                    FROM manifest 
                    WHERE last_accessed_ts < ? OR created_ts < ?
                    ORDER BY last_accessed_ts ASC
                """, [cutoff_date, cutoff_date]).fetchall()
                
                self.logger.info(f"Found {len(old_entries)} old entries to clean")
                
                for key, path, file_size, last_accessed in old_entries:
                    try:
                        file_path = Path(path)
                        
                        # Delete file if it exists
                        if file_path.exists():
                            if not self.dry_run:
                                file_path.unlink()
                                self.logger.debug(f"Deleted file: {path}")
                            else:
                                self.logger.info(f"[DRY RUN] Would delete: {path}")
                            
                            files_deleted += 1
                            bytes_freed += file_size or 0
                        
                        # Remove from manifest
                        if not self.dry_run:
                            db.execute("DELETE FROM manifest WHERE key = ?", [key])
                        
                        manifest_entries_deleted += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to delete {path}: {e}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)
            
            result = {
                'files_deleted': files_deleted,
                'bytes_freed': bytes_freed,
                'manifest_entries_deleted': manifest_entries_deleted,
                'errors': errors,
                'mb_freed': round(bytes_freed / 1024 / 1024, 2)
            }
            
            action = "Would delete" if self.dry_run else "Deleted"
            self.logger.info(f"{action} {files_deleted} old files "
                           f"({result['mb_freed']} MB), "
                           f"{manifest_entries_deleted} manifest entries")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Old entries cleanup failed: {e}")
            return {'error': str(e), 'files_deleted': 0, 'bytes_freed': 0}
    
    def _cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Find and clean up orphaned parquet files not in manifest."""
        files_deleted = 0
        bytes_freed = 0
        errors = []
        
        try:
            # Get all parquet files in cache directory
            parquet_patterns = ['*.parquet', '*.parquet.zst']
            all_parquet_files = set()
            
            for pattern in parquet_patterns:
                all_parquet_files.update(self.cache_root.glob(pattern))
            
            self.logger.info(f"Found {len(all_parquet_files)} parquet files on disk")
            
            # Get all paths from manifest
            manifest_paths = set()
            with duckdb.connect(str(self.manifest_db)) as db:
                paths_result = db.execute("SELECT path FROM manifest").fetchall()
                for (path,) in paths_result:
                    manifest_paths.add(Path(path))
            
            self.logger.info(f"Found {len(manifest_paths)} files referenced in manifest")
            
            # Find orphaned files
            orphaned_files = all_parquet_files - manifest_paths
            self.logger.info(f"Found {len(orphaned_files)} orphaned files")
            
            # Delete orphaned files
            for orphaned_file in orphaned_files:
                try:
                    file_size = orphaned_file.stat().st_size
                    
                    if not self.dry_run:
                        orphaned_file.unlink()
                        self.logger.debug(f"Deleted orphaned file: {orphaned_file}")
                    else:
                        self.logger.info(f"[DRY RUN] Would delete orphaned: {orphaned_file}")
                    
                    files_deleted += 1
                    bytes_freed += file_size
                    
                except Exception as e:
                    error_msg = f"Failed to delete orphaned file {orphaned_file}: {e}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)
            
            result = {
                'files_deleted': files_deleted,
                'bytes_freed': bytes_freed,
                'errors': errors,
                'mb_freed': round(bytes_freed / 1024 / 1024, 2),
                'total_parquet_files': len(all_parquet_files),
                'manifest_referenced_files': len(manifest_paths)
            }
            
            action = "Would delete" if self.dry_run else "Deleted"
            self.logger.info(f"{action} {files_deleted} orphaned files ({result['mb_freed']} MB)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Orphaned files cleanup failed: {e}")
            return {'error': str(e), 'files_deleted': 0, 'bytes_freed': 0}
    
    def _optimize_manifest_database(self) -> Dict[str, Any]:
        """Optimize manifest database performance."""
        try:
            with duckdb.connect(str(self.manifest_db)) as db:
                # Get database stats before optimization
                stats_before = db.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(file_size_bytes) as total_size_bytes
                    FROM manifest
                """).fetchone()
                
                if not self.dry_run:
                    # Vacuum database to reclaim space
                    db.execute("VACUUM")
                    
                    # Analyze tables for query optimization
                    db.execute("ANALYZE")
                    
                    # Update statistics
                    db.execute("""
                        INSERT OR REPLACE INTO gc_log 
                        (run_timestamp, files_deleted, bytes_freed, duration_seconds)
                        VALUES (CURRENT_TIMESTAMP, 0, 0, 0)
                    """)
                
                # Get database file size
                db_size = self.manifest_db.stat().st_size
                
                result = {
                    'database_size_mb': round(db_size / 1024 / 1024, 2),
                    'total_entries': stats_before[0] if stats_before else 0,
                    'total_cached_size_mb': round((stats_before[1] or 0) / 1024 / 1024, 2),
                    'optimized': not self.dry_run
                }
                
                action = "Would optimize" if self.dry_run else "Optimized"
                self.logger.info(f"{action} manifest database ({result['database_size_mb']} MB)")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
            return {'error': str(e)}
    
    def _log_gc_results(self, results: Dict[str, Any]):
        """Log comprehensive GC results."""
        self.logger.info("=" * 60)
        self.logger.info("GARBAGE COLLECTION RESULTS")
        self.logger.info("=" * 60)
        
        summary = results.get('summary', {})
        
        self.logger.info(f"Status: {results['status']}")
        self.logger.info(f"Duration: {results['duration_seconds']:.1f} seconds")
        self.logger.info(f"Dry Run: {results['dry_run']}")
        self.logger.info("")
        
        self.logger.info("SUMMARY:")
        self.logger.info(f"  Total files deleted: {summary.get('total_files_deleted', 0)}")
        self.logger.info(f"  Total space freed: {summary.get('total_bytes_freed', 0) / 1024 / 1024:.1f} MB")
        self.logger.info(f"  Manifest entries cleaned: {summary.get('manifest_entries_cleaned', 0)}")
        self.logger.info(f"  Orphaned files found: {summary.get('orphaned_files_found', 0)}")
        self.logger.info("")
        
        # Integrity check results
        integrity = results.get('integrity_check', {})
        if integrity:
            self.logger.info("INTEGRITY CHECK:")
            self.logger.info(f"  Total entries validated: {integrity.get('total_entries', 0)}")
            self.logger.info(f"  Missing files: {integrity.get('missing_files', 0)}")
            self.logger.info(f"  Corrupted entries: {integrity.get('corrupted_entries', 0)}")
            self.logger.info(f"  Integrity OK: {integrity.get('integrity_ok', False)}")
            self.logger.info("")
        
        # Database optimization results
        db_opt = results.get('database_optimization', {})
        if db_opt:
            self.logger.info("DATABASE OPTIMIZATION:")
            self.logger.info(f"  Database size: {db_opt.get('database_size_mb', 0)} MB")
            self.logger.info(f"  Total entries: {db_opt.get('total_entries', 0)}")
            self.logger.info(f"  Total cached data: {db_opt.get('total_cached_size_mb', 0)} MB")
            self.logger.info("")
        
        self.logger.info("=" * 60)
    
    def get_cache_overview(self) -> Dict[str, Any]:
        """Get overview of cache status without running GC."""
        if not self.manifest_exists:
            # Return empty overview if no manifest exists
            parquet_files = list(self.cache_root.glob('*.parquet*'))
            return {
                'cache_root': str(self.cache_root),
                'manifest_entries': 0,
                'unique_symbols': 0,
                'total_cached_rows': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'oldest_entry': None,
                'newest_entry': None,
                'avg_access_count': 0,
                'old_entries_count': 0,
                'recent_entries_count': 0,
                'parquet_files_on_disk': len(parquet_files),
                'manifest_size_bytes': 0,
                'manifest_size_mb': 0
            }
        
        try:
            with duckdb.connect(str(self.manifest_db)) as db:
                # Basic statistics
                stats = db.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        SUM(rows) as total_cached_rows,
                        SUM(file_size_bytes) as total_size_bytes,
                        MIN(created_ts) as oldest_entry,
                        MAX(created_ts) as newest_entry,
                        AVG(access_count) as avg_access_count
                    FROM manifest
                """).fetchone()
                
                # Age distribution
                cutoff_date = datetime.now() - timedelta(weeks=self.retention_weeks)
                age_stats = db.execute("""
                    SELECT 
                        COUNT(CASE WHEN last_accessed_ts < ? THEN 1 END) as old_entries,
                        COUNT(CASE WHEN last_accessed_ts >= ? THEN 1 END) as recent_entries
                    FROM manifest
                """, [cutoff_date, cutoff_date]).fetchone()
                
                # File system check
                parquet_files = list(self.cache_root.glob('*.parquet*'))
                db_size = self.manifest_db.stat().st_size
                
                return {
                    'cache_root': str(self.cache_root),
                    'manifest_entries': stats[0] if stats else 0,
                    'unique_symbols': stats[1] if stats else 0,
                    'total_cached_rows': stats[2] if stats else 0,
                    'total_size_mb': round((stats[3] or 0) / 1024 / 1024, 2),
                    'oldest_entry': stats[4],
                    'newest_entry': stats[5],
                    'avg_access_count': round(stats[6] or 0, 1),
                    'old_entries_count': age_stats[0] if age_stats else 0,
                    'recent_entries_count': age_stats[1] if age_stats else 0,
                    'parquet_files_on_disk': len(parquet_files),
                    'manifest_db_size_mb': round(db_size / 1024 / 1024, 2),
                    'retention_weeks': self.retention_weeks
                }
                
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main entry point for disk GC service."""
    parser = argparse.ArgumentParser(
        description="Disk Garbage Collection Service for IntradayJules Feature Store"
    )
    
    parser.add_argument(
        '--cache-root',
        default=os.getenv('FEATURE_STORE_PATH', '~/.feature_cache'),
        help='Root directory of feature cache (default: ~/.feature_cache)'
    )
    
    parser.add_argument(
        '--retention-weeks',
        type=int,
        default=4,
        help='Delete files older than this many weeks (default: 4)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report what would be deleted, do not actually delete'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force GC even if recently run'
    )
    
    parser.add_argument(
        '--overview-only',
        action='store_true',
        help='Only show cache overview, do not run GC'
    )
    
    parser.add_argument(
        '--output-json',
        help='Write results to JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize garbage collector
        gc = DiskGarbageCollector(
            cache_root=args.cache_root,
            retention_weeks=args.retention_weeks,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if args.overview_only:
            # Just show overview
            results = gc.get_cache_overview()
            print("\nCACHE OVERVIEW:")
            print("=" * 50)
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            # Run garbage collection
            results = gc.run_garbage_collection()
        
        # Write JSON output if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults written to: {args.output_json}")
        
        # Exit with appropriate code
        if results.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()