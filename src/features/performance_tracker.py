# src/features/performance_tracker.py
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import pandas as pd
import numpy as np


class PerformanceTracker:
    """
    Tracks performance metrics for feature computation and data processing.
    Provides insights into computation times, memory usage, and bottlenecks.
    """
    
    def __init__(self, max_history: int = 1000, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_history = max_history
        
        # Performance metrics storage
        self.computation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.data_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # Context tracking
        self._active_contexts: Dict[str, float] = {}
    
    def start_timing(self, operation: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation being timed
            
        Returns:
            Context key for stopping the timer
        """
        context_key = f"{operation}_{time.time()}"
        self._active_contexts[context_key] = time.time()
        return context_key
    
    def stop_timing(self, context_key: str, operation: str, 
                   data_size: Optional[int] = None, success: bool = True):
        """
        Stop timing an operation and record metrics.
        
        Args:
            context_key: Context key from start_timing
            operation: Name of the operation
            data_size: Size of data processed (optional)
            success: Whether the operation succeeded
        """
        if context_key not in self._active_contexts:
            self.logger.warning(f"No active timing context for: {context_key}")
            return
        
        start_time = self._active_contexts.pop(context_key)
        duration = time.time() - start_time
        
        self.computation_times[operation].append(duration)
        
        if data_size is not None:
            self.data_sizes[operation].append(data_size)
        
        if success:
            self.success_counts[operation] += 1
        else:
            self.error_counts[operation] += 1
    
    def record_error(self, operation: str, error: Exception):
        """Record an error for an operation."""
        self.error_counts[operation] += 1
        self.logger.debug(f"Recorded error for {operation}: {error}")
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with operation statistics
        """
        times = list(self.computation_times[operation])
        sizes = list(self.data_sizes[operation])
        
        if not times:
            return {'operation': operation, 'no_data': True}
        
        stats = {
            'operation': operation,
            'total_calls': len(times),
            'success_count': self.success_counts[operation],
            'error_count': self.error_counts[operation],
            'success_rate': self.success_counts[operation] / (self.success_counts[operation] + self.error_counts[operation]) if (self.success_counts[operation] + self.error_counts[operation]) > 0 else 0,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'total_time': np.sum(times)
        }
        
        if sizes:
            stats.update({
                'avg_data_size': np.mean(sizes),
                'min_data_size': np.min(sizes),
                'max_data_size': np.max(sizes),
                'avg_throughput': np.mean(sizes) / np.mean(times) if np.mean(times) > 0 else 0
            })
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations."""
        all_operations = set(self.computation_times.keys()) | set(self.error_counts.keys()) | set(self.success_counts.keys())
        return {op: self.get_operation_stats(op) for op in all_operations}
    
    def get_summary_report(self) -> str:
        """
        Generate a summary report of all performance metrics.
        
        Returns:
            Formatted string report
        """
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return "No performance data available."
        
        report_lines = ["Performance Summary Report", "=" * 50]
        
        # Sort operations by total time
        sorted_ops = sorted(all_stats.items(), 
                          key=lambda x: x[1].get('total_time', 0), 
                          reverse=True)
        
        for operation, stats in sorted_ops:
            if stats.get('no_data'):
                continue
                
            report_lines.extend([
                f"\nOperation: {operation}",
                f"  Calls: {stats['total_calls']} (Success: {stats['success_count']}, Errors: {stats['error_count']})",
                f"  Success Rate: {stats['success_rate']:.2%}",
                f"  Timing: Avg={stats['avg_time']:.4f}s, Min={stats['min_time']:.4f}s, Max={stats['max_time']:.4f}s",
                f"  Total Time: {stats['total_time']:.4f}s"
            ])
            
            if 'avg_throughput' in stats:
                report_lines.append(f"  Throughput: {stats['avg_throughput']:.2f} records/sec")
        
        return "\n".join(report_lines)
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            top_n: Number of top bottlenecks to return
            
        Returns:
            List of bottleneck information
        """
        all_stats = self.get_all_stats()
        
        # Sort by total time to find bottlenecks
        bottlenecks = []
        for operation, stats in all_stats.items():
            if stats.get('no_data'):
                continue
                
            bottleneck_score = stats.get('total_time', 0) * (1 - stats.get('success_rate', 1))
            bottlenecks.append({
                'operation': operation,
                'bottleneck_score': bottleneck_score,
                'total_time': stats.get('total_time', 0),
                'avg_time': stats.get('avg_time', 0),
                'success_rate': stats.get('success_rate', 1),
                'error_count': stats.get('error_count', 0)
            })
        
        return sorted(bottlenecks, key=lambda x: x['bottleneck_score'], reverse=True)[:top_n]
    
    def reset_stats(self, operation: Optional[str] = None):
        """
        Reset statistics for an operation or all operations.
        
        Args:
            operation: Specific operation to reset, or None for all
        """
        if operation:
            self.computation_times[operation].clear()
            self.data_sizes[operation].clear()
            self.error_counts[operation] = 0
            self.success_counts[operation] = 0
            self.logger.info(f"Reset stats for operation: {operation}")
        else:
            self.computation_times.clear()
            self.data_sizes.clear()
            self.error_counts.clear()
            self.success_counts.clear()
            self._active_contexts.clear()
            self.logger.info("Reset all performance statistics")
    
    def export_stats_to_dataframe(self) -> pd.DataFrame:
        """
        Export performance statistics to a pandas DataFrame.
        
        Returns:
            DataFrame with performance metrics
        """
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return pd.DataFrame()
        
        # Flatten the stats dictionary
        rows = []
        for operation, stats in all_stats.items():
            if stats.get('no_data'):
                continue
            row = {'operation': operation}
            row.update({k: v for k, v in stats.items() if k != 'operation'})
            rows.append(row)
        
        return pd.DataFrame(rows)


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: PerformanceTracker, operation: str, 
                 data_size: Optional[int] = None):
        self.tracker = tracker
        self.operation = operation
        self.data_size = data_size
        self.context_key = None
        self.success = True
    
    def __enter__(self):
        self.context_key = self.tracker.start_timing(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.tracker.record_error(self.operation, exc_val)
        
        if self.context_key:
            self.tracker.stop_timing(
                self.context_key, 
                self.operation, 
                self.data_size, 
                self.success
            )
        
        return False  # Don't suppress exceptions