"""
Streaming Trade Log Wrapper

Persists trade logs incrementally to Arrow/Parquet format
to avoid keeping entire trading history in RAM.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime


class StreamingTradeLogWrapper(gym.Wrapper):
    """
    Streams trade logs to disk in Arrow/Parquet format.
    
    Features:
    - Incremental writes to avoid RAM buildup
    - Batch writing for performance
    - Automatic compression and columnar storage
    - Fast read-back for analysis
    - Configurable write frequency
    
    This wrapper eliminates the memory bottleneck of keeping
    the entire trade_log list in RAM during long training runs.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 log_dir: str = "logs/trade_logs",
                 log_filename: Optional[str] = None,
                 batch_size: int = 100,
                 compression: str = "snappy",
                 enable_logging: bool = True,
                 write_frequency: int = 10,
                 max_memory_records: int = 1000):
        """
        Initialize streaming trade log wrapper.
        
        Args:
            env: Base trading environment
            log_dir: Directory to store trade log files
            log_filename: Custom filename (auto-generated if None)
            batch_size: Number of trades to batch before writing
            compression: Parquet compression ('snappy', 'gzip', 'lz4', 'brotli')
            enable_logging: Enable trade logging (can disable for testing)
            write_frequency: Write to disk every N trades
            max_memory_records: Maximum trades to keep in memory before forced write
        """
        super().__init__(env)
        self.log_dir = Path(log_dir)
        self.batch_size = batch_size
        self.compression = compression
        self.enable_logging = enable_logging
        self.write_frequency = write_frequency
        self.max_memory_records = max_memory_records
        
        self.logger = logging.getLogger("StreamingTradeLogWrapper")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"trades_{timestamp}.parquet"
        
        self.log_file_path = self.log_dir / log_filename
        self.temp_file_path = self.log_dir / f"temp_{log_filename}"
        
        # In-memory buffer for trades
        self.trade_buffer: List[Dict[str, Any]] = []
        self.total_trades = 0
        self.last_write_time = time.time()
        
        # Schema for Arrow table
        self.schema = self._create_trade_schema()
        
        # Initialize empty file if it doesn't exist
        if not self.log_file_path.exists() and self.enable_logging:
            self._initialize_log_file()
        
        if self.enable_logging:
            self.logger.info(f"Streaming trade log initialized: {self.log_file_path}")
            self.logger.info(f"Batch size: {batch_size}, Compression: {compression}")
        else:
            self.logger.info("Trade logging disabled")
    
    def step(self, action):
        """Execute action and log trade if it occurs."""
        # Execute base action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Check if a trade occurred
        if self.enable_logging and self._trade_occurred(action, info):
            trade_record = self._create_trade_record(action, obs, reward, info)
            self.trade_buffer.append(trade_record)
            self.total_trades += 1
            
            # Write to disk if batch is full or max memory reached
            if (len(self.trade_buffer) >= self.batch_size or 
                len(self.trade_buffer) >= self.max_memory_records or
                self.total_trades % self.write_frequency == 0):
                self._flush_buffer()
        
        # Add trade log info
        info.update({
            'trades_in_buffer': len(self.trade_buffer),
            'total_trades_logged': self.total_trades,
            'log_file_path': str(self.log_file_path),
            'trade_logging_enabled': self.enable_logging
        })
        
        return obs, reward, done, truncated, info
    
    def _trade_occurred(self, action: int, info: Dict[str, Any]) -> bool:
        """Check if a trade actually occurred (not just HOLD)."""
        # HOLD action (1) doesn't generate trades
        if action == 1:
            return False
        
        # Check if action was modified by other wrappers
        if info.get('action_modified_by_cooldown', False):
            return False
        
        if info.get('size_limit_violated', False):
            return False
        
        # Check if position actually changed
        if hasattr(self.env, 'current_position') and hasattr(self.env, 'previous_position'):
            return self.env.current_position != self.env.previous_position
        
        # Default: assume trade occurred for non-HOLD actions
        return True
    
    def _create_trade_record(self, action: int, obs: np.ndarray, 
                           reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trade record from current step data."""
        timestamp = time.time()
        step_count = getattr(self.env, 'step_count', 0)
        
        # Extract environment state
        current_position = getattr(self.env, 'current_position', 0)
        previous_position = getattr(self.env, 'previous_position', 0)
        current_price = getattr(self.env, 'current_price', 0.0)
        portfolio_value = getattr(self.env, 'portfolio_value', 0.0)
        cash = getattr(self.env, 'cash', 0.0)
        
        # Calculate trade details
        position_change = current_position - previous_position
        trade_value = abs(position_change) * current_price
        
        record = {
            'timestamp': timestamp,
            'step': step_count,
            'action': action,
            'position_change': position_change,
            'previous_position': previous_position,
            'current_position': current_position,
            'price': current_price,
            'trade_value': trade_value,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'reward': reward,
            'pnl': info.get('pnl', 0.0),
            'transaction_cost': info.get('transaction_cost', 0.0),
            'turnover': info.get('daily_turnover', 0.0),
            'volatility': info.get('daily_volatility', 0.0),
            'sharpe_ratio': info.get('sharpe_ratio', 0.0),
            'drawdown': info.get('current_drawdown', 0.0)
        }
        
        return record
    
    def _create_trade_schema(self) -> pa.Schema:
        """Create Arrow schema for trade records."""
        return pa.schema([
            ('timestamp', pa.float64()),
            ('step', pa.int64()),
            ('action', pa.int32()),
            ('position_change', pa.int32()),
            ('previous_position', pa.int32()),
            ('current_position', pa.int32()),
            ('price', pa.float64()),
            ('trade_value', pa.float64()),
            ('portfolio_value', pa.float64()),
            ('cash', pa.float64()),
            ('reward', pa.float64()),
            ('pnl', pa.float64()),
            ('transaction_cost', pa.float64()),
            ('turnover', pa.float64()),
            ('volatility', pa.float64()),
            ('sharpe_ratio', pa.float64()),
            ('drawdown', pa.float64())
        ])
    
    def _initialize_log_file(self):
        """Initialize empty log file with schema."""
        try:
            empty_df = pd.DataFrame(columns=[field.name for field in self.schema])
            table = pa.Table.from_pandas(empty_df, schema=self.schema)
            pq.write_table(table, self.log_file_path, compression=self.compression)
            self.logger.debug(f"Initialized empty log file: {self.log_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize log file: {e}")
    
    def _flush_buffer(self):
        """Write buffered trades to disk."""
        if not self.trade_buffer:
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.trade_buffer)
            
            # Convert to Arrow table with schema
            table = pa.Table.from_pandas(df, schema=self.schema)
            
            # Append to existing file or create new one
            if self.log_file_path.exists():
                # Read existing data
                existing_table = pq.read_table(self.log_file_path)
                # Combine with new data
                combined_table = pa.concat_tables([existing_table, table])
            else:
                combined_table = table
            
            # Write atomically using temporary file
            pq.write_table(combined_table, self.temp_file_path, compression=self.compression)
            shutil.move(str(self.temp_file_path), str(self.log_file_path))
            
            self.logger.debug(f"Flushed {len(self.trade_buffer)} trades to disk")
            
            # Clear buffer
            self.trade_buffer.clear()
            self.last_write_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to flush trade buffer: {e}")
            # Don't clear buffer on error - try again next time
    
    def reset(self, **kwargs):
        """Reset and flush any remaining trades."""
        # Flush remaining trades
        if self.trade_buffer:
            self._flush_buffer()
        
        # Reset counters
        self.total_trades = 0
        
        return self.env.reset(**kwargs)
    
    def close(self):
        """Close wrapper and flush final trades."""
        if self.trade_buffer:
            self._flush_buffer()
        super().close()
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """Read complete trade log as DataFrame."""
        if not self.log_file_path.exists():
            return pd.DataFrame()
        
        try:
            # Read from disk and combine with buffer
            disk_df = pd.read_parquet(self.log_file_path)
            
            if self.trade_buffer:
                buffer_df = pd.DataFrame(self.trade_buffer)
                return pd.concat([disk_df, buffer_df], ignore_index=True)
            else:
                return disk_df
        except Exception as e:
            self.logger.error(f"Failed to read trade log: {e}")
            return pd.DataFrame()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get trade logging statistics."""
        file_size = self.log_file_path.stat().st_size if self.log_file_path.exists() else 0
        
        return {
            'log_file_path': str(self.log_file_path),
            'total_trades': self.total_trades,
            'trades_in_buffer': len(self.trade_buffer),
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'compression': self.compression,
            'batch_size': self.batch_size,
            'write_frequency': self.write_frequency,
            'last_write_time': self.last_write_time
        }