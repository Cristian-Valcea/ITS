"""
Batch processing module for IntradayJules trading system.

This module provides end-of-day batch processing functionality including:
- Scheduled job execution
- MiFID II compliance report generation
- Data validation and archival
- Error handling and retry mechanisms
- Notification and alerting
"""

from .end_of_day_processor import EndOfDayProcessor, BatchJobConfig, BatchExecutionResult, create_batch_config_file

__all__ = [
    'EndOfDayProcessor',
    'BatchJobConfig',
    'BatchExecutionResult',
    'create_batch_config_file'
]