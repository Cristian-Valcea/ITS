# src/risk/calculators/feed_staleness_calculator.py
"""
Feed Staleness Calculator - CRITICAL priority sensor (<20µs target)

Monitors market data feed staleness to detect when feeds become stale,
which can lead to phantom fills and incorrect risk calculations.

Priority: CRITICAL (kill-switch capable)
Latency Target: <20µs
Action: KILL_SWITCH when feeds are too stale
"""

import time
import numpy as np
from typing import Dict, Any, List
from .base_calculator import BaseRiskCalculator, RiskCalculationResult, RiskMetricType


class FeedStalenessCalculator(BaseRiskCalculator):
    """
    Feed Staleness Calculator - Ultra-low latency feed monitoring.
    
    Calculates maximum staleness across all monitored feeds.
    Designed for microsecond-level latency in critical path.
    
    Formula: max_staleness = max(current_time - last_update_time) across all feeds
    """
    
    def _get_metric_type(self) -> RiskMetricType:
        return RiskMetricType.FEED_STALENESS
    
    def _validate_config(self) -> None:
        """Validate calculator configuration."""
        # No specific config validation needed for feed staleness
        pass
    
    def get_required_inputs(self) -> List[str]:
        """Return list of required input data keys."""
        return ['feed_timestamps', 'current_time']
    
    def calculate(self, data: Dict[str, Any]) -> RiskCalculationResult:
        """
        Calculate feed staleness with ultra-low latency.
        
        Args:
            data: Must contain 'feed_timestamps' dict and 'current_time'
            
        Returns:
            RiskCalculationResult with staleness metrics
        """
        feed_timestamps = data['feed_timestamps']
        current_time = data.get('current_time', time.time())
        
        if not feed_timestamps:
            return RiskCalculationResult(
                metric_type=self.metric_type,
                values={'max_staleness_ms': 0.0, 'stale_feeds': []},
                metadata={'feed_count': 0}
            )
        
        # Calculate staleness for each feed (vectorized for speed)
        staleness_values = []
        stale_feeds = []
        
        for feed_name, last_update in feed_timestamps.items():
            if last_update is None:
                staleness_ms = float('inf')
                stale_feeds.append(feed_name)
            else:
                staleness_ms = (current_time - last_update) * 1000.0  # Convert to ms
                staleness_values.append(staleness_ms)
                
                # Track feeds that are stale (>1 second)
                if staleness_ms > 1000.0:
                    stale_feeds.append(feed_name)
        
        # Calculate maximum staleness
        if staleness_values:
            max_staleness_ms = max(staleness_values)
            avg_staleness_ms = np.mean(staleness_values)
        else:
            max_staleness_ms = float('inf')
            avg_staleness_ms = float('inf')
        
        return RiskCalculationResult(
            metric_type=self.metric_type,
            values={
                'max_staleness_ms': max_staleness_ms,
                'avg_staleness_ms': avg_staleness_ms,
                'stale_feeds': stale_feeds,
                'active_feeds': len([t for t in feed_timestamps.values() if t is not None])
            },
            metadata={
                'feed_count': len(feed_timestamps),
                'stale_feed_count': len(stale_feeds),
                'vectorized': True
            }
        )