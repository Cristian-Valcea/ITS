"""
Price Feed Interface - PLACEHOLDER

This is a placeholder for pluggable price feed abstraction.
Will implement standardized interface for live and simulated data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator
import logging


class PriceFeedInterface(ABC):
    """
    Abstract interface for price feeds.
    
    TODO: Implement concrete classes for:
    - Historical L2 order book replay
    - Live market data feeds
    - Synthetic data generation
    """
    
    @abstractmethod
    def iter_ticks(self) -> Iterator[Dict]:
        """
        Yield price ticks in standardized format.
        
        Format: {ts, bid, ask, last, volume}
        """
        pass


class HistoricalReplayFeed(PriceFeedInterface):
    """Placeholder for historical data replay."""
    
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date
        logging.info(f"HistoricalReplayFeed placeholder initialized for {symbol} {date}")
    
    def iter_ticks(self) -> Iterator[Dict]:
        """Placeholder tick generator."""
        # TODO: Implement actual historical data replay
        yield {
            'ts': 1234567890,
            'bid': 100.0,
            'ask': 100.1,
            'last': 100.05,
            'volume': 1000
        }