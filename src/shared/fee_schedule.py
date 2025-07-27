"""
Venue-Specific Fee Engine for Trading System

Provides accurate transaction cost modeling for different venues and instruments.
Supports tiered fee structures based on volume thresholds.
"""

from __future__ import annotations
import yaml
import logging
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fee:
    """Represents a trading fee structure."""
    fee_per_side: float        # USD per contract/share per side
    currency: str = "USD"
    description: str = ""
    
    def __post_init__(self):
        if self.fee_per_side < 0:
            raise ValueError(f"Fee cannot be negative: {self.fee_per_side}")


@dataclass
class VolumeTier:
    """Volume tier for tiered fee structures."""
    volume_threshold: int      # Minimum volume for this tier
    fee: float                # Fee at this tier
    
    def __post_init__(self):
        if self.volume_threshold < 0:
            raise ValueError(f"Volume threshold cannot be negative: {self.volume_threshold}")
        if self.fee < 0:
            raise ValueError(f"Fee cannot be negative: {self.fee}")


class FeeSchedule:
    """
    Venue-specific fee schedule with support for tiered pricing.
    
    Loads fee configuration from YAML and provides fast lookups with caching.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize fee schedule from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.table: Dict = {}
        self.venue: str = "UNKNOWN"
        self.last_updated: Optional[str] = None
        
        self._load_config()
        logger.info(f"Loaded fee schedule for {self.venue} with {len(self.table)} instruments")
    
    def _load_config(self):
        """Load fee configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Fee config not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError("Fee config must be a dictionary")
            
            # Extract metadata
            metadata = config.get('metadata', {})
            self.venue = metadata.get('venue', 'UNKNOWN')
            self.last_updated = metadata.get('last_updated')
            
            # Store fee table (exclude metadata)
            self.table = {k: v for k, v in config.items() if k != 'metadata'}
            
            # Validate required DEFAULT entry
            if 'DEFAULT' not in self.table:
                raise ValueError("Fee config must include a DEFAULT entry")
            
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Failed to load fee config from {self.config_path}: {e}")
            raise
    
    def _validate_config(self):
        """Validate fee configuration structure."""
        for symbol, config in self.table.items():
            if not isinstance(config, dict):
                raise ValueError(f"Fee config for {symbol} must be a dictionary")
            
            # Check for either simple fee or tiered structure
            has_simple_fee = 'trade_fee' in config
            has_tiered_fee = 'tiers' in config
            
            if not (has_simple_fee or has_tiered_fee):
                raise ValueError(f"Fee config for {symbol} must have either 'trade_fee' or 'tiers'")
            
            if has_simple_fee and has_tiered_fee:
                raise ValueError(f"Fee config for {symbol} cannot have both 'trade_fee' and 'tiers'")
            
            # Validate tiered structure
            if has_tiered_fee:
                tiers = config['tiers']
                if not isinstance(tiers, list) or len(tiers) == 0:
                    raise ValueError(f"Tiers for {symbol} must be a non-empty list")
                
                for i, tier in enumerate(tiers):
                    if not isinstance(tier, dict):
                        raise ValueError(f"Tier {i} for {symbol} must be a dictionary")
                    if 'vol' not in tier or 'fee' not in tier:
                        raise ValueError(f"Tier {i} for {symbol} must have 'vol' and 'fee' keys")
    
    @lru_cache(maxsize=512)
    def lookup(self, symbol: str, volume_ytd: int = 0) -> Fee:
        """
        Look up fee for a given symbol and volume.
        
        Args:
            symbol: Trading symbol (e.g., 'MES', 'MNQ')
            volume_ytd: Year-to-date volume for tiered pricing
            
        Returns:
            Fee object with fee per side and currency
        """
        # Get symbol config or fall back to DEFAULT
        symbol_config = self.table.get(symbol, self.table['DEFAULT'])
        
        # Handle simple fee structure
        if 'trade_fee' in symbol_config:
            return Fee(
                fee_per_side=float(symbol_config['trade_fee']),
                currency=symbol_config.get('currency', 'USD'),
                description=symbol_config.get('description', f'Fee for {symbol}')
            )
        
        # Handle tiered fee structure
        if 'tiers' in symbol_config:
            tiers = symbol_config['tiers']
            
            # Sort tiers by volume threshold
            sorted_tiers = sorted(tiers, key=lambda t: t['vol'])
            
            # Find applicable tier
            applicable_fee = sorted_tiers[0]['fee']  # Start with lowest tier
            
            for tier in sorted_tiers:
                if volume_ytd >= tier['vol']:
                    applicable_fee = tier['fee']
                else:
                    break
            
            return Fee(
                fee_per_side=float(applicable_fee),
                currency=symbol_config.get('currency', 'USD'),
                description=symbol_config.get('description', f'Tiered fee for {symbol}')
            )
        
        # Should not reach here due to validation
        raise ValueError(f"Invalid fee configuration for symbol {symbol}")
    
    def get_fee_info(self, symbol: str) -> Dict:
        """
        Get detailed fee information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with fee configuration details
        """
        config = self.table.get(symbol, self.table['DEFAULT'])
        
        info = {
            'symbol': symbol,
            'venue': self.venue,
            'currency': config.get('currency', 'USD'),
            'description': config.get('description', ''),
            'is_tiered': 'tiers' in config
        }
        
        if 'trade_fee' in config:
            info['base_fee'] = config['trade_fee']
        
        if 'tiers' in config:
            info['tiers'] = config['tiers']
        
        return info
    
    def list_symbols(self) -> List[str]:
        """Get list of all configured symbols (excluding DEFAULT)."""
        return [symbol for symbol in self.table.keys() if symbol != 'DEFAULT']
    
    def calculate_total_fee(self, symbol: str, quantity: int, volume_ytd: int = 0) -> float:
        """
        Calculate total fee for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Number of contracts/shares (absolute value)
            volume_ytd: Year-to-date volume for tiered pricing
            
        Returns:
            Total fee in USD
        """
        fee = self.lookup(symbol, volume_ytd)
        return abs(quantity) * fee.fee_per_side
    
    def get_venue_info(self) -> Dict:
        """Get venue metadata information."""
        return {
            'venue': self.venue,
            'last_updated': self.last_updated,
            'config_path': str(self.config_path),
            'symbols_count': len(self.list_symbols())
        }


# Global fee schedule instances (lazy loaded)
_cme_fee_schedule: Optional[FeeSchedule] = None


def get_cme_fee_schedule() -> FeeSchedule:
    """Get the global CME fee schedule instance."""
    global _cme_fee_schedule
    
    if _cme_fee_schedule is None:
        config_path = Path(__file__).parent.parent.parent / "fees" / "cme_futures.yaml"
        _cme_fee_schedule = FeeSchedule(config_path)
    
    return _cme_fee_schedule


def calculate_cme_fee(symbol: str, quantity: int, volume_ytd: int = 0) -> float:
    """
    Convenience function to calculate CME futures fee.
    
    Args:
        symbol: CME futures symbol (e.g., 'MES', 'MNQ')
        quantity: Number of contracts (absolute value)
        volume_ytd: Year-to-date volume for tiered pricing
        
    Returns:
        Total fee in USD
    """
    schedule = get_cme_fee_schedule()
    return schedule.calculate_total_fee(symbol, quantity, volume_ytd)


# Fee calculation utilities
def format_fee_summary(symbol: str, quantity: int, fee: float) -> str:
    """Format a human-readable fee summary."""
    return f"{symbol}: {abs(quantity)} contracts × ${fee/abs(quantity):.2f} = ${fee:.2f}"


def validate_fee_config(config_path: Union[str, Path]) -> bool:
    """
    Validate a fee configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        True if valid, raises exception if invalid
    """
    try:
        FeeSchedule(config_path)
        return True
    except Exception as e:
        logger.error(f"Fee config validation failed: {e}")
        raise