"""
Market Impact Features Module

This module provides market microstructure features for training observations,
including spread, queue imbalance, market impact, and Kyle's lambda.

Features implemented:
- spread_bps: (ask - bid) / mid * 10000 - Raw microstructure cost
- queue_imbalance: (q_bid1 - q_ask1) / (q_bid1 + q_ask1) - Tan & Lehalle imbalance
- impact_10k: depth_to_move(10k USD) / mid - Price move for 10k notional
- kyle_lambda: abs(Δmid) / (signed volume) - Classical impact slope proxy
"""

import numpy as np
import pandas as pd
from typing import TypedDict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ImpactFeatures(TypedDict):
    """Market impact features structure."""
    spread_bps: float
    queue_imbalance: float
    impact_10k: float
    kyle_lambda: float


def calc_market_impact_features(
    book: Union[pd.DataFrame, pd.Series],
    mid: float,
    last_mid: Optional[float] = None,
    signed_vol: Optional[float] = None,
    notional: float = 10_000
) -> ImpactFeatures:
    """
    Calculate market impact features from order book data.
    
    Args:
        book: Order book data with columns:
            - bid_px1, bid_sz1, ask_px1, ask_sz1 (level-1, required)
            - bid_pxN, bid_szN, ask_pxN, ask_szN (optional deeper levels)
        mid: Mid price (bid_px1 + ask_px1) / 2
        last_mid: Previous mid price for Kyle's lambda calculation
        signed_vol: Signed volume for Kyle's lambda calculation
        notional: Notional amount for impact calculation (default: 10,000 USD)
    
    Returns:
        ImpactFeatures dictionary with calculated features
    """
    try:
        # Handle both DataFrame and Series input
        if isinstance(book, pd.DataFrame):
            if len(book) == 0:
                return _get_default_features()
            book_data = book.iloc[0] if len(book) > 1 else book.iloc[0]
        else:
            book_data = book
        
        # Extract level-1 data (required)
        try:
            bid_px1 = float(book_data["bid_px1"])
            bid_sz1 = float(book_data["bid_sz1"])
            ask_px1 = float(book_data["ask_px1"])
            ask_sz1 = float(book_data["ask_sz1"])
        except KeyError as e:
            logger.warning(f"Missing required level-1 data: {e}")
            return _get_default_features()
        
        # Validate inputs
        if mid <= 0 or bid_px1 <= 0 or ask_px1 <= 0:
            logger.warning("Invalid price data")
            return _get_default_features()
        
        if bid_sz1 < 0 or ask_sz1 < 0:
            logger.warning("Invalid size data")
            return _get_default_features()
        
        # 1. Spread in basis points
        spread_bps = (ask_px1 - bid_px1) / mid * 10_000
        
        # 2. Queue imbalance (Tan & Lehalle)
        total_size = bid_sz1 + ask_sz1
        if total_size > 0:
            queue_imbalance = (bid_sz1 - ask_sz1) / total_size
        else:
            queue_imbalance = 0.0
        
        # 3. Market impact for specified notional
        impact_10k = _calculate_market_impact(book_data, mid, notional)
        
        # 4. Kyle's lambda (requires historical data)
        kyle_lambda = _calculate_kyle_lambda(mid, last_mid, signed_vol)
        
        return ImpactFeatures(
            spread_bps=spread_bps,
            queue_imbalance=queue_imbalance,
            impact_10k=impact_10k,
            kyle_lambda=kyle_lambda
        )
        
    except Exception as e:
        logger.error(f"Error calculating market impact features: {e}")
        return _get_default_features()


def _calculate_market_impact(book_data: pd.Series, mid: float, notional: float) -> float:
    """
    Calculate market impact for sweeping specified notional.
    
    Assumes we are BUYING (consuming ask side).
    For selling, would consume bid side.
    """
    try:
        usd_left = notional
        impact_px = mid
        
        # Try to consume up to 5 levels of the order book
        for level in range(1, 6):
            px_col = f"ask_px{level}"
            sz_col = f"ask_sz{level}"
            
            # Check if this level exists
            if px_col not in book_data or sz_col not in book_data:
                break
                
            px = float(book_data[px_col])
            sz = float(book_data[sz_col])
            
            if px <= 0 or sz <= 0:
                continue
                
            # Calculate USD value at this level
            usd_at_level = sz * px
            take = min(usd_at_level, usd_left)
            
            if take > 0:
                impact_px = px
                usd_left -= take
                
            if usd_left <= 0:
                break
        
        # Return relative price impact
        if mid > 0:
            return (impact_px - mid) / mid
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Error calculating market impact: {e}")
        return 0.0


def _calculate_kyle_lambda(mid: float, last_mid: Optional[float], signed_vol: Optional[float]) -> float:
    """
    Calculate Kyle's lambda (price impact per unit volume).
    
    Kyle's lambda = |Δmid| / |signed_volume|
    """
    if last_mid is None or signed_vol is None or signed_vol == 0:
        return np.nan
    
    try:
        price_change = abs(mid - last_mid)
        volume_abs = abs(signed_vol)
        
        if volume_abs > 0:
            return price_change / volume_abs
        else:
            return np.nan
            
    except Exception as e:
        logger.warning(f"Error calculating Kyle's lambda: {e}")
        return np.nan


def _get_default_features() -> ImpactFeatures:
    """Return default/fallback feature values."""
    return ImpactFeatures(
        spread_bps=0.0,
        queue_imbalance=0.0,
        impact_10k=0.0,
        kyle_lambda=np.nan
    )


def validate_book_data(book: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Validate that order book data has required columns.
    
    Args:
        book: Order book data
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ["bid_px1", "bid_sz1", "ask_px1", "ask_sz1"]
    
    try:
        if isinstance(book, pd.DataFrame):
            return all(col in book.columns for col in required_cols)
        else:  # Series
            return all(col in book.index for col in required_cols)
    except Exception:
        return False


def get_available_levels(book: Union[pd.DataFrame, pd.Series]) -> int:
    """
    Get the number of available order book levels.
    
    Args:
        book: Order book data
        
    Returns:
        Number of available levels (1-5)
    """
    try:
        if isinstance(book, pd.DataFrame):
            columns = book.columns
        else:
            columns = book.index
            
        max_level = 0
        for level in range(1, 6):
            if f"bid_px{level}" in columns and f"ask_px{level}" in columns:
                max_level = level
            else:
                break
                
        return max_level
        
    except Exception:
        return 0


def calc_market_impact_features_batch(
    book_df: pd.DataFrame,
    mid_series: pd.Series,
    notional: float = 10_000
) -> pd.DataFrame:
    """
    Calculate market impact features for a batch of observations.
    
    Args:
        book_df: DataFrame with order book data (multiple rows)
        mid_series: Series with mid prices for each row
        notional: Notional amount for impact calculation
        
    Returns:
        DataFrame with impact features for each row
    """
    results = []
    
    for i in range(len(book_df)):
        try:
            book_row = book_df.iloc[i]
            mid = mid_series.iloc[i]
            
            # Calculate Kyle's lambda with previous mid
            last_mid = mid_series.iloc[i-1] if i > 0 else None
            
            # For signed volume, we'd need trade data - using None for now
            signed_vol = None
            
            features = calc_market_impact_features(
                book_row, mid, last_mid, signed_vol, notional
            )
            
            results.append(features)
            
        except Exception as e:
            logger.warning(f"Error processing row {i}: {e}")
            results.append(_get_default_features())
    
    return pd.DataFrame(results)


# Performance-optimized version for live trading
def calc_market_impact_features_fast(
    bid_px1: float, bid_sz1: float,
    ask_px1: float, ask_sz1: float,
    mid: float
) -> tuple[float, float]:
    """
    Fast calculation of critical features for live trading.
    
    Only calculates spread_bps and queue_imbalance for minimal latency.
    
    Args:
        bid_px1, bid_sz1: Best bid price and size
        ask_px1, ask_sz1: Best ask price and size
        mid: Mid price
        
    Returns:
        Tuple of (spread_bps, queue_imbalance)
    """
    try:
        # Spread in basis points
        spread_bps = (ask_px1 - bid_px1) / mid * 10_000
        
        # Queue imbalance
        total_size = bid_sz1 + ask_sz1
        if total_size > 0:
            queue_imbalance = (bid_sz1 - ask_sz1) / total_size
        else:
            queue_imbalance = 0.0
            
        return spread_bps, queue_imbalance
        
    except Exception:
        return 0.0, 0.0


if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    # Sample order book data
    sample_book = pd.Series({
        "bid_px1": 1.1998, "bid_sz1": 500_000,
        "ask_px1": 1.2000, "ask_sz1": 400_000,
        "bid_px2": 1.1996, "bid_sz2": 300_000,
        "ask_px2": 1.2002, "ask_sz2": 350_000,
    })
    
    mid_price = (sample_book["bid_px1"] + sample_book["ask_px1"]) / 2
    
    features = calc_market_impact_features(sample_book, mid_price)
    
    print("Market Impact Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")