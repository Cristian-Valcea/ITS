"""
FX Survivorship Bias Integration Examples
=========================================

This module demonstrates how to integrate the FX lifecycle guard into
the existing IntradayJules trading system to eliminate survivorship bias
in FX spot trading.

INTEGRATION POINTS:
1. ETL / DataAgent: Filter data before storing in FeatureStore
2. Training: Apply bias-free filtering in environment builder
3. Live Execution: Check pair activity before trading decisions

USAGE:
- Import and use in your existing data pipelines
- Replace raw FX data loading with bias-aware loading
- Integrate into backtesting and live trading workflows
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from data.fx_lifecycle import FxLifecycle
from data.survivorship_bias_handler import SurvivorshipBiasHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasAwareDataAgent:
    """
    Enhanced DataAgent with FX survivorship bias protection.
    
    This class demonstrates how to integrate FX lifecycle filtering
    into the existing data pipeline.
    """
    
    def __init__(self, fx_lifecycle_path: str = "data/fx_lifecycle.parquet"):
        """Initialize with FX lifecycle guard."""
        self.fx_lifecycle = FxLifecycle(fx_lifecycle_path)
        self.survivorship_handler = SurvivorshipBiasHandler(
            fx_lifecycle_path=fx_lifecycle_path
        )
        logger.info("BiasAwareDataAgent initialized with FX survivorship protection")
    
    def load_fx_data(self, 
                     pair: str, 
                     start_date: datetime, 
                     end_date: datetime,
                     apply_bias_filter: bool = True) -> pd.DataFrame:
        """
        Load FX data with optional survivorship bias filtering.
        
        Args:
            pair: FX pair symbol (e.g., "EURUSD")
            start_date: Start date for data
            end_date: End date for data
            apply_bias_filter: Whether to apply survivorship bias filtering
            
        Returns:
            DataFrame with FX OHLC data
        """
        logger.info(f"Loading FX data for {pair}: {start_date.date()} to {end_date.date()}")
        
        # Simulate loading raw FX data (replace with actual data source)
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(pair) % 2**32)  # Consistent random data per pair
        
        # Generate sample OHLC data
        n_days = len(dates)
        returns = np.random.normal(0.0001, 0.01, n_days)
        base_price = 1.2 if pair == "EURUSD" else np.random.uniform(0.5, 2.0)
        prices = base_price * np.exp(np.cumsum(returns))
        
        raw_df = pd.DataFrame({
            'open': prices * np.random.uniform(0.999, 1.001, n_days),
            'high': prices * np.random.uniform(1.001, 1.005, n_days),
            'low': prices * np.random.uniform(0.995, 0.999, n_days),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_days)
        }, index=dates)
        
        if apply_bias_filter:
            # Apply FX lifecycle filtering
            try:
                filtered_df = self.fx_lifecycle.apply(raw_df, pair, mode="drop")
                logger.info(f"Applied survivorship bias filter: {len(raw_df)} → {len(filtered_df)} rows")
                return filtered_df
            except KeyError:
                logger.warning(f"FX pair {pair} not in lifecycle database - returning raw data")
                return raw_df
        else:
            return raw_df
    
    def warm_feature_store_cache(self, pairs: List[str], lookback_days: int = 252):
        """
        Warm FeatureStore cache with bias-free FX data.
        
        This method demonstrates ETL integration point.
        """
        logger.info(f"Warming FeatureStore cache for {len(pairs)} FX pairs")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for pair in pairs:
            try:
                # Load bias-free data
                clean_data = self.load_fx_data(pair, start_date, end_date, apply_bias_filter=True)
                
                # Simulate storing in FeatureStore
                logger.info(f"Cached {len(clean_data)} rows for {pair} in FeatureStore")
                
                # Here you would call: FeatureStore.store(pair, clean_data)
                
            except Exception as e:
                logger.error(f"Failed to cache data for {pair}: {e}")


class BiasAwareEnvironmentBuilder:
    """
    Enhanced environment builder with FX survivorship bias protection.
    
    This class demonstrates how to integrate FX lifecycle filtering
    into the training pipeline.
    """
    
    def __init__(self, fx_lifecycle_path: str = "data/fx_lifecycle.parquet"):
        """Initialize with FX lifecycle guard."""
        self.fx_lifecycle = FxLifecycle(fx_lifecycle_path)
        logger.info("BiasAwareEnvironmentBuilder initialized")
    
    def make_training_env(self, 
                         pairs: List[str], 
                         start_date: datetime, 
                         end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Create training environment with bias-free FX data.
        
        Args:
            pairs: List of FX pairs to include
            start_date: Training data start date
            end_date: Training data end date
            
        Returns:
            Dictionary of bias-free DataFrames for training
        """
        logger.info(f"Creating bias-free training environment for {len(pairs)} pairs")
        
        training_data = {}
        
        for pair in pairs:
            try:
                # Generate sample data (replace with actual data loading)
                dates = pd.date_range(start_date, end_date, freq='D')
                np.random.seed(hash(pair) % 2**32)
                
                n_days = len(dates)
                returns = np.random.normal(0.0001, 0.01, n_days)
                base_price = np.random.uniform(0.5, 2.0)
                prices = base_price * np.exp(np.cumsum(returns))
                
                raw_df = pd.DataFrame({
                    'open': prices * np.random.uniform(0.999, 1.001, n_days),
                    'high': prices * np.random.uniform(1.001, 1.005, n_days),
                    'low': prices * np.random.uniform(0.995, 0.999, n_days),
                    'close': prices,
                    'volume': np.random.randint(1000, 10000, n_days)
                }, index=dates)
                
                # Apply survivorship bias filtering
                clean_df = self.fx_lifecycle.apply(raw_df, pair, mode="drop")
                training_data[pair] = clean_df
                
                logger.info(f"Prepared training data for {pair}: {len(clean_df)} rows")
                
            except KeyError:
                logger.warning(f"Skipping {pair} - not in FX lifecycle database")
            except Exception as e:
                logger.error(f"Failed to prepare training data for {pair}: {e}")
        
        return training_data


class BiasAwareLiveTrader:
    """
    Enhanced live trader with FX survivorship bias protection.
    
    This class demonstrates how to integrate FX lifecycle checking
    into live trading execution.
    """
    
    def __init__(self, fx_lifecycle_path: str = "data/fx_lifecycle.parquet"):
        """Initialize with FX lifecycle guard."""
        self.fx_lifecycle = FxLifecycle(fx_lifecycle_path)
        self.active_pairs = set()
        logger.info("BiasAwareLiveTrader initialized")
    
    def update_tradeable_pairs(self):
        """Update list of tradeable FX pairs based on current date."""
        now = pd.Timestamp.now()
        
        # Get all pairs from lifecycle database
        all_pairs = list(self.fx_lifecycle._idx.keys())
        
        # Filter to only active pairs
        self.active_pairs = {
            pair for pair in all_pairs 
            if self.fx_lifecycle.is_active(pair, now)
        }
        
        logger.info(f"Updated tradeable pairs: {len(self.active_pairs)} active")
        return self.active_pairs
    
    def should_trade_pair(self, pair: str) -> bool:
        """
        Check if a pair should be traded based on lifecycle status.
        
        Args:
            pair: FX pair symbol
            
        Returns:
            True if pair should be traded, False otherwise
        """
        now = pd.Timestamp.now()
        
        try:
            is_active = self.fx_lifecycle.is_active(pair, now)
            if not is_active:
                logger.info(f"Skipping {pair} - inactive/discontinued")
            return is_active
        except:
            logger.warning(f"Unknown pair {pair} - skipping for safety")
            return False
    
    def process_trading_signals(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Process trading signals, filtering out inactive pairs.
        
        Args:
            signals: Dictionary of pair -> signal strength
            
        Returns:
            Filtered signals for only active pairs
        """
        logger.info(f"Processing {len(signals)} trading signals")
        
        filtered_signals = {}
        
        for pair, signal in signals.items():
            if self.should_trade_pair(pair):
                filtered_signals[pair] = signal
            else:
                logger.info(f"Filtered out signal for inactive pair: {pair}")
        
        logger.info(f"Filtered to {len(filtered_signals)} active pair signals")
        return filtered_signals


def demonstrate_etl_integration():
    """Demonstrate ETL pipeline integration."""
    print("\n🔄 ETL Pipeline Integration")
    print("=" * 40)
    
    # Initialize bias-aware data agent
    data_agent = BiasAwareDataAgent()
    
    # Example: Load data for multiple pairs
    pairs = ["EURUSD", "GBPUSD", "USDDEM", "USDVEF"]
    
    print("Loading FX data with survivorship bias protection:")
    for pair in pairs:
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        try:
            data = data_agent.load_fx_data(pair, start_date, end_date)
            print(f"  {pair}: {len(data):,} rows loaded")
        except Exception as e:
            print(f"  {pair}: Error - {e}")
    
    # Warm FeatureStore cache
    print("\nWarming FeatureStore cache:")
    data_agent.warm_feature_store_cache(pairs)


def demonstrate_training_integration():
    """Demonstrate training pipeline integration."""
    print("\n🎓 Training Pipeline Integration")
    print("=" * 40)
    
    # Initialize bias-aware environment builder
    env_builder = BiasAwareEnvironmentBuilder()
    
    # Create training environment
    pairs = ["EURUSD", "GBPUSD", "USDDEM", "USDFRF", "USDVEF"]
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    training_data = env_builder.make_training_env(pairs, start_date, end_date)
    
    print(f"Training environment created with {len(training_data)} pairs:")
    for pair, data in training_data.items():
        print(f"  {pair}: {len(data):,} training samples")


def demonstrate_live_trading_integration():
    """Demonstrate live trading integration."""
    print("\n📈 Live Trading Integration")
    print("=" * 40)
    
    # Initialize bias-aware live trader
    trader = BiasAwareLiveTrader()
    
    # Update tradeable pairs
    active_pairs = trader.update_tradeable_pairs()
    print(f"Active tradeable pairs: {sorted(active_pairs)}")
    
    # Simulate trading signals
    all_signals = {
        "EURUSD": 0.75,
        "GBPUSD": -0.45,
        "USDDEM": 0.85,  # Discontinued pair
        "USDVEF": -0.90,  # Discontinued pair
        "USDJPY": 0.30,
    }
    
    print(f"\nOriginal signals: {len(all_signals)}")
    for pair, signal in all_signals.items():
        print(f"  {pair}: {signal:+.2f}")
    
    # Filter signals through survivorship bias protection
    filtered_signals = trader.process_trading_signals(all_signals)
    
    print(f"\nFiltered signals: {len(filtered_signals)}")
    for pair, signal in filtered_signals.items():
        print(f"  {pair}: {signal:+.2f}")


def main():
    """Run FX survivorship bias integration examples."""
    print("🚀 FX SURVIVORSHIP BIAS INTEGRATION EXAMPLES")
    print("=" * 60)
    print()
    print("This demonstrates how to integrate FX lifecycle filtering")
    print("into existing IntradayJules components to eliminate")
    print("survivorship bias in FX spot trading.")
    
    try:
        demonstrate_etl_integration()
        demonstrate_training_integration()
        demonstrate_live_trading_integration()
        
        print("\n" + "=" * 60)
        print("✅ INTEGRATION EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print("🔧 NEXT STEPS:")
        print("1. Replace existing FX data loading with BiasAwareDataAgent")
        print("2. Update training pipelines to use BiasAwareEnvironmentBuilder")
        print("3. Integrate BiasAwareLiveTrader into execution system")
        print("4. Update FeatureStore to use survivorship-filtered data")
        print()
        print("📊 BENEFITS:")
        print("• Eliminates survivorship bias in FX backtesting")
        print("• Prevents trading discontinued/inactive FX pairs")
        print("• Provides realistic performance expectations")
        print("• Maintains data quality and integrity")
        
    except Exception as e:
        print(f"\n❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)