# src/features/feature_manager.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Type
from .base_calculator import BaseFeatureCalculator
from .rsi_calculator import RSICalculator
from .ema_calculator import EMACalculator
from .vwap_calculator import VWAPCalculator
from .time_calculator import TimeFeatureCalculator
from .market_impact_calculator import MarketImpactCalculator
from .atr_calculator import ATRCalculator
from .vwap_ratio_calculator import VWAPRatioCalculator
from .micro_price_imbalance_calculator import MicroPriceImbalanceCalculator
from .feature_registry import FeatureRegistry, get_global_registry
from .performance_tracker import PerformanceTracker, TimingContext


class FeatureManager:
    """
    Manages and orchestrates multiple feature calculators.
    Provides a plugin-like system for feature computation with performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None,
                 use_performance_tracking: bool = True):
        """
        Initialize the feature manager.
        
        Args:
            config: Configuration dictionary containing feature settings
            logger: Logger instance
            use_performance_tracking: Whether to enable performance tracking
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.feature_config = self.config.get('feature_engineering', {})
        
        # If feature_engineering is not nested, use the config itself
        if not self.feature_config:
            self.feature_config = self.config
            
        # Support both 'features' and 'features_to_calculate' for backward compatibility
        self.feature_list = (
            self.feature_config.get('features', []) or 
            self.feature_config.get('features_to_calculate', [])
        )
        self.calculators: Dict[str, BaseFeatureCalculator] = {}
        
        # Initialize registry and performance tracking
        self.registry = get_global_registry()
        self.performance_tracker = PerformanceTracker() if use_performance_tracking else None
        
        # Register default calculators if not already registered
        self._register_default_calculators()
        self._initialize_calculators()
    
    def _register_default_calculators(self):
        """Register default calculators in the registry."""
        default_calculators = {
            'RSI': (RSICalculator, {'description': 'Relative Strength Index', 'category': 'momentum'}),
            'EMA': (EMACalculator, {'description': 'Exponential Moving Average', 'category': 'trend'}),
            'VWAP': (VWAPCalculator, {'description': 'Volume Weighted Average Price', 'category': 'volume'}),
            'Time': (TimeFeatureCalculator, {'description': 'Time-based features', 'category': 'temporal'}),
            'MarketImpact': (MarketImpactCalculator, {'description': 'Market microstructure features', 'category': 'microstructure'}),
            'ATR': (ATRCalculator, {'description': 'Average True Range volatility features', 'category': 'volatility'}),
            'VWAPRatio': (VWAPRatioCalculator, {'description': 'VWAP ratio and microstructural features', 'category': 'microstructure'}),
            'MicroPriceImbalance': (MicroPriceImbalanceCalculator, {'description': 'Micro-price imbalance and order flow features', 'category': 'microstructure'}),
        }
        
        for name, (calc_class, metadata) in default_calculators.items():
            if not self.registry.get_calculator_class(name):
                self.registry.register(name, calc_class, metadata)
        
    def _initialize_calculators(self):
        """Initialize feature calculators based on configuration."""
        for feature_type in self.feature_list:
            calculator_class = self.registry.get_calculator_class(feature_type)
            if calculator_class:
                # Get specific config for this feature type
                feature_specific_config = self.feature_config.get(
                    feature_type.lower(), {}
                )
                
                # Special handling for Time features
                if feature_type == 'Time' and not feature_specific_config:
                    feature_specific_config = self.feature_config.copy()  # Pass full config
                    self.logger.debug(f"Time calculator getting full config: {feature_specific_config}")
                
                calculator = self.registry.create_calculator(
                    feature_type, feature_specific_config, self.logger
                )
                
                if calculator:
                    self.calculators[feature_type] = calculator
                    self.logger.info(f"Initialized {feature_type} calculator")
                else:
                    self.logger.error(f"Failed to create {feature_type} calculator")
            else:
                self.logger.warning(f"Unknown feature type: {feature_type}")
    
    def register_calculator(self, name: str, calculator_class: Type[BaseFeatureCalculator],
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Register a new feature calculator.
        
        Args:
            name: Name of the feature calculator
            calculator_class: Calculator class to register
            metadata: Optional metadata about the calculator
        """
        self.registry.register(name, calculator_class, metadata)
        self.logger.info(f"Registered new calculator: {name}")
    
    def compute_features(self, raw_data_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Compute all configured features on the raw data with performance tracking.
        
        Args:
            raw_data_df: DataFrame with OHLCV data and DatetimeIndex
            
        Returns:
            DataFrame with original data and computed features, or None if error
        """
        if not isinstance(raw_data_df.index, pd.DatetimeIndex):
            self.logger.error("Raw data DataFrame must have a DatetimeIndex")
            return None
            
        df = raw_data_df.copy()
        
        # Standardize column names to lowercase for consistent feature calculation
        column_mapping = {}
        for col in df.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col.lower()
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.logger.debug(f"Standardized column names: {column_mapping}")
        
        data_size = len(df)
        
        # Apply each calculator in sequence with performance tracking
        for feature_type, calculator in self.calculators.items():
            if self.performance_tracker:
                with TimingContext(self.performance_tracker, f"compute_{feature_type}", data_size):
                    try:
                        self.logger.debug(f"Computing {feature_type} features")
                        df = calculator.calculate(df)
                    except Exception as e:
                        self.logger.error(f"Error computing {feature_type} features: {e}", exc_info=True)
                        continue
            else:
                try:
                    self.logger.debug(f"Computing {feature_type} features")
                    df = calculator.calculate(df)
                except Exception as e:
                    self.logger.error(f"Error computing {feature_type} features: {e}", exc_info=True)
                    continue
        
        # Drop rows with NaNs created by indicators with warm-up periods
        original_len = len(df)
        df.dropna(inplace=True)
        dropped_rows = original_len - len(df)
        
        if dropped_rows > 0:
            self.logger.info(f"Dropped {dropped_rows} rows due to NaNs after feature calculation")
        
        if df.empty:
            self.logger.warning("DataFrame is empty after feature calculation and NaN drop")
            return None
            
        self.logger.info(f"Computed features. DataFrame columns: {df.columns.tolist()}")
        return df
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get names of all features that will be computed.
        
        Returns:
            List of all feature names
        """
        feature_names = []
        for calculator in self.calculators.values():
            feature_names.extend(calculator.get_feature_names())
        return feature_names
    
    def get_max_lookback(self) -> int:
        """
        Get the maximum lookback period required by any calculator.
        
        Returns:
            Maximum lookback period needed
        """
        max_lookback = 0
        for calculator in self.calculators.values():
            max_lookback = max(max_lookback, calculator.get_max_lookback())
        return max_lookback
    
    def get_calculator(self, feature_type: str) -> Optional[BaseFeatureCalculator]:
        """
        Get a specific feature calculator.
        
        Args:
            feature_type: Type of feature calculator to get
            
        Returns:
            Calculator instance or None if not found
        """
        return self.calculators.get(feature_type)
    
    def list_available_calculators(self) -> List[str]:
        """
        List all available calculator types.
        
        Returns:
            List of available calculator names
        """
        return self.registry.list_calculators()
    
    def list_active_calculators(self) -> List[str]:
        """
        List currently active calculator types.
        
        Returns:
            List of active calculator names
        """
        return list(self.calculators.keys())
    
    def get_performance_report(self) -> str:
        """
        Get performance report if tracking is enabled.
        
        Returns:
            Performance report string
        """
        if self.performance_tracker:
            return self.performance_tracker.get_summary_report()
        else:
            return "Performance tracking is disabled"
    
    def get_calculator_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a calculator.
        
        Args:
            name: Calculator name
            
        Returns:
            Calculator information dictionary
        """
        return self.registry.get_calculator_info(name)


if __name__ == "__main__":
    import sys

    # Setup logging to stdout for debugging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout
    )
    logger = logging.getLogger("FeatureManagerDebug")

    # Example config
    config = {
        "feature_engineering": {
            "features": ["RSI", "EMA", "VWAP", "Time", "MarketImpact"],
            "rsi": {"period": 14},
            "ema": {"span": 20},
            "vwap": {},
            "time": {},
            "marketimpact": {}
        }
    }

    # Create a sample OHLCV DataFrame with DatetimeIndex
    dates = pd.date_range("2024-01-01", periods=100, freq="min")
    data = {
        "Open": np.random.rand(100) * 100,
        "High": np.random.rand(100) * 100,
        "Low": np.random.rand(100) * 100,
        "Close": np.random.rand(100) * 100,
        "Volume": np.random.randint(100, 1000, size=100)
    }
    df = pd.DataFrame(data, index=dates)

    # Initialize and run FeatureManager
    fm = FeatureManager(config, logger=logger, use_performance_tracking=True)
    result_df = fm.compute_features(df)

    if result_df is not None:
        print("Computed features DataFrame head:")
        print(result_df.head())
        print("\nFeature columns:", result_df.columns.tolist())
        print("\nPerformance report:")
        print(fm.get_performance_report())
    else:
        print("Feature computation failed or returned empty DataFrame.")