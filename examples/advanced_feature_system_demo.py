# examples/advanced_feature_system_demo.py
"""
Advanced demonstration of the modular feature system including:
- Feature registry usage
- Configuration validation
- Performance tracking
- Custom calculator creation
- Error handling and diagnostics
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add project root to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.column_names import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME
from src.agents.feature_agent import FeatureAgent
from src.features import (
    FeatureManager, DataProcessor, FeatureRegistry, ConfigValidator,
    PerformanceTracker, BaseFeatureCalculator, register_calculator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_realistic_data(num_days: int = 5, bars_per_day: int = 390) -> pd.DataFrame:
    """Create realistic intraday trading data."""
    total_bars = num_days * bars_per_day
    
    # Create datetime index for trading hours (9:30 AM to 4:00 PM)
    dates = []
    current_date = pd.Timestamp('2023-01-01')
    
    for day in range(num_days):
        if current_date.weekday() < 5:  # Only weekdays
            day_start = current_date.replace(hour=9, minute=30)
            day_bars = pd.date_range(day_start, periods=bars_per_day, freq='1min')
            dates.extend(day_bars)
        current_date += pd.Timedelta(days=1)
    
    dates = dates[:total_bars]  # Ensure we have the right number
    
    # Generate realistic price data with trends and volatility
    base_price = 100.0
    returns = np.random.normal(0, 0.001, len(dates))  # Small random returns
    
    # Add some trend and mean reversion
    trend = np.sin(np.arange(len(dates)) * 2 * np.pi / (bars_per_day * 2)) * 0.002
    returns += trend
    
    # Calculate prices
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = {
        COL_CLOSE: close_prices,
        COL_OPEN: np.roll(close_prices, 1),  # Previous close as open
        COL_HIGH: close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        COL_LOW: close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        COL_VOLUME: np.random.lognormal(8, 1, len(dates)).astype(int)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure OHLC relationships are correct
    df[COL_HIGH] = df[[COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]].max(axis=1)
    df[COL_LOW] = df[[COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]].min(axis=1)
    
    return df


class BollingerBandsCalculator(BaseFeatureCalculator):
    """Custom Bollinger Bands calculator."""
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data, [COL_CLOSE]):
            return data
            
        df = data.copy()
        window = self.config.get('window', 20)
        std_dev = self.config.get('std_dev', 2)
        
        try:
            # Calculate moving average and standard deviation
            sma = df[COL_CLOSE].rolling(window=window).mean()
            std = df[COL_CLOSE].rolling(window=window).std()
            
            # Calculate bands
            df[f'bb_upper_{window}'] = sma + (std * std_dev)
            df[f'bb_lower_{window}'] = sma - (std * std_dev)
            df[f'bb_middle_{window}'] = sma
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
            df[f'bb_position_{window}'] = (df[COL_CLOSE] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            self.logger.info(f"Calculated Bollinger Bands with window {window}")
            
        except Exception as e:
            # Handle errors for each feature
            for feature in self.get_feature_names():
                df[feature] = self.handle_calculation_error(e, feature, data.index)
        
        return df
    
    def get_feature_names(self):
        window = self.config.get('window', 20)
        return [
            f'bb_upper_{window}', f'bb_lower_{window}', f'bb_middle_{window}',
            f'bb_width_{window}', f'bb_position_{window}'
        ]
    
    def get_max_lookback(self):
        return self.config.get('window', 20)


def demo_feature_registry():
    """Demonstrate feature registry functionality."""
    print("=== Feature Registry Demo ===")
    
    registry = FeatureRegistry()
    
    # Register our custom calculator
    registry.register(
        'BollingerBands', 
        BollingerBandsCalculator,
        metadata={
            'description': 'Bollinger Bands with position and width indicators',
            'category': 'volatility',
            'author': 'Demo',
            'version': '1.0'
        }
    )
    
    print(f"Available calculators: {registry.list_calculators()}")
    
    # Get detailed info about a calculator
    bb_info = registry.get_calculator_info('BollingerBands')
    print(f"Bollinger Bands info: {bb_info}")
    
    # Validate configuration
    valid_config = {'window': 20, 'std_dev': 2}
    invalid_config = {'window': 'invalid'}
    
    print(f"Valid config validation: {registry.validate_config('BollingerBands', valid_config)}")
    print(f"Invalid config validation: {registry.validate_config('BollingerBands', invalid_config)}")


def demo_config_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Demo ===")
    
    validator = ConfigValidator()
    
    # Valid configuration
    valid_config = {
        'features': ['RSI', 'EMA', 'BollingerBands'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True},
        'bollingerbands': {'window': 20, 'std_dev': 2},
        'lookback_window': 5,
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26', 'bb_width_20'],
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26', 'bb_position_20']
    }
    
    is_valid, errors = validator.validate_config(valid_config)
    print(f"Valid config check: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Get suggestions
    suggestions = validator.suggest_improvements(valid_config)
    print(f"Suggestions: {suggestions}")
    
    # Get summary
    summary = validator.get_config_summary(valid_config)
    print(f"Config summary: {summary}")
    
    # Invalid configuration
    invalid_config = {
        'features': ['RSI', 'UnknownFeature'],
        'rsi': {'window': 'invalid'},
        'lookback_window': -1
    }
    
    is_valid, errors = validator.validate_config(invalid_config)
    print(f"\nInvalid config check: {is_valid}")
    print(f"Errors: {errors}")


def demo_performance_tracking():
    """Demonstrate performance tracking."""
    print("\n=== Performance Tracking Demo ===")
    
    # Register custom calculator globally
    register_calculator(
        'BollingerBands', 
        BollingerBandsCalculator,
        metadata={'category': 'volatility'}
    )
    
    config = {
        'features': ['RSI', 'EMA', 'BollingerBands'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True},
        'bollingerbands': {'window': 20, 'std_dev': 2},
        'lookback_window': 3
    }
    
    # Create feature manager with performance tracking
    feature_manager = FeatureManager(config, use_performance_tracking=True)
    
    # Generate test data
    test_data = create_realistic_data(num_days=2, bars_per_day=100)
    print(f"Created test data with {len(test_data)} bars")
    
    # Process data multiple times to get performance metrics
    for i in range(3):
        print(f"Processing iteration {i+1}...")
        features_df = feature_manager.compute_features(test_data)
        if features_df is not None:
            print(f"  Computed {len(features_df.columns)} features")
    
    # Get performance report
    print("\nPerformance Report:")
    print(feature_manager.get_performance_report())


def demo_complete_pipeline():
    """Demonstrate complete feature engineering pipeline."""
    print("\n=== Complete Pipeline Demo ===")
    
    # Comprehensive configuration
    config = {
        'data_dir_processed': 'data/demo_processed',
        'scalers_dir': 'data/demo_scalers',
        'features': ['RSI', 'EMA', 'VWAP', 'Time', 'BollingerBands'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True},
        'vwap': {'window': 20},
        'time': {
            'time_features': ['hour_of_day', 'day_of_week'],
            'sin_cos_encode': ['hour_of_day']
        },
        'bollingerbands': {'window': 20, 'std_dev': 2},
        'lookback_window': 10,
        'feature_cols_to_scale': [
            'rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation',
            'bb_width_20', 'bb_position_20'
        ],
        'observation_feature_cols': [
            'rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation',
            'bb_position_20', 'hour_sin', 'hour_cos', 'day_of_week'
        ]
    }
    
    # Validate configuration first
    validator = ConfigValidator()
    is_valid, errors = validator.validate_config(config)
    
    if not is_valid:
        print(f"Configuration errors: {errors}")
        return
    
    print("Configuration is valid!")
    
    # Create FeatureAgent
    agent = FeatureAgent(config)
    
    # Generate realistic data
    raw_data = create_realistic_data(num_days=3, bars_per_day=200)
    print(f"Generated {len(raw_data)} bars of realistic data")
    
    # Process data
    final_df, sequences, prices = agent.run(
        raw_data,
        symbol="DEMO_STOCK",
        fit_scaler=True,
        start_date_str="20230101",
        end_date_str="20230103",
        interval_str="1min"
    )
    
    if final_df is not None:
        print(f"\nProcessed features:")
        print(f"  DataFrame shape: {final_df.shape}")
        print(f"  Feature columns: {len(final_df.columns)}")
        print(f"  Sample features: {final_df.columns.tolist()[:10]}...")
    
    if sequences is not None:
        print(f"\nSequences for ML:")
        print(f"  Shape: {sequences.shape}")
        print(f"  Ready for model training!")
    
    if prices is not None:
        print(f"\nPrice data:")
        print(f"  Shape: {prices.shape}")
        print(f"  Aligned with sequences: {len(prices) == sequences.shape[0] if sequences is not None else 'N/A'}")
    
    # Get performance metrics
    print(f"\nPerformance metrics:")
    print(agent.feature_manager.get_performance_report())


def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Error Handling Demo ===")
    
    # Create configuration with potential issues
    config = {
        'features': ['RSI', 'EMA'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26]},
        'lookback_window': 5
    }
    
    agent = FeatureAgent(config)
    
    # Test with insufficient data
    small_data = create_realistic_data(num_days=1, bars_per_day=10)  # Only 10 bars
    print(f"Testing with insufficient data ({len(small_data)} bars)...")
    
    final_df, sequences, prices = agent.run(
        small_data,
        symbol="ERROR_TEST",
        fit_scaler=True
    )
    
    if final_df is None:
        print("  Correctly handled insufficient data")
    else:
        print(f"  Processed {len(final_df)} rows despite small dataset")
    
    # Test with invalid data
    print("\nTesting with invalid data...")
    invalid_data = pd.DataFrame({
        'invalid_col': [1, 2, 3, 4, 5]
    }, index=pd.date_range('2023-01-01', periods=5, freq='1min'))
    
    final_df, sequences, prices = agent.run(
        invalid_data,
        symbol="INVALID_TEST",
        fit_scaler=True
    )
    
    if final_df is None:
        print("  Correctly handled invalid data structure")


if __name__ == "__main__":
    print("Advanced Feature System Demonstration")
    print("=" * 60)
    
    try:
        demo_feature_registry()
        demo_config_validation()
        demo_performance_tracking()
        demo_complete_pipeline()
        demo_error_handling()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("\nKey benefits of the modular system:")
        print("✓ Plugin-like feature calculators")
        print("✓ Centralized registry management")
        print("✓ Configuration validation")
        print("✓ Performance tracking and optimization")
        print("✓ Robust error handling")
        print("✓ Easy extensibility")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()