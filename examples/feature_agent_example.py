# examples/feature_agent_example.py
"""
Example demonstrating the new modular FeatureAgent system.
Shows how to use the plugin-like feature calculators and data processing components.
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
from src.features import FeatureManager, DataProcessor
from src.features.base_calculator import BaseFeatureCalculator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1min')
    num_rows = len(dates)
    
    # Generate realistic OHLCV data
    base_price = 100
    price_changes = np.random.randn(num_rows) * 0.5
    close_prices = base_price + np.cumsum(price_changes)
    
    data = {
        COL_OPEN: close_prices + np.random.randn(num_rows) * 0.1,
        COL_HIGH: close_prices + np.abs(np.random.randn(num_rows)) * 0.3,
        COL_LOW: close_prices - np.abs(np.random.randn(num_rows)) * 0.3,
        COL_CLOSE: close_prices,
        COL_VOLUME: np.random.randint(1000, 10000, size=num_rows)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure OHLC relationships are correct
    df[COL_HIGH] = df[[COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]].max(axis=1)
    df[COL_LOW] = df[[COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]].min(axis=1)
    
    return df


def example_basic_usage():
    """Example of basic FeatureAgent usage."""
    print("=== Basic FeatureAgent Usage ===")
    
    # Configuration for the FeatureAgent
    config = {
        'data_dir_processed': 'data/processed_example',
        'scalers_dir': 'data/scalers_example',
        'features': ['RSI', 'EMA', 'VWAP', 'Time'],  # Which feature types to compute
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True},
        'vwap': {'window': 20},  # Rolling VWAP with 20-period window
        'time': {
            'time_features': ['hour_of_day', 'day_of_week'], 
            'sin_cos_encode': ['hour_of_day']
        },
        'lookback_window': 5,
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation'],
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff', 'vwap_deviation', 'hour_sin', 'hour_cos']
    }
    
    # Create FeatureAgent
    feature_agent = FeatureAgent(config=config)
    
    # Create sample data
    raw_data = create_sample_data()
    print(f"Created sample data with shape: {raw_data.shape}")
    
    # Process the data
    final_df, sequences, prices = feature_agent.run(
        raw_data, 
        symbol="EXAMPLE_STOCK", 
        fit_scaler=True,  # Fit new scaler (training mode)
        start_date_str="20230101", 
        end_date_str="20230101", 
        interval_str="1min"
    )
    
    if final_df is not None:
        print(f"\nProcessed features DataFrame shape: {final_df.shape}")
        print(f"Feature columns: {final_df.columns.tolist()}")
        print("\nFirst few rows of processed features:")
        print(final_df.head())
    
    if sequences is not None:
        print(f"\nFeature sequences shape: {sequences.shape}")
        print("This is ready for ML model input!")
    
    if prices is not None:
        print(f"\nPrice data shape: {prices.shape}")
        print("This is aligned with sequences for reward calculation")


def example_modular_components():
    """Example of using individual modular components."""
    print("\n=== Using Modular Components Separately ===")
    
    # Create sample data
    raw_data = create_sample_data()
    
    # 1. Using FeatureManager directly
    feature_config = {
        'features': ['RSI', 'EMA'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26], 'ema_diff': True}
    }
    
    feature_manager = FeatureManager(config=feature_config)
    print(f"Available calculators: {feature_manager.list_available_calculators()}")
    print(f"Active calculators: {feature_manager.list_active_calculators()}")
    
    # Compute features
    features_df = feature_manager.compute_features(raw_data)
    if features_df is not None:
        print(f"Computed features: {feature_manager.get_all_feature_names()}")
        print(f"Max lookback needed: {feature_manager.get_max_lookback()}")
    
    # 2. Using DataProcessor directly
    data_config = {
        'lookback_window': 3,
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26', 'ema_diff']
    }
    
    data_processor = DataProcessor(
        config=data_config, 
        scalers_dir='data/scalers_example'
    )
    
    if features_df is not None:
        # Normalize features
        normalized_df = data_processor.normalize_features(
            features_df, fit_scaler=True, symbol="EXAMPLE"
        )
        
        if normalized_df is not None:
            # Create sequences
            obs_features = data_processor.prepare_observation_features(normalized_df)
            sequences = data_processor.create_sequences(obs_features)
            
            if sequences is not None:
                print(f"Created sequences with shape: {sequences.shape}")


def example_custom_calculator():
    """Example of creating and registering a custom feature calculator."""
    print("\n=== Custom Feature Calculator Example ===")
    
    class SimpleMACalculator(BaseFeatureCalculator):
        """Simple Moving Average calculator."""
        
        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            if not self.validate_data(data, [COL_CLOSE]):
                return data
                
            df = data.copy()
            window = self.config.get('window', 20)
            
            try:
                df[f'sma_{window}'] = df[COL_CLOSE].rolling(window=window).mean()
                self.logger.info(f"Calculated SMA with window {window}")
            except Exception as e:
                df[f'sma_{window}'] = self.handle_calculation_error(
                    e, f'sma_{window}', data.index
                )
            
            return df
        
        def get_feature_names(self):
            window = self.config.get('window', 20)
            return [f'sma_{window}']
        
        def get_max_lookback(self):
            return self.config.get('window', 20)
    
    # Register the custom calculator
    feature_manager = FeatureManager(config={})
    feature_manager.register_calculator('SMA', SimpleMACalculator)
    
    print(f"Available calculators after registration: {feature_manager.list_available_calculators()}")
    
    # Use the custom calculator
    config_with_sma = {
        'features': ['SMA', 'RSI'],
        'sma': {'window': 20},
        'rsi': {'window': 14}
    }
    
    feature_manager_with_sma = FeatureManager(config=config_with_sma)
    raw_data = create_sample_data()
    
    features_with_sma = feature_manager_with_sma.compute_features(raw_data)
    if features_with_sma is not None:
        print(f"Features with custom SMA: {features_with_sma.columns.tolist()}")


def example_live_trading_simulation():
    """Example of live trading simulation."""
    print("\n=== Live Trading Simulation ===")
    
    config = {
        'features': ['RSI', 'EMA'],
        'rsi': {'window': 14},
        'ema': {'windows': [12, 26]},
        'lookback_window': 3,
        'feature_cols_to_scale': ['rsi_14', 'ema_12', 'ema_26'],
        'observation_feature_cols': ['rsi_14', 'ema_12', 'ema_26']
    }
    
    feature_agent = FeatureAgent(config=config)
    
    # Create historical data for warmup
    historical_data = create_sample_data()
    warmup_data = historical_data.iloc[:50]  # First 50 bars for warmup
    live_data = historical_data.iloc[50:]    # Remaining bars for live simulation
    
    # First, train the scaler with historical data
    print("Training scaler with historical data...")
    _, _, _ = feature_agent.run(
        warmup_data, 
        symbol="LIVE_EXAMPLE", 
        fit_scaler=True
    )
    
    # Initialize live session
    print("Initializing live trading session...")
    feature_agent.initialize_live_session(
        symbol="LIVE_EXAMPLE",
        historical_data_for_warmup=warmup_data.iloc[-20:]  # Last 20 bars for warmup
    )
    
    # Simulate live bars
    print("Processing live bars...")
    for i in range(min(5, len(live_data))):  # Process first 5 live bars
        new_bar = live_data.iloc[i:i+1]  # Single row DataFrame
        
        observation, price_info = feature_agent.process_live_bar(new_bar, "LIVE_EXAMPLE")
        
        if observation is not None:
            print(f"Bar {i+1}: Got observation with shape {observation.shape}")
        else:
            print(f"Bar {i+1}: No observation yet (building history)")


if __name__ == "__main__":
    print("FeatureAgent Modular System Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_modular_components()
        example_custom_calculator()
        example_live_trading_simulation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()