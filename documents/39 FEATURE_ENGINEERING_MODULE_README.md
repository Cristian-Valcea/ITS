# Feature Engineering Module

This module provides a modular, plugin-like system for feature computation and data processing in the IntradayJules trading system.

## Architecture Overview

The feature engineering system is split into several modular components:

### 1. Base Components

- **`BaseFeatureCalculator`**: Abstract base class for all feature calculators
- **`FeatureManager`**: Orchestrates multiple feature calculators
- **`DataProcessor`**: Handles normalization, scaling, and sequencing

### 2. Feature Calculators (Plugins)

- **`RSICalculator`**: Relative Strength Index
- **`EMACalculator`**: Exponential Moving Averages with MACD-like differences
- **`VWAPCalculator`**: Volume Weighted Average Price with deviation
- **`TimeFeatureCalculator`**: Time-based features with sin/cos encoding

### 3. Main Interface

- **`FeatureAgent`**: Refactored main agent that uses the modular components

## Key Benefits

1. **Modularity**: Each feature calculator is independent and can be used separately
2. **Extensibility**: Easy to add new feature calculators by extending `BaseFeatureCalculator`
3. **Plugin System**: Feature calculators are registered and can be dynamically loaded
4. **Separation of Concerns**: Feature computation is separate from data processing
5. **Reusability**: Components can be reused across different contexts
6. **Testability**: Each component can be tested independently

## Usage Examples

### Basic Usage

```python
from src.agents.feature_agent import FeatureAgent

config = {
    'features': ['RSI', 'EMA', 'VWAP', 'Time'],
    'rsi': {'window': 14},
    'ema': {'windows': [12, 26], 'ema_diff': True},
    'vwap': {'window': 20},
    'time': {'time_features': ['hour_of_day'], 'sin_cos_encode': ['hour_of_day']},
    'lookback_window': 5
}

agent = FeatureAgent(config)
features_df, sequences, prices = agent.run(raw_data, symbol="AAPL", fit_scaler=True)
```

### Using Components Separately

```python
from src.features import FeatureManager, DataProcessor

# Feature computation
feature_manager = FeatureManager(config)
features_df = feature_manager.compute_features(raw_data)

# Data processing
data_processor = DataProcessor(config, scalers_dir="data/scalers")
normalized_df = data_processor.normalize_features(features_df, fit_scaler=True)
sequences = data_processor.create_sequences(normalized_df)
```

### Creating Custom Feature Calculators

```python
from src.features.base_calculator import BaseFeatureCalculator

class CustomCalculator(BaseFeatureCalculator):
    def calculate(self, data):
        # Your custom feature calculation logic
        return data_with_features
    
    def get_feature_names(self):
        return ['custom_feature_1', 'custom_feature_2']
    
    def get_max_lookback(self):
        return self.config.get('window', 20)

# Register and use
feature_manager.register_calculator('Custom', CustomCalculator)
```

## Configuration

The system uses a hierarchical configuration structure:

```python
config = {
    # Global settings
    'features': ['RSI', 'EMA', 'VWAP', 'Time'],  # Which calculators to use
    'lookback_window': 5,                        # Sequence length
    'feature_cols_to_scale': [...],              # Columns to normalize
    'observation_feature_cols': [...],           # Final feature set for model
    
    # Calculator-specific settings
    'rsi': {'window': 14},
    'ema': {'windows': [12, 26], 'ema_diff': True},
    'vwap': {'window': 20, 'group_by_day': False},
    'time': {
        'time_features': ['hour_of_day', 'day_of_week'],
        'sin_cos_encode': ['hour_of_day']
    }
}
```

## Live Trading Support

The system supports live trading with proper state management:

```python
# Initialize live session
agent.initialize_live_session(
    symbol="AAPL",
    historical_data_for_warmup=warmup_data
)

# Process live bars
for new_bar in live_data_stream:
    observation, price_info = agent.process_live_bar(new_bar, "AAPL")
    if observation is not None:
        # Use observation for trading decision
        action = model.predict(observation)
```

## Error Handling

- Each calculator handles its own errors gracefully
- Failed calculations return NaN series with proper logging
- Data validation ensures input requirements are met
- Comprehensive logging at different levels (DEBUG, INFO, WARNING, ERROR)

## Testing

Each component can be tested independently:

```python
# Test individual calculator
calculator = RSICalculator({'window': 14})
result = calculator.calculate(test_data)

# Test feature manager
manager = FeatureManager(config)
features = manager.compute_features(test_data)

# Test data processor
processor = DataProcessor(config, scalers_dir)
normalized = processor.normalize_features(features, fit_scaler=True)
```

## Migration from Original FeatureAgent

The original `FeatureAgent` has been refactored to use this modular system while maintaining the same public interface. Existing code should continue to work without changes.

Key differences:
- Internal implementation is now modular
- Better error handling and logging
- Easier to extend with new features
- More efficient live trading processing
- Better separation of concerns

## File Structure

```
src/features/
├── __init__.py                 # Module exports
├── README.md                   # This file
├── base_calculator.py          # Abstract base class
├── feature_manager.py          # Feature orchestration
├── data_processor.py           # Data processing utilities
├── rsi_calculator.py           # RSI implementation
├── ema_calculator.py           # EMA implementation
├── vwap_calculator.py          # VWAP implementation
└── time_calculator.py          # Time features implementation
```

## Future Extensions

The modular design makes it easy to add:
- Bollinger Bands calculator
- MACD calculator
- Stochastic oscillator
- Order book imbalance features
- Custom technical indicators
- Alternative normalization methods
- Different sequence generation strategies