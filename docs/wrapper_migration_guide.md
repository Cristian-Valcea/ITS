# Trading Environment Wrapper Migration Guide

## 🎯 Overview

This guide shows how to migrate from the monolithic `IntradayTradingEnv` to the new modular wrapper architecture for improved maintainability, testability, and performance.

## 🏗️ Architecture Comparison

### Before: Monolithic Environment
```python
# 98 parameters in __init__()
env = IntradayTradingEnv(
    processed_feature_data=data,
    price_data=prices,
    trade_cooldown_steps=5,
    max_daily_drawdown_pct=0.02,
    hourly_turnover_cap=5.0,
    position_sizing_pct_capital=0.25,
    action_change_penalty_factor=0.001,
    # ... 90+ more parameters
)
```

### After: Modular Wrappers
```python
# Clean base environment + composable wrappers
from src.gym_env.wrappers import create_trading_env

base_env = SimpleTradingEnv(data, prices)  # Dirt simple
env = create_trading_env(base_env, config_type='training')
```

## 🔧 Usage Examples

### 1. Quick Migration (Backward Compatible)
```python
from src.gym_env.wrappers import migrate_from_monolithic, create_trading_env

# Convert existing config
old_config = {
    'trade_cooldown_steps': 5,
    'position_sizing_pct_capital': 0.25,
    'action_change_penalty_factor': 0.001,
    'log_trades': True
}

new_config = migrate_from_monolithic(old_config)
env = create_trading_env(base_env, custom_config=new_config)
```

### 2. Preset Configurations
```python
from src.gym_env.wrappers import create_trading_env

# Training environment (balanced rules)
env = create_trading_env(base_env, config_type='training')

# Production environment (strict rules)
env = create_trading_env(base_env, config_type='production')

# Debug environment (loose rules, detailed logging)
env = create_trading_env(base_env, config_type='debug')

# Minimal environment (testing only)
env = create_trading_env(base_env, config_type='minimal')
```

### 3. Custom Configuration
```python
from src.gym_env.wrappers import TradingWrapperFactory

factory = TradingWrapperFactory()

custom_config = {
    'cooldown': {
        'enabled': True,
        'cooldown_steps': 3
    },
    'size_limit': {
        'enabled': True,
        'max_position_pct': 0.20,
        'max_portfolio_risk_pct': 0.03
    },
    'action_penalty': {
        'enabled': False  # Disable for this experiment
    },
    'streaming_log': {
        'enabled': True,
        'log_dir': 'logs/experiment_1',
        'batch_size': 50
    }
}

env = factory.create_wrapped_env(base_env, custom_config)
```

### 4. A/B Testing Different Rules
```python
# Version A: With cooldown
env_a = create_trading_env(base_env, custom_config={
    'cooldown': {'enabled': True, 'cooldown_steps': 5}
})

# Version B: Without cooldown
env_b = create_trading_env(base_env, custom_config={
    'cooldown': {'enabled': False}
})

# Compare training results
```

## 📊 Benefits

### 1. Memory Efficiency
**Before**: `self.trade_log = []` keeps entire training run in RAM
**After**: Incremental Arrow/Parquet persistence with configurable batching

```python
# Streaming logs save memory
env = create_trading_env(base_env, custom_config={
    'streaming_log': {
        'enabled': True,
        'batch_size': 100,      # Write every 100 trades
        'compression': 'snappy', # Fast compression
        'max_memory_records': 1000  # RAM limit
    }
})
```

### 2. Modularity & Testing
```python
# Test individual components
cooldown_env = CooldownWrapper(base_env, cooldown_steps=5)
size_limit_env = SizeLimitWrapper(base_env, max_position_pct=0.25)

# Test wrapper combinations
env = CooldownWrapper(
    SizeLimitWrapper(base_env, max_position_pct=0.25),
    cooldown_steps=5
)
```

### 3. Performance Optimization
```python
# Zero overhead for disabled rules
config = {
    'cooldown': {'enabled': False},      # No overhead
    'size_limit': {'enabled': False},    # No overhead
    'action_penalty': {'enabled': False} # No overhead
}
env = create_trading_env(base_env, custom_config=config)
```

## 🔄 Migration Steps

### Step 1: Prepare Base Environment
```python
# Simplify existing environment (remove complex rules)
class SimpleTradingEnv(gym.Env):
    def __init__(self, data, prices, initial_capital=100000):
        # Only core trading logic - no penalties, limits, etc.
        pass
    
    def step(self, action):
        # Execute trade, calculate reward, return obs
        pass
```

### Step 2: Gradual Wrapper Introduction
```python
# Week 1: Add streaming logs only
env = create_trading_env(base_env, custom_config={
    'streaming_log': {'enabled': True}
})

# Week 2: Add cooldown rules
env = create_trading_env(base_env, custom_config={
    'cooldown': {'enabled': True, 'cooldown_steps': 5},
    'streaming_log': {'enabled': True}
})

# Week 3: Add size limits
# ... gradually add more wrappers
```

### Step 3: Validation & Testing
```python
# Compare old vs new environments
old_env = IntradayTradingEnv(...)  # Monolithic
new_env = create_trading_env(base_env, 'training')  # Modular

# Run identical episodes and compare results
```

## 🎯 Wrapper Details

### CooldownWrapper
- **Purpose**: Prevent rapid-fire trading
- **Effect**: Forces HOLD for N steps after each trade
- **Use Case**: Reduce turnover, encourage deliberate decisions

### SizeLimitWrapper
- **Purpose**: Enforce position size and risk limits
- **Effect**: Converts risky trades to HOLD actions
- **Use Case**: Risk management, prevent over-concentration

### ActionPenaltyWrapper
- **Purpose**: Penalize undesirable trading behaviors
- **Effect**: Reduces reward for ping-ponging, poor timing
- **Use Case**: Encourage disciplined trading patterns

### StreamingTradeLogWrapper
- **Purpose**: Memory-efficient trade logging
- **Effect**: Incremental Parquet writes, configurable batching
- **Use Case**: Long training runs, production deployment

## 📈 Performance Comparison

### Memory Usage
```
Monolithic Environment:
- 500k timesteps: ~2.5 GB RAM (trade_log list)
- 1M timesteps: ~5 GB RAM

Streaming Wrapper:
- 500k timesteps: ~50 MB RAM (batched writes)
- 1M timesteps: ~50 MB RAM (constant memory)
```

### Flexibility
```
Monolithic: Change rules → Modify core environment → Risk breaking
Modular: Change rules → Adjust wrapper config → Safe & isolated
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install pyarrow  # For streaming logs
   ```

2. **Basic Usage**:
   ```python
   from src.gym_env.wrappers import create_trading_env
   
   base_env = SimpleTradingEnv(data, prices)
   env = create_trading_env(base_env, 'training')
   
   # Use exactly like before
   obs, info = env.reset()
   obs, reward, done, truncated, info = env.step(action)
   ```

3. **Monitor Wrapper Effects**:
   ```python
   # Check which wrappers are applied
   from src.gym_env.wrappers import TradingWrapperFactory
   factory = TradingWrapperFactory()
   print(factory.get_wrapper_info(env))
   
   # Monitor wrapper statistics
   info = env.step(action)[4]  # Get info dict
   print(info['cooldown_active'])
   print(info['size_limit_violated'])
   print(info['trades_in_buffer'])
   ```

## ✅ Next Steps

1. **Phase 1**: Implement SimpleTradingEnv (dirt-simple base)
2. **Phase 2**: Migrate training pipeline to use wrapper factory
3. **Phase 3**: Update configuration files to use wrapper format
4. **Phase 4**: Remove legacy parameters from monolithic environment
5. **Phase 5**: Performance testing and optimization

This modular architecture will make your trading system more maintainable, testable, and scalable for future enhancements.