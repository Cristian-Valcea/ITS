# Market Impact Features Implementation

This document describes the implementation of market microstructure features in the IntradayJules trading system, addressing the requirement for order-book spread/depth features in training observations.

## Overview

The market impact features provide the trading system with microstructure information that helps:
- **Training**: RL models learn to avoid trading in poor liquidity conditions
- **Risk Management**: Automatic throttling when market impact becomes excessive  
- **Execution**: Better timing of trades based on order book conditions

## Features Implemented

### 1. Core Features

| Feature Name | Formula | Range | Description |
|--------------|---------|-------|-------------|
| `spread_bps` | `(ask - bid) / mid * 10,000` | [0, ∞) | Bid-ask spread in basis points |
| `queue_imbalance` | `(bid_sz1 - ask_sz1) / (bid_sz1 + ask_sz1)` | [-1, +1] | Order book imbalance (Tan & Lehalle) |
| `impact_10k` | `depth_to_move(10k USD) / mid` | [0, ∞) | Price impact for 10k USD notional |
| `kyle_lambda` | `abs(Δmid) / signed_volume` | [0, ∞) | Kyle's lambda (price impact slope) |

### 2. Performance Characteristics

- **Fast calculation** (spread_bps, queue_imbalance): < 5 μs
- **Full calculation** (all features): < 10 μs for L5 order book
- **Memory efficient**: Minimal state tracking
- **Robust**: Handles missing/invalid data gracefully

## Architecture

### 1. Core Module: `src/shared/market_impact.py`

```python
from src.shared.market_impact import calc_market_impact_features

# Full feature calculation
features = calc_market_impact_features(
    book=order_book_series,
    mid=mid_price,
    last_mid=previous_mid,  # For Kyle's lambda
    signed_vol=trade_volume,  # For Kyle's lambda
    notional=10_000
)

# Fast calculation (critical path)
spread_bps, queue_imbalance = calc_market_impact_features_fast(
    bid_px1, bid_sz1, ask_px1, ask_sz1, mid
)
```

### 2. Feature Calculator: `src/features/market_impact_calculator.py`

Integrates with the feature engineering pipeline:

```python
from src.features import MarketImpactCalculator

calculator = MarketImpactCalculator({
    'notional_amount': 10_000,
    'enable_kyle_lambda': True
})

enhanced_df = calculator.calculate(ohlcv_df)
```

### 3. Data Enhancement: `src/agents/data_agent.py`

Simulates order book data from OHLCV for training:

```python
# Enhance historical data with simulated order book
enhanced_data = data_agent.enhance_data_with_order_book(ohlcv_df, symbol)
```

### 4. Live Processing: `src/execution/core/live_data_loader.py`

Optimized for low-latency live trading:

```python
# Critical path - only fast features
features = live_loader.process_live_market_data(
    book_snapshot, symbol, include_heavy_features=False
)

# Monitoring path - all features
full_features = live_loader.process_live_market_data(
    book_snapshot, symbol, include_heavy_features=True
)
```

### 5. Risk Integration: `src/risk/calculators/market_impact_calculator.py`

Automatic risk management based on market conditions:

```python
risk_result = market_impact_risk_calculator.calculate({
    'spread_bps': 75.0,      # Wide spread
    'queue_imbalance': 0.9,  # Heavy imbalance
    'impact_10k': 0.002,     # High impact
    'kyle_lambda': 2e-6      # High Kyle's lambda
})
# Result: {'action': 'THROTTLE', 'risk_level': 'HIGH'}
```

## Integration Points

### 1. Training Environment

Market impact features are automatically added to observations when enabled:

```yaml
# config/training.yaml
environment:
  include_market_impact_features: true
  observation_feature_cols:
    - rsi_14
    - ema_10
    # Market impact features added automatically:
    # - spread_bps
    # - queue_imbalance
    # - impact_10k
    # - kyle_lambda
```

### 2. Feature Pipeline

Register and use in feature engineering:

```python
from src.features import FeatureManager

config = {
    'features': ['RSI', 'EMA', 'MarketImpact'],
    'marketimpact': {
        'notional_amount': 10_000,
        'enable_kyle_lambda': True
    }
}

feature_manager = FeatureManager(config)
enhanced_data = feature_manager.calculate_features(raw_data)
```

### 3. Risk Management

Automatic throttling based on market conditions:

```yaml
# config/risk.yaml
risk:
  calculators:
    market_impact:
      enabled: true
      max_spread_bps: 50.0
      max_impact_threshold: 0.001
      min_queue_balance: -0.8
      max_queue_balance: 0.8
```

## Configuration

### Complete Example

See `config/market_impact_example.yaml` for a comprehensive configuration example.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `notional_amount` | 10,000 | USD notional for impact calculation |
| `enable_kyle_lambda` | true | Calculate Kyle's lambda |
| `max_spread_bps` | 50.0 | Risk threshold for spread |
| `max_impact_threshold` | 0.001 | Risk threshold for market impact |
| `include_market_impact_features` | true | Add to training observations |

## Usage Examples

### 1. Basic Feature Calculation

```python
import pandas as pd
from src.shared.market_impact import calc_market_impact_features

# Order book data
book = pd.Series({
    'bid_px1': 100.0, 'bid_sz1': 1000,
    'ask_px1': 100.1, 'ask_sz1': 1200,
    'bid_px2': 99.9, 'bid_sz2': 2000,
    'ask_px2': 100.2, 'ask_sz2': 1800
})

features = calc_market_impact_features(book, mid=100.05)
print(f"Spread: {features['spread_bps']:.1f} bps")
print(f"Imbalance: {features['queue_imbalance']:.3f}")
```

### 2. Training with Market Impact Features

```python
from src.training import TrainerAgent

config = {
    'environment': {
        'include_market_impact_features': True,
        'observation_feature_cols': ['rsi_14', 'ema_10']
    },
    'feature_engineering': {
        'features': ['RSI', 'EMA', 'MarketImpact']
    }
}

trainer = TrainerAgent(config)
model = trainer.train()  # Model will see market impact features
```

### 3. Live Trading with Risk Management

```python
from src.execution import OrchestratorAgent

config = {
    'risk': {
        'calculators': {
            'market_impact': {
                'enabled': True,
                'max_spread_bps': 50.0
            }
        }
    }
}

orchestrator = OrchestratorAgent(config)
# Trades will be automatically throttled in poor liquidity conditions
```

## Testing

### Run Tests

```bash
# Run comprehensive test suite
python -m pytest tests/test_market_impact_features.py -v

# Run simple validation
python test_market_impact_simple.py

# Performance test
python -c "
from src.shared.market_impact import calc_market_impact_features_fast
import time
start = time.perf_counter_ns()
for _ in range(1000):
    calc_market_impact_features_fast(100.0, 1000, 100.1, 1000, 100.05)
print(f'Avg time: {(time.perf_counter_ns() - start) / 1000 / 1000:.2f} μs')
"
```

### Expected Results

- **Basic calculation**: Spread ~10 bps, imbalance ~0.0 for balanced book
- **Performance**: < 5 μs for fast calculation, < 10 μs for full calculation
- **Risk integration**: Automatic throttling when thresholds exceeded

## Performance Monitoring

### Metrics

The system tracks several performance metrics:

- `market_impact_calc_time`: Calculation latency
- `market_impact_risk_score`: Current risk level
- `spread_bps`: Current spread
- `queue_imbalance`: Current imbalance

### Alerts

Risk system generates alerts when:
- Spread > 50 bps (configurable)
- Market impact > 0.1% (configurable)  
- Queue imbalance > 80% (configurable)
- Kyle's lambda > threshold (configurable)

## Troubleshooting

### Common Issues

1. **Missing order book data**: System falls back to simulated data
2. **Performance issues**: Check if heavy features are enabled in critical path
3. **Risk false positives**: Adjust thresholds in risk configuration
4. **NaN values**: Kyle's lambda requires historical data

### Debug Mode

Enable debug logging for detailed information:

```yaml
logging:
  loggers:
    market_impact:
      level: DEBUG
```

## Future Enhancements

1. **Real-time order book integration**: Direct L2 feed processing
2. **Advanced impact models**: Non-linear impact functions
3. **Cross-asset calibration**: Asset-specific parameters
4. **Machine learning impact prediction**: Learned impact models

## References

- Tan, Z. & Lehalle, C.A. (2016). "Order Book Imbalance and Market Microstructure"
- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
- Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions"