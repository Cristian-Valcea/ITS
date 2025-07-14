# Model v1.0.0-baseline

## Overview
This is the baseline DQN model trained on NVDA intraday data (2024-01-01 to 2024-01-31). While the performance is negative, it represents a stable training configuration that can serve as a reference point for future improvements.

## Key Metrics
- **Average Return**: -5.16%
- **Win Rate**: 0.0%
- **Episodes**: 159
- **Training Duration**: 2:09:32

## Model Files
- `DQN_2025-07-14_14-15-45.zip`: Complete model bundle
- `policy.pt`: PyTorch policy model
- `DQN_2025-07-14_14-15-45_torchscript.pt`: TorchScript export
- `checkpoints/`: Training checkpoints

## Configuration Highlights
- **Algorithm**: DQN with 500K timesteps
- **Trade Cooldown**: 12 steps (12 minutes)
- **Position Sizing**: 25% of capital
- **Transaction Costs**: 0.1%
- **Reward Scaling**: 0.0001

## Known Issues
1. **Overtrading**: 196.9 trades/episode despite cooldowns
2. **Multi-day Episodes**: 225 hours/episode instead of intraday
3. **Negative Learning**: Performance deteriorated over training
4. **High Turnover**: 4.59x ratio indicating excessive activity

## Usage
```python
from stable_baselines3 import DQN
model = DQN.load("models/registry/v1.0.0-baseline/model/DQN_2025-07-14_14-15-45.zip")
```

## Next Steps
- Fix episode duration to single trading days
- Increase trade cooldown to reduce overtrading
- Improve reward function design
- Add position size limits