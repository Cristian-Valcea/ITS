# TensorBoard Telemetry for Lambda Multiplier Monitoring

## Overview

The trading environment now emits comprehensive telemetry metrics for TensorBoard monitoring, providing real-time insights into lambda multiplier behavior, penalty applications, and drawdown dynamics.

## Telemetry Categories

### 🔥 Lambda Multiplier Metrics

**Always Recorded** (when lambda is calculated):
- `lambda/multiplier`: Current lambda multiplier value (1.0x to 8.0x via sigmoid)
- `lambda/base_lambda`: Base lambda from schedule (e.g., 1000-10000)
- `lambda/penalty_lambda`: Final penalty lambda (base × multiplier, capped at 500k)
- `lambda/ceiling_hit`: Binary flag (1.0 if absolute ceiling hit, 0.0 otherwise)
- `lambda/sigmoid_multiplier`: Raw sigmoid multiplier for current excess
- `lambda/excess_pct`: Current drawdown excess percentage

**Sigmoid Events** (when multiplier changes):
- `lambda/multiplier_growth`: Percentage growth from sigmoid response
- `lambda/sigmoid_multiplier`: Raw sigmoid multiplier calculation
- `lambda/excess_pct`: Current excess percentage driving sigmoid response

### ⚠️ Penalty Metrics

**Soft DD Penalties**:
- `penalties/soft_dd_penalty`: Penalty amount applied to reward
- `penalties/soft_dd_excess_pct`: DD excess above soft limit (percentage)

**Hard DD Penalties**:
- `penalties/hard_dd_penalty`: Penalty amount applied to reward
- `penalties/hard_dd_excess_pct`: DD excess above hard limit (percentage)

### 📊 Drawdown Metrics

**Always Recorded** (every step):
- `drawdown/current_pct`: Current drawdown percentage
- `drawdown/soft_limit_pct`: Soft DD limit (e.g., 2.0%)
- `drawdown/hard_limit_pct`: Hard DD limit (e.g., 4.0%)
- `drawdown/consecutive_steps`: Consecutive steps in drawdown

### 🔄 Baseline Reset Metrics (DD Purgatory Escape)

**Always Recorded** (when baseline reset enabled):
- `baseline/current_value`: Current baseline value for DD calculation
- `baseline/steps_since_reset`: Steps since last baseline reset
- `baseline/portfolio_vs_baseline`: Portfolio performance vs baseline (%)
- `baseline/recovery_threshold`: Threshold for purgatory escape
- `baseline/distance_to_escape`: Distance to escape threshold (%)

**Reset Events** (when baseline resets occur):
- `baseline/reset_triggered`: Binary flag (1.0 when reset occurs)
- `baseline/reset_reason`: Reset reason code (1=purgatory_escape, 2=flat_timeout, 3=legacy_recovery)
- `baseline/old_baseline`: Previous baseline value
- `baseline/new_baseline`: New baseline value after reset

## Usage with Stable-Baselines3

### Basic Setup

```python
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# Configure TensorBoard logging
logger = configure("./tensorboard_logs/", ["tensorboard"])

# Create environment (telemetry is automatic)
env = IntradayTradingEnv(...)

# Create model with logger
model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(logger)

# Train (telemetry will be recorded)
model.learn(total_timesteps=100000)
```

### Advanced Monitoring

```python
# Custom callback to log additional metrics
from stable_baselines3.common.callbacks import BaseCallback

class LambdaMonitoringCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Access environment telemetry
        if hasattr(self.training_env.envs[0], 'current_lambda_multiplier'):
            multiplier = self.training_env.envs[0].current_lambda_multiplier
            self.logger.record('custom/lambda_multiplier', multiplier)
        return True

# Use callback during training
callback = LambdaMonitoringCallback()
model.learn(total_timesteps=100000, callback=callback)
```

## TensorBoard Visualization

### Launch TensorBoard

```bash
tensorboard --logdir=./tensorboard_logs/
```

### Key Visualizations

1. **Lambda Multiplier Evolution**:
   - Plot `lambda/multiplier` over time
   - Look for exponential growth patterns
   - Monitor cap hits (`lambda/multiplier_capped`)

2. **Penalty Impact**:
   - Compare `penalties/soft_dd_penalty` vs `penalties/hard_dd_penalty`
   - Correlate with `drawdown/current_pct`

3. **Drawdown Dynamics**:
   - Plot `drawdown/current_pct` with limits
   - Monitor `drawdown/consecutive_steps` for persistence

4. **System Health**:
   - Watch `lambda/ceiling_hit` (should be rare)
   - Monitor `lambda/penalty_lambda` for explosions

## Example TensorBoard Queries

### Multiplier Growth Events
```
lambda/multiplier_growth > 0
```

### High Penalty Periods
```
penalties/soft_dd_penalty > 0.01
```

### Ceiling Breaches
```
lambda/ceiling_hit == 1.0
```

## Telemetry Compatibility

The telemetry system is **backward compatible**:
- ✅ Works with standard Python loggers (telemetry disabled)
- ✅ Works with Stable-Baselines3 loggers (telemetry enabled)
- ✅ Works with custom loggers that implement `record()` method
- ✅ No performance impact when telemetry is disabled

## Performance Considerations

- **Minimal Overhead**: Telemetry only records when `logger.record()` exists
- **Selective Recording**: Lambda metrics only recorded when lambda is calculated
- **Efficient Storage**: Uses float32 precision for TensorBoard compatibility

## Troubleshooting

### No Telemetry Data
- Ensure logger has `record()` method
- Check that environment is being used with SB3 or compatible framework
- Verify TensorBoard log directory permissions

### Missing Lambda Metrics
- Lambda telemetry only appears when penalties are calculated
- Ensure drawdown is occurring to trigger lambda calculations
- Check that `dynamic_lambda_schedule=True` in config

### Performance Issues
- Telemetry has minimal impact (~0.1% overhead)
- If needed, disable by using standard Python logger
- Consider reducing logging frequency for very long training runs

## Integration Examples

### Ray/RLlib Integration
```python
import ray
from ray.rllib.algorithms.ppo import PPO

# Ray automatically handles TensorBoard logging
config = {
    "env": IntradayTradingEnv,
    "framework": "torch",
    "logger_config": {
        "type": "ray.tune.logger.TBXLogger",
        "logdir": "./ray_results"
    }
}

trainer = PPO(config=config)
trainer.train()  # Telemetry automatically recorded
```

### Weights & Biases Integration
```python
import wandb
from stable_baselines3.common.logger import configure

# Configure W&B logging
logger = configure("./logs/", ["wandb"])

model = PPO("MlpPolicy", env, verbose=1)
model.set_logger(logger)
model.learn(total_timesteps=100000)  # Metrics sync to W&B
```

This telemetry system provides comprehensive monitoring of the lambda multiplier system, enabling data-driven optimization and debugging of the penalty mechanism.