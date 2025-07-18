# 📊 TensorBoard Integration for Turnover Penalty System

## 🎯 Overview

The IntradayJules trading system now includes comprehensive TensorBoard integration for visualizing:
- **Training Progress**: Loss, learning rate, Q-values, gradients
- **Performance Metrics**: Total rewards, win rate, Sharpe ratio, drawdown
- **Turnover Penalty Evolution**: Penalty values, normalized turnover, excess tracking
- **Risk Management**: Volatility, VaR, expected shortfall, portfolio metrics
- **System Monitoring**: Buffer sizes, computation time, model parameters

## 🚀 Quick Start

### 1. Launch TensorBoard
```bash
# Simple launch (opens browser automatically)
python launch_tensorboard.py

# Custom port
python launch_tensorboard.py --port 6007

# List available runs
python launch_tensorboard.py --list-runs

# Don't open browser
python launch_tensorboard.py --no-browser
```

### 2. Access TensorBoard
Open your browser to: **http://localhost:6006**

### 3. Alternative Launch Methods
```bash
# Direct TensorBoard command
tensorboard --logdir runs

# Using the exporter module
python src/training/core/tensorboard_exporter.py --launch
```

## 📈 Available Visualizations

### 🎯 **Turnover Penalty Metrics**
- `turnover/penalty_mean` - Average turnover penalty
- `turnover/penalty_current` - Current penalty value
- `turnover/normalized_mean` - Average normalized turnover
- `turnover/normalized_current` - Current normalized turnover
- `turnover/target` - Target turnover ratio
- `turnover/excess_mean` - Average excess over target
- `turnover/excess_current` - Current excess over target

### 📊 **Performance Metrics**
- `episode/total_reward` - Episode rewards
- `episode/portfolio_value` - Portfolio value evolution
- `performance/win_rate` - Percentage of profitable episodes
- `performance/sharpe_ratio_current` - Current Sharpe ratio
- `performance/sharpe_ratio_mean` - Average Sharpe ratio
- `performance/max_drawdown_current` - Current maximum drawdown
- `performance/max_drawdown_worst` - Worst drawdown experienced

### 🧠 **Training Metrics**
- `training/loss` - Training loss convergence
- `training/learning_rate` - Learning rate schedule
- `training/q_value_mean` - Average Q-values
- `training/q_value_variance` - Q-value spread (max - min)
- `training/gradient_norm` - Gradient magnitude

### ⚠️ **Risk Metrics**
- `risk/volatility` - Portfolio volatility
- `risk/var_95` - Value at Risk (95% confidence)
- `risk/var_99` - Value at Risk (99% confidence)
- `risk/expected_shortfall` - Expected shortfall (CVaR)
- `risk/beta` - Portfolio beta
- `risk/correlation` - Market correlation

### 🔍 **Monitoring Metrics**
- `monitoring/vol_penalty` - Volatility penalty tracking
- `monitoring/drawdown_pct_mean` - Average drawdown percentage
- `monitoring/q_variance_mean` - Q-value variance monitoring
- `monitoring/reward_magnitude_mean` - Reward magnitude tracking

## 🛠️ Integration in Training Code

### Basic Usage
```python
from src.training.core.tensorboard_exporter import TensorBoardExporter

# Create exporter
with TensorBoardExporter(
    log_dir="runs",
    model_name="DQN_Trading",
    comment="turnover_penalty_v2"
) as exporter:
    
    # Training loop
    for episode in range(num_episodes):
        # ... run episode ...
        
        # Log episode metrics
        exporter.log_episode_metrics(
            episode=episode,
            total_reward=episode_reward,
            portfolio_value=portfolio_value,
            turnover_penalty=turnover_penalty,
            normalized_turnover=normalized_turnover,
            turnover_target=target_ratio,
            win=episode_reward > 0
        )
        
        # Log training metrics
        if episode % 10 == 0:
            exporter.log_training_metrics(
                step=episode * steps_per_episode,
                loss=training_loss,
                learning_rate=current_lr,
                q_values=q_value_list
            )
```

### Advanced Usage
```python
# Log risk metrics
exporter.log_risk_metrics(
    step=current_step,
    volatility=portfolio_volatility,
    var_95=value_at_risk_95,
    expected_shortfall=cvar
)

# Log hyperparameters
exporter.log_hyperparameters(
    hparams={
        'learning_rate': 0.001,
        'target_turnover': 0.02,
        'penalty_weight': 50.0
    },
    metrics={
        'final_reward': final_episode_reward,
        'final_sharpe': final_sharpe_ratio
    }
)

# Log model graph
exporter.log_model_graph(model, sample_input)

# Log custom histograms
exporter.log_histogram('weights/layer1', model.layer1.weight, step)
```

## 🔧 Automatic Integration

The TensorBoard monitoring is automatically enabled in the training system:

### In TrainerAgent
```python
# Monitoring callbacks are automatically added
monitoring_callbacks = create_monitoring_callbacks(self.config)
callbacks.extend(monitoring_callbacks)
```

### Enhanced Monitoring Callback
The `TensorBoardMonitoringCallback` automatically extracts:
- Environment metrics from the trading environment
- Turnover penalty data from the penalty calculator
- Portfolio performance metrics
- Risk management indicators

## 📁 Log Structure

TensorBoard logs are organized as:
```
runs/
├── ModelName_YYYYMMDD_HHMMSS_comment/
│   ├── events.out.tfevents.* (scalar data)
│   ├── events.out.tfevents.* (histograms)
│   └── events.out.tfevents.* (graphs)
└── ModelName_YYYYMMDD_HHMMSS_comment2/
    └── ...
```

## 🎛️ Configuration

### Environment Variables
```bash
# Set custom log directory
export TENSORBOARD_LOG_DIR="custom_logs"

# Set custom port
export TENSORBOARD_PORT=6007
```

### Training Configuration
```yaml
training:
  tensorboard_log: "runs"
  log_frequency: 100  # Log every 100 steps
  
monitoring:
  enable_tensorboard: true
  log_frequency: 100
  buffer_size: 1000
```

## 📊 Key Visualizations to Monitor

### 1. **Turnover Penalty Learning**
- Watch `turnover/penalty_current` decrease over time
- Monitor `turnover/normalized_current` converging to `turnover/target`
- Check `turnover/excess_current` approaching zero

### 2. **Performance Improvement**
- Track `episode/total_reward` trending upward
- Monitor `performance/win_rate` increasing
- Watch `performance/sharpe_ratio_current` improving

### 3. **Training Stability**
- Ensure `training/loss` is decreasing smoothly
- Check `training/q_value_variance` for stability
- Monitor `training/gradient_norm` for explosion/vanishing

### 4. **Risk Management**
- Watch `performance/max_drawdown_current` staying controlled
- Monitor `risk/volatility` for acceptable levels
- Check `risk/var_95` for risk exposure

## 🔍 Troubleshooting

### TensorBoard Not Starting
```bash
# Check if TensorBoard is installed
pip install tensorboard

# Check if port is available
netstat -an | grep 6006

# Try different port
python launch_tensorboard.py --port 6007
```

### No Data Visible
- Ensure training has started and logged some episodes
- Check that log directory exists and contains event files
- Verify TensorBoard is pointing to correct log directory

### Performance Issues
- Reduce log frequency in configuration
- Clear old log directories
- Use smaller buffer sizes for monitoring

## 🚀 Advanced Features

### Custom Metrics
```python
# Log custom business metrics
exporter.writer.add_scalar('business/profit_factor', profit_factor, step)
exporter.writer.add_scalar('business/max_position_size', max_position, step)
```

### Comparative Analysis
```python
# Log multiple experiments for comparison
exporter.log_hyperparameters(
    hparams={'experiment': 'baseline'},
    metrics={'final_sharpe': baseline_sharpe}
)

exporter.log_hyperparameters(
    hparams={'experiment': 'turnover_penalty'},
    metrics={'final_sharpe': enhanced_sharpe}
)
```

### Real-time Monitoring
```python
# Flush logs frequently for real-time viewing
exporter.flush()  # Force write to disk
```

## 📈 Best Practices

1. **Consistent Naming**: Use consistent metric names across experiments
2. **Meaningful Tags**: Group related metrics with prefixes (training/, performance/, etc.)
3. **Regular Flushing**: Flush logs periodically for real-time monitoring
4. **Hyperparameter Logging**: Always log hyperparameters for reproducibility
5. **System Info**: Log system information for debugging
6. **Clean Organization**: Use timestamped directories for different runs

## 🎯 Expected Results

With proper turnover penalty tuning, you should see:

1. **Early Training**: High turnover penalty, poor performance
2. **Mid Training**: Decreasing penalty as agent learns control
3. **Late Training**: Stable penalty near target, improved performance
4. **Convergence**: Consistent performance with controlled turnover

The TensorBoard visualizations will clearly show this learning progression! 🚀

---

## 📞 Support

For issues or questions about TensorBoard integration:
1. Check the troubleshooting section above
2. Review the example code in `test_tensorboard_integration.py`
3. Examine the monitoring callback in `src/training/core/tensorboard_monitoring.py`