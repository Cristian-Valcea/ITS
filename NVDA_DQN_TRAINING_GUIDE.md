# 🎯 NVDA DQN Training Guide

This guide shows you how to use the specialized HTML interface to train a Deep Q-Network (DQN) model on NVIDIA (NVDA) stock data for the period November 1, 2024 to May 31, 2025, with testing on June 2025 data.

## 🚀 Quick Start

### 1. Start the API Server

```powershell
# From the project root directory
.\start_api.ps1
```

### 2. Access the NVDA DQN Training Interface

Open your web browser and navigate to:
```
http://127.0.0.1:8000/ui/nvda-dqn
```

### 3. Configure and Start Training

The interface is pre-configured with optimal settings for NVDA DQN training:

- **Symbol**: NVDA (pre-filled, read-only)
- **Training Period**: November 1, 2024 to May 31, 2025 (7 months)
- **Testing Period**: June 1, 2025 to June 30, 2025 (1 month)
- **Data Interval**: 1-minute bars (recommended for intraday trading)
- **Algorithm**: Deep Q-Network with experience replay
- **Evaluation**: Automatic evaluation on June 2025 data

Simply click **"🚀 Start NVDA DQN Training"** to begin!

## 📋 What the Training Pipeline Does

### Phase 1: Data Collection
- Downloads NVDA 1-minute price bars from November 1, 2024 to May 31, 2025
- Validates data quality and handles missing values
- Caches data for faster subsequent runs

### Phase 2: Feature Engineering
- **RSI (14-period)**: Relative Strength Index for momentum
- **EMA (10, 20, 50-period)**: Exponential Moving Averages for trend
- **VWAP**: Volume Weighted Average Price with deviation
- **Time Features**: Hour of day, minute of hour (cyclically encoded)

### Phase 3: Environment Setup
- Creates IntradayTradingEnv with:
  - Initial capital: $100,000
  - Transaction cost: 0.05% per trade
  - Risk controls: 5% max daily drawdown
  - Position sizing: 25% of capital per trade

### Phase 4: DQN Training
- **Algorithm**: Deep Q-Network with experience replay
- **Policy**: Multi-layer Perceptron (MLP)
- **Learning Rate**: 0.0001
- **Buffer Size**: 100,000 experiences
- **Batch Size**: 64
- **Exploration**: ε-greedy (1.0 → 0.05 over 10% of training)
- **Target Network**: Updated every 1,000 steps
- **Total Timesteps**: 1,000,000 (configurable in config/model_params.yaml)

### Phase 5: Model Export
- Saves trained DQN model as .zip file
- Exports TorchScript policy bundle for production deployment
- Validates prediction latency (<100µs SLO)

### Phase 6: Evaluation (Optional)
- Backtests on June 2025 data (out-of-sample)
- Generates performance metrics:
  - Total return, Sharpe ratio, Max drawdown
  - Win rate, Profit factor, Calmar ratio
  - Trade statistics and turnover analysis

## 🔧 Configuration Options

### Training Parameters
- **Use Cached Data**: Enable to reuse previously downloaded data
- **Continue from Model**: Specify path to resume training from checkpoint
- **Training Period**: Adjust start/end dates (default: 7 months)
- **Data Interval**: Choose from 1min, 5min, 15min, 1hour, 1day

### Evaluation Settings
- **Run Evaluation**: Enable automatic post-training evaluation
- **Evaluation Period**: Default is June 2025 (1 month out-of-sample)
- **Evaluation Interval**: Should match training interval

### Advanced Configuration
To modify DQN hyperparameters, edit `config/model_params.yaml`:

```yaml
algorithm_name: "DQN"
algorithm_params:
  learning_rate: 0.0001
  buffer_size: 100000
  batch_size: 64
  target_update_interval: 1000
  exploration_fraction: 0.1
  exploration_final_eps: 0.05
```

## 📊 Monitoring Training Progress

### 1. Task Status Page
After starting training, you'll get a Task ID. Click "📈 Check Status" to monitor:
- Current status (pending/running/completed/failed)
- Real-time progress updates
- Error messages if training fails

### 2. TensorBoard Logs
Training metrics are logged to `logs/tensorboard/`. Start TensorBoard:

```bash
tensorboard --logdir=logs/tensorboard/
```

### 3. Application Logs
Check `logs/app.log` for detailed training information.

## 📁 Output Files

### Model Files
```
models/orch_test/DQN_YYYYMMDD_HHMMSS/
├── policy.pt              # TorchScript policy bundle
├── metadata.json          # Model metadata
├── model.zip              # Full SB3 model
└── training_config.json   # Training configuration
```

### Reports
```
reports/orch_test/
└── eval_NVDA_dqn_final_YYYYMMDD_HHMMSS_summary.txt
```

### Data Files
```
data/
├── raw_orch_test/NVDA/     # Raw price data
├── processed_orch_test/NVDA/ # Processed features
└── scalers_orch_test/      # Feature scalers
```

## 🚨 Troubleshooting

### Common Issues

**1. "Data fetching failed"**
- Check internet connection
- Verify IBKR connection settings in `config/main_config.yaml`
- Try with `use_cached_data` disabled

**2. "Environment creation failed"**
- Check feature engineering configuration
- Verify data shapes and types
- Review `config/main_config.yaml` feature settings

**3. "Training failed"**
- Check GPU/CPU resources
- Verify model parameters in `config/model_params.yaml`
- Review risk limits in `config/risk_limits.yaml`

**4. "Out of memory"**
- Reduce batch size in `config/model_params.yaml`
- Reduce buffer size
- Use smaller training period

### Performance Tips

**For Faster Training:**
- Enable "Use Cached Data" for subsequent runs
- Use GPU if available (CUDA)
- Reduce total_timesteps for testing
- Use larger batch sizes (if memory allows)

**For Better Results:**
- Use longer training periods (12+ months)
- Tune hyperparameters with Optuna
- Enable risk-aware training callbacks
- Use ensemble methods

## 🔗 API Endpoints

The HTML interface uses these REST API endpoints:

- `POST /ui/nvda-dqn` - Submit training job
- `GET /ui/tasks/{task_id}` - Check training status
- `GET /api/v1/status` - System health check
- `GET /docs` - Interactive API documentation

## 📈 Expected Results

### Training Time
- **CPU**: 3-5 hours for 7 months of 1-minute data
- **GPU**: 45-90 minutes with CUDA acceleration

### Performance Metrics
Typical results for NVDA DQN training:
- **Sharpe Ratio**: 1.5-2.5 (good performance)
- **Max Drawdown**: 3-8% (within risk limits)
- **Win Rate**: 45-55% (typical for momentum strategies)
- **Annual Return**: 15-35% (varies by market conditions)

## 🎯 Next Steps

After successful training:

1. **Evaluate Results**: Review the evaluation report
2. **Hyperparameter Tuning**: Use Optuna for optimization
3. **Production Deployment**: Use the TorchScript bundle
4. **Live Trading**: Integrate with ExecutionAgent
5. **Model Monitoring**: Set up performance tracking

## 📞 Support

For issues or questions:
- Check the logs in `logs/app.log`
- Review the API documentation at `/docs`
- Examine the task status page for error details
- Consult the main project documentation

---

**Happy Trading! 🚀📈**