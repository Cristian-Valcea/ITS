# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IntradayJules is a sophisticated algorithmic trading system built with Python, featuring reinforcement learning, multi-agent architecture, and comprehensive risk management for intraday trading strategies.

## Core Architecture

### Multi-Agent System
- **Orchestrator Agent** (`src/execution/orchestrator_agent.py`) - Central coordinator managing the entire trading workflow
- **Data Agent** (`src/agents/data_agent.py`) - Handles market data collection and preprocessing
- **Feature Agent** (`src/agents/feature_agent.py`) - Generates technical indicators and market features
- **Risk Agent** (`src/agents/risk_agent.py`) - Manages risk controls and position sizing
- **Trainer Agent** (`src/agents/trainer_agent.py`) - Handles model training and optimization
- **Evaluator Agent** (`src/agents/evaluator_agent.py`) - Performs backtesting and performance analysis

### Key Components
- **Trading Environment** (`src/gym_env/intraday_trading_env.py`) - Custom OpenAI Gym environment for RL training
- **Feature Engineering** (`src/features/`) - Advanced technical indicators (RSI, EMA, VWAP, market microstructure)
- **Risk Management** (`src/risk/`) - Comprehensive risk sensors, calculators, and enforcement
- **Feature Store** (`src/shared/feature_store.py`) - High-performance feature caching with DuckDB
- **FastAPI Server** (`src/api/`) - REST API for monitoring and control

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
.\activate_venv.ps1

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Start clean training with emergency fix
.\start_training_clean.bat

# Manual training command
python src/main.py train --main_config config/emergency_fix_orchestrator_gpu.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31
```

### API Server
```bash
# Start API server
.\start_api.bat
# or
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Testing
```bash
# Run comprehensive tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_risk_integration.py
python -m pytest tests/test_feature_store.py
```

### Monitoring
```bash
# TensorBoard for training metrics
tensorboard --logdir logs/tensorboard_gpu_recurrent_ppo_microstructural --port 6006

# Live log monitoring
python monitor_live_logs.py
```

## Configuration Management

### Primary Config Files
- `config/emergency_fix_orchestrator_gpu.yaml` - Main training configuration with emergency reward fix
- `config/model_params.yaml` - ML model parameters and hyperparameters  
- `config/risk_limits.yaml` - Risk management settings and thresholds

### Configuration Structure
```yaml
# Main config includes:
data_config:          # Data source and preprocessing settings
feature_config:       # Feature engineering parameters
model_config:         # RL model architecture and training parameters
risk_config:          # Risk management and position sizing
execution_config:     # Order execution and broker settings
```

## Key Development Patterns

### Agent Communication
- Agents communicate through standardized interfaces defined in `src/agents/base_agent.py`
- Use dependency injection for agent initialization
- Follow the orchestrator pattern for workflow coordination

### Configuration Loading
```python
# Standard config loading pattern
from src.shared.config_loader import load_config

config = load_config("config/main_config.yaml")
```

### Risk Management Integration
```python
# Risk checks should be integrated at key decision points
from src.risk.risk_agent_v2 import RiskAgentV2

risk_agent = RiskAgentV2(config['risk_config'])
risk_result = risk_agent.evaluate_action(action, state)
```

### Feature Store Usage
```python
# High-performance feature retrieval
from src.shared.feature_store import FeatureStore

feature_store = FeatureStore()
features = feature_store.get_features(symbol, start_date, end_date)
```

## Emergency Fix System

The system includes an emergency reward fix to address excessive turnover:
- **Config**: Set `use_emergency_reward_fix: true` in main config
- **Parameters**: `emergency_transaction_cost_pct` and `emergency_holding_bonus`
- **Monitoring**: Look for "🚨 EMERGENCY REWARD ACTIVE" messages in logs
- **Expected Impact**: Turnover should drop from 65x to <3x

## Data Pipeline

### Data Flow
1. **Raw Data** → `data/raw_*` (market data from brokers)
2. **Processing** → `data/processed_*` (cleaned and normalized)
3. **Feature Engineering** → Feature Store (cached features)
4. **Training** → Models stored in `models/`
5. **Evaluation** → Reports in `reports/`

### Supported Data Sources
- Interactive Brokers (via `ib_insync`)
- Yahoo Finance (`yfinance`)
- Alpha Vantage API
- Custom CSV data

## Model Training

### Training Process
1. **Data Collection** - Agent fetches and preprocesses market data
2. **Feature Engineering** - Technical indicators and market microstructure features
3. **Environment Setup** - Custom Gym environment with trading logic
4. **RL Training** - Stable-Baselines3 with custom callbacks
5. **Evaluation** - Backtesting and performance analysis
6. **Model Deployment** - Automated model versioning and deployment

### Key Training Parameters
- **Algorithm**: QR-DQN with Rainbow components
- **Environment**: Custom intraday trading gym environment
- **Features**: Technical indicators + market microstructure + risk metrics
- **Reward Function**: PnL-based with transaction costs and risk penalties

## Risk Management

### Risk Architecture
- **Calculators** (`src/risk/calculators/`) - Quantitative risk metrics
- **Sensors** (`src/risk/sensors/`) - Real-time risk monitoring
- **Enforcement** (`src/risk/enforcement/`) - Automated risk controls
- **Audit** (`src/risk/audit/`) - Compliance and audit logging

### Key Risk Metrics
- VaR (Value at Risk) and Expected Shortfall
- Drawdown and drawdown velocity
- Concentration and exposure limits
- Kyle's Lambda (market impact)
- Operational risk sensors

## Troubleshooting

### Common Issues
1. **DuckDB Lock Files** - Run cleanup in `start_training_clean.bat`
2. **Port Conflicts** - Kill processes on ports 6006 (TensorBoard) and 8000 (API)
3. **Memory Issues** - Adjust batch sizes in model config
4. **Feature Store Corruption** - Clear cache: `del /F /Q %USERPROFILE%\.feature_cache\*`

### Debug Commands
```bash
# Check system status
python test_system_status.py

# Validate configuration
python test_production_config.py

# Test GPU readiness
python scripts/test_gpu_readiness.py
```

## Performance Optimization

### High-Performance Features
- **DuckDB** - For feature store with columnar storage
- **Vectorized Operations** - NumPy/Pandas optimizations
- **Feature Caching** - Persistent feature store with compression
- **Batch Processing** - Efficient data loading and processing
- **GPU Acceleration** - CUDA support for training

### Memory Management
- Use `zstandard` compression for feature storage
- Implement proper connection pooling for databases
- Clear intermediate data structures promptly
- Monitor memory usage with built-in profiling

## Deployment

### Production Deployment
- **Model Registry** - Versioned model storage in `models/registry/`
- **Blue-Green Deployment** - Zero-downtime model updates
- **Health Checks** - Automated system health monitoring
- **Rollback Support** - Automatic fallback to previous models

### Monitoring in Production
- **Prometheus Metrics** - System and trading metrics
- **Grafana Dashboards** - Real-time monitoring
- **Audit Logging** - Compliance and trade audit trails
- **Alert Management** - Automated alert system

## Testing Strategy

### Test Categories
- **Unit Tests** - Individual component testing
- **Integration Tests** - Agent interaction testing
- **Risk Tests** - Risk management validation
- **Performance Tests** - System performance validation
- **Compliance Tests** - Regulatory requirement validation

### Test Execution
```bash
# Quick smoke test
python tests/simple_test.py

# Comprehensive test suite
python tests/run_comprehensive_tests.py

# Risk-specific tests
python tests/run_risk_tests.py
```

## Code Quality

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints throughout the codebase
- Implement proper error handling and logging
- Document all public APIs and complex algorithms
- Use descriptive variable and function names

### Development Tools
- **Linting**: `ruff` for code quality
- **Formatting**: `black` for consistent formatting
- **Type Checking**: `mypy` for static type analysis
- **Testing**: `pytest` for comprehensive testing
- **Documentation**: Docstrings for all public methods