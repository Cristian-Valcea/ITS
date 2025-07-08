# IntradayJules Information

## Summary
IntradayJules is a sophisticated intraday trading system built with Python, featuring a multi-agent architecture for data processing, feature engineering, risk management, and automated trading execution using reinforcement learning. The system includes a FastAPI REST API for configuration management and pipeline control.

## Structure
- **src/**: Core application code organized by functionality
  - **agents/**: Trading system agents (data, feature, risk, etc.)
  - **api/**: FastAPI implementation for REST endpoints
  - **gym_env/**: Custom OpenAI Gym environment for trading
  - **risk/**: Risk management system with calculators and sensors
  - **features/**: Feature engineering pipeline
  - **training/**: Model training and optimization
- **config/**: YAML configuration files
- **data/**: Raw and processed market data
- **models/**: Trained ML models
- **scripts/**: Automation scripts for deployment and maintenance
- **tests/**: Test suite for the application

## Language & Runtime
**Language**: Python
**Version**: 3.10+
**Build System**: Standard Python setuptools
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- **Machine Learning**: tensorflow, torch, scikit-learn
- **Reinforcement Learning**: gym, gymnasium, stable-baselines3
- **Data Handling**: pandas, numpy, yfinance, ib_insync
- **Feature Store**: duckdb, pyarrow, zstandard
- **API Framework**: fastapi, uvicorn, pydantic
- **Configuration**: PyYAML, python-dotenv

**Development Dependencies**:
- pytest, black, flake8, ruff, mypy, isort

## Build & Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## API
**Framework**: FastAPI
**Entry Point**: src/api/main.py
**Endpoints**:
- `/api/v1/status`: System status
- `/api/v1/config/*`: Configuration management
- `/api/v1/pipelines/*`: Pipeline control (training, evaluation)
- `/ui/*`: Web interface endpoints

**Run Command**:
```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Testing
**Framework**: pytest
**Test Location**: tests/
**Naming Convention**: test_*.py
**Configuration**: pytest.ini
**Markers**: slow, integration, unit

**Run Command**:
```bash
pytest
# Or specific tests
pytest tests/run_tests.py
```

## Main Components

### Data Processing
- **Data Agent**: Handles market data collection and preprocessing
- **Feature Agent**: Generates technical indicators (RSI, EMA, VWAP, time features)
- **Feature Store**: DuckDB-based storage for efficient feature retrieval

### Trading Environment
- **Custom Gym Environment**: OpenAI Gym-compatible trading environment
- **Environment Agent**: Interfaces with the trading environment

### Risk Management
- **Risk Agent**: Manages position sizing and risk controls
- **Risk Calculators**: Specialized risk metrics computation
- **Risk Sensors**: Real-time monitoring of risk parameters

### Model Training
- **Trainer Agent**: Handles model training and optimization
- **Evaluator Agent**: Performs backtesting and performance analysis
- **Hyperparameter Search**: Distributed optimization with Ray/Optuna

### Orchestration
- **Orchestrator Agent**: Coordinates all agents and manages workflow
- **Pipeline Control**: Training, evaluation, and live trading pipelines
- **Configuration Management**: YAML-based configuration system

### Interactive Brokers Integration
- **IBKR Tools**: Real-time data and order execution via ib_insync
- **Execution Agent**: Handles order placement and management

### Survivorship Bias Handling
- **Bias-Free Backtester**: Accounts for survivorship bias in historical data
- **CRSP Integration**: Delisting returns integration