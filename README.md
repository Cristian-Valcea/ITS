# IntradayJules - Intraday Trading System

A sophisticated intraday trading system built with Python, featuring multiple specialized agents for data processing, feature engineering, risk management, and automated trading execution using reinforcement learning.

## 🎯 **Current Status: v0.3.0 - Production-Ready API**

✅ **Working Components:**
- **FastAPI REST API** - Complete backend with interactive documentation
- **Multi-agent system** architecture fully functional
- **Configuration Management API** - CRUD operations for all YAML configs
- **Pipeline Control API** - Trigger training/evaluation via REST endpoints
- **Feature engineering pipeline** (RSI, EMA, VWAP, Time features)
- **Virtual environment** setup with all dependencies
- **Interactive Documentation** - Swagger UI and ReDoc

🔧 **In Development:**
- Environment shape validation for small datasets
- Enhanced dummy data generation for testing
- Asynchronous pipeline execution
- Frontend web interface

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different trading tasks
- **Reinforcement Learning**: Custom OpenAI Gym environment with Stable-Baselines3
- **Risk Management**: Built-in risk controls and position sizing
- **Feature Engineering**: Advanced technical indicators (RSI, EMA, VWAP, time features)
- **Interactive Brokers Integration**: Real-time data and order execution
- **Automated Training**: Configurable model retraining capabilities
- **Configuration Management**: Comprehensive YAML-based configuration system

## 🚀 **Quick Start**

### Prerequisites
- Python 3.10+
- Windows (PowerShell scripts provided)

### Setup
```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/IntradayJules.git
cd IntradayJules

# 2. Create and activate virtual environment (automated)
.\activate_venv.ps1

# 3. Run a quick training test
.\run_training.ps1 AAPL 2023-01-01 2023-01-31 1min

# Or manually:
python src/main.py train --symbol AAPL --start_date 2023-01-01 --end_date 2023-01-31 --interval 1min --main_config config/main_config.yaml --model_params config/model_params.yaml --risk_limits config/risk_limits.yaml
```

### Configuration
All system behavior is controlled through YAML configuration files:
- `config/main_config.yaml` - Main system configuration
- `config/model_params.yaml` - ML model parameters  
- `config/risk_limits.yaml` - Risk management settings

## 🌐 **API Usage**

### Start the API Server
```powershell
# Option 1: Using the launch script
.\scripts\run_api.ps1

# Option 2: Direct uvicorn command
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Access the API
- **Interactive Documentation**: http://127.0.0.1:8000/docs
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc
- **API Status**: http://127.0.0.1:8000/api/v1/status

### Example API Calls
```bash
# Check API status
curl http://127.0.0.1:8000/api/v1/status

# Get main configuration
curl http://127.0.0.1:8000/api/v1/config/main_config

# Trigger training pipeline
curl -X POST http://127.0.0.1:8000/api/v1/pipelines/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "start_date": "2023-01-01", "end_date": "2023-01-31", "interval": "1min"}'
```

### Test the API
```powershell
# Run the API test script
.\venv\Scripts\python.exe test_api.py
```

## 📁 Project Structure

```
IntradayJules/
├── src/
│   ├── agents/           # Trading agents
│   │   ├── base_agent.py
│   │   ├── data_agent.py
│   │   ├── env_agent.py
│   │   ├── evaluator_agent.py
│   │   ├── feature_agent.py
│   │   ├── orchestrator_agent.py
│   │   ├── risk_agent.py
│   │   └── trainer_agent.py
│   ├── gym_env/          # Custom trading environment
│   │   └── intraday_trading_env.py
│   └── main.py           # Main application entry point
├── config/               # Configuration files
│   ├── main_config.yaml
│   ├── model_params.yaml
│   └── risk_limits.yaml
└── scripts/              # Automation scripts
    ├── weekly_retrain.ps1
    └── weekly_retrain.sh
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IntradayJules.git
cd IntradayJules
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Configure your settings in the `config/` directory
2. Run the main application:
```bash
python src/main.py
```

## 📊 Agents Overview

- **Data Agent**: Handles market data collection and preprocessing
- **Feature Agent**: Generates technical indicators and market features
- **Risk Agent**: Manages position sizing and risk controls
- **Environment Agent**: Interfaces with the trading environment
- **Trainer Agent**: Handles model training and optimization
- **Evaluator Agent**: Performs backtesting and performance analysis
- **Orchestrator Agent**: Coordinates all agents and manages workflow

## ⚙️ Configuration

The system uses YAML configuration files:
- `main_config.yaml`: General system settings
- `model_params.yaml`: Machine learning model parameters
- `risk_limits.yaml`: Risk management settings

## 🔄 Automated Training

Weekly retraining is supported through:
- `weekly_retrain.sh` (Linux/Mac)
- `weekly_retrain.ps1` (Windows PowerShell)

## 📈 Performance

[Add performance metrics and backtesting results here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

[Your contact information here]