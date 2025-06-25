# IntradayJules - Intraday Trading System

A sophisticated intraday trading system built with Python, featuring multiple specialized agents for data processing, feature engineering, risk management, and automated trading execution.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different trading tasks
- **Gym Environment**: Custom OpenAI Gym environment for reinforcement learning
- **Risk Management**: Built-in risk controls and position sizing
- **Feature Engineering**: Advanced technical indicators and market features
- **Automated Training**: Weekly model retraining capabilities
- **Configuration Management**: YAML-based configuration system

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