# IntradayJules Project Structure

## Overview
A comprehensive Python-based intraday trading system with reinforcement learning capabilities, risk management, and enterprise-grade deployment features.

```
IntradayJules/
├── 📁 Root Configuration & Scripts
│   ├── 📄 requirements*.txt                 # Dependencies (training, execution, GPU)
│   ├── 📄 IntradayJules.pyproj/.sln        # Visual Studio project files
│   ├── 📄 pytest.ini                       # Test configuration
│   ├── 📄 .env.example                     # Environment template
│   ├── 📄 activate_venv.ps1                # Virtual environment activation
│   └── 📄 *.py                            # Root-level utilities & scripts
│
├── 📁 src/                                 # Main source code
│   ├── 📄 main.py                         # Application entry point
│   ├── 📄 column_names.py                 # Data schema definitions
│   │
│   ├── 📁 agents/                         # Core agent orchestration
│   │   ├── 📄 base_agent.py               # Abstract base agent
│   │   ├── 📄 orchestrator_agent.py       # Main coordinator
│   │   ├── 📄 data_agent*.py              # Data management agents
│   │   ├── 📄 risk_agent.py               # Risk management
│   │   ├── 📄 trainer_agent.py            # Training coordination
│   │   ├── 📄 evaluator_agent.py          # Model evaluation
│   │   ├── 📄 env_agent.py                # Environment management
│   │   └── 📄 feature_agent.py            # Feature engineering
│   │
│   ├── 📁 ai_agents/                      # AI/ML specialized agents
│   │   └── 📄 dqn_data_agent_system.py    # DQN-based data agent
│   │
│   ├── 📁 api/                            # REST API & web interface
│   │   ├── 📄 main.py                     # FastAPI application
│   │   ├── 📄 *_endpoints.py              # API endpoints
│   │   ├── 📄 *_models.py                 # Request/response models
│   │   ├── 📄 services.py                 # Business logic
│   │   └── 📁 templates/                  # HTML templates
│   │       ├── 📄 dashboard.html          # Main dashboard
│   │       ├── 📄 config_editor.html      # Configuration UI
│   │       └── 📄 *.html                  # Other UI components
│   │
│   ├── 📁 backtesting/                    # Backtesting framework
│   │   └── 📄 bias_free_backtester.py     # Unbiased backtesting
│   │
│   ├── 📁 batch/                          # Batch processing
│   │   └── 📄 end_of_day_processor.py     # EOD data processing
│   │
│   ├── 📁 cli/                            # Command-line interface
│   │   └── 📄 model_cli.py                # Model management CLI
│   │
│   ├── 📁 compliance/                     # Regulatory compliance
│   │   └── 📄 mifid_ii_exporter.py        # MiFID II reporting
│   │
│   ├── 📁 data/                           # Data management
│   │   ├── 📄 bias_aware_data_agent.py    # Bias-aware data handling
│   │   ├── 📄 survivorship_bias_handler.py # Survivorship bias correction
│   │   ├── 📄 fx_lifecycle.py             # FX data lifecycle
│   │   └── 📄 *.py                        # Other data utilities
│   │
│   ├── 📁 deployment/                     # Deployment & DevOps
│   │   ├── 📄 blue_green_rollout.py       # Blue-green deployments
│   │   └── 📄 model_deployment_service.py # Model deployment
│   │
│   ├── 📁 evaluation/                     # Model evaluation
│   │   ├── 📄 backtest_runner.py          # Backtesting execution
│   │   ├── 📄 metrics_calculator.py       # Performance metrics
│   │   ├── 📄 model_loader.py             # Model loading utilities
│   │   ├── 📄 performance_visualizer.py   # Visualization tools
│   │   └── 📄 report_generator.py         # Report generation
│   │
│   ├── 📁 execution/                      # Trade execution
│   │   ├── 📄 orchestrator_agent.py       # Execution orchestration
│   │   ├── 📄 order_throttling.py         # Order rate limiting
│   │   ├── 📄 *_execution_agent.py        # Execution agents
│   │   └── 📁 core/                       # Core execution components
│   │       ├── 📄 execution_loop.py       # Main execution loop
│   │       ├── 📄 order_router.py         # Order routing
│   │       ├── 📄 latency_monitor.py      # Latency tracking
│   │       ├── 📄 pnl_tracker.py          # P&L monitoring
│   │       ├── 📄 risk_callbacks.py       # Risk integration
│   │       └── 📄 *.py                    # Other core components
│   │
│   ├── 📁 features/                       # Feature engineering
│   │   ├── 📄 feature_manager.py          # Feature orchestration
│   │   ├── 📄 feature_registry.py         # Feature catalog
│   │   ├── 📄 *_calculator.py             # Feature calculators
│   │   └── 📄 *.py                        # Other feature utilities
│   │
│   ├── 📁 governance/                     # Model governance
│   │   ├── 📄 model_lineage.py            # Model versioning
│   │   ├── 📄 audit_immutable.py          # Audit trail
│   │   └── 📄 release_approval.py         # Release management
│   │
│   ├── 📁 graph_ai_agents/               # Graph-based AI agents
│   │   └── 📄 *.py                        # Graph AI implementations
│   │
│   ├── 📁 gym_env/                        # Reinforcement learning environment
│   │   ├── 📄 intraday_trading_env.py     # Trading environment
│   │   └── 📄 kyle_lambda_fill_simulator.py # Market impact simulation
│   │
│   ├── 📁 monitoring/                     # System monitoring
│   │   └── 📁 alerting/                   # Alert management
│   │       └── 📄 alert_manager.py        # Alert orchestration
│   │
│   ├── 📁 risk/                           # Risk management system
│   │   ├── 📄 risk_agent_v2.py            # Main risk orchestrator
│   │   ├── 📄 rules_engine.py             # Risk rules engine
│   │   ├── 📄 event_bus.py                # Event-driven architecture
│   │   ├── 📄 stress_runner.py            # Stress testing
│   │   ├── 📁 calculators/                # Risk calculators
│   │   │   ├── 📄 var_calculator.py       # Value at Risk
│   │   │   ├── 📄 drawdown_calculator.py  # Drawdown analysis
│   │   │   ├── 📄 kyle_lambda_calculator.py # Market impact
│   │   │   └── 📄 *.py                    # Other risk metrics
│   │   ├── 📁 config/                     # Risk configuration
│   │   ├── 📁 enforcement/                # Risk enforcement
│   │   ├── 📁 sensors/                    # Risk sensors
│   │   └── 📁 obs/                        # Observability
│   │
│   ├── 📁 security/                       # Security components
│   │   └── 📄 secrets_manager.py          # Secrets management
│   │
│   ├── 📁 shared/                         # Shared utilities
│   │   ├── 📄 constants.py                # System constants
│   │   ├── 📄 feature_store*.py           # Feature store
│   │   ├── 📄 fee_schedule.py             # Trading fees
│   │   ├── 📄 market_impact.py            # Market impact models
│   │   └── 📄 *.py                        # Other shared utilities
│   │
│   ├── 📁 storage/                        # Data storage
│   │   ├── 📄 audit_storage_manager.py    # Audit data storage
│   │   └── 📄 s3_audit_replication.py     # S3 replication
│   │
│   ├── 📁 tools/                          # External tools integration
│   │   ├── 📄 GetIBKRData.py              # Interactive Brokers API
│   │   └── 📄 ibkr_tools.py               # IBKR utilities
│   │
│   ├── 📁 training/                       # Model training
│   │   ├── 📄 trainer_agent.py            # Training orchestration
│   │   ├── 📄 enhanced_trainer_agent.py   # Advanced training
│   │   ├── 📄 reward_pnl_audit.py         # Reward system audit
│   │   ├── 📁 core/                       # Training core
│   │   │   ├── 📄 trainer_core.py         # Core training logic
│   │   │   ├── 📄 policy_export.py        # Model export
│   │   │   ├── 📄 env_builder.py          # Environment setup
│   │   │   └── 📄 *.py                    # Other training components
│   │   ├── 📁 callbacks/                  # Training callbacks
│   │   ├── 📁 interfaces/                 # Training interfaces
│   │   └── 📁 policies/                   # Policy implementations
│   │
│   └── 📁 utils/                          # General utilities
│       └── 📄 db.py                       # Database utilities
│
├── 📁 Infrastructure & Configuration
│   ├── 📁 config/                         # Configuration files
│   ├── 📁 k8s/ & 📁 kubernetes/           # Kubernetes deployments
│   ├── 📁 deployment_artifacts/           # Deployment assets
│   ├── 📁 secrets/                        # Secret management
│   └── 📁 scripts/                        # Utility scripts
│
├── 📁 Data & Models
│   ├── 📁 data/                           # Training/test data
│   ├── 📁 models/                         # Trained models
│   ├── 📁 cache_ibkr/                     # IBKR data cache
│   └── 📁 fees/                           # Fee data
│
├── 📁 Monitoring & Observability
│   ├── 📁 logs/                           # Application logs
│   ├── 📁 tensorboard_logs/               # TensorBoard data
│   ├── 📁 monitoring/                     # Monitoring configs
│   └── 📁 reports/                        # Generated reports
│
├── 📁 Testing & Quality Assurance
│   ├── 📁 tests/                          # Test suites
│   ├── 📁 examples/                       # Usage examples
│   └── 📄 test_*.py                       # Individual test files
│
├── 📁 Documentation & Planning
│   ├── 📁 docs/                           # Documentation
│   ├── 📁 documents/                      # Project documents
│   ├── 📁 recaps/                         # Progress summaries
│   └── 📄 *.md                            # Markdown documentation
│
└── 📁 Development Environment
    ├── 📁 .git/                           # Git repository
    ├── 📁 .github/                        # GitHub workflows
    ├── 📁 .vscode/                        # VS Code settings
    ├── 📁 venv/                           # Python virtual environment
    └── 📁 __pycache__/ & cache files      # Build artifacts
```

## Key Components Description

### 🎯 **Core Architecture**
- **Orchestrator Agent**: Central coordinator managing all system components
- **Trading Environment**: Reinforcement learning environment for strategy training
- **Execution Engine**: High-performance order execution with <50ms latency
- **Risk Management**: Real-time risk monitoring and enforcement

### 🧠 **AI/ML Pipeline**
- **Training Core**: DQN-based reinforcement learning with risk-aware callbacks
- **Feature Engineering**: Real-time feature calculation (RSI, VWAP, EMA, etc.)
- **Model Export**: TorchScript deployment for production
- **Evaluation**: Comprehensive backtesting and performance analysis

### 🛡️ **Risk & Compliance**
- **Risk Calculators**: VaR, drawdown, market impact, volatility analysis
- **Event-Driven Architecture**: Real-time risk event processing
- **MiFID II Compliance**: Regulatory reporting and audit trails
- **Stress Testing**: Portfolio stress testing and scenario analysis

### 🚀 **Production Ready**
- **API Layer**: FastAPI-based REST API with web dashboard
- **Kubernetes Deployment**: Container orchestration and scaling
- **Blue-Green Deployments**: Zero-downtime model updates
- **Monitoring**: Prometheus metrics, TensorBoard integration, audit logs

### 📊 **Data Management**
- **Feature Store**: Optimized feature storage and retrieval
- **Survivorship Bias Handling**: Data quality and bias correction
- **IBKR Integration**: Interactive Brokers data pipeline
- **S3 Replication**: Distributed data storage and backup

## Technology Stack
- **Language**: Python 3.10+
- **ML Framework**: Stable-Baselines3, PyTorch
- **Data**: Pandas, DuckDB, ccxt
- **API**: FastAPI, asyncio
- **Deployment**: Kubernetes, Docker
- **Monitoring**: Prometheus, TensorBoard, loguru
- **Testing**: pytest (≥85% coverage)
