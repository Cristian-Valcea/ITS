# IntradayJules Trading System - User Manual

**Version:** 1.0  
**Last Updated:** August 2, 2025  
**Status:** Production Ready  

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [System Overview](#system-overview)
3. [Monitoring & Visualization](#monitoring--visualization)
4. [Automated Validation](#automated-validation)
5. [Development Workflows](#development-workflows)
6. [Production Operations](#production-operations)
7. [Troubleshooting](#troubleshooting)
8. [Management Reporting](#management-reporting)

---

## Quick Start Guide

### Prerequisites
- Python 3.10+ with virtual environment
- Docker and Docker Compose
- Polygon.io API key (for live data)
- TimescaleDB instance

### 🚀 Fast Setup (5 minutes)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start monitoring infrastructure
docker compose -f docker-compose.grafana.yml up -d

# 3. Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus

# 4. Generate sample data
python monitoring/prometheus_data_generator.py --mode sample
```

### 🧪 Test System Validation

```bash
# Quick validation test
python scripts/nightly_polygon_backtest.py --test-mode

# View results
cat reports/backtest/nightly_results.json
```

---

## System Overview

**IntradayJules** is an institutional-grade algorithmic trading system featuring:

- **AI-Driven Trading**: 201K-step trained RecurrentPPO model
- **Dual-Ticker Portfolio**: NVDA + MSFT with 26-dimensional observations
- **Real-Time Monitoring**: Executive dashboards with live metrics
- **Continuous Validation**: Nightly backtests on live market data
- **Risk Management**: Multi-layer safety controls and position limits

### Architecture Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Live Market   │───▶│   AI Trading     │───▶│   Monitoring    │
│   Data (IBKR)   │    │   Engine         │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   TimescaleDB   │    │  Risk Controls   │    │   Executive     │
│   (Market Data) │    │  & Validation    │    │   Reports       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Monitoring & Visualization

### 🤖 AI Trading System - Executive Monitoring Dashboard

Complete monitoring infrastructure for the AI trading system with real-time portfolio performance, trading activity, and AI model metrics.

#### 📊 Dashboard Features

##### Panel 1: Portfolio Performance
- **Real-time Portfolio Value**: Current total portfolio value in USD
- **Daily P&L**: Daily profit/loss tracking with trend analysis
- **Maximum Drawdown**: Risk management with drawdown percentage monitoring
- **Time Series Visualization**: Historical performance trends

##### Panel 2: Trading Activity  
- **Position Tracking**: Real-time NVDA and MSFT position sizes
- **Trade Frequency**: Hourly trade count with activity patterns
- **Order Flow Analysis**: Buy/sell activity visualization
- **Position Management**: Share count monitoring

##### Panel 3: AI Model Performance
- **Model Confidence Score**: Real-time AI model confidence (0-100%)
- **Prediction Accuracy**: 1-hour rolling prediction accuracy
- **Episode Reward**: Reinforcement learning episode reward tracking
- **Model Health Indicators**: AI performance gauges with thresholds

#### 🚀 Quick Setup

##### 1. Start Monitoring Stack

```bash
# Start Prometheus + Grafana stack
docker compose -f docker-compose.grafana.yml up -d

# Verify services are running
docker ps --filter "name=prometheus\|grafana\|pushgateway"
```

##### 2. Import Dashboard

1. Open Grafana: http://localhost:3000 (admin/admin)
2. Go to **Dashboards** → **Import**
3. Upload `monitoring/grafana_dashboard.json`
4. Select Prometheus as data source

##### 3. Start Data Generation

```bash
# Generate sample data
python monitoring/prometheus_data_generator.py --mode sample

# Start continuous live data (for demo)
python monitoring/prometheus_data_generator.py --mode continuous

# Generate historical data (4 hours)
python monitoring/prometheus_data_generator.py --mode historical --hours 4
```

#### 📈 Metrics Reference

##### Portfolio Metrics
```prometheus
# Portfolio value in USD
trading_portfolio_value_usd

# Cash balance
trading_cash_balance_usd  

# Daily profit/loss
trading_daily_pnl_usd

# Maximum drawdown percentage
trading_max_drawdown_pct
```

##### Position Metrics
```prometheus
# NVDA position (shares)
trading_nvda_position_shares

# MSFT position (shares)  
trading_msft_position_shares

# Current prices
trading_nvda_price_usd
trading_msft_price_usd
```

##### Trading Activity
```prometheus
# Total trade count
trading_total_trades_count
```

##### AI Model Metrics
```prometheus
# Model confidence (0-1)
ai_model_confidence_score

# Prediction accuracy (1-hour rolling)
ai_prediction_accuracy_1h

# Episode reward (reinforcement learning)
ai_episode_reward_mean
```

#### 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Trading    │───▶│   Prometheus     │───▶│    Grafana      │
│   System        │    │   Pushgateway    │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Metrics        │    │   Time Series    │    │  Executive      │
│  Generation     │    │   Database       │    │  Visualization  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

##### Components

1. **AI Trading System**: Generates real trading metrics
2. **Prometheus Pushgateway**: Receives and stores metrics
3. **Prometheus Server**: Time-series database and query engine  
4. **Grafana Dashboard**: Executive visualization and alerting

#### 🔧 Configuration Files

##### `grafana_dashboard.json`
- Complete Grafana dashboard configuration
- 3 panels with realistic trading metrics
- Executive-ready visualization theme
- Auto-refresh every 30 seconds

##### `prometheus.yml`  
- Prometheus server configuration
- Metric collection from pushgateway
- 30-second scrape intervals

##### `prometheus_data_generator.py`
- Realistic trading data simulation
- Market hours volatility modeling
- AI model performance simulation
- Continuous or batch data generation

#### 🎯 Production Deployment

##### Real Integration Points

1. **AI Trading System Integration**:
```python
# In your trading system
from monitoring.prometheus_data_generator import TradingMetricsGenerator

metrics = TradingMetricsGenerator()
metrics.portfolio_value = current_portfolio_value
metrics.nvda_position = current_nvda_shares
metrics.push_metrics_to_prometheus()
```

2. **Alerting Rules**:
```yaml
# Add to Prometheus alerting rules
- alert: HighDrawdown
  expr: trading_max_drawdown_pct > 0.05
  labels:
    severity: critical 
  annotations:
    summary: "Portfolio drawdown > 5%"
```

3. **Historical Data Storage**:
```bash
# Configure Prometheus retention
--storage.tsdb.retention.time=30d
--storage.tsdb.retention.size=10GB
```

#### 🏭 Management Demo Ready

##### Executive Dashboard Features
- **Real-time Performance**: Live P&L and portfolio tracking
- **Risk Monitoring**: Drawdown alerts and position limits
- **AI Health**: Model confidence and prediction accuracy
- **Trading Efficiency**: Activity levels and execution monitoring

##### Demo Talking Points
1. **Performance Transparency**: Real-time portfolio value and P&L
2. **Risk Management**: Automated drawdown monitoring with alerts
3. **AI Reliability**: Model confidence tracking with accuracy metrics  
4. **Operational Insight**: Trading activity and position management
5. **Scalability**: Enterprise-grade monitoring infrastructure

#### 📚 URLs & Access

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus UI**: http://localhost:9090
- **Pushgateway**: http://localhost:9091
- **Dashboard JSON**: `monitoring/grafana_dashboard.json`

**Perfect for CRO/CIO presentations with live data and professional visualization.**

---

## Automated Validation

### 🌙 Nightly Rolling-Window Backtest on Live Polygon Data

#### Overview

This CI job provides **continuous validation** of the trading system using real market data from Polygon.io. It runs automatically every weeknight after market close and validates system performance against live market conditions.

#### 🎯 Purpose

- **Real-world validation**: Tests the system with actual market data, not synthetic data
- **Performance monitoring**: Tracks system reliability over time
- **Risk management**: Ensures the system maintains acceptable risk profiles
- **Management reporting**: Provides ongoing evidence of system stability

#### 🔄 Schedule

**Automatic**: Runs at 1:00 AM EST (6:00 AM UTC) Tuesday-Saturday (covering Monday-Friday market days)

**Manual**: Can be triggered manually with custom parameters via GitHub Actions UI

#### 📊 What It Does

##### 1. Data Validation Phase
- Validates Polygon API credentials
- Checks data availability and quality for recent trading days
- Calculates data completeness scores for NVDA and MSFT
- Fails early if data quality is below 70% threshold

##### 2. Rolling-Window Backtest Phase
- Downloads live minute-level market data from Polygon API
- Stores data in TimescaleDB for backtesting
- Loads the latest production model (201K dual-ticker model)
- Runs rolling-window backtests with configurable window sizes
- Calculates performance metrics (returns, Sharpe ratio, drawdown)

##### 3. Reporting Phase
- Generates performance charts and markdown reports
- Archives results as GitHub Actions artifacts
- Validates performance against predefined gates
- Sends notification of success/failure

#### 🛠️ Configuration

##### Environment Variables (CI)
```yaml
# Required GitHub Secrets
POLYGON_API_KEY: Your Polygon.io API key

# Database (auto-configured in CI)
TEST_DB_HOST: localhost
TEST_DB_PORT: 5432
TEST_DB_NAME: intradayjules
TEST_DB_USER: postgres
TEST_DB_PASSWORD: ci_test_password
```

##### Manual Trigger Parameters
- **`lookback_days`**: Number of days to backtest (default: 5)
- **`test_mode`**: Run with smaller dataset for testing (default: false)

#### 📈 Performance Gates

The CI job enforces these performance requirements:

1. **Success Rate**: ≥70% of rolling windows must complete successfully
2. **Data Quality**: ≥70% data completeness from Polygon API
3. **Return Volatility**: Absolute returns must be <50% (sanity check)
4. **Minimum Windows**: At least 1 successful backtest window required

#### 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Polygon API   │───▶│   GitHub Actions │───▶│   TimescaleDB   │
│  (Live Data)    │    │   Runner (CI)    │    │  (Test Instance)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Market Data    │    │  Rolling Window  │    │   Performance   │
│   Validation    │    │   Backtesting    │    │    Reports      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### 📋 Output Artifacts

Each run generates these artifacts (retained for 30 days):

##### 1. Performance Report (`nightly_report.md`)
```markdown
# Nightly Polygon Backtest Report

**Generated:** 2025-08-02 06:15:23 UTC
**Data Quality:** 89.2%

## Summary
- **Windows Tested:** 3/3
- **Success Rate:** 100.0%
- **Average Returns:** 1.23% ± 0.87%
- **Average Sharpe:** 1.45
- **Win Rate:** 66.7%

## Window Details
| Window | Start Date | End Date | Returns (%) | Sharpe | Steps |
|--------|------------|----------|-------------|--------|-------|
| 1      | 2025-07-29 | 2025-07-31 | 2.10      | 1.67   | 742   |
| 2      | 2025-07-30 | 2025-08-01 | 0.89      | 1.23   | 698   |
| 3      | 2025-07-31 | 2025-08-02 | 0.71      | 1.45   | 734   |
```

##### 2. Performance Chart (`nightly_performance.png`)
- Line charts showing returns and portfolio value progression
- Visual validation of system stability

##### 3. Raw Results (`nightly_results.json`)
```json
{
  "summary": {
    "total_windows": 3,
    "successful_windows": 3,
    "success_rate": 1.0,
    "avg_returns_pct": 1.23,
    "data_quality_score": 89.2
  },
  "windows": [...]
}
```

#### 🚀 Local Testing

You can run the backtest locally for development:

```bash
# Activate virtual environment
source venv/bin/activate

# Set up environment
export POLYGON_API_KEY="your_api_key_here"

# Test mode (quick validation)
python scripts/nightly_polygon_backtest.py --test-mode

# Full backtest (5 days, 3-day windows)
python scripts/nightly_polygon_backtest.py --lookback-days 5 --window-size 3

# Custom configuration
python scripts/nightly_polygon_backtest.py \
  --lookback-days 7 \
  --window-size 4 \
  --output reports/my_backtest.json
```

#### 🔧 Setup Requirements

##### 1. Polygon API Key
1. Sign up at [polygon.io](https://polygon.io)
2. Get your API key from the dashboard
3. Add as GitHub Secret: `POLYGON_API_KEY`

##### 2. GitHub Actions Permissions
Ensure your repository has these permissions enabled:
- Actions: Read and write
- Contents: Read
- Metadata: Read

##### 3. Model Dependencies
The workflow requires these production models:
- Primary: `deploy_models/dual_ticker_prod_20250731_step201k_stable.zip`
- Fallback: `models/phase1_fast_recovery_model.zip`

#### 📊 Monitoring

##### Success Indicators
✅ All jobs complete successfully  
✅ Data quality >70%  
✅ Success rate >70%  
✅ Performance metrics within expected ranges  

##### Failure Scenarios
❌ **Polygon API issues**: Check API key, rate limits, or service status  
❌ **Data quality low**: Market holidays, API changes, or connectivity issues  
❌ **Model loading failures**: Missing model files or corrupted archives  
❌ **Performance degradation**: System changes affecting trading performance  

##### Troubleshooting
1. **API Errors**: Check Polygon.io status page and API key validity
2. **Database Errors**: Verify TimescaleDB schema matches expectations  
3. **Model Errors**: Ensure production models are properly committed
4. **Performance Issues**: Review changes since last successful run

#### 🎯 Management Value

This automated validation provides:

1. **Confidence**: Continuous proof the system works with real market data
2. **Risk Control**: Early detection of performance degradation
3. **Audit Trail**: Complete record of system performance over time
4. **Professional Presentation**: Charts and reports suitable for executive review

The nightly backtest transforms development confidence into **institutional-grade validation** with real market data evidence.

---

## Development Workflows

### 🧪 Development Environment Setup

```bash
# 1. Environment preparation
source venv/bin/activate
source scripts/secure_docker_setup.sh

# 2. Database validation
python test_secure_database_passwords.py

# 3. AI system validation
python test_ai_integration.py

# 4. Run comprehensive tests
pytest tests/ -v
```

### 📊 Model Training Workflow

```bash
# Start training with monitoring
python launch_200k_dual_ticker_training.py

# Monitor progress
python launch_tensorboard.py

# Validate model performance
python validate_implementations_basic.py
```

### 🔄 Code Quality Gates

```bash
# Linting and formatting
black src/ tests/
flake8 src/ tests/ --max-line-length=100

# Security validation
python tests/EXHAUSTIVE_SECRETS_VALIDATION.py

# Feature validation
pytest tests/test_feature_lag_validation.py
```

---

## Production Operations

### 🚀 Live Trading Deployment

```bash
# 1. Start AI inference service
python ai_inference_service.py

# 2. Launch live trading
python live_trader.py

# 3. Monitor with Grafana
docker compose -f docker-compose.grafana.yml up -d
```

### 🔐 Security Management

```bash
# Rotate vault password
python scripts/setup_secure_docker_env.py --rotate

# Validate all credentials
python scripts/validate_credentials.py

# Check security compliance
python tests/test_secure_passwords.py
```

### 📈 Performance Monitoring

```bash
# Real-time metrics
curl http://localhost:9091/metrics

# Historical analysis
python scripts/end_of_day_validation.py

# Executive reports
python create_performance_plots.py
```

---

## Troubleshooting

### Common Issues

#### 🔌 Connection Errors
```bash
# Database connectivity
python test_secure_database_passwords.py

# IBKR gateway status
python test_ibkr_connection.py

# API validations
python test_live_data_integration.py
```

#### 🤖 AI Model Issues
```bash
# Model loading test
python test_ai_integration.py

# Inference validation
python simple_inference_api.py

# Performance validation
python validate_50k_acceptance.py
```

#### 📊 Monitoring Issues
```bash
# Prometheus connectivity
curl http://localhost:9090/-/healthy

# Grafana dashboard import
# Navigate to http://localhost:3000 and import monitoring/grafana_dashboard.json

# Data generation test
python monitoring/prometheus_data_generator.py --mode sample
```

### Support Resources

- **Documentation**: `/docs/` directory
- **Configuration**: `/config/` directory  
- **Examples**: `/examples/` directory
- **Tests**: `/tests/` directory

---

## Management Reporting

### 📊 Executive Dashboard Access

**Grafana Dashboard**: http://localhost:3000 (admin/admin)
- Real-time portfolio performance
- Trading activity monitoring
- AI model health indicators
- Risk management metrics

### 📈 Performance Reports

#### Daily Reports
- End-of-day validation: `scripts/end_of_day_validation.py`
- Performance plots: `create_performance_plots.py`
- Portfolio analysis: `portfolio_loss_detailed_analysis.py`

#### Weekly Reports
- Rolling backtest results: `reports/backtest/`
- Model performance analysis: `logs/run_metadata/`
- Risk assessment: Generated automatically

#### Monthly Reports
- Comprehensive validation: `COMPREHENSIVE_VALIDATION_REPORT.json`
- Performance benchmarking: Available in `/reports/validation/`
- Executive summary: Generated from monitoring data

### 🎯 Key Performance Indicators

1. **Trading Performance**
   - Portfolio returns vs benchmark
   - Sharpe ratio (target: >1.0)
   - Maximum drawdown (limit: <5%)
   - Win rate (target: >55%)

2. **System Reliability**
   - Uptime during trading hours (target: >99%)
   - Data quality scores (target: >90%)
   - Model confidence levels (target: >70%)
   - Error rates (target: <1%)

3. **Risk Management**
   - Position limits compliance (100%)
   - Daily loss limits adherence (100%)
   - Automated stop-loss triggers
   - Real-time risk monitoring

### 📋 Audit Trail

All system activities are logged and auditable:
- Trading decisions and executions
- Model predictions and confidence scores
- Risk control activations
- Performance metrics over time
- System configuration changes

---

## Appendix

### 📚 Additional Resources

- **CLAUDE.md**: Complete project documentation
- **PHASE2B_IMPLEMENTATION_COMPLETE.md**: Live trading system status
- **AI_INTEGRATION_DELIVERY.md**: AI system technical documentation
- **CRITICAL_REVIEWER_SUCCESS_REPORT.md**: Validation evidence

### 🔗 External Dependencies

- **Polygon.io**: Live market data provider
- **Interactive Brokers**: Trading execution platform
- **TimescaleDB**: Market data storage
- **Prometheus/Grafana**: Monitoring infrastructure

### 📞 Support Contacts

For technical support and system maintenance:
- System Architecture: See `PROJECT_STRUCTURE.md`
- Security Protocols: See `SECURE_PASSWORD_MANAGEMENT_IMPLEMENTATION.md`
- Trading Operations: See `IBKR Paper Trading + Live Data Pipeline`

---

**Document Version**: 1.0  
**Last Updated**: August 2, 2025  
**System Status**: ✅ Production Ready - AI Integration Complete