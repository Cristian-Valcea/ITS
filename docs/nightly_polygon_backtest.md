# Nightly Rolling-Window Backtest on Live Polygon Data

## Overview

This CI job provides **continuous validation** of the trading system using real market data from Polygon.io. It runs automatically every weeknight after market close and validates system performance against live market conditions.

## 🎯 Purpose

- **Real-world validation**: Tests the system with actual market data, not synthetic data
- **Performance monitoring**: Tracks system reliability over time
- **Risk management**: Ensures the system maintains acceptable risk profiles
- **Management reporting**: Provides ongoing evidence of system stability

## 🔄 Schedule

**Automatic**: Runs at 1:00 AM EST (6:00 AM UTC) Tuesday-Saturday (covering Monday-Friday market days)

**Manual**: Can be triggered manually with custom parameters via GitHub Actions UI

## 📊 What It Does

### 1. Data Validation Phase
- Validates Polygon API credentials
- Checks data availability and quality for recent trading days
- Calculates data completeness scores for NVDA and MSFT
- Fails early if data quality is below 70% threshold

### 2. Rolling-Window Backtest Phase
- Downloads live minute-level market data from Polygon API
- Stores data in TimescaleDB for backtesting
- Loads the latest production model (201K dual-ticker model)
- Runs rolling-window backtests with configurable window sizes
- Calculates performance metrics (returns, Sharpe ratio, drawdown)

### 3. Reporting Phase
- Generates performance charts and markdown reports
- Archives results as GitHub Actions artifacts
- Validates performance against predefined gates
- Sends notification of success/failure

## 🛠️ Configuration

### Environment Variables (CI)
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

### Manual Trigger Parameters
- **`lookback_days`**: Number of days to backtest (default: 5)
- **`test_mode`**: Run with smaller dataset for testing (default: false)

## 📈 Performance Gates

The CI job enforces these performance requirements:

1. **Success Rate**: ≥70% of rolling windows must complete successfully
2. **Data Quality**: ≥70% data completeness from Polygon API
3. **Return Volatility**: Absolute returns must be <50% (sanity check)
4. **Minimum Windows**: At least 1 successful backtest window required

## 🏗️ Architecture

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

## 📋 Output Artifacts

Each run generates these artifacts (retained for 30 days):

### 1. Performance Report (`nightly_report.md`)
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

### 2. Performance Chart (`nightly_performance.png`)
- Line charts showing returns and portfolio value progression
- Visual validation of system stability

### 3. Raw Results (`nightly_results.json`)
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

## 🚀 Local Testing

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

## 🔧 Setup Requirements

### 1. Polygon API Key
1. Sign up at [polygon.io](https://polygon.io)
2. Get your API key from the dashboard
3. Add as GitHub Secret: `POLYGON_API_KEY`

### 2. GitHub Actions Permissions
Ensure your repository has these permissions enabled:
- Actions: Read and write
- Contents: Read
- Metadata: Read

### 3. Model Dependencies
The workflow requires these production models:
- Primary: `deploy_models/dual_ticker_prod_20250731_step201k_stable.zip`
- Fallback: `models/phase1_fast_recovery_model.zip`

## 📊 Monitoring

### Success Indicators
✅ All jobs complete successfully  
✅ Data quality >70%  
✅ Success rate >70%  
✅ Performance metrics within expected ranges  

### Failure Scenarios
❌ **Polygon API issues**: Check API key, rate limits, or service status  
❌ **Data quality low**: Market holidays, API changes, or connectivity issues  
❌ **Model loading failures**: Missing model files or corrupted archives  
❌ **Performance degradation**: System changes affecting trading performance  

### Troubleshooting
1. **API Errors**: Check Polygon.io status page and API key validity
2. **Database Errors**: Verify TimescaleDB schema matches expectations  
3. **Model Errors**: Ensure production models are properly committed
4. **Performance Issues**: Review changes since last successful run

## 🎯 Management Value

This automated validation provides:

1. **Confidence**: Continuous proof the system works with real market data
2. **Risk Control**: Early detection of performance degradation
3. **Audit Trail**: Complete record of system performance over time
4. **Professional Presentation**: Charts and reports suitable for executive review

The nightly backtest transforms development confidence into **institutional-grade validation** with real market data evidence.