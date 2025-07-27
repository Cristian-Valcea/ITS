# 🔄 3-Month Rolling Window Walk-Forward Backtest System

## Overview

This document describes the comprehensive 3-month rolling window walk-forward backtest system implemented for IntradayJules to verify model robustness across different time periods and market conditions.

## 🎯 Key Features

### 1. Walk-Forward Analysis
- **3-month training windows** with **1-month evaluation periods**
- **1-month step size** for progressive walk-forward testing
- **Overlapping windows** to ensure comprehensive coverage
- **Minimum 20 trading days** per evaluation window

### 2. Comprehensive Metrics
- **Performance Metrics**: Return, Sharpe ratio, drawdown, win rate
- **Risk Metrics**: Volatility, Ulcer Index, Calmar ratio, Sortino ratio
- **Trading Metrics**: Number of trades, turnover, position sizing
- **Market Regime Analysis**: Trend classification and volatility regimes

### 3. Robustness Analysis
- **Consistency Scoring**: Statistical measures of performance stability
- **Market Adaptability**: Performance across different market conditions
- **Time Pattern Analysis**: Seasonal and temporal performance patterns
- **Deployment Recommendations**: Automated assessment for production readiness

## 🏗️ Architecture

### Core Components

#### 1. RollingWindowBacktest
```python
class RollingWindowBacktest:
    """
    Main class implementing rolling-window walk-forward backtesting.
    
    Process:
    1. Split historical data into overlapping 3-month training windows
    2. Use each window to evaluate model performance
    3. Evaluate on subsequent 1-month period
    4. Walk forward by 1 month and repeat
    5. Aggregate results for robustness analysis
    """
```

#### 2. WindowResult
```python
@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    eval_start: str
    eval_end: str
    
    # Performance metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    # ... additional metrics
```

### Key Methods

#### Window Generation
```python
def _generate_time_windows(self, start_date: str, end_date: str) -> List[Dict[str, str]]:
    """Generate overlapping time windows for walk-forward analysis."""
```

#### Performance Evaluation
```python
def _evaluate_window(self, model, window, symbol, window_id) -> Optional[WindowResult]:
    """Evaluate model performance on a single time window."""
```

#### Robustness Analysis
```python
def _calculate_robustness_stats(self) -> Dict[str, Any]:
    """Calculate robustness statistics across all windows."""
```

## 📊 Metrics and Analysis

### Performance Metrics (Per Window)

#### Core Performance
- **Total Return %**: Portfolio return for the evaluation period
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown %**: Maximum peak-to-trough decline
- **Win Rate %**: Percentage of profitable trades

#### Risk Metrics
- **Volatility %**: Annualized return volatility
- **Ulcer Index**: Downside risk measure
- **Calmar Ratio**: Return/Max Drawdown ratio
- **Sortino Ratio**: Return/Downside deviation ratio

#### Trading Activity
- **Number of Trades**: Total trades executed
- **Turnover Ratio**: Trading volume relative to portfolio value
- **Average Position Size**: Mean position size in shares
- **Trade Frequency**: Trades per day

#### Market Context
- **Market Volatility**: Realized market volatility during period
- **Market Trend**: Classified as Up (+1), Sideways (0), or Down (-1)
- **Market Return %**: Benchmark return for the period

### Robustness Statistics (Across All Windows)

#### Consistency Scores
```python
robustness_scores = {
    'return_consistency': float,      # 1 / (1 + return_std/return_mean)
    'sharpe_consistency': float,      # 1 / (1 + sharpe_std)
    'drawdown_control': float,        # 1 - max_drawdown/20%
    'win_rate_stability': float,      # 1 / (1 + win_rate_std)
    'overall_robustness': float       # Weighted average of above
}
```

#### Market Regime Analysis
```python
regime_analysis = {
    'up_market': {
        'count': int,
        'avg_return': float,
        'avg_sharpe': float,
        'win_rate': float
    },
    'down_market': {...},
    'sideways_market': {...},
    'low_vol_regime': {...},
    'high_vol_regime': {...}
}
```

#### Deployment Recommendations
- **DEPLOY_FULL_CAPITAL**: Excellent robustness (>80% score, >80% profitable windows)
- **DEPLOY_REDUCED_CAPITAL**: Good robustness (>60% score, >70% profitable windows)
- **PAPER_TRADE_FIRST**: Fair robustness (>40% score, >60% profitable windows)
- **REQUIRES_IMPROVEMENT**: Poor robustness (below thresholds)

## 🚀 Usage

### 1. Command Line Interface

```bash
# Run rolling backtest with default settings
python scripts/run_rolling_backtest.py \
    --model_path models/best_model.zip \
    --start_date 2023-01-01 \
    --end_date 2024-01-01

# Custom window configuration
python scripts/run_rolling_backtest.py \
    --model_path models/best_model.zip \
    --start_date 2023-01-01 \
    --end_date 2024-01-01 \
    --training_window_months 6 \
    --evaluation_window_months 2 \
    --step_size_months 2
```

### 2. Programmatic Interface

```python
from src.agents.evaluator_agent import EvaluatorAgent

# Initialize evaluator
evaluator = EvaluatorAgent(config)

# Run rolling window backtest
results = evaluator.run_rolling_window_backtest(
    model_path="models/best_model.zip",
    data_start_date="2023-01-01",
    data_end_date="2024-01-01",
    symbol="SPY",
    training_window_months=3,
    evaluation_window_months=1,
    step_size_months=1
)

# Access results
robustness_stats = results['robustness_stats']
window_results = results['window_results']
recommendation = robustness_stats['executive_summary']['overall_assessment']['recommendation']
```

### 3. Integration with Training Pipeline

```python
# After training completion, run robustness validation
if config.get('evaluation', {}).get('rolling_backtest', {}).get('enabled', False):
    evaluator_agent = EvaluatorAgent(config)
    
    backtest_results = evaluator_agent.run_rolling_window_backtest(
        model_path=best_model_path,
        data_start_date="2023-01-01",
        data_end_date="2024-01-01"
    )
    
    # Make deployment decision based on results
    recommendation = backtest_results['robustness_stats']['executive_summary']['overall_assessment']['recommendation']
    
    if recommendation == "DEPLOY_FULL_CAPITAL":
        logger.info("✅ Model ready for full deployment")
    elif recommendation == "DEPLOY_REDUCED_CAPITAL":
        logger.info("⚠️ Model ready for reduced capital deployment")
    elif recommendation == "PAPER_TRADE_FIRST":
        logger.info("📝 Recommend paper trading before live deployment")
    else:
        logger.info("❌ Model requires improvement before deployment")
```

## 📈 Configuration

### YAML Configuration

```yaml
evaluation:
  # Rolling Window Walk-Forward Backtest Configuration
  rolling_backtest:
    enabled: true                    # Enable rolling window backtest
    training_window_months: 3        # 3-month training windows
    evaluation_window_months: 1      # 1-month evaluation periods
    step_size_months: 1              # Walk forward by 1 month
    min_trading_days: 20             # Minimum trading days per window
    enable_regime_analysis: true     # Analyze performance across market regimes
    save_detailed_results: true      # Save detailed CSV results
    
    # Robustness thresholds for deployment recommendations
    deployment_thresholds:
      excellent_robustness: 0.8      # Threshold for full capital deployment
      good_robustness: 0.6           # Threshold for reduced capital deployment
      fair_robustness: 0.4           # Threshold for paper trading first
      profitable_window_pct: 70      # Minimum % of profitable windows
      max_acceptable_drawdown: 15    # Maximum acceptable drawdown %
```

### Programmatic Configuration

```python
config = create_rolling_backtest_config(base_config)

# Override specific settings
config['rolling_backtest'].update({
    'training_window_months': 6,     # Longer training windows
    'evaluation_window_months': 2,   # Longer evaluation periods
    'step_size_months': 2,           # Larger step size
    'min_trading_days': 30           # More stringent minimum
})
```

## 📊 Output and Reports

### 1. Console Output

```
🚀 Starting Rolling Window Walk-Forward Backtest
============================================================
📊 Processing Window 1/12
   Training: 2023-01-01 → 2023-04-01
   Evaluation: 2023-04-01 → 2023-05-01
   📈 Return: +2.34% | Sharpe: 1.45 | DD: 1.2% | Trades: 23 | Win Rate: 65.2%
   🌍 Market: +3.1% | Vol: 18.5% | Regime: Up

...

🎉 Rolling Window Backtest Complete!
📊 Processed 12 windows successfully
============================================================
📊 ROLLING WINDOW BACKTEST SUMMARY
============================================================
Total Windows Processed: 12
Average Return: 2.15%
Average Sharpe Ratio: 1.23
Worst Drawdown: 3.45%
Profitable Windows: 10/12 (83.3%)
Consistency Rating: GOOD
Overall Robustness Score: 0.742

🚀 DEPLOYMENT RECOMMENDATION: DEPLOY_REDUCED_CAPITAL
   ⚠️ Model shows good robustness - deploy with reduced position sizing

📄 Detailed Report: reports/rolling_backtest/rolling_backtest_report_20241201_143022.json
============================================================
```

### 2. JSON Report Structure

```json
{
  "metadata": {
    "generated_at": "2024-12-01T14:30:22",
    "training_window_months": 3,
    "evaluation_window_months": 1,
    "step_size_months": 1,
    "total_windows_processed": 12
  },
  "window_results": [
    {
      "window_id": 1,
      "train_start": "2023-01-01",
      "train_end": "2023-04-01",
      "eval_start": "2023-04-01",
      "eval_end": "2023-05-01",
      "total_return_pct": 2.34,
      "sharpe_ratio": 1.45,
      "max_drawdown_pct": 1.2,
      "num_trades": 23,
      "win_rate_pct": 65.2,
      "market_trend": 1,
      "market_return_pct": 3.1,
      "market_volatility": 18.5
    }
  ],
  "robustness_analysis": {
    "performance_stats": {
      "total_return_pct": {
        "mean": 2.15,
        "std": 1.23,
        "min": -0.45,
        "max": 4.67,
        "consistency_score": 0.635
      }
    },
    "robustness_scores": {
      "return_consistency": 0.635,
      "sharpe_consistency": 0.712,
      "drawdown_control": 0.827,
      "win_rate_stability": 0.789,
      "overall_robustness": 0.742
    },
    "regime_analysis": {
      "up_market": {
        "count": 5,
        "avg_return": 3.24,
        "avg_sharpe": 1.56,
        "win_rate": 80.0
      },
      "down_market": {
        "count": 3,
        "avg_return": 0.45,
        "avg_sharpe": 0.67,
        "win_rate": 66.7
      },
      "sideways_market": {
        "count": 4,
        "avg_return": 1.89,
        "avg_sharpe": 1.12,
        "win_rate": 75.0
      }
    }
  },
  "executive_summary": {
    "overall_assessment": {
      "consistency_rating": "GOOD",
      "robustness_score": 0.742,
      "profitable_windows_pct": 83.3,
      "recommendation": "DEPLOY_REDUCED_CAPITAL"
    },
    "key_findings": {
      "average_return": "2.15%",
      "average_sharpe": "1.23",
      "worst_drawdown": "3.45%",
      "total_windows_tested": 12
    },
    "risk_assessment": {
      "drawdown_control": 0.827,
      "return_consistency": 0.635,
      "market_adaptability": 0.756
    }
  }
}
```

### 3. CSV Export

Detailed results are also exported to CSV format for easy analysis:

```csv
window_id,train_start,train_end,eval_start,eval_end,total_return_pct,sharpe_ratio,max_drawdown_pct,num_trades,win_rate_pct,market_trend,market_return_pct,market_volatility
1,2023-01-01,2023-04-01,2023-04-01,2023-05-01,2.34,1.45,1.2,23,65.2,1,3.1,18.5
2,2023-02-01,2023-05-01,2023-05-01,2023-06-01,1.87,1.23,2.1,19,58.3,0,0.8,22.1
...
```

## 🧪 Testing

### Unit Tests

```bash
# Run rolling window backtest tests
cd c:/Projects/IntradayJules
python tests/test_rolling_window_backtest.py

# Expected output:
🧪 Running Rolling Window Backtest Tests
==================================================
1️⃣ Testing Rolling Window Backtest System...
   ✅ Rolling window backtest system tests passed
2️⃣ Testing Robustness Analysis...
   ✅ Robustness analysis tests passed
3️⃣ Testing Configuration...
   ✅ Configuration tests passed
4️⃣ Testing Data Structures...
   ✅ Data structure tests passed

🎉 All Rolling Window Backtest tests passed!
```

### Test Coverage

- **Window Generation**: Time window creation and validation
- **Market Regime Analysis**: Trend and volatility classification
- **Robustness Calculations**: Statistical consistency measures
- **Configuration Management**: Setup and parameter validation
- **Data Structures**: WindowResult serialization and handling

## 🎯 Key Benefits

### 1. **Robustness Validation**
- Tests model performance across multiple time periods
- Identifies performance degradation over time
- Validates consistency across different market conditions

### 2. **Market Regime Analysis**
- Evaluates performance in up, down, and sideways markets
- Assesses adaptability to different volatility regimes
- Provides insights into model strengths and weaknesses

### 3. **Deployment Readiness**
- Automated assessment of production readiness
- Clear recommendations for capital allocation
- Risk-adjusted deployment strategies

### 4. **Performance Insights**
- Identifies seasonal patterns in performance
- Highlights potential overfitting issues
- Provides comprehensive risk assessment

## 📈 Interpretation Guide

### Robustness Scores

#### Overall Robustness Score
- **0.8 - 1.0**: Excellent - Ready for full deployment
- **0.6 - 0.8**: Good - Deploy with reduced capital
- **0.4 - 0.6**: Fair - Paper trade first
- **0.0 - 0.4**: Poor - Requires improvement

#### Component Scores
- **Return Consistency**: Lower variance in returns is better
- **Sharpe Consistency**: Stable risk-adjusted returns
- **Drawdown Control**: Maximum drawdown management
- **Win Rate Stability**: Consistent profitability

### Market Regime Performance

#### Ideal Characteristics
- **Positive returns** in up markets (>2%)
- **Capital preservation** in down markets (>-1%)
- **Steady performance** in sideways markets (>1%)
- **Adaptability** across volatility regimes

#### Warning Signs
- **Large losses** in any single regime
- **Inconsistent performance** across regimes
- **High volatility** in returns
- **Declining performance** over time

## ✅ Implementation Status

### ✅ Completed Features

1. **🔄 Rolling Window System**
   - ✅ 3-month training windows
   - ✅ 1-month evaluation periods
   - ✅ Walk-forward progression
   - ✅ Configurable window sizes

2. **📊 Comprehensive Metrics**
   - ✅ Performance metrics (return, Sharpe, drawdown)
   - ✅ Risk metrics (volatility, Ulcer Index, Calmar)
   - ✅ Trading metrics (trades, turnover, positions)
   - ✅ Market regime indicators

3. **🎯 Robustness Analysis**
   - ✅ Consistency scoring system
   - ✅ Market regime analysis
   - ✅ Time pattern analysis
   - ✅ Deployment recommendations

4. **🛠️ Integration & Tools**
   - ✅ Command line interface
   - ✅ Programmatic API
   - ✅ EvaluatorAgent integration
   - ✅ Configuration management

5. **📄 Reporting & Output**
   - ✅ JSON report generation
   - ✅ CSV data export
   - ✅ Console summary output
   - ✅ Executive summary

6. **🧪 Testing Framework**
   - ✅ Comprehensive unit tests
   - ✅ Mock data generation
   - ✅ Robustness validation
   - ✅ Integration testing

### 🎉 Production Ready

The 3-month rolling window walk-forward backtest system is **fully implemented** and **production-ready** with:

- **Comprehensive robustness validation** across multiple time periods and market conditions
- **Automated deployment recommendations** based on statistical analysis
- **Market regime analysis** for understanding model adaptability
- **Detailed reporting and visualization** for decision-making support
- **Seamless integration** with existing training and evaluation pipeline
- **Robust testing framework** ensuring reliability and accuracy

The system provides **unprecedented insight** into model robustness and enables **data-driven deployment decisions** based on comprehensive historical performance analysis.

---

**Status**: ✅ **COMPLETE** - 3-month rolling window walk-forward backtest fully implemented and tested
**Next Steps**: Run backtest on trained models before deployment to validate robustness