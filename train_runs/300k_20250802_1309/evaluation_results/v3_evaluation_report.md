# 🌟 V3 Model Evaluation Report

**Evaluation Date**: 2025-08-02 19:34:05
**Model**: V3-trained (200K→300K with improved reward system)
**Episodes Evaluated**: 20

## 🎯 V3 Environment Features

The V3 environment includes several improvements:
- **Risk-free baseline**: Prevents cost-blind trading
- **Embedded impact costs**: Kyle lambda model (68bp calibrated)
- **Hold bonus**: Incentivizes doing nothing when no alpha
- **Action change penalties**: Reduces overtrading
- **Ticket costs**: Fixed costs per trade ($25)
- **Downside penalties**: Risk management

## 📊 Performance Summary

- **Average Return**: 2.22%
- **Win Rate**: 100.0%
- **Sharpe Ratio**: 0.000
- **Max Drawdown**: 7.99%
- **Average Trades per Episode**: 14.0

## 🎯 Trading Behavior

### Action Distribution
- **SELL_BOTH**: 140 (24.1%)
- **SELL_NVDA_HOLD_MSFT**: 20 (3.4%)
- **SELL_NVDA_BUY_MSFT**: 0 (0.0%)
- **HOLD_NVDA_SELL_MSFT**: 0 (0.0%)
- **HOLD_BOTH**: 300 (51.7%)
- **HOLD_NVDA_BUY_MSFT**: 0 (0.0%)
- **BUY_NVDA_SELL_MSFT**: 120 (20.7%)
- **BUY_NVDA_HOLD_MSFT**: 0 (0.0%)
- **BUY_BOTH**: 0 (0.0%)

### Trading Frequency
- **Total Trades**: 280
- **Trade Frequency**: 0.483

## 📈 Episode Results

| Episode | Return | Reward | Trades | Steps |
|---------|--------|--------|-----------|-------|
| 1 | 2.22% | -8812.1244 | 14 | 29 |
| 2 | 2.22% | -8812.1244 | 14 | 29 |
| 3 | 2.22% | -8812.1244 | 14 | 29 |
| 4 | 2.22% | -8812.1244 | 14 | 29 |
| 5 | 2.22% | -8812.1244 | 14 | 29 |
| 6 | 2.22% | -8812.1244 | 14 | 29 |
| 7 | 2.22% | -8812.1244 | 14 | 29 |
| 8 | 2.22% | -8812.1244 | 14 | 29 |
| 9 | 2.22% | -8812.1244 | 14 | 29 |
| 10 | 2.22% | -8812.1244 | 14 | 29 |
| 11 | 2.22% | -8812.1244 | 14 | 29 |
| 12 | 2.22% | -8812.1244 | 14 | 29 |
| 13 | 2.22% | -8812.1244 | 14 | 29 |
| 14 | 2.22% | -8812.1244 | 14 | 29 |
| 15 | 2.22% | -8812.1244 | 14 | 29 |
| 16 | 2.22% | -8812.1244 | 14 | 29 |
| 17 | 2.22% | -8812.1244 | 14 | 29 |
| 18 | 2.22% | -8812.1244 | 14 | 29 |
| 19 | 2.22% | -8812.1244 | 14 | 29 |
| 20 | 2.22% | -8812.1244 | 14 | 29 |
