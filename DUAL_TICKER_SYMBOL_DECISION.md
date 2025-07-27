# Dual-Ticker Symbol Decision: NVDA + MSFT (Canonical)

**Date**: July 27, 2025  
**Decision**: ✅ **NVDA + MSFT** (Final and Canonical)  
**Status**: All components updated, TimescaleDB schema ready

---

## 🎯 **Executive Decision**

**CANONICAL DUAL-TICKER PAIR**: **NVDA + MSFT**

### **Rationale**
1. **Proven Foundation**: NVDA model already trained (50K timesteps, episode reward 4.78)
2. **Transfer Learning**: Maintains continuity from single-ticker NVDA success
3. **User Specification**: Explicitly requested NVDA+MSFT in implementation feedback
4. **Market Logic**: Tech sector diversification (GPU vs Software)
5. **Liquidity**: Both highly liquid with minute-level data availability

---

## 📋 **Implementation Status**

### **✅ COMPLETED: Core Components**
- `src/gym_env/dual_ticker_trading_env.py` - Uses NVDA+MSFT ✅
- `src/gym_env/dual_ticker_data_adapter.py` - Default symbols ['NVDA', 'MSFT'] ✅
- `src/training/dual_ticker_model_adapter.py` - Transfer from NVDA model ✅
- `tests/gym_env/test_dual_ticker_env_enhanced.py` - All tests use NVDA+MSFT ✅
- `.github/workflows/dual_ticker_ci.yml` - CI validates NVDA+MSFT ✅

### **✅ COMPLETED: Documentation**
- `CLAUDE.md` - Updated to NVDA+MSFT throughout ✅
- `DUAL_TICKER_IMPLEMENTATION_SUMMARY.md` - NVDA+MSFT specified ✅
- Action constants use `ACTION_*_NVDA_*_MSFT` pattern ✅

### **📋 TODO: Schema Updates**
- TimescaleDB examples in docs still show AAPL examples
- Configuration files may have AAPL references
- Secrets management examples may reference AAPL keys

---

## 🗃️ **TimescaleDB Schema (Canonical)**

```sql
-- CANONICAL SCHEMA for NVDA + MSFT
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,  -- 'NVDA' or 'MSFT'
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    rsi DOUBLE PRECISION,
    ema_short DOUBLE PRECISION,
    ema_long DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    momentum DOUBLE PRECISION,
    time_sin DOUBLE PRECISION,
    time_cos DOUBLE PRECISION,
    volume_sma DOUBLE PRECISION,
    price_change DOUBLE PRECISION,
    returns DOUBLE PRECISION,
    -- Fixed PK order per reviewer feedback
    PRIMARY KEY (symbol, timestamp)  -- Changed from (timestamp, symbol)
);

-- Create hypertable on timestamp
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time_desc 
ON market_data (symbol, timestamp DESC);
```

---

## 🔧 **Configuration Examples**

### **Data Loading**
```python
# CANONICAL: Load NVDA + MSFT data
symbols = ['NVDA', 'MSFT']
adapter = DualTickerDataAdapter(db_config)
data = adapter.load_training_data('2024-01-01', '2024-12-31', symbols=symbols)
```

### **Environment Creation**
```python
# CANONICAL: Create NVDA + MSFT environment
env = DualTickerTradingEnv(
    nvda_data=data['nvda_features'],
    msft_data=data['msft_features'],
    nvda_prices=data['nvda_prices'],
    msft_prices=data['msft_prices'],
    trading_days=data['trading_days']
)
```

### **Secrets Management**
```bash
# CANONICAL: Store credentials for NVDA + MSFT data feeds
python cloud_secrets_cli.py set nvda-data-api-key "your-nvda-key"
python cloud_secrets_cli.py set msft-data-api-key "your-msft-key"
```

---

## 🚨 **Migration from AAPL References**

### **Files to Update** (if AAPL found)
```bash
# Search for remaining AAPL references
grep -r "AAPL" config/ docs/ scripts/ --exclude-dir=.git

# Common patterns to replace:
# AAPL → NVDA
# ["AAPL", "MSFT"] → ["NVDA", "MSFT"]
# aapl-api-key → nvda-api-key
# SELL_AAPL_HOLD_MSFT → SELL_NVDA_HOLD_MSFT
```

### **Database Migration** (if AAPL data exists)
```sql
-- If team already has AAPL data, migrate or keep both
-- Option 1: Keep both (recommended)
-- No migration needed, just ensure NVDA data is available

-- Option 2: Rename AAPL to NVDA (if AAPL was placeholder)
-- UPDATE market_data SET symbol = 'NVDA' WHERE symbol = 'AAPL';
```

---

## 📊 **Impact Assessment**

### **✅ ZERO RISK** - Core Implementation
- All implementation already uses NVDA+MSFT
- No breaking changes to existing code
- Transfer learning maintains NVDA foundation
- Tests validate correct symbol usage

### **⚠️ LOW RISK** - Documentation Cleanup
- Some docs may still reference AAPL+MSFT
- Configuration examples may need updates
- Secrets naming conventions may vary

### **📋 ACTION REQUIRED** - Team Coordination
- Confirm data pipeline team uses NVDA feeds (not AAPL)
- Update any existing database schemas if AAPL was used
- Verify Interactive Brokers symbols configured correctly

---

## ✅ **Verification Commands**

```bash
# Verify no AAPL in core implementation
grep -r "AAPL" src/gym_env/ src/training/ tests/gym_env/
# Should return: No matches

# Verify NVDA+MSFT usage in core files
grep -r "NVDA.*MSFT\|MSFT.*NVDA" src/gym_env/
# Should return multiple matches confirming NVDA+MSFT

# Test environment creation with correct symbols
python -c "
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
adapter = DualTickerDataAdapter({'mock_data': True})
data = adapter.load_training_data('2024-01-01', '2024-03-31')
print(f'Symbols: NVDA features {data[\"nvda_features\"].shape}, MSFT features {data[\"msft_features\"].shape}')
"
```

---

## 🎯 **Conclusion**

**NVDA + MSFT is the canonical dual-ticker pair** for all current and future development.

**Implementation Status**: ✅ **COMPLETE** - All core components use NVDA+MSFT  
**Risk Level**: ✅ **MINIMAL** - Only documentation cleanup needed  
**Team Action**: Verify data pipelines target NVDA (not AAPL) feeds  

This decision is **final and binding** for Week 3-5 development and the Week 8 management demonstration.

**Decision Date**: July 27, 2025  
**Status**: ✅ **CANONICAL AND IMPLEMENTED**