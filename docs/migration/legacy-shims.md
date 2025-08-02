# 🔄 Legacy Shims Migration Guide

## ⚠️ **URGENT: REMOVAL DEADLINE 2026-06-30**

Legacy compatibility shims will be **PERMANENTLY REMOVED** on **June 30, 2026**. 
All downstream code must migrate to the new API before this date.

---

## 📋 **Migration Checklist**

### **1. TickVsMinuteAlphaStudy**

| ❌ **DEPRECATED** | ✅ **NEW API** | **Migration** |
|------------------|----------------|---------------|
| `resample_to_timeframe(data, '5T')` | `aggregate_to_bars(tick_data, '5T')` | Change method name + ensure tick data format |
| `calculate_strategy_metrics(returns, '1T')` | `calculate_performance_metrics(bars_data, '1T')` | Pass DataFrame with required columns |

**Example Migration:**
```python
# ❌ OLD (deprecated)
study = TickVsMinuteAlphaStudy()
resampled = study.resample_to_timeframe(ohlc_data, '5T')
metrics = study.calculate_strategy_metrics(returns, '1T')

# ✅ NEW (recommended)
study = TickVsMinuteAlphaStudy()
bars = study.aggregate_to_bars(tick_data, '5T')
metrics = study.calculate_performance_metrics(bars_data, '1T')
```

### **2. FilteringAblationStudy**

| ❌ **DEPRECATED** | ✅ **NEW API** | **Migration** |
|------------------|----------------|---------------|
| `get_earnings_dates(symbols, start, end)` | `_generate_earnings_dates(symbols, start, end)` | Internal method - use `run_comprehensive_ablation()` |
| `apply_earnings_filter(data, config)` | `run_comprehensive_ablation()` | Integrated filtering in comprehensive study |
| `calculate_performance_metrics(returns, config)` | `calculate_strategy_performance(data, config)` | Pass DataFrame with required columns |
| `generate_lockbox_hash(data)` | Integrated in `run_comprehensive_ablation()` | Hash generation is automatic |

**Example Migration:**
```python
# ❌ OLD (deprecated)
study = FilteringAblationStudy()
earnings = study.get_earnings_dates(['NVDA'], start, end)
filtered = study.apply_earnings_filter(data, earnings)
metrics = study.calculate_performance_metrics(returns, "config")
hash_val = study.generate_lockbox_hash(data)

# ✅ NEW (recommended)
study = FilteringAblationStudy()
results = study.run_comprehensive_ablation(
    symbols=['NVDA'],
    start_date=start,
    end_date=end,
    configs=['earnings_included', 'earnings_excluded']
)
# All metrics, filtering, and hashing handled automatically
```

---

## 🚨 **Breaking Changes**

### **Data Format Requirements**
- **Old**: Accepts Series or any DataFrame
- **New**: Requires specific column names (`close`, `returns`, `signal`, etc.)

### **Return Types**
- **Old**: Returns simple dictionaries or Series
- **New**: Returns structured dataclasses (`AlphaStudyResult`, `AblationResult`)

### **Configuration**
- **Old**: String-based config names
- **New**: Dictionary-based configuration objects

---

## 🔧 **Automated Migration Tools**

### **Migration Script**
```bash
# Run automated migration (updates imports and method calls)
python scripts/migrate_legacy_shims.py --path src/ --dry-run
python scripts/migrate_legacy_shims.py --path src/ --apply
```

### **Validation Script**
```bash
# Check for remaining deprecated usage
python scripts/check_deprecated_usage.py --path src/
```

---

## 📅 **Timeline**

| **Date** | **Action** | **Impact** |
|----------|------------|------------|
| **2025-08-02** | Deprecation warnings added | Runtime warnings in logs |
| **2025-10-01** | Import warnings | Warnings on `import legacy_shims` |
| **2026-01-01** | Final migration reminder | Escalated warnings |
| **2026-06-30** | **REMOVAL** | ❌ **Code will break** |

---

## 🆘 **Support**

- **Migration Issues**: Create ticket with label `legacy-migration`
- **API Questions**: See `docs/api/` directory
- **Emergency Support**: Contact platform team before 2026-06-30

---

*Last Updated: 2025-08-02*  
*Next Review: 2025-10-01*