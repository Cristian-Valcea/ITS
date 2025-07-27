# Baseline Model Integrity Report

**Generated**: 2025-07-21  
**Model**: RECURRENTPPO_2025-07-20_09-11-12  
**Status**: ✅ VERIFIED

## Model Verification

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| **Model File** | baseline_model.zip | baseline_model.zip | ✅ EXISTS |
| **SHA256** | DABFD70C21315CC8B5F3D09445FE850E6F7B699D1F658D8466102330CACF44C3 | DABFD70C21315CC8B5F3D09445FE850E6F7B699D1F658D8466102330CACF44C3 | ✅ MATCH |
| **File Size** | 401,478 bytes | 401,478 bytes | ✅ MATCH |
| **Creation Date** | 2025-07-20 11:07:58 | 2025-07-20 11:07:58 | ✅ MATCH |

## Performance Metrics Validation

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| **Total Return** | 1.2778% | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Sharpe Ratio** | -2.2288 | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Max Drawdown** | 2.6427% | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Turnover Ratio** | 5.9047 | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Number of Trades** | 307 | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Win Rate** | 34.25% | eval_NVDA_RECURRENTPPO_2025-07-20_09-11-12_20250720_110830_summary.txt | ✅ VERIFIED |
| **Training Timesteps** | 150,000 | Model metadata | ✅ VERIFIED |

## Model Loading Test

```python
# Model successfully loads without errors
# Policy file: policy.pt (verified)
# TorchScript file: RECURRENTPPO_2025-07-20_09-11-12_torchscript.pt (verified)
# Metadata file: RECURRENTPPO_2025-07-20_09-11-12_metadata.json (verified)
```

## Environment Integrity

| Component | Status | Details |
|-----------|--------|---------|
| **Requirements Lock** | ✅ COMPLETE | requirements-baseline.txt created |
| **Pip Freeze** | ✅ COMPLETE | pip-freeze-baseline.txt created |
| **Config Checksum** | ✅ VERIFIED | SHA256 matches baseline_config.yaml |

## Institutional Assessment

**Overall Status**: STABLE_BUT_SUBOPTIMAL

### ✅ Strengths
- Model integrity verified with SHA256 checksums
- Training completed successfully (150,000 timesteps)
- Reasonable drawdown control (2.64% < 5% threshold)
- Complete infrastructure working (evaluation, model saving)
- Environment fully reproducible

### ❌ Critical Issues
- Negative Sharpe ratio (-2.2288) indicates poor risk-adjusted returns
- High turnover (5.9047x daily) creates excessive transaction costs
- Low win rate (34.25%) suggests weak predictive signal

### 🎯 Compliance Status
- **Sharpe Requirement**: ❌ FAIL (target ≥1.0, actual -2.2288)
- **Drawdown Requirement**: ✅ PASS (target ≤5%, actual 2.6427%)
- **Turnover Requirement**: ❌ FAIL (target ≤2x, actual 5.9047x)
- **Win Rate Requirement**: ❌ FAIL (target ≥50%, actual 34.25%)

## Conclusion

✅ **Baseline model integrity is VERIFIED and COMPLETE**  
✅ **All artifacts properly preserved and checksummed**  
✅ **Environment is fully reproducible**  
⚠️ **Model performance requires improvement in Phase 1+**

**Ready for institutional enhancement phases.**