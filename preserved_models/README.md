# Preserved Models Archive

## Phase 1A Success Model

**File**: `phase1a_5k_success_model.zip`
**Source**: Phase 1A Freeze-Early Test, 5K checkpoint
**Performance**: Mean reward 26.59, Episode length 259.7 steps
**Date**: 2025-08-04
**Status**: ✅ **PROVEN SUCCESSFUL MODEL**

### Key Metrics:
- **Mean Reward**: 26.586512 (well above 0.5 threshold)
- **Episode Length**: 259.7 steps (well above 40 step minimum)
- **Evaluation Period**: Feb 2024 (out-of-sample)
- **Episodes Completed**: 19
- **Max Reward Achieved**: 105.23
- **Training Steps**: 5,000

### Why This Model is Important:
1. **Proves early learning is genuine** - not statistical noise
2. **Demonstrates stable trading behavior** - long episodes, positive rewards
3. **Baseline for comparison** - shows what good performance looks like
4. **Warm-start candidate** - can be used to initialize future training

### Usage:
```python
from stable_baselines3 import PPO
model = PPO.load("preserved_models/phase1a_5k_success_model.zip")
```

### Next Steps:
- Use for paper trading validation
- Analyze action patterns for insights
- Consider as warm-start for future training
- Compare against Phase 1B variants to isolate degradation cause