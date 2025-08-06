# 🔬 COMPREHENSIVE DIAGNOSTIC ROADMAP
## Stairways V4 Training Issues - Root Cause Analysis

**Objective**: Determine why early positive rewards (3.33 → 3.08 → 0.98) consistently degrade over training time, and validate whether the foundation is solid before implementing advanced learning rate schedules.

**Timeline**: 3.5 days wall-clock with proper analysis buffers
**Compute**: Assumes 500 FPS, 16 envs, GPU acceleration available

---

## 📋 EXECUTIVE SUMMARY

We've observed a consistent pattern across multiple training runs:
- **Early episodes (0-20K steps)**: Strong positive rewards (0.8-3.3 range)
- **Mid-training (50K+ steps)**: Gradual degradation to negative rewards
- **Late training (100K+ steps)**: Consistent negative performance

**Key Questions to Answer**:
1. Are early "wins" real learning or statistical noise?
2. Is the RefinedRewardSystem causing the degradation?
3. Is there data leakage giving false early signals?
4. Does the agent learn genuine strategies that later degrade due to policy drift?

---

## 🧪 PHASE 1: FOUNDATION VALIDATION (Day 1-2)

### 1A. Freeze-Early Validity Test
**Purpose**: Determine if early checkpoints contain real learned behavior vs. lucky noise

**Method**:
- Train with current RefinedRewardSystem + 30% DD cap
- Save checkpoints at 5K, 10K, 15K steps (seed=0)
- For each checkpoint: Run evaluation-only (LR=0, entropy=0) for 5K steps on **Feb 2024 data** (never seen before)

**Expected Runtime**: 3 × (0.5h train + 0.4h eval) = 2.7 hours

**Success Criteria**: 
- ≥0.5 mean reward maintained on Feb 2024 eval for ≥3 checkpoints
- Episode lengths >40 steps consistently
- No immediate collapse to negative rewards

**Deliverables**:
```bash
scripts/run_freeze_early.sh
config/freeze_early_eval.yaml
```

### 1B. Reward Engine A/B/C Test
**Purpose**: Isolate whether RefinedRewardSystem is causing degradation

**Variants** (same data slice, seed=0, 50K steps each):
- **Variant A**: Stock V3Enhanced reward (baseline)
- **Variant B**: RefinedRewardSystem as wrapper (current shim approach)  
- **Variant C**: RefinedRewardSystem integrated into environment reward calculation

**Expected Runtime**: 3 × 4h = 12 hours (can run in parallel)

**Success Criteria**:
- If only B degrades → RefinedRewardSystem wrapper issue
- If B+C degrade but A stable → RefinedRewardSystem logic issue
- If all degrade → environment or data issue

**Deliverables**:
```bash
scripts/run_reward_abc.sh
config/reward_v3.yaml          # Stock V3Enhanced baseline
config/reward_shim.yaml        # Shim variant (current approach)
config/reward_internal.yaml   # RefinedReward inside-env variant
```

### 1C. Data Leakage Audit
**Purpose**: Ensure agent cannot peek at future information

**Method**:
- Shuffle episode start offsets (±0.5 trading day)
- Shift all derived features (VWAP, rolling returns, etc.) by +1 step
- Run 10K sanity job to confirm no crashes
- Compare reward distribution pre/post shift

**Expected Runtime**: 1 hour

**Success Criteria**: Post-shift reward ≈ pre-shift reward (within 10%)

**Deliverables**:
```python
scripts/audit_data_leakage.py
data/processed/shifted_features_*.parquet
```

### 1D. Multi-Seed Robustness
**Purpose**: Quantify variance and ensure results aren't seed-dependent

**Method**:
- Rerun best variant from 1B for 10K steps each on seeds 1-4
- Calculate mean and standard deviation across seeds

**Expected Runtime**: 4 × 1h = 4 hours

**Success Criteria**: Std-dev < 30% of |mean| AND std-dev < 0.5 absolute

**Deliverables**:
```python
analysis/multi_seed_variance.ipynb
results/seed_comparison_stats.csv
```

---

## 🎯 PHASE 2: TEMPORAL VALIDATION (Day 2-3)

### 2. Out-of-Sample Temporal Test
**Purpose**: Test generalization to unseen time periods

**Method**:
- Train best variant from Phase 1 on 2022-2023 data
- Test on full 2024 data with LR=0 (evaluation only)
- Use 30-minute rolling P&L windows for Sharpe calculation

**Expected Runtime**: 4h train + 1h eval = 5 hours

**Success Criteria**: 
- OOS Sharpe ≥ 0.3 
- Positive mean reward on 2024 data
- No significant regime-specific overfitting

**Deliverables**:
```python
scripts/temporal_validation.py
analysis/oos_performance.ipynb
```

---

## 🔍 PHASE 3: BEHAVIORAL ANALYSIS (Day 3)

### 3. Action-Trace Analysis
**Purpose**: Understand what the agent actually does during "good" vs "bad" periods

**Method**:
- Log actions, positions, prices for all episodes during Phase 1A
- Compare patterns between:
  - Early good episodes (≤15K steps, reward >0.5)
  - Later bad episodes (≥100K steps, reward <0)
- Look for reward hacking patterns (flip-flopping, position gaming)

**Expected Runtime**: 2 hours offline analysis

**Success Criteria**: 
- Trades align with price movements
- No obvious exploit patterns (excessive flip-flopping)
- Position sizing makes economic sense

**Deliverables**:
```python
notebooks/action_trace.ipynb
analysis/trade_pattern_comparison.html
```

---

## 📊 PHASE 4: ROOT CAUSE SYNTHESIS (Day 3-4)

### 4. Decision Matrix
Based on Phase 1-3 results, determine root cause and next actions:

| Test Result | Interpretation | Next Action |
|-------------|----------------|-------------|
| Freeze-early passes, A/B shows only shim degrades | RefinedRewardSystem wrapper bug | Fix shim implementation |
| Freeze-early passes, A/B shows all degrade | Environment or data issue | Deep dive into env dynamics |
| Freeze-early fails | Early rewards are noise | Redesign reward system |
| Temporal validation fails | Overfitting to 2022-2023 regime | Add regime-aware features |
| Action analysis shows exploits | Reward hacking | Patch reward loopholes |

---

## 🛠️ IMPLEMENTATION DETAILS

### Optimization Tweaks
1. **Parallel Execution**: Run 1A & 1C simultaneously (GPU-0 for training, GPU-1/CPU for leakage audit)
2. **Data Loader Reuse**: Share TickDataLoader across 1B variants via forked subprocess envs (15% I/O savings)
3. **Inline Logging**: Capture action traces during 1A evaluation to avoid replay overhead

### File Structure
```
diagnostic_runs/
├── phase1a_freeze_early/
│   ├── checkpoint_5k.zip
│   ├── checkpoint_10k.zip  
│   ├── checkpoint_15k.zip
│   └── eval_results_feb2024.csv
├── phase1b_reward_abc/
│   ├── variant_a_v3enhanced/
│   ├── variant_b_shim/
│   └── variant_c_internal/
├── phase1c_leakage_audit/
│   └── shifted_data_results.csv
├── phase1d_multi_seed/
│   └── seed_variance_analysis.csv
├── phase2_temporal/
│   └── oos_2024_results.csv
└── phase3_actions/
    ├── action_traces.parquet
    └── trade_analysis.html
```

### Configuration Templates

**freeze_early_eval.yaml**:
```yaml
training:
  total_timesteps: 5000
  learning_rate: 0.0      # Frozen for evaluation
  ent_coef: 0.0          # No exploration
  eval_only: true
  
data:
  start_date: '2024-02-01'
  end_date: '2024-02-29'   # Feb 2024 only
  
environment:
  max_daily_drawdown_pct: 0.30
  initial_capital: 10000.0
```

**reward_v3.yaml**:
```yaml
reward_system: "DualTickerRewardV3Tuned"
reward_params:
  hold_bonus: 0.02
  transaction_cost: 0.5
  base_version: "v3_gold_standard_400k_20250802_202736"
```

**reward_shim.yaml**:
```yaml
reward_system: "RefinedRewardWrapper"
reward_params:
  pnl_epsilon: 750.0
  holding_alpha: 0.05
  penalty_beta: 0.10
  exploration_coef: 0.05
```

### Scripts to Prepare Tonight

**scripts/run_freeze_early.sh**:
```bash
#!/bin/bash
# Phase 1A: Freeze-Early Validity Test

echo "🔬 Starting Freeze-Early Validity Test"
echo "Training checkpoints at 5K, 10K, 15K steps..."

for steps in 5000 10000 15000; do
    echo "Training to ${steps} steps..."
    python train_diagnostic.py \
        --config config/freeze_early_train.yaml \
        --total_timesteps ${steps} \
        --save_path diagnostic_runs/phase1a_freeze_early/checkpoint_${steps}k.zip
    
    echo "Evaluating checkpoint ${steps}k on Feb 2024 data..."
    python evaluate_checkpoint.py \
        --checkpoint diagnostic_runs/phase1a_freeze_early/checkpoint_${steps}k.zip \
        --config config/freeze_early_eval.yaml \
        --output diagnostic_runs/phase1a_freeze_early/eval_${steps}k_feb2024.csv
done

echo "✅ Freeze-Early Validity Test Complete"
```

**scripts/run_reward_abc.sh**:
```bash
#!/bin/bash
# Phase 1B: Reward Engine A/B/C Test

echo "🔬 Starting Reward Engine A/B/C Test"

# Run all variants in parallel
python train_diagnostic.py \
    --config config/reward_v3.yaml \
    --total_timesteps 50000 \
    --save_path diagnostic_runs/phase1b_reward_abc/variant_a_v3enhanced &

python train_diagnostic.py \
    --config config/reward_shim.yaml \
    --total_timesteps 50000 \
    --save_path diagnostic_runs/phase1b_reward_abc/variant_b_shim &

python train_diagnostic.py \
    --config config/reward_internal.yaml \
    --total_timesteps 50000 \
    --save_path diagnostic_runs/phase1b_reward_abc/variant_c_internal &

wait  # Wait for all background jobs to complete

echo "✅ Reward Engine A/B/C Test Complete"
```

**notebooks/action_trace.ipynb** (template):
```python
"""
Action Trace Analysis Notebook
Analyzes agent behavior patterns during good vs bad episodes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load action traces from Phase 1A
action_traces = pd.read_parquet('diagnostic_runs/phase1a_freeze_early/action_traces.parquet')

# Separate good vs bad episodes
good_episodes = action_traces[action_traces['episode_reward'] > 0.5]
bad_episodes = action_traces[action_traces['episode_reward'] < 0]

# Analysis functions
def analyze_position_patterns(df, title):
    """Analyze position holding patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position distribution
    axes[0,0].hist(df['nvda_position'], bins=20, alpha=0.7, label='NVDA')
    axes[0,0].hist(df['msft_position'], bins=20, alpha=0.7, label='MSFT')
    axes[0,0].set_title(f'{title}: Position Distribution')
    axes[0,0].legend()
    
    # Position changes (flip-flop detection)
    df['nvda_pos_change'] = df['nvda_position'].diff().abs()
    df['msft_pos_change'] = df['msft_position'].diff().abs()
    
    axes[0,1].plot(df['nvda_pos_change'].rolling(10).mean(), label='NVDA')
    axes[0,1].plot(df['msft_pos_change'].rolling(10).mean(), label='MSFT')
    axes[0,1].set_title(f'{title}: Position Change Frequency')
    axes[0,1].legend()
    
    # P&L vs positions
    axes[1,0].scatter(df['nvda_position'], df['step_pnl'], alpha=0.5)
    axes[1,0].set_title(f'{title}: NVDA Position vs P&L')
    
    axes[1,1].scatter(df['msft_position'], df['step_pnl'], alpha=0.5)
    axes[1,1].set_title(f'{title}: MSFT Position vs P&L')
    
    plt.tight_layout()
    return fig

# Generate comparison plots
good_fig = analyze_position_patterns(good_episodes, "Good Episodes (Reward > 0.5)")
bad_fig = analyze_position_patterns(bad_episodes, "Bad Episodes (Reward < 0)")

# Save analysis
good_fig.savefig('diagnostic_runs/phase3_actions/good_episodes_analysis.png')
bad_fig.savefig('diagnostic_runs/phase3_actions/bad_episodes_analysis.png')

print("✅ Action trace analysis complete")
```

---

## 🎯 SUCCESS METRICS & DECISION TREE

### Pass/Fail Criteria Summary

| Phase | Test | Success Threshold | Failure Action |
|-------|------|------------------|----------------|
| 1A | Freeze-Early | ≥0.5 reward on 3+ checkpoints | Early rewards are noise → redesign |
| 1B | Reward A/B/C | Clear isolation of degradation source | Fix identified component |
| 1C | Leakage Audit | Post-shift ≈ pre-shift performance | Fix data pipeline |
| 1D | Multi-Seed | Std-dev < 30% of mean, < 0.5 absolute | Increase sample size |
| 2 | Temporal OOS | Sharpe ≥ 0.3, positive reward | Add regime awareness |
| 3 | Action Trace | No obvious exploits, sensible trades | Patch reward system |

### Final Decision Matrix

**If ALL tests pass**: 
- Foundation is solid
- Proceed with 3-phase learning rate schedule
- Implement adaptive LR decay (drop by 50% if reward falls >20% over 10K steps)
- Keep KL guard (0.010) and frequent checkpoints

**If Freeze-Early fails**:
- Early rewards are statistical noise
- Need fundamental reward system redesign
- Consider curriculum learning approach

**If Reward A/B/C shows shim-specific issues**:
- Bug in RefinedRewardSystem wrapper
- Fix implementation, retest
- May need to integrate reward logic directly into environment

**If Temporal validation fails**:
- Model overfits to training period
- Add regime-aware features
- Consider ensemble approaches

---

## 📅 EXECUTION SCHEDULE

**Tonight (Day 0)**:
- [ ] Prepare all config files and scripts
- [ ] Set up diagnostic_runs/ directory structure
- [ ] Queue Phase 1A jobs to start overnight

**Day 1 Morning**:
- [ ] Check Phase 1A results
- [ ] Launch Phase 1B (A/B/C) jobs in parallel
- [ ] Run Phase 1C leakage audit

**Day 1 Evening**:
- [ ] Analyze Phase 1A-1C results
- [ ] Launch Phase 1D multi-seed jobs
- [ ] Prepare Phase 2 temporal validation

**Day 2**:
- [ ] Complete Phase 1D analysis
- [ ] Run Phase 2 temporal validation
- [ ] Begin Phase 3 action trace analysis

**Day 3**:
- [ ] Complete all analysis
- [ ] Generate comprehensive report
- [ ] Make go/no-go decision on phased LR approach

**Day 4 (Buffer)**:
- [ ] Address any unexpected issues
- [ ] Finalize recommendations
- [ ] Plan next implementation phase

---

## 🚨 RISK MITIGATION

**Compute Resource Management**:
- Stagger jobs to avoid resource conflicts
- Monitor GPU memory usage
- Have fallback CPU-only configs ready

**Data Integrity**:
- Backup original datasets before any modifications
- Version control all config changes
- Maintain audit trail of all experiments

**Analysis Bias Prevention**:
- Designate independent reviewer for results
- Pre-commit to success criteria (no moving goalposts)
- Document all assumptions and limitations

**Timeline Contingencies**:
- 0.5 day buffer built into schedule
- Parallel execution where possible
- Simplified fallback analyses if needed

---

## 📋 DELIVERABLES CHECKLIST

### Scripts & Configs
- [ ] `scripts/run_freeze_early.sh`
- [ ] `scripts/run_reward_abc.sh`
- [ ] `scripts/audit_data_leakage.py`
- [ ] `config/freeze_early_eval.yaml`
- [ ] `config/reward_v3.yaml`
- [ ] `config/reward_shim.yaml`
- [ ] `config/reward_internal.yaml`

### Analysis Notebooks
- [ ] `notebooks/action_trace.ipynb`
- [ ] `analysis/multi_seed_variance.ipynb`
- [ ] `analysis/oos_performance.ipynb`

### Results & Reports
- [ ] Phase 1A: Freeze-early validation results
- [ ] Phase 1B: Reward system comparison
- [ ] Phase 1C: Data leakage audit report
- [ ] Phase 1D: Multi-seed variance analysis
- [ ] Phase 2: Temporal out-of-sample validation
- [ ] Phase 3: Action trace behavioral analysis
- [ ] Phase 4: Comprehensive root cause report with recommendations

---

**🎯 FINAL OBJECTIVE**: By end of Day 3, we will have definitive answers to whether the foundation is solid enough for advanced learning rate scheduling, or if we need to address more fundamental issues first.

This diagnostic roadmap is designed to be bias-resistant, scientifically rigorous, and actionable. Each phase builds on the previous one, and clear success/failure criteria prevent analysis paralysis.

**Ready to execute. Awaiting confirmation to proceed.**