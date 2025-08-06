# 🔬 COMPREHENSIVE DIAGNOSTIC ROADMAP - APPROVED VERSION
## Stairways V4 Training Issues - Root Cause Analysis

**Objective**: Determine why early positive rewards (3.33 → 3.08 → 0.98) consistently degrade over training time, and validate whether the foundation is solid before implementing advanced learning rate schedules.

**Timeline**: 4.2 days wall-clock with management-approved buffers (+20% padding)
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

## 🧪 PHASE 1: FOUNDATION VALIDATION (Day 1-2.4)

### 1A. Freeze-Early Validity Test ⭐ **MANAGEMENT ENHANCED**
**Purpose**: Determine if early checkpoints contain real learned behavior vs. lucky noise

**Method**:
- Train with current RefinedRewardSystem + 30% DD cap
- Save checkpoints at 5K, 10K, 15K steps (seed=0)
- For each checkpoint: Run evaluation-only (LR=0, entropy=0) for 5K steps on **Feb 2024 data** (never seen before)

**Expected Runtime**: 3 × (0.5h train + 0.4h eval) = 2.7 hours → **3.2 hours** (with 20% buffer)

**🔒 TIGHTENED SUCCESS CRITERIA** (Management Enhancement #1):
- ~~≥0.5 mean reward maintained on Feb 2024 eval for ≥3 checkpoints~~
- **NEW**: **ALL THREE** checkpoints must score ≥0.5 reward AND **NONE** fall below 0.3
- Episode lengths >40 steps consistently across all checkpoints
- No immediate collapse to negative rewards on any checkpoint

**Rationale**: Prevents single lucky checkpoint from masking inconsistent learning

**Deliverables**:
```bash
scripts/run_freeze_early.sh
config/freeze_early_eval.yaml
results/freeze_early_all_checkpoints.csv  # All three must pass
```

### 1B. Reward Engine A/B/C Test ⭐ **MANAGEMENT ENHANCED**
**Purpose**: Isolate whether RefinedRewardSystem is causing degradation

**Variants** (same data slice, seed=0, 50K steps each):
- **Variant A**: Stock V3Enhanced reward (baseline)
- **Variant B**: RefinedRewardSystem as wrapper (current shim approach)  
- **Variant C**: RefinedRewardSystem integrated into environment reward calculation

**Expected Runtime**: 3 × 4h = 12 hours → **14.4 hours** (with 20% buffer)

**🚨 SPEED-BUMP MONITORING** (Management Enhancement #2):
- Stream rolling 1K-step Sharpe to logger during training
- **Auto-abort criterion**: If Sharpe plunges below -0.5 before 30K steps, terminate that variant early
- Saves GPU hours on obviously failing runs
- Log abort reason and timestamp for analysis

**Success Criteria**:
- If only B degrades → RefinedRewardSystem wrapper issue
- If B+C degrade but A stable → RefinedRewardSystem logic issue
- If all degrade → environment or data issue
- **NEW**: No variant should trigger speed-bump abort (indicates fundamental instability)

**Deliverables**:
```bash
scripts/run_reward_abc.sh
scripts/monitor_sharpe_speedbump.py  # NEW: Real-time monitoring
config/reward_v3.yaml          # Stock V3Enhanced baseline
config/reward_shim.yaml        # Shim variant (current approach)
config/reward_internal.yaml   # RefinedReward inside-env variant
logs/speedbump_monitoring.log  # NEW: Abort tracking
```

### 1C. Data Leakage Audit ⭐ **MANAGEMENT ENHANCED**
**Purpose**: Ensure agent cannot peek at future information

**Method**:
- Shuffle episode start offsets (±0.5 trading day)
- Shift all derived features (VWAP, rolling returns, etc.) by +1 step
- Run 10K sanity job to confirm no crashes
- Compare reward distribution pre/post shift

**Expected Runtime**: 1 hour → **1.2 hours** (with 20% buffer)

**🔍 FEATURE IMPORTANCE ANALYSIS** (Management Enhancement #3):
- Run permutation importance on both original and +1-shift datasets
- Compare top 10 feature rankings between versions
- **Alert threshold**: >50% change in top-5 feature importance indicates hidden leakage
- Generate feature importance diff report

**Success Criteria**: 
- Post-shift reward ≈ pre-shift reward (within 10%)
- **NEW**: Feature importance rankings remain stable (≤50% change in top-5)
- No crashes or data pipeline errors

**Deliverables**:
```python
scripts/audit_data_leakage.py
scripts/feature_importance_diff.py  # NEW: Leakage detection
data/processed/shifted_features_*.parquet
analysis/feature_importance_comparison.html  # NEW: Visual diff report
```

### 1D. Multi-Seed Robustness ⭐ **MANAGEMENT ENHANCED**
**Purpose**: Quantify variance and ensure results aren't seed-dependent

**Method**:
- Rerun best variant from 1B for 10K steps each on seeds 1-4
- Calculate mean and standard deviation across seeds

**Expected Runtime**: 4 × 1h = 4 hours → **4.8 hours** (with 20% buffer)

**📊 TRAJECTORY VISUALIZATION** (Management Enhancement #4):
- ~~Calculate mean and standard deviation across seeds~~
- **NEW**: Plot full seed trajectories (ep_rew_mean vs. steps) for visual inspection
- Generate trajectory comparison plots for all 5 seeds (0-4)
- Flag outlier behaviors that single std-dev numbers might hide

**Success Criteria**: 
- Std-dev < 30% of |mean| AND std-dev < 0.5 absolute
- **NEW**: Visual trajectory inspection shows consistent patterns across seeds
- **NEW**: No single seed shows dramatically different behavior pattern

**Deliverables**:
```python
analysis/multi_seed_variance.ipynb
analysis/seed_trajectory_plots.html  # NEW: Visual trajectory comparison
results/seed_comparison_stats.csv
results/seed_outlier_analysis.csv    # NEW: Outlier detection
```

---

## 🎯 PHASE 2: TEMPORAL VALIDATION (Day 2.4-3.4)

### 2. Out-of-Sample Temporal Test
**Purpose**: Test generalization to unseen time periods

**Method**:
- Train best variant from Phase 1 on 2022-2023 data
- Test on full 2024 data with LR=0 (evaluation only)
- Use 30-minute rolling P&L windows for Sharpe calculation

**Expected Runtime**: 4h train + 1h eval = 5 hours → **6 hours** (with 20% buffer)

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

## 🔍 PHASE 3: BEHAVIORAL ANALYSIS (Day 3.4-4.0)

### 3. Action-Trace Analysis ⭐ **MANAGEMENT ENHANCED**
**Purpose**: Understand what the agent actually does during "good" vs "bad" periods

**Method**:
- Log actions, positions, prices for all episodes during Phase 1A
- Compare patterns between:
  - Early good episodes (≤15K steps, reward >0.5)
  - Later bad episodes (≥100K steps, reward <0)
- Look for reward hacking patterns (flip-flopping, position gaming)

**Expected Runtime**: 2 hours → **2.4 hours** (with 20% buffer)

**🚨 FLIP-RATE KPI MONITORING** (Management Enhancement #5):
- **NEW**: Compute flip-rate per episode: `flip_rate = Σ[aₜ ≠ aₜ₋₁] / episode_length`
- **Alert threshold**: flip_rate > 0.3 while net P&L ≤ 0 indicates reward hacking
- Generate flip-rate distribution plots for good vs bad episodes
- Flag episodes with suspicious flip patterns

**Success Criteria**: 
- Trades align with price movements
- No obvious exploit patterns (excessive flip-flopping)
- Position sizing makes economic sense
- **NEW**: Flip-rate < 0.3 for profitable episodes, or justified by market conditions

**Deliverables**:
```python
notebooks/action_trace.ipynb
analysis/trade_pattern_comparison.html
analysis/flip_rate_analysis.html      # NEW: Reward hacking detection
results/suspicious_episodes.csv       # NEW: Flagged episodes
```

---

## 📊 PHASE 4: ROOT CAUSE SYNTHESIS (Day 4.0-4.2)

### 4. Decision Matrix
Based on Phase 1-3 results, determine root cause and next actions:

| Test Result | Interpretation | Next Action |
|-------------|----------------|-------------|
| Freeze-early passes (ALL checkpoints), A/B shows only shim degrades | RefinedRewardSystem wrapper bug | Fix shim implementation |
| Freeze-early passes, A/B shows all degrade | Environment or data issue | Deep dive into env dynamics |
| Freeze-early fails (ANY checkpoint) | Early rewards are noise | Redesign reward system |
| Temporal validation fails | Overfitting to 2022-2023 regime | Add regime-aware features |
| Action analysis shows exploits (flip-rate alerts) | Reward hacking | Patch reward loopholes |
| Speed-bump aborts triggered | Fundamental instability | Revisit hyperparameters |
| Feature importance shifts detected | Data leakage present | Fix data pipeline |

---

## 🛠️ IMPLEMENTATION DETAILS

### Management-Approved Optimization Tweaks
1. **Parallel Execution**: Run 1A & 1C simultaneously (GPU-0 for training, GPU-1/CPU for leakage audit)
2. **Data Loader Reuse**: Share TickDataLoader across 1B variants via forked subprocess envs (15% I/O savings)
3. **Inline Logging**: Capture action traces during 1A evaluation to avoid replay overhead
4. **Speed-bump Monitoring**: Real-time Sharpe tracking with auto-abort capability
5. **Enhanced Analytics**: Feature importance, trajectory plots, flip-rate KPIs

### File Structure ⭐ **ENHANCED**
```
diagnostic_runs/
├── phase1a_freeze_early/
│   ├── checkpoint_5k.zip
│   ├── checkpoint_10k.zip  
│   ├── checkpoint_15k.zip
│   ├── eval_results_feb2024.csv
│   └── all_checkpoints_summary.csv     # NEW: Consolidated results
├── phase1b_reward_abc/
│   ├── variant_a_v3enhanced/
│   ├── variant_b_shim/
│   ├── variant_c_internal/
│   └── speedbump_monitoring.log        # NEW: Auto-abort tracking
├── phase1c_leakage_audit/
│   ├── shifted_data_results.csv
│   └── feature_importance_diff.html    # NEW: Leakage detection
├── phase1d_multi_seed/
│   ├── seed_variance_analysis.csv
│   └── trajectory_comparison.html      # NEW: Visual trajectories
├── phase2_temporal/
│   └── oos_2024_results.csv
├── phase3_actions/
│   ├── action_traces.parquet
│   ├── trade_analysis.html
│   ├── flip_rate_analysis.html         # NEW: Reward hacking detection
│   └── compressed_traces.tar.gz        # NEW: Storage hygiene
└── storage_cleanup/
    └── cleanup_traces.sh               # NEW: Automated cleanup
```

### Configuration Templates

**freeze_early_eval.yaml** ⭐ **ENHANCED**:
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

# NEW: Tightened success criteria
validation:
  min_reward_all_checkpoints: 0.5    # ALL must pass
  min_reward_floor: 0.3              # NONE can fall below
  required_episode_length: 40
```

**speedbump_monitoring.yaml** ⭐ **NEW**:
```yaml
monitoring:
  sharpe_window: 1000           # Rolling window for Sharpe calculation
  abort_threshold: -0.5         # Auto-abort if Sharpe falls below
  check_interval: 1000          # Check every N steps
  min_steps_before_abort: 30000 # Don't abort before this many steps
  
logging:
  log_level: INFO
  log_file: "logs/speedbump_monitoring.log"
  metrics_to_track:
    - "ep_rew_mean"
    - "ep_len_mean" 
    - "rolling_sharpe"
    - "policy_loss"
```

### Scripts to Prepare Tonight ⭐ **ENHANCED**

**scripts/run_freeze_early.sh** ⭐ **ENHANCED**:
```bash
#!/bin/bash
# Phase 1A: Freeze-Early Validity Test (Management Enhanced)

echo "🔬 Starting Freeze-Early Validity Test (Enhanced)"
echo "Training checkpoints at 5K, 10K, 15K steps..."
echo "⚠️  ALL THREE checkpoints must pass ≥0.5 reward, none below 0.3"

declare -a results=()

for steps in 5000 10000 15000; do
    echo "Training to ${steps} steps..."
    python train_diagnostic.py \
        --config config/freeze_early_train.yaml \
        --total_timesteps ${steps} \
        --save_path diagnostic_runs/phase1a_freeze_early/checkpoint_${steps}k.zip
    
    echo "Evaluating checkpoint ${steps}k on Feb 2024 data..."
    result=$(python evaluate_checkpoint.py \
        --checkpoint diagnostic_runs/phase1a_freeze_early/checkpoint_${steps}k.zip \
        --config config/freeze_early_eval.yaml \
        --output diagnostic_runs/phase1a_freeze_early/eval_${steps}k_feb2024.csv \
        --return_mean_reward)
    
    results+=($result)
    echo "Checkpoint ${steps}k result: ${result}"
done

# NEW: Validate ALL checkpoints pass criteria
python scripts/validate_all_checkpoints.py \
    --results "${results[@]}" \
    --min_reward 0.5 \
    --min_floor 0.3 \
    --output diagnostic_runs/phase1a_freeze_early/all_checkpoints_summary.csv

echo "✅ Freeze-Early Validity Test Complete (Enhanced)"
```

**scripts/monitor_sharpe_speedbump.py** ⭐ **NEW**:
```python
#!/usr/bin/env python3
"""
Speed-bump monitoring for reward A/B/C tests
Auto-aborts training runs if Sharpe falls below threshold
"""

import time
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

class SpeedBumpMonitor:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_file']),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def monitor_training_run(self, monitor_csv_path, variant_name):
        """Monitor a training run and abort if Sharpe falls below threshold"""
        
        self.logger.info(f"🚨 Starting speed-bump monitoring for {variant_name}")
        self.logger.info(f"   Abort threshold: Sharpe < {self.config['abort_threshold']}")
        self.logger.info(f"   Min steps before abort: {self.config['min_steps_before_abort']}")
        
        last_check_step = 0
        
        while True:
            try:
                # Check if monitor file exists and has new data
                if not Path(monitor_csv_path).exists():
                    time.sleep(10)
                    continue
                
                df = pd.read_csv(monitor_csv_path, skiprows=1)
                if len(df) == 0:
                    time.sleep(10)
                    continue
                
                current_step = len(df) * 512  # Assuming 512 steps per episode
                
                # Only check every N steps
                if current_step - last_check_step < self.config['check_interval']:
                    time.sleep(10)
                    continue
                
                # Don't abort too early
                if current_step < self.config['min_steps_before_abort']:
                    time.sleep(10)
                    continue
                
                # Calculate rolling Sharpe
                window_size = min(self.config['sharpe_window'] // 512, len(df))
                recent_rewards = df['r'].tail(window_size).values
                
                if len(recent_rewards) < 10:  # Need minimum data
                    time.sleep(10)
                    continue
                
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                sharpe = mean_reward / (std_reward + 1e-8)
                
                self.logger.info(f"📊 {variant_name} @ {current_step} steps: Sharpe = {sharpe:.3f}")
                
                # Check abort condition
                if sharpe < self.config['abort_threshold']:
                    self.logger.warning(f"🚨 SPEED-BUMP ABORT: {variant_name}")
                    self.logger.warning(f"   Sharpe {sharpe:.3f} < {self.config['abort_threshold']}")
                    self.logger.warning(f"   Aborting at {current_step} steps")
                    
                    # Create abort signal file
                    abort_file = Path(monitor_csv_path).parent / "ABORT_SIGNAL"
                    abort_file.write_text(f"Aborted at {current_step} steps, Sharpe={sharpe:.3f}")
                    
                    return "ABORTED"
                
                # Check if training completed normally
                if current_step >= 50000:  # Expected completion
                    self.logger.info(f"✅ {variant_name} completed normally")
                    return "COMPLETED"
                
                last_check_step = current_step
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error monitoring {variant_name}: {e}")
                time.sleep(30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor_csv', required=True)
    parser.add_argument('--variant_name', required=True)
    parser.add_argument('--config', default='config/speedbump_monitoring.yaml')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)['monitoring']
    
    # Start monitoring
    monitor = SpeedBumpMonitor(config)
    result = monitor.monitor_training_run(args.monitor_csv, args.variant_name)
    
    print(f"Monitoring result: {result}")

if __name__ == "__main__":
    main()
```

**scripts/feature_importance_diff.py** ⭐ **NEW**:
```python
#!/usr/bin/env python3
"""
Feature importance analysis for data leakage detection
Compares feature rankings before/after +1 step shift
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def calculate_feature_importance(X, y, feature_names):
    """Calculate permutation importance for features"""
    
    # Use RandomForest as proxy model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df

def compare_feature_importance(original_data, shifted_data, output_path):
    """Compare feature importance between original and shifted data"""
    
    print("🔍 Analyzing feature importance for leakage detection...")
    
    # Load data
    orig_df = pd.read_parquet(original_data)
    shift_df = pd.read_parquet(shifted_data)
    
    # Assume last column is target (reward)
    feature_cols = orig_df.columns[:-1]
    target_col = orig_df.columns[-1]
    
    # Calculate importance for both datasets
    orig_importance = calculate_feature_importance(
        orig_df[feature_cols].values, 
        orig_df[target_col].values,
        feature_cols
    )
    
    shift_importance = calculate_feature_importance(
        shift_df[feature_cols].values,
        shift_df[target_col].values, 
        feature_cols
    )
    
    # Merge and compare
    comparison = orig_importance.merge(
        shift_importance, 
        on='feature', 
        suffixes=('_original', '_shifted')
    )
    
    # Calculate ranking changes
    comparison['rank_original'] = comparison['importance_mean_original'].rank(ascending=False)
    comparison['rank_shifted'] = comparison['importance_mean_shifted'].rank(ascending=False)
    comparison['rank_change'] = abs(comparison['rank_original'] - comparison['rank_shifted'])
    
    # Flag potential leakage
    top_5_orig = set(comparison.nsmallest(5, 'rank_original')['feature'])
    top_5_shift = set(comparison.nsmallest(5, 'rank_shifted')['feature'])
    
    overlap = len(top_5_orig.intersection(top_5_shift))
    change_pct = (5 - overlap) / 5 * 100
    
    print(f"📊 Top-5 feature overlap: {overlap}/5 ({100-change_pct:.1f}%)")
    print(f"📊 Top-5 change percentage: {change_pct:.1f}%")
    
    if change_pct > 50:
        print("🚨 WARNING: >50% change in top-5 features - potential leakage detected!")
    else:
        print("✅ Feature rankings stable - no obvious leakage")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top 10 features comparison
    top_features = comparison.head(10)
    
    axes[0,0].barh(top_features['feature'], top_features['importance_mean_original'])
    axes[0,0].set_title('Original Data - Top 10 Features')
    axes[0,0].set_xlabel('Importance')
    
    axes[0,1].barh(top_features['feature'], top_features['importance_mean_shifted'])
    axes[0,1].set_title('Shifted Data - Top 10 Features')
    axes[0,1].set_xlabel('Importance')
    
    # Ranking change plot
    axes[1,0].scatter(comparison['rank_original'], comparison['rank_shifted'], alpha=0.6)
    axes[1,0].plot([0, len(comparison)], [0, len(comparison)], 'r--', alpha=0.5)
    axes[1,0].set_xlabel('Original Ranking')
    axes[1,0].set_ylabel('Shifted Ranking')
    axes[1,0].set_title('Feature Ranking Changes')
    
    # Change magnitude histogram
    axes[1,1].hist(comparison['rank_change'], bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Absolute Rank Change')
    axes[1,1].set_ylabel('Number of Features')
    axes[1,1].set_title('Distribution of Ranking Changes')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    
    # Save detailed results
    comparison.to_csv(f"{output_path}/feature_importance_comparison.csv", index=False)
    
    # Summary report
    summary = {
        'top_5_overlap': overlap,
        'change_percentage': change_pct,
        'leakage_detected': change_pct > 50,
        'max_rank_change': comparison['rank_change'].max(),
        'mean_rank_change': comparison['rank_change'].mean()
    }
    
    pd.DataFrame([summary]).to_csv(f"{output_path}/leakage_summary.csv", index=False)
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_data', required=True)
    parser.add_argument('--shifted_data', required=True) 
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    
    summary = compare_feature_importance(args.original_data, args.shifted_data, args.output_path)
    
    print(f"\n📋 LEAKAGE ANALYSIS SUMMARY:")
    print(f"   Top-5 overlap: {summary['top_5_overlap']}/5")
    print(f"   Change percentage: {summary['change_percentage']:.1f}%")
    print(f"   Leakage detected: {summary['leakage_detected']}")
    print(f"   Max rank change: {summary['max_rank_change']:.0f}")
    print(f"   Mean rank change: {summary['mean_rank_change']:.1f}")

if __name__ == "__main__":
    main()
```

**scripts/cleanup_traces.sh** ⭐ **NEW** (Management Enhancement #6):
```bash
#!/bin/bash
# Storage hygiene - compress and cleanup large trace files

echo "🧹 Starting storage cleanup for diagnostic traces..."

DIAGNOSTIC_DIR="diagnostic_runs"
CLEANUP_DIR="${DIAGNOSTIC_DIR}/storage_cleanup"

mkdir -p ${CLEANUP_DIR}

# Find large parquet files (>100MB)
echo "📊 Finding large trace files..."
find ${DIAGNOSTIC_DIR} -name "*.parquet" -size +100M -exec ls -lh {} \;

# Compress action traces
echo "🗜️ Compressing action traces..."
cd ${DIAGNOSTIC_DIR}/phase3_actions/
tar -czf action_traces_compressed.tar.gz *.parquet
if [ $? -eq 0 ]; then
    echo "✅ Compression successful"
    # Move originals to cleanup directory (don't delete immediately)
    mv *.parquet ${CLEANUP_DIR}/
    echo "📁 Original files moved to ${CLEANUP_DIR}/"
else
    echo "❌ Compression failed - keeping originals"
fi

# Generate cleanup report
echo "📋 Generating cleanup report..."
du -sh ${DIAGNOSTIC_DIR}/* > ${CLEANUP_DIR}/storage_report.txt
echo "💾 Storage report saved to ${CLEANUP_DIR}/storage_report.txt"

echo "✅ Storage cleanup complete"
echo "⚠️  Original files preserved in ${CLEANUP_DIR}/ for 7 days"
echo "🗑️  Run 'rm -rf ${CLEANUP_DIR}/' after verification to permanently delete"
```

---

## 🎯 SUCCESS METRICS & DECISION TREE ⭐ **ENHANCED**

### Pass/Fail Criteria Summary (Management Approved)

| Phase | Test | Success Threshold | Failure Action |
|-------|------|------------------|----------------|
| 1A | Freeze-Early | **ALL** checkpoints ≥0.5, **NONE** <0.3 | Early rewards are noise → redesign |
| 1B | Reward A/B/C | Clear isolation + no speed-bump aborts | Fix identified component |
| 1C | Leakage Audit | Post-shift ≈ pre-shift + feature stability | Fix data pipeline |
| 1D | Multi-Seed | Std-dev criteria + trajectory consistency | Increase sample size |
| 2 | Temporal OOS | Sharpe ≥ 0.3, positive reward | Add regime awareness |
| 3 | Action Trace | No exploits + flip-rate < 0.3 | Patch reward system |

### Enhanced Decision Matrix

**If ALL tests pass**: 
- Foundation is solid
- Proceed with 3-phase learning rate schedule
- Implement adaptive LR decay (drop by 50% if reward falls >20% over 10K steps)
- Keep KL guard (0.010) and frequent checkpoints

**If Freeze-Early fails (ANY checkpoint)**:
- Early rewards are statistical noise
- Need fundamental reward system redesign
- Consider curriculum learning approach

**If Speed-bump aborts triggered**:
- Fundamental instability in training
- Revisit hyperparameters before proceeding
- May indicate reward system issues

**If Feature importance shifts >50%**:
- Data leakage present in pipeline
- Fix feature engineering before proceeding
- Rerun all tests after fix

**If Flip-rate alerts triggered**:
- Reward hacking detected
- Patch reward system loopholes
- Add position change penalties

---

## 📅 EXECUTION SCHEDULE ⭐ **ENHANCED** (+20% Time Buffers)

**Tonight (Day 0)**:
- [ ] Prepare all config files and scripts
- [ ] Set up diagnostic_runs/ directory structure  
- [ ] Queue Phase 1A jobs to start overnight
- [ ] Set up speed-bump monitoring infrastructure

**Day 1 Morning**:
- [ ] Check Phase 1A results (validate ALL checkpoints pass)
- [ ] Launch Phase 1B (A/B/C) jobs in parallel with speed-bump monitoring
- [ ] Run Phase 1C leakage audit with feature importance analysis

**Day 1 Evening**:
- [ ] Analyze Phase 1A-1C results
- [ ] Check for any speed-bump aborts
- [ ] Launch Phase 1D multi-seed jobs
- [ ] Prepare Phase 2 temporal validation

**Day 2**:
- [ ] Complete Phase 1D analysis with trajectory visualization
- [ ] Run Phase 2 temporal validation
- [ ] Begin Phase 3 action trace analysis with flip-rate monitoring

**Day 3**:
- [ ] Complete all analysis including enhanced KPIs
- [ ] Run storage cleanup procedures
- [ ] Generate comprehensive report with all enhancements

**Day 4 (Enhanced Buffer)**:
- [ ] Address any unexpected issues
- [ ] Validate all enhanced criteria
- [ ] Finalize recommendations with management input
- [ ] Plan next implementation phase

**Day 4.2 (Final Buffer)**:
- [ ] Final review and sign-off
- [ ] Archive results and cleanup temporary files

---

## 🚨 RISK MITIGATION ⭐ **ENHANCED**

**Compute Resource Management**:
- Stagger jobs to avoid resource conflicts
- Monitor GPU memory usage with alerts
- Have fallback CPU-only configs ready
- **NEW**: Speed-bump monitoring prevents wasted GPU hours

**Data Integrity**:
- Backup original datasets before any modifications
- Version control all config changes
- Maintain audit trail of all experiments
- **NEW**: Feature importance tracking for leakage detection

**Analysis Bias Prevention**:
- Designate independent reviewer for results
- Pre-commit to success criteria (no moving goalposts)
- Document all assumptions and limitations
- **NEW**: Tightened criteria prevent single-checkpoint luck

**Timeline Contingencies**:
- **Enhanced**: 20% buffer built into all phases (management approved)
- Parallel execution where possible
- Simplified fallback analyses if needed
- **NEW**: Auto-abort capability saves time on failing runs

**Storage Management** ⭐ **NEW**:
- Automated compression of large trace files
- Cleanup procedures to prevent disk space issues
- Staged deletion with 7-day grace period
- Storage monitoring and reporting

---

## 📋 DELIVERABLES CHECKLIST ⭐ **ENHANCED**

### Scripts & Configs
- [ ] `scripts/run_freeze_early.sh` (enhanced with ALL-checkpoint validation)
- [ ] `scripts/run_reward_abc.sh` (enhanced with speed-bump monitoring)
- [ ] `scripts/monitor_sharpe_speedbump.py` ⭐ **NEW**
- [ ] `scripts/audit_data_leakage.py` (enhanced with feature importance)
- [ ] `scripts/feature_importance_diff.py` ⭐ **NEW**
- [ ] `scripts/validate_all_checkpoints.py` ⭐ **NEW**
- [ ] `scripts/cleanup_traces.sh` ⭐ **NEW**
- [ ] `config/freeze_early_eval.yaml` (enhanced criteria)
- [ ] `config/speedbump_monitoring.yaml` ⭐ **NEW**
- [ ] `config/reward_v3.yaml`
- [ ] `config/reward_shim.yaml`
- [ ] `config/reward_internal.yaml`

### Analysis Notebooks ⭐ **ENHANCED**
- [ ] `notebooks/action_trace.ipynb` (enhanced with flip-rate analysis)
- [ ] `analysis/multi_seed_variance.ipynb` (enhanced with trajectory plots)
- [ ] `analysis/oos_performance.ipynb`
- [ ] `analysis/feature_importance_comparison.html` ⭐ **NEW**
- [ ] `analysis/flip_rate_analysis.html` ⭐ **NEW**
- [ ] `analysis/seed_trajectory_plots.html` ⭐ **NEW**

### Results & Reports ⭐ **ENHANCED**
- [ ] Phase 1A: Freeze-early validation (ALL checkpoints must pass)
- [ ] Phase 1B: Reward system comparison (with speed-bump monitoring)
- [ ] Phase 1C: Data leakage audit (with feature importance analysis)
- [ ] Phase 1D: Multi-seed variance (with trajectory visualization)
- [ ] Phase 2: Temporal out-of-sample validation
- [ ] Phase 3: Action trace behavioral analysis (with flip-rate KPIs)
- [ ] Phase 4: Comprehensive root cause report with management enhancements
- [ ] Storage cleanup report and compressed archives ⭐ **NEW**

---

## 🎯 MANAGEMENT ENHANCEMENTS SUMMARY

**✅ Applied Management Feedback:**

1. **Tightened Freeze-Early Gate**: ALL three checkpoints must pass ≥0.5, none below 0.3
2. **Speed-bump Monitoring**: Auto-abort if Sharpe <-0.5 before 30K steps  
3. **Feature Importance Analysis**: Detect hidden leakage via ranking changes
4. **Trajectory Visualization**: Full seed plots instead of just mean±std
5. **Flip-rate KPI**: Reward hacking detection with 0.3 threshold
6. **Storage Hygiene**: Automated compression and cleanup procedures
7. **Timeline Buffers**: +20% padding on all phases for realistic scheduling

**🎯 Key Benefits:**
- **Bias-resistant**: Tightened criteria prevent false positives
- **Resource-efficient**: Speed-bump monitoring saves GPU hours
- **Comprehensive**: Enhanced analytics catch subtle issues
- **Practical**: Storage management and realistic timelines
- **Actionable**: Clear decision matrix with enhanced criteria

---

**🚀 FINAL OBJECTIVE**: By end of Day 4.2, we will have definitive, management-approved answers to whether the foundation is solid enough for advanced learning rate scheduling, or if we need to address more fundamental issues first.

This enhanced diagnostic roadmap incorporates all management feedback while maintaining scientific rigor and practical execution. The tightened criteria and enhanced monitoring will provide conclusive results we can act on with confidence.

**✅ APPROVED FOR EXECUTION - Ready to proceed with Phase 1A tonight.**