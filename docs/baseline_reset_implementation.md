# Enhanced Baseline Reset Logic: Escaping DD Purgatory

## Overview

The trading environment now includes sophisticated baseline reset logic to prevent agents from getting stuck in "DD purgatory" - a state where the drawdown baseline never resets, keeping the agent perpetually penalized even after recovery.

## The Problem: DD Purgatory

**Before Enhancement:**
- Baseline only reset after significant recovery (+0.5%) or very long timeout (800 steps)
- Agents could get stuck in permanent penalty state
- No proportional escape mechanism based on risk limits
- Recovery requirements were fixed, not adaptive to risk settings

**DD Purgatory Symptoms:**
- Portfolio recovers but baseline never resets
- Continuous penalties despite good performance
- Agent discouraged from taking any risk
- Training stagnation due to permanent penalty state

## The Solution: Three-Tier Escape Mechanism

### **Method 1: Purgatory Escape (Primary)**
```
Reset when: equity > baseline + (soft_dd_limit × 0.5)
```

**Logic:**
- If soft DD limit is 2%, escape requires 1% gain above baseline
- If soft DD limit is 3%, escape requires 1.5% gain above baseline
- **Proportional to risk tolerance**: Higher risk limits = higher escape thresholds

**Example:**
```
Baseline: $50,000
Soft DD: 2.0%
Buffer: $50,000 × 2.0% × 0.5 = $500
Escape threshold: $50,500
Required gain: 1.0%
```

### **Method 2: Flat Timeout (Safety Net)**
```
Reset after: 200 steps of flat performance
```

**Logic:**
- Prevents permanent purgatory if recovery is too slow
- Reasonable timeout that allows for natural market cycles
- Ensures baseline eventually progresses forward

### **Method 3: Legacy Recovery (Compatibility)**
```
Reset when: equity > baseline + 0.5%
```

**Logic:**
- Maintains backward compatibility
- Fixed 0.5% recovery threshold
- Fallback mechanism for edge cases

## Implementation Details

### **Constants**
```python
BASELINE_RESET_BUFFER_FACTOR = 0.5   # Purgatory escape factor
BASELINE_RESET_FLAT_STEPS = 200      # Flat timeout steps
```

### **Reset Logic Flow**
```python
# Method 1: Purgatory escape
soft_dd_buffer = baseline × soft_limit × BASELINE_RESET_BUFFER_FACTOR
recovery_threshold = baseline + soft_dd_buffer
purgatory_escape = portfolio_value > recovery_threshold

# Method 2: Flat timeout
flat_timeout = steps_since_baseline >= BASELINE_RESET_FLAT_STEPS

# Method 3: Legacy recovery
legacy_recovery = (portfolio_value - baseline) / baseline >= 0.005

# Reset if any condition met
if purgatory_escape or flat_timeout or legacy_recovery:
    reset_baseline()
```

### **Reset Reasons**
1. **`purgatory_escape`**: Portfolio exceeded proportional threshold
2. **`flat_timeout`**: Too many steps without reset
3. **`legacy_recovery`**: Traditional 0.5% recovery threshold

## Telemetry Integration

### **Always-Recorded Metrics**
```python
baseline/current_value          # Current baseline for DD calculation
baseline/steps_since_reset      # Steps since last reset
baseline/portfolio_vs_baseline  # Performance vs baseline (%)
baseline/recovery_threshold     # Current escape threshold
baseline/distance_to_escape     # Distance to escape (%)
```

### **Reset Event Metrics**
```python
baseline/reset_triggered        # 1.0 when reset occurs
baseline/reset_reason          # 1=purgatory, 2=timeout, 3=legacy
baseline/old_baseline          # Previous baseline value
baseline/new_baseline          # New baseline value
```

## Benefits

### **1. Proportional Escape Thresholds**
- **2% soft DD**: Requires 1.0% gain to escape
- **3% soft DD**: Requires 1.5% gain to escape
- **1.5% soft DD**: Requires 0.75% gain to escape

**Adaptive to risk tolerance**: Higher risk limits require proportionally higher recovery.

### **2. Multiple Safety Mechanisms**
- **Primary**: Proportional purgatory escape
- **Secondary**: Flat timeout prevention
- **Tertiary**: Legacy compatibility

**No single point of failure**: If one mechanism fails, others provide backup.

### **3. Smooth Baseline Progression**
- Baseline moves forward with portfolio performance
- No backwards jumps (baseline can only increase or stay same)
- Natural adaptation to market conditions

### **4. Rich Monitoring**
- Detailed telemetry for all reset events
- TensorBoard visualization of escape dynamics
- Clear reason codes for analysis

## Usage Examples

### **Environment Configuration**
```python
env = IntradayTradingEnv(
    # ... other parameters ...
    dd_baseline_reset_enabled=True,        # Enable enhanced reset logic
    dd_recovery_threshold_pct=0.005,       # Legacy 0.5% threshold
    dd_reset_timeout_steps=800,            # Legacy timeout (unused)
    institutional_safeguards_config=config # Provides soft_dd_pct
)
```

### **Monitoring Reset Events**
```python
# TensorBoard visualization
tensorboard --logdir=./logs

# Key metrics to watch:
# - baseline/reset_triggered: When resets occur
# - baseline/reset_reason: Why resets occurred
# - baseline/distance_to_escape: How close to escape
# - baseline/portfolio_vs_baseline: Performance tracking
```

## Mathematical Properties

### **Escape Threshold Calculation**
```
threshold = baseline + (baseline × soft_dd_pct × buffer_factor)
required_gain = soft_dd_pct × buffer_factor
```

### **Examples**
| Baseline | Soft DD | Buffer | Threshold | Required Gain |
|----------|---------|--------|-----------|---------------|
| $50,000  | 2.0%    | $500   | $50,500   | 1.00%         |
| $100,000 | 2.5%    | $1,250 | $101,250  | 1.25%         |
| $25,000  | 1.5%    | $187   | $25,187   | 0.75%         |

### **Validation**
- ✅ Escape always requires portfolio growth
- ✅ Threshold proportional to risk tolerance
- ✅ Reasonable recovery requirements
- ✅ Multiple safety mechanisms

## Testing

### **Test Coverage**
1. **Purgatory escape via recovery**: Portfolio growth triggers reset
2. **Flat timeout reset**: Long periods trigger automatic reset
3. **Mathematical validation**: All calculations correct
4. **Telemetry verification**: All metrics recorded properly

### **Test Results**
```
✅ Enhanced baseline reset logic implemented
✅ Three escape mechanisms available
✅ Proportional escape thresholds
✅ Multiple safety mechanisms
✅ Smooth baseline progression
✅ Rich telemetry integration
```

## Integration with Sigmoid Risk Gain

The baseline reset logic works seamlessly with the sigmoid risk gain schedule:

1. **Baseline determines DD reference**: Current baseline used for excess calculation
2. **Sigmoid scales penalties**: Smooth penalty scaling based on excess
3. **Reset prevents purgatory**: Baseline resets prevent permanent penalty state
4. **Combined benefits**: Nuanced risk management + escape mechanisms

## Conclusion

The enhanced baseline reset logic solves the DD purgatory problem through:

- **Smart escape thresholds** proportional to risk tolerance
- **Multiple safety mechanisms** preventing permanent purgatory
- **Rich telemetry** for monitoring and analysis
- **Seamless integration** with existing risk management systems

This ensures agents can recover from drawdowns naturally without getting stuck in permanent penalty states, leading to more effective training and better risk management.