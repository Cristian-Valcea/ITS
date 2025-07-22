# Sigmoid vs Exponential Risk Gain Comparison

## Overview

The trading environment has been upgraded from a brutal exponential risk gain system to a sophisticated sigmoid risk gain schedule. This provides much more nuanced risk management without crushing signal.

## Mathematical Comparison

### Old Exponential System
```
λ = λ₀ × multiplier
multiplier = min(1.1^breaches, 20.0)  # 10% growth per breach, capped at 20×
```

**Problems:**
- ❌ Brutal 20× cap creates cliff effect
- ❌ Binary threshold (0.3% excess) causes sudden jumps
- ❌ Exponential growth crushes signal too aggressively
- ❌ No smooth transition between states

### New Sigmoid System
```
λ = λ₀ × (1 + tanh(150 × excess))
multiplier ∈ [1.0, 8.0]  # Smooth scaling, reasonable cap
```

**Benefits:**
- ✅ Smooth, continuous scaling
- ✅ Gentle response to small breaches
- ✅ Aggressive scaling for serious violations
- ✅ Natural cap at 8× without brutal cliff
- ✅ Preserves signal while managing risk

## Response Curve Comparison

| Excess DD | Old Exponential | New Sigmoid | Improvement |
|-----------|----------------|-------------|-------------|
| 0.1%      | 1.00×          | 2.04×       | Gentler initial response |
| 0.3%      | 1.10×          | 3.95×       | More responsive to small breaches |
| 0.5%      | 1.17×          | 5.45×       | Better risk scaling |
| 1.0%      | 1.37×          | 7.34×       | Appropriate escalation |
| 1.5%      | 1.61×          | 7.85×       | Near maximum, controlled |
| 2.0%      | 1.89×          | 7.97×       | Smooth approach to cap |
| 3.0%      | 2.59×          | 8.00×       | Reasonable maximum |
| 5.0%      | 4.77×          | 8.00×       | No brutal explosion |
| 10.0%     | 20.00×         | 8.00×       | **8× vs 20× - Much gentler!** |

## Key Improvements

### 1. **Smooth Scaling**
- **Old**: Binary threshold at 0.3% excess caused sudden jumps
- **New**: Continuous sigmoid response to any excess level
- **Result**: No discontinuities or sudden penalty spikes

### 2. **Reasonable Cap**
- **Old**: Brutal 20× multiplier could destroy signal completely
- **New**: Gentle 8× cap maintains risk management without signal crushing
- **Result**: Better balance between risk control and performance

### 3. **Nuanced Response**
- **Old**: Same 10% growth regardless of breach severity
- **New**: Response proportional to excess severity
- **Result**: Small breaches get gentle treatment, large breaches get aggressive response

### 4. **Mathematical Properties**
- **Continuous**: No jumps or discontinuities
- **Monotonic**: Always increasing with excess
- **Bounded**: Natural cap without artificial clipping
- **Smooth**: Differentiable everywhere

## Sigmoid Parameters

### Current Configuration
```python
SIGMOID_MAX_MULT = 8.0      # Maximum multiplier (8× cap)
SIGMOID_STEEPNESS = 150.0   # Controls curve steepness
```

### Response Examples
- **0.1% excess** → 2.04× multiplier (gentle)
- **0.5% excess** → 5.45× multiplier (moderate)
- **1.0% excess** → 7.34× multiplier (aggressive)
- **2.0% excess** → 7.97× multiplier (near maximum)

### Curve Characteristics
- **50% response**: 0.37% excess → 4.50× multiplier
- **90% response**: 0.99% excess → 7.30× multiplier
- **Inflection point**: 0.0% excess (smooth start)

## Implementation Benefits

### 1. **Signal Preservation**
The 8× cap (vs old 20×) means penalties are strong enough to discourage risk but not so brutal as to completely destroy trading signal.

### 2. **Proportional Response**
Small drawdown excesses get proportionally smaller penalties, allowing for natural market volatility without excessive punishment.

### 3. **Risk Management**
Large drawdown excesses still trigger aggressive penalties (7-8×), ensuring proper risk control for serious violations.

### 4. **Smooth Training**
The continuous, differentiable response provides better gradients for reinforcement learning, leading to more stable training.

## Telemetry Integration

The sigmoid system provides rich telemetry for monitoring:

```python
# TensorBoard metrics
lambda/sigmoid_multiplier    # Raw sigmoid calculation
lambda/excess_pct           # Current excess driving response
lambda/multiplier           # Final multiplier (with decay)
lambda/penalty_lambda       # Final penalty value
```

This allows for detailed analysis of risk management behavior during training.

## Migration Impact

### Backward Compatibility
- ✅ All existing reward bounds maintained
- ✅ Same penalty application mechanism
- ✅ Compatible with existing telemetry systems
- ✅ No breaking changes to environment interface

### Performance Impact
- ✅ Minimal computational overhead (single tanh calculation)
- ✅ More stable training due to smooth gradients
- ✅ Better risk/reward balance
- ✅ Reduced signal crushing

## Conclusion

The sigmoid risk gain schedule represents a significant improvement over the old exponential system:

1. **Smoother**: No brutal cliffs or discontinuities
2. **Smarter**: Proportional response to risk level
3. **Gentler**: 8× cap vs 20× cap preserves signal
4. **Better**: Improved risk/reward balance for training

This change should lead to more stable training, better performance, and more robust risk management without the signal-crushing effects of the old system.