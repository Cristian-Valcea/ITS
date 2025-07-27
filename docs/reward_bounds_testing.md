# Reward Bounds Regression Testing

## Overview

The reward bounds unit test (`tests/test_reward_bounds.py`) provides critical regression protection for the trading environment's reward system. It ensures all rewards stay within institutional safeguards bounds of **[-150, 150]**.

### Lambda Protection System

The system includes multiple layers of lambda protection:
- **Multiplier Cap**: 20× maximum multiplier (MAX_LAMBDA_MULT)
- **Absolute Ceiling**: 500,000 ultimate safety limit (ABS_MAX_LAMBDA)
- **Growth Control**: 10% growth per breach above 0.3% excess DD
- **Time Decay**: 1% decay per step for gradual forgiveness

## Test Coverage

### Core Test: `test_reward_bounds_random_actions`
- **Runs 1000 random trading steps**
- **Validates every reward is within bounds**
- **Catches reward calculation bugs immediately**
- **Execution time: ~5 seconds**

### Additional Tests
- `test_reward_bounds_extreme_scenarios`: Tests edge cases (aggressive trading, hold-only, etc.)
- `test_reward_bounds_configuration_consistency`: Validates config parameters
- `test_reward_calculation_components`: Checks individual reward components

## Quick Test Execution

```bash
# Run just the core regression test
python run_reward_bounds_test.py

# Run all reward bounds tests
pytest tests/test_reward_bounds.py -v

# Run with specific marker
pytest -m reward_bounds -v
```

## Integration with CI/CD

### Pre-commit Hook
Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: reward-bounds-test
      name: Reward Bounds Regression Test
      entry: python run_reward_bounds_test.py
      language: system
      pass_filenames: false
```

### GitHub Actions
Add to workflow:
```yaml
- name: Run Reward Bounds Test
  run: python run_reward_bounds_test.py
```

## Expected Results

### Healthy System
```
✅ REWARD BOUNDS TEST PASSED!
   Reward range: [-1.93, 2.05]
   Expected range: [-150.0, 150.0]
   Mean reward: 0.09
   Violations: 0
```

### Regression Detected
```
❌ REWARD BOUNDS TEST FAILED!
AssertionError: Reward 250.5 above maximum 150.0
```

## Troubleshooting

### Common Issues

1. **Environment Constructor Changes**
   - Update `environment` fixture in test file
   - Match parameter names with `IntradayTradingEnv.__init__()`

2. **Config Structure Changes**
   - Update `test_reward_bounds_configuration_consistency`
   - Ensure YAML structure matches expectations

3. **Reward Component Changes**
   - Review `test_reward_calculation_components`
   - Update component bounds if needed

### Test Maintenance

- **When adding new reward components**: Update component bounds checks
- **When changing institutional safeguards**: Update expected bounds
- **When modifying environment constructor**: Update test fixtures

## Benefits

1. **Immediate Regression Detection**: Catches reward bugs in seconds
2. **Confidence in Changes**: Safe to modify reward system knowing bounds are protected
3. **Documentation**: Test serves as living documentation of reward bounds
4. **CI/CD Integration**: Automated protection in deployment pipeline

## Performance

- **Execution Time**: ~5 seconds
- **Memory Usage**: Minimal (single environment instance)
- **Deterministic**: Uses fixed random seeds for reproducibility
- **Lightweight**: No external dependencies beyond pytest

This test is a critical safety net for the trading system's reward mechanism.