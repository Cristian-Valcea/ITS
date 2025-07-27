# Model Versioning Guide

## 🎯 **Quick Start**

### **Check Current Models**
```bash
python scripts/model_registry.py list
python scripts/model_registry.py latest
```

### **After Training a New Model**
```bash
# 1. Evaluate new model performance
python analyze_episodes.py

# 2. If improved, register new version
python scripts/model_registry.py register \
  --model-path "src/models/DQN_2025-07-15_10-30-00" \
  --version "v1.1.0-improved" \
  --description "Fixed overtrading with 30-step cooldown"

# 3. Compare with baseline
python scripts/model_registry.py compare v1.0.0-baseline v1.1.0-improved

# 4. If better, promote to latest
python scripts/model_registry.py promote v1.1.0-improved

# 5. Commit to git
python scripts/model_registry.py commit v1.1.0-improved
```

## 📁 **Registry Structure**
```
models/registry/
├── v1.0.0-baseline/          # Your current baseline
│   ├── model/                # Model files (Git LFS)
│   ├── metadata.json         # Training config
│   ├── performance.json      # Metrics & analysis
│   └── README.md            # Documentation
├── v1.1.0-improved/         # Future versions
├── LATEST.txt               # Points to current best
└── README.md               # Registry documentation
```

## 🏷️ **Version Naming Convention**
- **v1.0.0**: Major architecture changes (DQN → PPO)
- **v1.1.0**: Significant improvements (hyperparameters, features)
- **v1.0.1**: Bug fixes or minor tweaks

## 🔄 **Workflow for New Experiments**

### **1. Before Training**
```bash
# Check current baseline
python scripts/model_registry.py latest
```

### **2. After Training**
```bash
# Analyze results
python analyze_episodes.py

# Compare key metrics:
# - Average return improvement
# - Win rate increase  
# - Reduced overtrading
# - Better risk metrics
```

### **3. If Performance Regresses**
```bash
# Revert to last good model
git checkout v1.0.0-baseline
# Or load from registry
cp models/registry/v1.0.0-baseline/model/* models/current/
```

### **4. If Performance Improves**
```bash
# Register new version
# Promote to latest
# Commit with git tag
```

## 📊 **Performance Tracking**

### **Key Metrics to Track**
- **Average Return %**: Primary profitability metric
- **Win Rate %**: Consistency indicator  
- **Trades/Episode**: Overtrading detection
- **Turnover Ratio**: Risk management
- **Max Drawdown**: Risk assessment

### **Regression Detection**
A model has regressed if:
- Average return decreases by >1%
- Win rate drops by >5%
- Trades/episode increases by >20%
- Turnover ratio increases significantly

## 🚀 **Current Baseline: v1.0.0-baseline**

### **Performance**
- **Average Return**: -5.16%
- **Win Rate**: 0.0%
- **Episodes**: 159
- **Avg Trades/Episode**: 196.9

### **Known Issues**
1. Overtrading (196.9 trades/episode)
2. Multi-day episodes (225 hours)
3. Zero win rate
4. Negative learning progression

### **Improvement Targets**
- **Return**: Target >0% average
- **Win Rate**: Target >40%
- **Trading**: Target <50 trades/episode
- **Episodes**: Target 6-8 hours (single day)

## 🛠️ **Model Recovery**

### **If Training Fails**
```bash
# Quick recovery to baseline
git checkout v1.0.0-baseline
python -c "
from stable_baselines3 import DQN
model = DQN.load('models/registry/v1.0.0-baseline/model/DQN_2025-07-14_14-15-45.zip')
print('✅ Baseline model loaded successfully')
"
```

### **Emergency Rollback**
```bash
# Rollback to previous version
git log --oneline --grep="feat.*model"  # Find model commits
git reset --hard <commit-hash>
```

This system ensures you never lose a working model and can always compare new experiments against your stable baseline! 🎯