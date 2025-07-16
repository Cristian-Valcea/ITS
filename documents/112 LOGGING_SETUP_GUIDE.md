# 📋 IntradayJules Logging Setup Guide

## 🎯 Answer: Logs Go to BOTH stdout AND Files

Your IntradayJules system is configured for **dual logging**:
- ✅ **Console Output (stdout)**: Real-time monitoring while training
- ✅ **Log Files**: Persistent storage for analysis and debugging

---

## 📁 Log File Locations

### Main Training Logs
```bash
# GPU Fixed Configuration
logs/orchestrator_gpu_fixed.log          # Main orchestrator log
logs/orchestrator_gpu_fixed.log.1        # Rotated backup (when > 50MB)
logs/orchestrator_gpu_fixed.log.2        # Older backup
# ... up to 5 backups (backup_count: 5)

# TensorBoard Logs (for monitoring)
logs/tensorboard_gpu_fixed/              # TensorBoard training metrics
```

### Other Log Types
```bash
logs/risk_audit.jsonl                    # Risk management audit trail
logs/audit/audit_records.jsonl           # System audit logs
logs/run_metadata/                       # Episode metadata
logs/test/                               # Test run logs
```

---

## 🔧 Current Logging Configuration

From `config/main_config_orchestrator_gpu_fixed.yaml`:
```yaml
logging:
  level: INFO                             # Log level (DEBUG, INFO, WARNING, ERROR)
  log_to_file: true                       # Enable file logging
  log_file_path: logs/orchestrator_gpu_fixed.log  # Main log file
  max_file_size_mb: 50                    # Rotate when file > 50MB
  backup_count: 5                         # Keep 5 backup files
```

---

## 🚀 Pre-Run Preparation Commands

### 1. Create/Clear Log Directory
```bash
# Ensure logs directory exists
mkdir -p logs

# Optional: Clear old logs for fresh start
# rm -f logs/orchestrator_gpu_fixed.log*
# rm -rf logs/tensorboard_gpu_fixed/
```

### 2. Check Current Log Files
```bash
# View existing log files
python view_logs.py

# Or manually check
ls -la logs/
```

### 3. Set Up Real-Time Monitoring
```bash
# Terminal 1: Start training
python -m src.execution.orchestrator_agent --config config/main_config_orchestrator_gpu_fixed.yaml

# Terminal 2: Monitor logs in real-time
tail -f logs/orchestrator_gpu_fixed.log

# Terminal 3: Monitor TensorBoard
tensorboard --logdir logs/tensorboard_gpu_fixed --port 6006
```

---

## 📊 Monitoring During Training

### Real-Time Log Monitoring
```bash
# Follow main log file
tail -f logs/orchestrator_gpu_fixed.log

# Filter for specific events
tail -f logs/orchestrator_gpu_fixed.log | grep -E "(Episode|Reward|Loss)"

# Monitor risk events
tail -f logs/risk_audit.jsonl | jq '.'
```

### TensorBoard Monitoring
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard_gpu_fixed

# Open browser to: http://localhost:6006
```

### Live Log Monitor Script
```bash
# Use the existing monitor script
python monitor_live_logs.py
```

---

## 🔍 Log Analysis Commands

### View Recent Training Progress
```bash
# Last 100 lines of main log
tail -n 100 logs/orchestrator_gpu_fixed.log

# Search for specific patterns
grep -n "Episode" logs/orchestrator_gpu_fixed.log | tail -20
grep -n "Kyle Lambda" logs/orchestrator_gpu_fixed.log
grep -n "ERROR\|WARNING" logs/orchestrator_gpu_fixed.log
```

### Performance Metrics
```bash
# Extract episode rewards
grep "Episode.*reward" logs/orchestrator_gpu_fixed.log | tail -20

# Check training convergence
grep "Training.*complete" logs/orchestrator_gpu_fixed.log

# Risk management events
grep -E "(turnover|drawdown|risk)" logs/orchestrator_gpu_fixed.log
```

---

## ⚙️ Customizing Log Output

### Increase Verbosity (More Detailed Logs)
```yaml
# In config/main_config_orchestrator_gpu_fixed.yaml
logging:
  level: DEBUG  # Change from INFO to DEBUG
```

### Separate Component Logs
```yaml
# Add component-specific logging
logging:
  level: INFO
  log_to_file: true
  log_file_path: logs/orchestrator_gpu_fixed.log
  component_logs:
    kyle_lambda: logs/kyle_lambda.log
    risk_agent: logs/risk_agent.log
    training: logs/training.log
```

### Reduce Log File Size
```yaml
logging:
  max_file_size_mb: 20  # Smaller files, more frequent rotation
  backup_count: 10      # Keep more backup files
```

---

## 📈 Log File Structure

### Main Log Format
```
2025-07-15 14:30:15,123 - INFO - OrchestratorAgent - Starting training episode 1
2025-07-15 14:30:15,124 - INFO - KyleLambdaFillSimulator - Fill calculated: mid=100.0, fill=100.05
2025-07-15 14:30:15,125 - INFO - RiskAgent - Turnover check: 0.85 < 3.0 (OK)
2025-07-15 14:30:15,126 - INFO - TrainerAgent - Episode 1 reward: 0.0234
```

### Risk Audit Log Format (JSON)
```json
{
  "timestamp": "2025-07-15T14:30:15.123Z",
  "event_type": "turnover_check",
  "current_turnover": 0.85,
  "turnover_cap": 3.0,
  "status": "OK"
}
```

---

## 🛠️ Troubleshooting Log Issues

### Log Files Not Created
```bash
# Check permissions
ls -la logs/
chmod 755 logs/

# Verify config
python -c "
import yaml
with open('config/main_config_orchestrator_gpu_fixed.yaml') as f:
    config = yaml.safe_load(f)
print('Log config:', config['logging'])
"
```

### Log Files Too Large
```bash
# Check current log sizes
du -h logs/*.log

# Manually rotate if needed
mv logs/orchestrator_gpu_fixed.log logs/orchestrator_gpu_fixed.log.backup
```

### Missing TensorBoard Logs
```bash
# Verify TensorBoard directory
ls -la logs/tensorboard_gpu_fixed/

# Check training config
grep -A5 "tensorboard_log" config/main_config_orchestrator_gpu_fixed.yaml
```

---

## 🎯 Recommended Pre-Run Checklist

### Before Starting Training:
- [ ] Verify logs directory exists: `mkdir -p logs`
- [ ] Check available disk space: `df -h`
- [ ] Clear old logs if needed: `rm -f logs/orchestrator_gpu_fixed.log*`
- [ ] Set up monitoring terminals
- [ ] Start TensorBoard: `tensorboard --logdir logs/tensorboard_gpu_fixed`

### During Training:
- [ ] Monitor real-time logs: `tail -f logs/orchestrator_gpu_fixed.log`
- [ ] Check TensorBoard metrics: http://localhost:6006
- [ ] Watch for error patterns: `grep ERROR logs/orchestrator_gpu_fixed.log`

### After Training:
- [ ] Archive important logs: `cp logs/orchestrator_gpu_fixed.log logs/training_$(date +%Y%m%d_%H%M%S).log`
- [ ] Analyze performance: `python view_logs.py`
- [ ] Extract key metrics for reporting

---

## 🚀 Ready for Your Next Run!

Your logging system is fully configured for comprehensive monitoring:
- **Real-time console output** for immediate feedback
- **Persistent log files** for detailed analysis
- **TensorBoard integration** for visual metrics
- **Risk audit trails** for compliance
- **Automatic log rotation** to prevent disk issues

Start your training with confidence knowing every detail will be captured! 📊