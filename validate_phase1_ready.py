#!/usr/bin/env python3
"""
Phase 1 Training Readiness Validation
Confirms all components are ready for training
"""

import os
import sys
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_phase1_readiness():
    """Validate Phase 1 training readiness."""
    
    logger.info("🔍 PHASE 1 TRAINING READINESS CHECK")
    logger.info("=" * 50)
    
    checks = []
    
    # Check 1: Phase 1 config exists
    config_path = "config/phase1_reality_grounding.yaml"
    if os.path.exists(config_path):
        checks.append(("Phase 1 Config", True, config_path))
        
        # Load and validate config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check key Phase 1 features
            has_safeguards = 'validation' in config
            has_model_validation = 'model_validation' in config
            reward_scaling = config.get('environment', {}).get('reward_scaling', 1.0)
            
            checks.append(("Institutional Safeguards", has_safeguards, "validation section"))
            checks.append(("Model Validation", has_model_validation, "model_validation section"))
            checks.append(("Reward Scaling", reward_scaling == 0.08, f"0.08 (tuned for ep_rew_mean > +40, found: {reward_scaling})"))
            
        except Exception as e:
            checks.append(("Config Validation", False, f"Error: {e}"))
    else:
        checks.append(("Phase 1 Config", False, "Missing"))
    
    # Check 2: Training script exists
    training_script = "phase1_fast_recovery_training.py"
    if os.path.exists(training_script):
        checks.append(("Training Script", True, training_script))
    else:
        checks.append(("Training Script", False, "Missing"))
    
    # Check 3: Batch file exists
    batch_file = "start_training_phase1.bat"
    if os.path.exists(batch_file):
        checks.append(("Batch File", True, batch_file))
    else:
        checks.append(("Batch File", False, "Missing"))
    
    # Check 4: Required directories
    required_dirs = [
        "logs/tensorboard_phase1_fix1",
        "models",
        "src/gym_env",
        "src/validation"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            checks.append((f"Directory: {dir_path}", True, "Exists"))
        else:
            checks.append((f"Directory: {dir_path}", False, "Missing"))
            # Create if missing
            os.makedirs(dir_path, exist_ok=True)
            checks[-1] = (f"Directory: {dir_path}", True, "Created")
    
    # Check 5: Key source files
    key_files = [
        "src/gym_env/institutional_safeguards.py",
        "src/models/compatibility_validator.py",
        "src/validation/observation_consistency.py"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            checks.append((f"Source: {os.path.basename(file_path)}", True, "Exists"))
        else:
            checks.append((f"Source: {os.path.basename(file_path)}", False, "Missing"))
    
    # Display results
    logger.info("📊 READINESS CHECK RESULTS:")
    logger.info("")
    
    all_passed = True
    for check_name, passed, detail in checks:
        status = "✅ READY" if passed else "❌ MISSING"
        logger.info(f"   {check_name:<35} {status:<10} {detail}")
        if not passed:
            all_passed = False
    
    logger.info("")
    
    if all_passed:
        logger.info("🎉 ALL CHECKS PASSED!")
        logger.info("")
        logger.info("✅ PHASE 1 TRAINING IS READY TO RUN!")
        logger.info("")
        logger.info("🚀 TO START TRAINING:")
        logger.info("   1. Run: start_training_phase1.bat")
        logger.info("   2. Monitor: http://localhost:6006")
        logger.info("   3. Watch for institutional safeguards in logs")
        logger.info("")
        logger.info("🎯 EXPECTED RESULTS:")
        logger.info("   • Episode rewards: 8,000 - 19,000")
        logger.info("   • Reward bounds: -40 to +100 (scaled)")
        logger.info("   • No silent training failures")
        logger.info("   • Institutional-grade behavior")
        
        return True
    else:
        logger.error("❌ SOME CHECKS FAILED!")
        logger.error("Fix the missing components before training.")
        return False

if __name__ == "__main__":
    success = validate_phase1_readiness()
    sys.exit(0 if success else 1)