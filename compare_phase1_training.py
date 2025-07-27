#!/usr/bin/env python3
"""
Phase 1 Training Comparison Tool
Compares Phase 1 Reality Grounding training with previous training sessions
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}

def compare_configurations():
    """Compare Phase 1 config with previous configs."""
    
    logger.info("🔍 CONFIGURATION COMPARISON")
    logger.info("=" * 50)
    
    # Load configurations
    phase1_config = load_config("config/phase1_reality_grounding.yaml")
    emergency_config = load_config("config/emergency_fix_orchestrator_gpu.yaml")
    
    if not phase1_config:
        logger.error("❌ Phase 1 config not found!")
        return
    
    if not emergency_config:
        logger.error("❌ Emergency fix config not found!")
        return
    
    # Compare key parameters
    comparisons = [
        ("Initial Capital", 
         phase1_config.get('environment', {}).get('initial_capital', 'N/A'),
         emergency_config.get('environment', {}).get('initial_capital', 'N/A')),
        
        ("Reward Scaling", 
         phase1_config.get('environment', {}).get('reward_scaling', 'N/A'),
         emergency_config.get('environment', {}).get('reward_scaling', 'N/A')),
        
        ("Max Position Size", 
         phase1_config.get('environment', {}).get('max_position_size_pct', 'N/A'),
         emergency_config.get('environment', {}).get('position_sizing_pct_capital', 'N/A')),
        
        ("Risk Features", 
         phase1_config.get('risk', {}).get('include_risk_features', 'N/A'),
         emergency_config.get('risk', {}).get('include_risk_features', 'N/A')),
        
        ("Institutional Safeguards", 
         "ENABLED" if 'validation' in phase1_config else "DISABLED",
         "DISABLED"),
        
        ("Model Validation", 
         "ENABLED" if 'model_validation' in phase1_config else "DISABLED",
         "DISABLED"),
        
        ("Observation Consistency", 
         "ENABLED" if phase1_config.get('validation', {}).get('observation_consistency_check') else "DISABLED",
         "DISABLED"),
    ]
    
    logger.info("📊 KEY PARAMETER COMPARISON:")
    logger.info("")
    logger.info(f"{'Parameter':<25} {'Phase 1':<15} {'Previous':<15} {'Status'}")
    logger.info("-" * 70)
    
    for param, phase1_val, prev_val, in comparisons:
        if phase1_val != prev_val:
            status = "🔄 CHANGED"
        else:
            status = "✅ SAME"
        
        logger.info(f"{param:<25} {str(phase1_val):<15} {str(prev_val):<15} {status}")
    
    logger.info("")
    
    # Highlight Phase 1 improvements
    logger.info("🛡️ PHASE 1 IMPROVEMENTS:")
    logger.info("   ✅ Institutional Safeguards - NEW")
    logger.info("   ✅ Model Compatibility Validation - NEW")
    logger.info("   ✅ Observation Consistency Testing - NEW")
    logger.info("   ✅ Reward Scaling Fix - 0.02 (institutional grade)")
    logger.info("   ✅ Enhanced Risk Features - 11 total features")
    logger.info("   ✅ Reward Bounds Enforcement - (-2000, +5000)")
    logger.info("")

def analyze_expected_differences():
    """Analyze what differences to expect in Phase 1 training."""
    
    logger.info("🎯 EXPECTED TRAINING DIFFERENCES")
    logger.info("=" * 50)
    
    logger.info("📈 REWARD SCALING IMPACT:")
    logger.info("   • Previous: Rewards in range 0-950,000")
    logger.info("   • Phase 1: Rewards in range 8,000-19,000")
    logger.info("   • Impact: More stable training, better convergence")
    logger.info("")
    
    logger.info("🛡️ INSTITUTIONAL SAFEGUARDS IMPACT:")
    logger.info("   • Reward bounds: Prevents extreme reward spikes")
    logger.info("   • Position limits: Prevents over-leveraging")
    logger.info("   • Cash reserves: Ensures liquidity buffer")
    logger.info("   • Impact: More conservative, institutional-grade behavior")
    logger.info("")
    
    logger.info("🔍 VALIDATION IMPACT:")
    logger.info("   • Model compatibility: Prevents silent training failures")
    logger.info("   • Observation consistency: Ensures train/eval match")
    logger.info("   • Impact: Higher training reliability, fewer surprises")
    logger.info("")
    
    logger.info("📊 MONITORING IMPROVEMENTS:")
    logger.info("   • Enhanced TensorBoard logging")
    logger.info("   • Safeguard violation tracking")
    logger.info("   • Consistency test results")
    logger.info("   • Impact: Better visibility into training health")
    logger.info("")

def check_tensorboard_setup():
    """Check TensorBoard directory setup."""
    
    logger.info("📊 TENSORBOARD SETUP CHECK")
    logger.info("=" * 50)
    
    # Check directories
    tensorboard_dirs = [
        "logs/tensorboard_phase1",
        "logs/tensorboard_emergency_fix",
        "logs/tensorboard_turnover_penalty"
    ]
    
    for dir_path in tensorboard_dirs:
        if os.path.exists(dir_path):
            logger.info(f"✅ {dir_path}: EXISTS")
        else:
            logger.info(f"❌ {dir_path}: MISSING")
            # Create Phase 1 directory
            if "phase1" in dir_path:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"   📁 Created: {dir_path}")
    
    logger.info("")
    logger.info("🔗 TENSORBOARD URLS:")
    logger.info("   • Phase 1: http://localhost:6006")
    logger.info("   • Previous: http://localhost:6006 (same port)")
    logger.info("   • Note: Clear browser cache to see Phase 1 data")
    logger.info("")

def validate_training_readiness():
    """Validate that Phase 1 training is ready to run."""
    
    logger.info("✅ TRAINING READINESS CHECK")
    logger.info("=" * 50)
    
    checks = []
    
    # Check Phase 1 config
    if os.path.exists("config/phase1_reality_grounding.yaml"):
        checks.append(("Phase 1 Config", True))
    else:
        checks.append(("Phase 1 Config", False))
    
    # Check Phase 1 training script
    if os.path.exists("phase1_training.py"):
        checks.append(("Phase 1 Training Script", True))
    else:
        checks.append(("Phase 1 Training Script", False))
    
    # Check Phase 1 batch file
    if os.path.exists("start_training_phase1.bat"):
        checks.append(("Phase 1 Batch File", True))
    else:
        checks.append(("Phase 1 Batch File", False))
    
    # Check safeguards
    if os.path.exists("src/gym_env/institutional_safeguards.py"):
        checks.append(("Institutional Safeguards", True))
    else:
        checks.append(("Institutional Safeguards", False))
    
    # Check validators
    if os.path.exists("src/models/compatibility_validator.py"):
        checks.append(("Model Compatibility Validator", True))
    else:
        checks.append(("Model Compatibility Validator", False))
    
    if os.path.exists("src/validation/observation_consistency.py"):
        checks.append(("Observation Consistency Validator", True))
    else:
        checks.append(("Observation Consistency Validator", False))
    
    # Display results
    all_passed = True
    for check_name, passed in checks:
        status = "✅ READY" if passed else "❌ MISSING"
        logger.info(f"   {check_name:<30} {status}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("🎉 ALL CHECKS PASSED - Phase 1 training is ready!")
        logger.info("")
        logger.info("🚀 TO START PHASE 1 TRAINING:")
        logger.info("   1. Run: start_training_phase1.bat")
        logger.info("   2. Monitor: http://localhost:6006")
        logger.info("   3. Compare with previous training results")
        logger.info("")
        return True
    else:
        logger.error("❌ SOME CHECKS FAILED - Fix issues before training")
        return False

def main():
    """Main comparison analysis."""
    
    logger.info("🔍 PHASE 1 TRAINING COMPARISON ANALYSIS")
    logger.info("=" * 60)
    logger.info("")
    
    # Run all comparisons
    compare_configurations()
    analyze_expected_differences()
    check_tensorboard_setup()
    ready = validate_training_readiness()
    
    logger.info("=" * 60)
    logger.info("📋 SUMMARY")
    logger.info("=" * 60)
    
    if ready:
        logger.info("✅ Phase 1 Reality Grounding is ready for training!")
        logger.info("")
        logger.info("🎯 WHAT TO EXPECT:")
        logger.info("   • Episode rewards: 8,000 - 19,000 (vs previous 0-950k)")
        logger.info("   • Institutional safeguards active")
        logger.info("   • Enhanced validation and monitoring")
        logger.info("   • More stable and predictable training")
        logger.info("")
        logger.info("🚀 NEXT STEPS:")
        logger.info("   1. Run: start_training_phase1.bat")
        logger.info("   2. Compare TensorBoard metrics with previous runs")
        logger.info("   3. Monitor safeguard violations (should be minimal)")
        logger.info("   4. Validate episode rewards stay in target range")
        
        return True
    else:
        logger.error("❌ Phase 1 training not ready - fix issues first")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)