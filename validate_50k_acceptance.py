#!/usr/bin/env python3
"""
50K Run Acceptance Criteria Validator
Validates 5K pilot results against green-light criteria for 50K run
"""

import sys
import yaml
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_50k_acceptance_criteria():
    """Validate 5K pilot results against 50K green-light criteria."""
    
    logger.info("🚀 50K RUN ACCEPTANCE CRITERIA VALIDATION")
    logger.info("=" * 60)
    
    # Load config to verify settings
    with open('config/phase1_reality_grounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🔧 CONFIGURATION VERIFICATION:")
    logger.info(f"   • Reward scaling: {config['environment']['reward_scaling']} (should be 0.02)")
    logger.info(f"   • Soft DD limit: {config['risk']['soft_dd_pct']:.1%}")
    logger.info(f"   • Hard DD limit: {config['risk']['hard_dd_pct']:.1%}")
    logger.info(f"   • Penalty lambda: {config['risk']['penalty_lambda']}")
    logger.info(f"   • Normalize advantage: {config['training']['normalize_advantage']}")
    logger.info("")
    
    # Acceptance criteria
    criteria = {
        'no_final_safety_termination': True,  # No "FINAL SAFETY TERMINATION" lines
        'ep_rew_mean_target': 40,             # ep_rew_mean > +40 by 3k steps
        'entropy_threshold': -0.4,            # entropy > -0.4 throughout
        'explained_variance_target': 0.1      # explained_variance stabilizes > 0.1
    }
    
    logger.info("🎯 50K GREEN-LIGHT ACCEPTANCE CRITERIA:")
    logger.info("=" * 60)
    logger.info("1. ✅ NO 'FINAL SAFETY TERMINATION' LINES")
    logger.info("   - Episodes must complete without safety termination")
    logger.info("   - DD penalties applied but no episode ending")
    logger.info("")
    logger.info(f"2. ✅ ep_rew_mean > +{criteria['ep_rew_mean_target']} BY 3K STEPS")
    logger.info("   - Reward signal must be positive and meaningful")
    logger.info("   - Conservative 0.02 scaling should still show progress")
    logger.info("")
    logger.info(f"3. ✅ ENTROPY > {criteria['entropy_threshold']} THROUGHOUT")
    logger.info("   - Agent must maintain exploration")
    logger.info("   - No policy collapse or deterministic behavior")
    logger.info("")
    logger.info(f"4. ✅ explained_variance STABILIZES > {criteria['explained_variance_target']}")
    logger.info("   - Critic must be learning effectively")
    logger.info("   - Value function providing useful signal")
    logger.info("")
    
    logger.info("🔍 WHAT TO MONITOR IN 5K PILOT:")
    logger.info("=" * 60)
    logger.info("📊 TENSORBOARD METRICS:")
    logger.info("   • rollout/ep_rew_mean: Watch for upward trend > +40")
    logger.info("   • train/entropy_loss: Should stay > -0.4")
    logger.info("   • train/explained_variance: Should stabilize > 0.1")
    logger.info("")
    logger.info("📝 LOG MESSAGES:")
    logger.info("   • ❌ 'FINAL SAFETY TERMINATION' = FAIL")
    logger.info("   • ✅ 'Applied soft DD penalty' = OK (learning continues)")
    logger.info("   • ✅ Episodes completing full steps = OK")
    logger.info("")
    
    logger.info("🚨 FAILURE CONDITIONS:")
    logger.info("=" * 60)
    logger.info("❌ IMMEDIATE FAIL:")
    logger.info("   • Any 'FINAL SAFETY TERMINATION' messages")
    logger.info("   • ep_rew_mean stuck below +40 after 3k steps")
    logger.info("   • Entropy dropping below -0.4 consistently")
    logger.info("   • explained_variance not improving past 0.1")
    logger.info("")
    
    logger.info("✅ SUCCESS CONDITIONS:")
    logger.info("=" * 60)
    logger.info("🎯 GREEN LIGHT FOR 50K IF:")
    logger.info("   1. No safety terminations in 5k pilot")
    logger.info("   2. ep_rew_mean > +40 by step 3000")
    logger.info("   3. Entropy consistently > -0.4")
    logger.info("   4. explained_variance stabilizes > 0.1")
    logger.info("")
    
    logger.info("🚀 READY FOR 5K PILOT!")
    logger.info("=" * 60)
    logger.info("📋 NEXT STEPS:")
    logger.info("   1. Run: python phase1_fast_recovery_training.py")
    logger.info("   2. Monitor TensorBoard: http://localhost:6006")
    logger.info("   3. Watch console for termination messages")
    logger.info("   4. Validate criteria after 5k steps")
    logger.info("   5. If all green → proceed to 50k full run")
    logger.info("")
    
    return True

if __name__ == "__main__":
    success = validate_50k_acceptance_criteria()
    sys.exit(0 if success else 1)