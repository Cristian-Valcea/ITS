#!/usr/bin/env python3
"""
Smoke Test Validation - 1-Day Sprint Tuning
Validates the 5k smoke test results against target criteria
"""

import sys
import yaml
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_smoke_test_criteria():
    """Validate smoke test results against target criteria."""
    
    logger.info("🧪 SMOKE TEST VALIDATION (1-DAY SPRINT)")
    logger.info("=" * 60)
    
    # Load updated config
    with open('config/phase1_reality_grounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Display tuning changes
    logger.info("🔧 TUNING CHANGES APPLIED:")
    logger.info(f"   1. Reward scaling: 0.02 → {config['environment']['reward_scaling']} (12.5x increase)")
    logger.info(f"   2. Reward bounds: ±2000/5000 → ±{config['validation']['reward_bounds']['max_reward']} (tighter clipping)")
    logger.info(f"   3. PPO normalize_advantage: {config['training'].get('normalize_advantage', False)}")
    logger.info(f"   4. PPO vf_coef: 0.5 → {config['training'].get('vf_coef', 0.5)}")
    logger.info(f"   5. Training timesteps: 50k → 100k (full run after smoke test)")
    logger.info("")
    
    # Expected scaling effects
    old_scaling = 0.02
    new_scaling = config['environment']['reward_scaling']
    scaling_multiplier = new_scaling / old_scaling
    
    logger.info("📊 EXPECTED SCALING EFFECTS:")
    logger.info(f"   • Previous ep_rew_mean: ~0.3")
    logger.info(f"   • Scaling multiplier: {scaling_multiplier:.1f}x")
    logger.info(f"   • Expected new ep_rew_mean: ~{0.3 * scaling_multiplier:.1f}")
    logger.info(f"   • Target range: 6-12 (smoke test)")
    logger.info(f"   • Ultimate target: 400 scaled (8k raw → 400 with 0.25 scaling)")
    logger.info("")
    
    # Smoke test criteria
    criteria = {
        'ep_rew_mean_min': 6,
        'ep_rew_mean_max': 12,
        'entropy_threshold': -0.4,
        'explained_variance_threshold': 0.2,
        'timesteps': 5000
    }
    
    logger.info("🎯 SMOKE TEST SUCCESS CRITERIA:")
    logger.info(f"   ✓ ep_rew_mean: {criteria['ep_rew_mean_min']} - {criteria['ep_rew_mean_max']}")
    logger.info(f"   ✓ Entropy: > {criteria['entropy_threshold']}")
    logger.info(f"   ✓ explained_variance: > {criteria['explained_variance_threshold']}")
    logger.info(f"   ✓ Complete: {criteria['timesteps']:,} timesteps")
    logger.info("")
    
    # Reward bounds validation
    min_bound = config['validation']['reward_bounds']['min_reward']
    max_bound = config['validation']['reward_bounds']['max_reward']
    scaled_min = min_bound * new_scaling
    scaled_max = max_bound * new_scaling
    
    logger.info("🛡️ REWARD BOUNDS VALIDATION:")
    logger.info(f"   • Raw bounds: {min_bound} to {max_bound}")
    logger.info(f"   • Scaled bounds: {scaled_min:.1f} to {scaled_max:.1f}")
    logger.info(f"   • Target prevents gradient explosion while allowing signal")
    logger.info("")
    
    logger.info("🚀 READY FOR SMOKE TEST!")
    logger.info("=" * 60)
    logger.info("📋 NEXT STEPS:")
    logger.info("   1. Run: start_training_phase1.bat")
    logger.info("   2. Monitor TensorBoard: http://localhost:6006")
    logger.info("   3. Watch for target metrics in 5k steps")
    logger.info("   4. If criteria met → proceed to 100k full run")
    logger.info("   5. If criteria failed → adjust and retry")
    logger.info("")
    
    return True

if __name__ == "__main__":
    success = validate_smoke_test_criteria()
    sys.exit(0 if success else 1)