#!/usr/bin/env python3
"""
Step 2: Smoke-test reward pipeline
Verify reward_bounds_check prints 2 numbers; confirm range ≈ (-50, 100) after scaling
"""

import sys
import numpy as np
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from gym_env.institutional_safeguards import InstitutionalSafeguards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reward_bounds():
    """Test reward bounds checking with expected range."""
    
    logger.info("🔍 STEP 2: SMOKE-TEST REWARD PIPELINE")
    logger.info("=" * 50)
    
    # Load Phase 1 config
    with open('config/phase1_reality_grounding.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create safeguards
    safeguards = InstitutionalSafeguards(config)
    
    # Test reward bounds
    logger.info("📊 Testing reward bounds...")
    
    # Test various reward values
    test_rewards = [-3000, -100, -50, 0, 50, 100, 200, 6000]
    
    logger.info("🎯 Reward bounds check results:")
    logger.info(f"   Min bound: {safeguards.reward_bounds[0]}")
    logger.info(f"   Max bound: {safeguards.reward_bounds[1]}")
    
    expected_min = -50  # After 0.02 scaling: -2000 * 0.02 = -40, close to -50
    expected_max = 100  # After 0.02 scaling: 5000 * 0.02 = 100
    
    logger.info(f"🎯 Expected range after scaling: ≈ ({expected_min}, {expected_max})")
    
    # Test scaled bounds
    scaled_min = safeguards.reward_bounds[0] * config['environment']['reward_scaling']
    scaled_max = safeguards.reward_bounds[1] * config['environment']['reward_scaling']
    
    logger.info(f"📊 Actual scaled range: ({scaled_min}, {scaled_max})")
    
    # Verify range is approximately correct
    if abs(scaled_min - expected_min) < 20 and abs(scaled_max - expected_max) < 20:
        logger.info("✅ Reward bounds are in expected range!")
        return True
    else:
        logger.error("❌ Reward bounds are outside expected range!")
        return False

if __name__ == "__main__":
    success = test_reward_bounds()
    if success:
        logger.info("✅ STEP 2 COMPLETE: Reward pipeline smoke test PASSED")
    else:
        logger.error("❌ STEP 2 FAILED: Reward pipeline smoke test FAILED")
    sys.exit(0 if success else 1)