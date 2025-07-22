#!/usr/bin/env python3
"""
Launch Corrected 50K Training
Fixes applied:
1. Episode structure: max_episode_steps=1000 (50 episodes × 1000 steps)
2. Reward scaling: 0.07 (target ep_rew_mean 4-6)
"""

import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Launch the corrected 50K training."""
    
    print("🚀 LAUNCHING CORRECTED 50K TRAINING")
    print("=" * 50)
    print("✅ FIXES APPLIED:")
    print("   1. Episode Structure: max_episode_steps=1000")
    print("   2. Reward Scaling: 0.07 (target ep_rew_mean 4-6)")
    print("   3. PPO Scaling: Enabled (working correctly)")
    print()
    print("📊 EXPECTED RESULTS:")
    print("   - Episode Length: ~1000 steps (not 50,000)")
    print("   - Episode Reward: 4-6 range (not 239)")
    print("   - Total Episodes: 50 episodes")
    print("   - Duration: 8-12 hours")
    print()
    
    # Verify fixes are in place
    config_path = Path("config/phase1_reality_grounding.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_content = f.read()
            if "reward_scaling: 0.07" in config_content:
                print("✅ Config reward_scaling: 0.07 ✓")
            else:
                print("❌ Config reward_scaling not set to 0.07")
                return
    
    training_script = Path("phase1_fast_recovery_training.py")
    if training_script.exists():
        with open(training_script, 'r') as f:
            script_content = f.read()
            if "max_episode_steps=1000" in script_content:
                print("✅ Training script max_episode_steps: 1000 ✓")
            else:
                print("❌ Training script missing max_episode_steps=1000")
                return
    
    print("\n🎯 LAUNCHING TRAINING...")
    print("Monitor progress with: .\\monitor_tensorboard.bat")
    print("Expected TensorBoard metrics:")
    print("   - rollout/ep_len_mean: ~1000 (not 50,000)")
    print("   - rollout/ep_rew_mean: 4-6 (not 239)")
    print()
    
    # Launch training
    try:
        result = subprocess.run([
            sys.executable, "phase1_fast_recovery_training.py"
        ], check=True, capture_output=False)
        
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

if __name__ == "__main__":
    main()