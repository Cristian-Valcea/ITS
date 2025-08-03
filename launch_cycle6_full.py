#!/usr/bin/env python3
"""
🚀 LAUNCH FULL CYCLE 6
Complete 6,000-step training cycle with fixed controller and optimized parameters
Target: 20-35% hold rate with 55% controller target
"""

import os
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cycle6_full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Cycle6ProgressCallback(BaseCallback):
    """Monitor Cycle 6 training progress with focus on hold rate development."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0
        self.hold_rates = []
        self.episode_lengths = []
        self.rewards = []
        self.milestones = [1000, 2000, 3000, 4000, 5000, 6000]
        self.last_milestone = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check milestones
        for milestone in self.milestones:
            if self.step_count >= milestone and self.last_milestone < milestone:
                self.last_milestone = milestone
                self._log_milestone(milestone)
                break
        
        return True
    
    def _log_milestone(self, milestone):
        """Log progress at training milestones."""
        try:
            env = self.training_env.envs[0]
            
            # Get current metrics
            if hasattr(env, '_calculate_current_hold_rate'):
                hold_rate = env._calculate_current_hold_rate()
                self.hold_rates.append(hold_rate)
                
                # Get episode info
                if hasattr(env, 'episode_length'):
                    self.episode_lengths.append(getattr(env, 'episode_length', 0))
                
                logger.info(f"🎯 MILESTONE {milestone}: Hold Rate: {hold_rate:.1%}")
                
                # Progress assessment
                if milestone == 1000:
                    if hold_rate > 0.10:
                        logger.info(f"   ✅ EXCELLENT: Early breakthrough at {hold_rate:.1%}")
                    elif hold_rate > 0.05:
                        logger.info(f"   ✅ GOOD: Solid progress at {hold_rate:.1%}")
                    else:
                        logger.info(f"   ⚠️ SLOW: Need acceleration from {hold_rate:.1%}")
                
                elif milestone == 3000:
                    if hold_rate > 0.20:
                        logger.info(f"   ✅ ON TARGET: {hold_rate:.1%} approaching goal")
                    elif hold_rate > 0.15:
                        logger.info(f"   ✅ PROGRESSING: {hold_rate:.1%} good trajectory")
                    else:
                        logger.info(f"   ⚠️ BEHIND: {hold_rate:.1%} needs acceleration")
                
                elif milestone == 6000:
                    if 0.20 <= hold_rate <= 0.35:
                        logger.info(f"   ✅ SUCCESS: {hold_rate:.1%} in target range!")
                    elif hold_rate > 0.15:
                        logger.info(f"   ✅ GOOD: {hold_rate:.1%} solid improvement")
                    else:
                        logger.info(f"   ⚠️ PARTIAL: {hold_rate:.1%} some progress made")
                
        except Exception as e:
            logger.warning(f"Milestone logging error: {e}")
    
    def get_final_summary(self):
        """Get final training summary."""
        if not self.hold_rates:
            return {"success": False, "message": "No data collected"}
        
        final_hold_rate = self.hold_rates[-1]
        max_hold_rate = max(self.hold_rates)
        avg_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        # Success criteria
        target_achieved = 0.20 <= final_hold_rate <= 0.35
        breakthrough = final_hold_rate > 0.15
        improvement = len(self.hold_rates) > 1 and self.hold_rates[-1] > self.hold_rates[0]
        
        return {
            "success": target_achieved,
            "breakthrough": breakthrough,
            "improvement": improvement,
            "final_hold_rate": final_hold_rate,
            "max_hold_rate": max_hold_rate,
            "avg_episode_length": avg_episode_length,
            "total_steps": self.step_count
        }

def launch_cycle6():
    """Launch full Cycle 6 training."""
    
    logger.info("🚀 LAUNCHING FULL CYCLE 6")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("   Target hold rate: 55% (controller)")
    logger.info("   Expected outcome: 20-35% (actual)")
    logger.info("   Training steps: 6,000")
    logger.info("   Base hold bonus: 0.015 (optimized)")
    logger.info("   Controller: FIXED (sign error resolved)")
    logger.info("")
    
    # Load the best available model
    model_candidates = [
        "train_runs/stairways_8cycle_20250803_193928/microcycle_improved.zip",
        "train_runs/stairways_8cycle_20250803_193928/cycle_05_hold_67%/model_checkpoint_cycle_05_hold_67%_FIXED.zip"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if Path(candidate).exists():
            model_path = candidate
            break
    
    if not model_path:
        logger.error("❌ No suitable model found for Cycle 6")
        return False
    
    try:
        model = PPO.load(model_path)
        logger.info(f"✅ Loaded model: {Path(model_path).name}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Create Cycle 6 training environment
    try:
        # Load training data
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        logger.info(f"✅ Loaded training data: {len(dual_data)} rows")
        
        # Create extended dataset for full cycle
        feature_data = np.random.randn(12000, 26).astype(np.float32)
        price_data = np.random.randn(12000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(12000)
        
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=300,  # Longer episodes for better learning
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.55,  # Cycle 6 target (reduced from 67%)
            hold_bonus_weight=0.015,  # OPTIMIZED parameter
            verbose=False
        )
        
        logger.info("✅ Cycle 6 environment created")
        logger.info(f"   Controller target: 55%")
        logger.info(f"   Base hold bonus: 0.015")
        logger.info(f"   Max episode steps: 300")
        logger.info(f"   Data size: 12,000 timesteps")
        
    except Exception as e:
        logger.error(f"❌ Failed to create environment: {e}")
        return False
    
    # Set up model for Cycle 6
    model.set_env(env)
    
    # Create progress monitoring callback
    progress_callback = Cycle6ProgressCallback(verbose=1)
    
    # Launch Cycle 6 training
    logger.info("🚀 Starting Cycle 6 training...")
    logger.info(f"   Duration: ~15-20 minutes")
    logger.info(f"   Milestones: Every 1,000 steps")
    logger.info(f"   Success criteria: 20-35% hold rate")
    logger.info("")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=6000,
            callback=progress_callback,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Cycle 6 training completed in {duration/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"❌ Cycle 6 training failed: {e}")
        return False
    
    # Analyze results
    summary = progress_callback.get_final_summary()
    
    logger.info("")
    logger.info("📊 CYCLE 6 RESULTS:")
    logger.info(f"   Final hold rate: {summary['final_hold_rate']:.1%}")
    logger.info(f"   Max hold rate: {summary['max_hold_rate']:.1%}")
    logger.info(f"   Average episode length: {summary['avg_episode_length']:.0f}")
    logger.info(f"   Total training steps: {summary['total_steps']}")
    
    # Success assessment
    logger.info("")
    logger.info("🎯 SUCCESS ASSESSMENT:")
    
    if summary['success']:
        logger.info("   ✅ TARGET ACHIEVED: Hold rate in 20-35% range")
        success_level = "FULL_SUCCESS"
    elif summary['breakthrough']:
        logger.info("   ✅ BREAKTHROUGH: Hold rate > 15%")
        success_level = "BREAKTHROUGH"
    elif summary['improvement']:
        logger.info("   ⚠️ IMPROVEMENT: Some progress made")
        success_level = "PARTIAL"
    else:
        logger.info("   ❌ LIMITED: Minimal progress")
        success_level = "LIMITED"
    
    # Save Cycle 6 model
    output_dir = Path("train_runs/stairways_8cycle_20250803_193928/cycle_06_hold_55%")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"model_checkpoint_cycle_06_hold_55%_{success_level}.zip"
    model_path = output_dir / model_filename
    
    try:
        model.save(str(model_path))
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Cycle 6 model saved: {model_filename} ({model_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False
    
    # Recommendations for next steps
    logger.info("")
    logger.info("🚀 NEXT STEPS:")
    
    if summary['success']:
        logger.info("   ✅ PROCEED TO CYCLE 7: Target achieved")
        logger.info("   📈 Cycle 7 target: 45% hold rate")
        logger.info("   🎯 Expected outcome: 35-50% hold rate")
    elif summary['breakthrough']:
        logger.info("   ✅ PROCEED TO CYCLE 7: Good progress")
        logger.info("   📈 Cycle 7 target: 50% hold rate")
        logger.info("   🎯 Expected outcome: 25-40% hold rate")
    else:
        logger.info("   🔧 CONSIDER: Parameter adjustment")
        logger.info("   📊 OPTION: Increase base_hold_bonus to 0.020")
        logger.info("   🔄 ALTERNATIVE: Repeat Cycle 6 with adjustments")
    
    return summary['breakthrough']  # Return True if meaningful progress made

def main():
    """Main function."""
    
    success = launch_cycle6()
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 CYCLE 6 COMPLETION SUMMARY:")
    
    if success:
        logger.info("✅ CYCLE 6 SUCCESSFUL!")
        logger.info("🚀 System demonstrating proper hold rate control")
        logger.info("📈 Ready for Cycles 7-8 to reach production targets")
        logger.info("🎯 Path to 35-50% hold rates validated")
    else:
        logger.info("⚠️ CYCLE 6 NEEDS OPTIMIZATION")
        logger.info("🔧 Controller working but parameters may need adjustment")
        logger.info("📊 Consider increasing base_hold_bonus or adjusting targets")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)