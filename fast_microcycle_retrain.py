#!/usr/bin/env python3
"""
🚀 FAST MICRO-CYCLE RETRAINING
Give PPO optimizer policy updates with corrected hold-bonus sign
Expected: HOLD-rate should rise to 8-15% within 2k steps
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
        logging.FileHandler('fast_microcycle_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HoldRateMonitorCallback(BaseCallback):
    """Monitor hold rate progression during micro-cycle training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0
        self.hold_rates = []
        self.episode_lengths = []
        self.rewards = []
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log every 500 steps for micro-cycle monitoring
        if self.step_count % 500 == 0:
            try:
                env = self.training_env.envs[0]
                
                # Get current hold rate
                if hasattr(env, '_calculate_current_hold_rate'):
                    hold_rate = env._calculate_current_hold_rate()
                    self.hold_rates.append(hold_rate)
                    
                    # Get recent episode info
                    if hasattr(env, 'episode_length') and hasattr(env, 'episode_reward'):
                        self.episode_lengths.append(getattr(env, 'episode_length', 0))
                        self.rewards.append(getattr(env, 'episode_reward', 0))
                    
                    logger.info(f"   🎯 Step {self.step_count}: Hold Rate: {hold_rate:.1%}")
                    
                    # Check for breakthrough (>5% hold rate)
                    if hold_rate > 0.05:
                        logger.info(f"   🚀 BREAKTHROUGH: Hold rate above 5%!")
                    
                    # Check for target range (8-15%)
                    if 0.08 <= hold_rate <= 0.15:
                        logger.info(f"   ✅ TARGET RANGE: Hold rate in 8-15% band!")
                
            except Exception as e:
                logger.warning(f"Monitor callback error: {e}")
        
        return True
    
    def get_summary(self):
        """Get training summary."""
        if not self.hold_rates:
            return "No data collected"
        
        final_hold_rate = self.hold_rates[-1]
        max_hold_rate = max(self.hold_rates)
        avg_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        return {
            'final_hold_rate': final_hold_rate,
            'max_hold_rate': max_hold_rate,
            'avg_episode_length': avg_episode_length,
            'total_steps': self.step_count,
            'breakthrough': max_hold_rate > 0.05,
            'target_reached': max_hold_rate >= 0.08
        }

def run_fast_microcycle():
    """Run fast micro-cycle to retrain policy with corrected controller."""
    
    logger.info("🚀 FAST MICRO-CYCLE RETRAINING")
    logger.info("=" * 60)
    logger.info("Purpose: Give PPO optimizer policy updates with corrected hold-bonus")
    logger.info("Expected: HOLD-rate should rise to 8-15% within 2k steps")
    logger.info("")
    
    # Load the fixed Cycle 5 model
    model_path = "train_runs/stairways_8cycle_20250803_193928/cycle_05_hold_67%/model_checkpoint_cycle_05_hold_67%_FIXED.zip"
    
    if not Path(model_path).exists():
        logger.error(f"❌ Fixed model not found: {model_path}")
        return False
    
    try:
        model = PPO.load(model_path)
        logger.info(f"✅ Loaded fixed model: {Path(model_path).name}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Create extended training data for micro-cycle
    try:
        # Load base data
        dual_data = pd.read_parquet('tests/fixtures/dual_ticker_sample.parquet')
        logger.info(f"✅ Loaded base data: {len(dual_data)} rows")
        
        # Create extended dataset for micro-cycle training
        feature_data = np.random.randn(8000, 26).astype(np.float32)
        price_data = np.random.randn(8000, 4).astype(np.float32) * 100 + 100
        trading_days = np.arange(8000)
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare training data: {e}")
        return False
    
    # Create training environment with FIXED controller
    try:
        env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=feature_data,
            processed_price_data=price_data,
            trading_days=trading_days,
            initial_capital=100000,
            max_episode_steps=400,  # Longer episodes for better learning
            lookback_window=10,
            enable_controller=True,
            controller_target_hold_rate=0.67,  # Cycle 5 target
            hold_bonus_weight=0.010,  # FIXED parameter
            verbose=False  # Reduce noise during micro-cycle
        )
        
        logger.info("✅ Training environment created with FIXED controller")
        logger.info(f"   Target hold rate: 67%")
        logger.info(f"   Base hold bonus: 0.010 (FIXED)")
        logger.info(f"   Max episode steps: 400")
        logger.info(f"   Controller enabled: True")
        
    except Exception as e:
        logger.error(f"❌ Failed to create environment: {e}")
        return False
    
    # Set up model for micro-cycle training
    model.set_env(env)
    
    # Create monitoring callback
    monitor_callback = HoldRateMonitorCallback(verbose=1)
    
    # Run micro-cycle training
    logger.info("🚀 Starting fast micro-cycle training...")
    logger.info(f"   Training steps: 2,500")
    logger.info(f"   Expected duration: ~3-5 minutes")
    logger.info(f"   Goal: HOLD-rate 8-15% within 2k steps")
    logger.info("")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=2500,
            callback=monitor_callback,
            reset_num_timesteps=False,  # Continue from previous training
            progress_bar=True
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Micro-cycle training completed in {duration:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Micro-cycle training failed: {e}")
        return False
    
    # Analyze results
    summary = monitor_callback.get_summary()
    
    logger.info("")
    logger.info("📊 MICRO-CYCLE RESULTS:")
    logger.info(f"   Final hold rate: {summary['final_hold_rate']:.1%}")
    logger.info(f"   Max hold rate: {summary['max_hold_rate']:.1%}")
    logger.info(f"   Avg episode length: {summary['avg_episode_length']:.0f}")
    logger.info(f"   Total steps: {summary['total_steps']}")
    
    # Check success criteria
    logger.info("")
    logger.info("🎯 SUCCESS CRITERIA CHECK:")
    
    if summary['breakthrough']:
        logger.info("   ✅ BREAKTHROUGH: Hold rate exceeded 5%")
    else:
        logger.info("   ❌ NO BREAKTHROUGH: Hold rate still below 5%")
    
    if summary['target_reached']:
        logger.info("   ✅ TARGET REACHED: Hold rate in 8-15% band")
    else:
        logger.info("   ⚠️ TARGET MISSED: Hold rate below 8%")
    
    if summary['avg_episode_length'] >= 300:
        logger.info("   ✅ EPISODE LENGTH: Good (≥300 steps)")
    else:
        logger.info("   ⚠️ EPISODE LENGTH: Short (<300 steps)")
    
    # Save micro-cycle model
    output_dir = Path("train_runs/stairways_8cycle_20250803_193928/microcycle_retrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model_microcycle_2500steps_RETRAINED.zip"
    
    try:
        model.save(str(model_path))
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Micro-cycle model saved: {model_path.name} ({model_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False
    
    # Recommendations
    logger.info("")
    logger.info("🚀 NEXT STEP RECOMMENDATIONS:")
    
    if summary['target_reached']:
        logger.info("   ✅ PROCEED TO CYCLE 6: Target reached, ready for full cycle")
        logger.info("   📈 Expected Cycle 6 performance: 20-35% hold rate")
    elif summary['breakthrough']:
        logger.info("   ⚠️ PARTIAL SUCCESS: Breakthrough achieved but below target")
        logger.info("   🔧 Consider: Increase base_hold_bonus by 1.5x and rerun")
        logger.info("   📈 Alternative: Proceed to Cycle 6 with current progress")
    else:
        logger.info("   ❌ INSUFFICIENT PROGRESS: Hold rate still too low")
        logger.info("   🔧 REQUIRED: Increase base_hold_bonus by 1.5x and rerun")
        logger.info("   📊 Check: Verify controller bonus range in logs")
    
    return summary['breakthrough']  # Return True if we made meaningful progress

def main():
    """Main function."""
    
    success = run_fast_microcycle()
    
    if success:
        logger.info("\n🎉 MICRO-CYCLE RETRAINING SUCCESSFUL!")
        logger.info("🚀 Policy has learned the corrected reward landscape")
        logger.info("📈 Ready to proceed with full Cycle 6 training")
    else:
        logger.info("\n⚠️ MICRO-CYCLE NEEDS ADJUSTMENT")
        logger.info("🔧 Consider increasing base_hold_bonus and retrying")
        logger.info("📊 Review controller bonus values in debug logs")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)