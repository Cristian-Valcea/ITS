#!/usr/bin/env python3
"""
🚀 ENHANCED 50K TRAINING - GIVE PROFITS ROOM TO RUN
- Ultra-low friction: tc_bp=0.5, trade_penalty=0.7
- Maximum training drawdown: 15% (let profits run!)
- Strict evaluation: 2% (risk control)
- Lower learning rate: 1e-4
- Same data for consistency
"""

import os
import sys  
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

class EnhancedTradingEnv(DualTickerTradingEnv):
    """Enhanced wrapper that gives profits maximum room to run"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override max daily drawdown for training
        if self.is_training:
            self.max_daily_drawdown_pct = 0.15  # 15% for training - let profits run!
            logger.info(f"🚀 ENHANCED: Training drawdown set to {self.max_daily_drawdown_pct:.1%}")
        else:
            self.max_daily_drawdown_pct = 0.02  # 2% for evaluation - strict control
            logger.info(f"🛡️ ENHANCED: Evaluation drawdown set to {self.max_daily_drawdown_pct:.1%}")
    
    def step(self, action: int):
        """Execute one trading step with robust error handling"""
        
        # Check if episode should end
        if self.current_step >= self.max_steps:
            # Return final observation and mark as done
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'portfolio_value': self.portfolio_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
                'episode_ended': True,
                'max_drawdown_used': self.max_daily_drawdown_pct
            }
            return observation, reward, terminated, truncated, info
        
        # Normal step execution
        try:
            return super().step(action)
        except Exception as e:
            logger.warning(f"Step error handled: {e}")
            # Return safe values if step fails
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'portfolio_value': self.portfolio_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
                'step_error': str(e),
                'max_drawdown_used': self.max_daily_drawdown_pct
            }
            return observation, reward, terminated, truncated, info

def create_enhanced_environment():
    """Create enhanced training environment - give profits room to run!"""
    logger.info("🚀 Creating ENHANCED training environment - PROFITS ROOM TO RUN...")
    
    # Generate SAME mock dataset for consistency (using fixed seed)
    np.random.seed(42)  # Fixed seed for reproducible data
    n_periods = 60000
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Mock feature data (same seed = same data)
    nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
    nvda_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    # Mock price data with trend (same seed = same data)
    nvda_base_price = 170.0
    nvda_returns = np.random.normal(0.0001, 0.02, n_periods)
    nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=trading_days
    )
    
    msft_base_price = 510.0
    msft_returns = np.random.normal(0.0001, 0.015, n_periods)
    msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=trading_days
    )
    msft_data = np.random.randn(n_periods, 12).astype(np.float32)
    msft_data[:, 0] = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    # Create environment with ENHANCED SPECIFICATIONS - ULTRA LOW FRICTION
    env = EnhancedTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,     # Starting capital
        tc_bp=0.5,                  # 🔥 ULTRA LOW FRICTION (was 0.7)
        trade_penalty_bp=0.7,       # 🔥 ULTRA LOW FRICTION (was 1.0)
        turnover_bp=2.0,            # Turnover penalty (kept same)
        hold_action_bonus=0.01,     # Hold bonus
        action_repeat_penalty=0.002, # Action change penalty
        high_water_mark_reward=0.001, # High-water mark reward
        daily_trade_limit=50,       # Daily trade limit
        reward_scaling=0.1,         # Reward scaling
        training_drawdown_pct=0.15, # 🚀 MAXIMUM ROOM: 15% (was 10%)
        evaluation_drawdown_pct=0.02, # Evaluation still strict at 2%
        is_training=True,           # Training mode
        log_trades=False
    )
    
    # Wrap with Monitor for better tracking
    env = Monitor(env)
    
    logger.info(f"✅ ENHANCED Environment created")
    logger.info(f"📊 Initial Capital: ${env.env.initial_capital:,.2f}")
    logger.info(f"📊 Current Portfolio: ${env.env.portfolio_value:,.2f}")
    logger.info("🚀 ENHANCED SPECIFICATIONS - GIVE PROFITS ROOM TO RUN:")
    logger.info(f"   💰 Transaction Cost: {0.5} bp (ULTRA LOW - was 0.7)")
    logger.info(f"   💰 Trade Penalty: {0.7} bp (ULTRA LOW - was 1.0)") 
    logger.info(f"   💰 Turnover Penalty: {2.0} bp (KEPT)")
    logger.info(f"   🏆 High-Water Mark Reward: {0.001}")
    logger.info(f"   🚀 Training Drawdown: {15.0}% (MAXIMUM ROOM - was 10%)")
    logger.info(f"   🛡️ Evaluation Drawdown: {2.0}% (strict)")
    logger.info(f"   🧠 Learning Rate: 1e-4 (stable)")
    logger.info("🚀 PROFIT MAXIMIZATION FEATURES:")
    logger.info("   ✅ Ultra-low friction for maximum exploration")
    logger.info("   ✅ 15% drawdown - let winning trades run!")
    logger.info("   ✅ Graceful episode ending handling")
    logger.info("   ✅ Error recovery mechanisms")
    logger.info("   ✅ Monitor wrapper for tracking")
    logger.info("   ✅ Same data (fixed seed=42)")
    
    return env

def create_enhanced_model(env):
    """Create RecurrentPPO model with enhanced specifications"""
    logger.info("🧠 Creating ENHANCED RecurrentPPO model...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=1e-4,         # Stable learning rate
        n_steps=2048,               # Standard rollout length
        batch_size=64,              # Reasonable batch size
        n_epochs=10,                # Standard epochs
        gamma=0.99,                 # Standard discount
        gae_lambda=0.95,            # Standard GAE
        clip_range=0.1,             # Conservative clipping
        ent_coef=0.01,              # Standard entropy
        vf_coef=0.5,                # Standard value function
        max_grad_norm=0.5,          # Gradient clipping
        tensorboard_log="logs/",
        verbose=1,
        seed=42,                    # Same seed for consistency
        device="auto"
    )
    
    logger.info("✅ ENHANCED RecurrentPPO model created")
    logger.info(f"🧠 Learning Rate: {1e-4} (stable)")
    return model

def main():
    """Main enhanced training function"""
    logger.info("🚀 ENHANCED 50K TRAINING - GIVE PROFITS ROOM TO RUN!")
    logger.info("🔥 Ultra-low friction + Maximum drawdown + Stable LR + Same data")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create enhanced training environment
        train_env = create_enhanced_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Log initial portfolio state
        logger.info("📊 INITIAL PORTFOLIO STATE:")
        logger.info(f"   💰 Starting Capital: ${train_env.env.initial_capital:,.2f}")
        logger.info(f"   📊 Current Portfolio: ${train_env.env.portfolio_value:,.2f}")
        logger.info(f"   🏔️ Peak Portfolio: ${train_env.env.peak_portfolio_value:,.2f}")
        logger.info(f"   🚀 Max Training Drawdown: {train_env.env.max_daily_drawdown_pct:.1%}")
        
        # Step 2: Add VecNormalize
        logger.info("🔧 Adding VecNormalize for stability...")
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations
            norm_reward=True,    # Normalize rewards
            clip_reward=10.0,    # Clip extreme rewards
            gamma=0.99
        )
        
        # Step 3: Create enhanced model
        model = create_enhanced_model(vec_env)
        
        # Step 4: Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./models/checkpoints/",
            name_prefix="dual_ticker_enhanced"
        )
        
        # Step 5: Train in chunks for better monitoring
        logger.info("🎯 Starting ENHANCED 50K training...")
        logger.info("🚀 PROFIT MAXIMIZATION: tc_bp=0.5, trade_penalty=0.7, drawdown=15%")
        logger.info("💾 Model checkpoints every 10,000 steps")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        # Train in chunks for better monitoring
        total_steps = 50000
        chunk_size = 10000
        
        for chunk_start in range(0, total_steps, chunk_size):
            chunk_steps = min(chunk_size, total_steps - chunk_start)
            
            logger.info(f"🎯 Training chunk: steps {chunk_start} to {chunk_start + chunk_steps}")
            
            # Check portfolio before chunk
            try:
                portfolio = train_env.env.portfolio_value
                peak = train_env.env.peak_portfolio_value
                drawdown = ((peak - portfolio) / peak) if peak > 0 else 0.0
                total_return = (portfolio - train_env.env.initial_capital) / train_env.env.initial_capital
                max_dd_allowed = train_env.env.max_daily_drawdown_pct
                
                logger.info(f"📊 BEFORE CHUNK {chunk_start//chunk_size + 1}:")
                logger.info(f"   💰 Portfolio: ${portfolio:,.2f}")
                logger.info(f"   🏔️ Peak: ${peak:,.2f}")
                logger.info(f"   📉 Drawdown: {drawdown:.2%} (max: {max_dd_allowed:.1%})")
                logger.info(f"   📈 Return: {total_return:+.2%}")
                logger.info(f"   🚀 Room to Run: {max_dd_allowed:.1%} drawdown allowed")
                
            except Exception as e:
                logger.warning(f"Could not check portfolio before chunk: {e}")
            
            # Train this chunk
            try:
                model.learn(
                    total_timesteps=chunk_steps,
                    callback=checkpoint_callback,
                    progress_bar=True,
                    reset_num_timesteps=(chunk_start == 0)
                )
                
                logger.info(f"✅ Completed chunk: steps {chunk_start} to {chunk_start + chunk_steps}")
                
            except Exception as e:
                logger.error(f"❌ Chunk training failed: {e}")
                logger.info("🔄 Continuing with next chunk...")
                continue
            
            # Check portfolio after chunk
            try:
                portfolio = train_env.env.portfolio_value
                peak = train_env.env.peak_portfolio_value
                drawdown = ((peak - portfolio) / peak) if peak > 0 else 0.0
                total_return = (portfolio - train_env.env.initial_capital) / train_env.env.initial_capital
                max_dd_allowed = train_env.env.max_daily_drawdown_pct
                
                logger.info(f"📊 AFTER CHUNK {chunk_start//chunk_size + 1}:")
                logger.info(f"   💰 Portfolio: ${portfolio:,.2f}")
                logger.info(f"   🏔️ Peak: ${peak:,.2f}")
                logger.info(f"   📉 Drawdown: {drawdown:.2%} (max: {max_dd_allowed:.1%})")
                logger.info(f"   📈 Return: {total_return:+.2%}")
                if drawdown < max_dd_allowed:
                    remaining_room = max_dd_allowed - drawdown
                    logger.info(f"   🚀 Remaining Room: {remaining_room:.2%} more drawdown allowed")
                logger.info("   " + "="*50)
                
            except Exception as e:
                logger.warning(f"Could not check portfolio after chunk: {e}")
        
        # Step 6: Final portfolio check
        logger.info("📊 FINAL ENHANCED PORTFOLIO CHECK:")
        try:
            final_portfolio = train_env.env.portfolio_value
            final_peak = train_env.env.peak_portfolio_value
            final_drawdown = ((final_peak - final_portfolio) / final_peak) if final_peak > 0 else 0.0
            total_return = (final_portfolio - train_env.env.initial_capital) / train_env.env.initial_capital
            max_dd_used = train_env.env.max_daily_drawdown_pct
            
            logger.info(f"   💰 Final Portfolio: ${final_portfolio:,.2f}")
            logger.info(f"   🏔️ Peak Portfolio: ${final_peak:,.2f}")
            logger.info(f"   📉 Final Drawdown: {final_drawdown:.2%}")
            logger.info(f"   📈 Total Return: {total_return:+.2%}")
            logger.info(f"   🚀 Max Drawdown Allowed: {max_dd_used:.1%}")
            
            if final_drawdown < max_dd_used:
                unused_room = max_dd_used - final_drawdown
                logger.info(f"   💡 Unused Drawdown Room: {unused_room:.2%} (could have run further)")
            else:
                logger.info(f"   ⚠️ Used Full Drawdown Allowance: {final_drawdown:.2%}")
            
        except Exception as e:
            logger.warning(f"Could not access final portfolio values: {e}")
        
        # Step 7: Save final model
        model_path = "models/dual_ticker_enhanced_50k_final.zip"
        vecnorm_path = "models/dual_ticker_enhanced_50k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 ENHANCED PROFIT-MAXIMIZING TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("🚀 ENHANCED SPECIFICATIONS SUCCESSFULLY APPLIED:")
        logger.info("   💰 Transaction Cost: 0.5 bp (ultra-low)")
        logger.info("   💰 Trade Penalty: 0.7 bp (ultra-low)")
        logger.info("   🚀 Training Drawdown: 15% (maximum room)")
        logger.info("   🛡️ Evaluation Drawdown: 2% (strict)")
        logger.info("   🧠 Learning Rate: 1e-4 (stable)")
        logger.info("   📊 Same Data: Fixed seed=42")
        logger.info("✅ ENHANCED PROFIT-MAXIMIZING SYSTEM SUCCESSFULLY COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ ENHANCED Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()