#!/usr/bin/env python3
"""
📊 50K TRAINING WITH PORTFOLIO MONITORING
Enhanced training script that shows portfolio values at key points
"""

import os
import sys  
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env_file()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training components
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

def _get_secure_db_password():
    """Get database password from secure vault with fallback"""
    try:
        from secrets_helper import SecretsHelper
        return SecretsHelper.get_timescaledb_password()
    except Exception as e:
        print(f"Could not get password from vault: {e}")
        return os.getenv('TIMESCALEDB_PASSWORD', 'password')

class PortfolioMonitoringCallback(BaseCallback):
    """Custom callback to monitor portfolio values during training"""
    
    def __init__(self, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.portfolio_history = []
        self.step_history = []
        
    def _on_step(self) -> bool:
        # Check every check_freq steps
        if self.n_calls % self.check_freq == 0:
            # Get current environment
            env = self.training_env.envs[0]  # Get the actual environment from VecEnv
            
            # Try to access portfolio value
            try:
                if hasattr(env, 'portfolio_value'):
                    portfolio_value = env.portfolio_value
                    peak_value = getattr(env, 'peak_portfolio_value', portfolio_value)
                    drawdown = ((peak_value - portfolio_value) / peak_value) if peak_value > 0 else 0.0
                    
                    self.portfolio_history.append(portfolio_value)
                    self.step_history.append(self.n_calls)
                    
                    logger.info(f"📊 STEP {self.n_calls:,}: Portfolio=${portfolio_value:,.2f}, Peak=${peak_value:,.2f}, DD={drawdown:.2%}")
                    
            except Exception as e:
                if self.verbose > 0:
                    logger.debug(f"Could not access portfolio value: {e}")
        
        return True
    
    def _on_training_start(self) -> None:
        logger.info("🚀 PORTFOLIO MONITORING STARTED")
        
    def _on_training_end(self) -> None:
        logger.info("🏁 PORTFOLIO MONITORING COMPLETED")
        if self.portfolio_history:
            initial_value = self.portfolio_history[0] if self.portfolio_history else 100000
            final_value = self.portfolio_history[-1] if self.portfolio_history else 100000
            total_return = (final_value - initial_value) / initial_value
            
            logger.info("📊 PORTFOLIO SUMMARY:")
            logger.info(f"   💰 Initial Portfolio: ${initial_value:,.2f}")
            logger.info(f"   💰 Final Portfolio: ${final_value:,.2f}")
            logger.info(f"   📈 Total Return: {total_return:+.2%}")
            logger.info(f"   📊 Monitoring Points: {len(self.portfolio_history)}")

def create_training_environment():
    """Create training environment with portfolio monitoring"""
    logger.info("📊 Creating PORTFOLIO MONITORING training environment...")
    
    # Try to connect to TimescaleDB first
    try:
        timescaledb_config = {
            'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
            'port': int(os.getenv('TIMESCALEDB_PORT', 5432)),
            'database': os.getenv('TIMESCALEDB_DATABASE', 'trading_data'),
            'user': os.getenv('TIMESCALEDB_USER', 'postgres'),
            'password': _get_secure_db_password()
        }
        
        logger.info("🔌 Attempting TimescaleDB connection...")
        data_adapter = DualTickerDataAdapter(**timescaledb_config)
        
        # Get training data
        nvda_data, msft_data, nvda_prices, msft_prices, trading_days = data_adapter.get_training_data(
            start_date='2024-01-01',
            end_date='2024-12-31',
            tickers=['NVDA', 'MSFT']
        )
        
        logger.info(f"✅ TimescaleDB data loaded: {len(trading_days)} periods")
        
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB failed ({e}), using LARGE mock dataset...")
        
        # Generate LARGE mock dataset for 50K training
        big_n_periods = 60000  # Much larger dataset
        big_trading_days = pd.date_range('2024-01-01', periods=big_n_periods, freq='1min')
        
        # Mock feature data with some realistic patterns
        big_nvda_data = np.random.randn(big_n_periods, 12).astype(np.float32)
        big_nvda_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
        
        # Mock price data with trend and volatility
        nvda_base_price = 170.0
        nvda_returns = np.random.normal(0.0001, 0.02, big_n_periods)  # Realistic returns
        nvda_prices = pd.Series(
            nvda_base_price * np.exp(np.cumsum(nvda_returns)),
            index=big_trading_days
        )
        
        msft_base_price = 510.0
        msft_returns = np.random.normal(0.0001, 0.015, big_n_periods)  # Realistic returns
        msft_prices = pd.Series(
            msft_base_price * np.exp(np.cumsum(msft_returns)),
            index=big_trading_days
        )
        big_msft_data = np.random.randn(big_n_periods, 12).astype(np.float32)
        big_msft_data[:, 0] = np.cumsum(np.random.randn(big_n_periods) * 0.01)  # Price-like feature
        
        # Use the big mock data
        nvda_data, msft_data = big_nvda_data, big_msft_data
        trading_days = big_trading_days
    
    # 📊 CREATE TRAINING ENVIRONMENT WITH ALL OPTIMIZATIONS
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,     # Starting capital
        tc_bp=1.0,                  # 📈 REDUCED: Lower transaction costs (10 bp)
        trade_penalty_bp=2.0,       # 📈 REDUCED: Lower trade penalty (20 bp)
        turnover_bp=2.0,            # 🔧 KEPT: Turnover penalty (20 bp)
        hold_action_bonus=0.01,     # 🔧 OPTIMIZED: Bonus for holding positions
        action_repeat_penalty=0.002, # 🔧 OPTIMIZED: Penalty for action changes
        high_water_mark_reward=0.001, # 🏆 NEW: High-water mark reward system
        daily_trade_limit=50,       # 🔧 OPTIMIZED: Daily trade limit
        reward_scaling=0.1,         # 🔧 OPTIMIZED: Better reward scaling
        training_drawdown_pct=0.07, # 🎓 NEW: 7% drawdown for training exploration
        evaluation_drawdown_pct=0.02, # 🛡️ NEW: 2% drawdown for evaluation/production
        is_training=True,           # 🎓 TRAINING MODE: Allow exploration
        log_trades=False            # Reduce logging spam
    )
    
    logger.info(f"✅ PORTFOLIO MONITORING Environment created")
    logger.info(f"📊 Initial Capital: ${env.initial_capital:,.2f}")
    logger.info(f"📊 Current Portfolio: ${env.portfolio_value:,.2f}")
    logger.info(f"📊 Peak Portfolio: ${env.peak_portfolio_value:,.2f}")
    logger.info("📈 ALL OPTIMIZATIONS APPLIED:")
    logger.info(f"   💰 Transaction Cost: {1.0} bp (REDUCED)")
    logger.info(f"   💰 Trade Penalty: {2.0} bp (REDUCED)") 
    logger.info(f"   💰 Turnover Penalty: {2.0} bp (KEPT)")
    logger.info(f"   🏆 High-Water Mark Reward: {0.001}")
    logger.info(f"   🎓 Training Drawdown: {7.0}% (exploration)")
    logger.info(f"   🛡️ Evaluation Drawdown: {2.0}% (strict)")
    
    return env

def create_evaluation_environment():
    """Create evaluation environment for monitoring"""
    logger.info("🛡️ Creating PORTFOLIO MONITORING evaluation environment...")
    
    # Generate smaller evaluation dataset
    eval_n_periods = 5000
    eval_trading_days = pd.date_range('2025-01-01', periods=eval_n_periods, freq='1min')
    
    # Mock evaluation data with different seed
    np.random.seed(12345)
    eval_nvda_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    eval_nvda_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    # Mock price data
    nvda_base_price = 175.0
    nvda_returns = np.random.normal(0.0001, 0.022, eval_n_periods)
    eval_nvda_prices = pd.Series(
        nvda_base_price * np.exp(np.cumsum(nvda_returns)),
        index=eval_trading_days
    )
    
    msft_base_price = 515.0
    msft_returns = np.random.normal(0.0001, 0.016, eval_n_periods)
    eval_msft_prices = pd.Series(
        msft_base_price * np.exp(np.cumsum(msft_returns)),
        index=eval_trading_days
    )
    eval_msft_data = np.random.randn(eval_n_periods, 12).astype(np.float32)
    eval_msft_data[:, 0] = np.cumsum(np.random.randn(eval_n_periods) * 0.01)
    
    # 🛡️ CREATE EVALUATION ENVIRONMENT
    env = DualTickerTradingEnv(
        nvda_data=eval_nvda_data,
        msft_data=eval_msft_data,
        nvda_prices=eval_nvda_prices,
        msft_prices=eval_msft_prices,
        trading_days=eval_trading_days,
        initial_capital=100000,     # Same as training
        tc_bp=1.0,                  # Same reduced friction as training
        trade_penalty_bp=2.0,       # Same reduced friction as training
        turnover_bp=2.0,            # Same turnover penalty as training
        hold_action_bonus=0.01,     # Same incentives as training
        action_repeat_penalty=0.002, # Same incentives as training
        high_water_mark_reward=0.001, # Same high-water mark system as training
        daily_trade_limit=50,       # Same limits as training
        reward_scaling=0.1,         # Same scaling as training
        training_drawdown_pct=0.07, # Not used in evaluation mode
        evaluation_drawdown_pct=0.02, # 🛡️ STRICT: 2% evaluation limit
        is_training=False,          # 🛡️ EVALUATION MODE: Strict risk controls
        log_trades=False            # Reduce logging spam
    )
    
    logger.info(f"✅ EVALUATION Environment created")
    logger.info(f"📊 Evaluation Initial Capital: ${env.initial_capital:,.2f}")
    
    return env

def create_model(env):
    """Create the RecurrentPPO model with optimized parameters"""
    logger.info("🧠 Creating PORTFOLIO MONITORING RecurrentPPO model...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=0.00015,      # Slightly lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,             # Tighter clipping for stability
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/",
        verbose=1,
        seed=42,
        device="auto"
    )
    
    logger.info("✅ PORTFOLIO MONITORING RecurrentPPO model created")
    return model

def setup_callbacks(eval_env):
    """Setup training callbacks with portfolio monitoring"""
    logger.info("📋 Setting up callbacks with PORTFOLIO MONITORING...")
    
    # Portfolio monitoring callback - check every 5K steps
    portfolio_callback = PortfolioMonitoringCallback(
        check_freq=5000,
        verbose=1
    )
    
    # Checkpoint callback - save every 10K steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dual_ticker_portfolio_monitored"
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5000,  # Evaluate every 5K steps
        deterministic=True,
        render=False,
        n_eval_episodes=3,  # Short evaluation episodes
        callback_on_new_best=None
    )
    
    callbacks = [portfolio_callback, checkpoint_callback, eval_callback]
    logger.info("✅ PORTFOLIO MONITORING Callbacks configured")
    return callbacks

def main():
    """Main training function with portfolio monitoring"""
    logger.info("📊 50K PORTFOLIO MONITORING TRAINING STARTED")
    logger.info("🎯 Goal: Monitor portfolio values throughout training")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create training environment
        train_env = create_training_environment()
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Log initial portfolio state
        logger.info("📊 INITIAL PORTFOLIO STATE:")
        logger.info(f"   💰 Starting Capital: ${train_env.initial_capital:,.2f}")
        logger.info(f"   📊 Current Portfolio: ${train_env.portfolio_value:,.2f}")
        logger.info(f"   🏔️ Peak Portfolio: ${train_env.peak_portfolio_value:,.2f}")
        
        # Step 2: Add VecNormalize for reward normalization
        logger.info("🔧 Adding VecNormalize for reward stability...")
        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False,      # Don't normalize observations (features are already scaled)
            norm_reward=True,    # 🔧 OPTIMIZED: Normalize rewards for better learning
            clip_reward=10.0,    # 🔧 OPTIMIZED: Clip extreme rewards
            gamma=0.99
        )
        
        # Step 3: Create evaluation environment
        eval_env = create_evaluation_environment()
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        # Step 4: Create model
        model = create_model(vec_env)
        
        # Step 5: Setup callbacks with portfolio monitoring
        callbacks = setup_callbacks(eval_vec_env)
        
        # Step 6: Train with portfolio monitoring
        logger.info("🎯 Starting 50K PORTFOLIO MONITORING training...")
        logger.info("📊 Portfolio values will be logged every 5,000 steps")
        logger.info("💾 Model checkpoints every 10,000 steps")
        logger.info("🛡️ Evaluation episodes every 5,000 steps")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=50000,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 7: Final portfolio check
        logger.info("📊 FINAL PORTFOLIO CHECK:")
        try:
            final_portfolio = train_env.portfolio_value
            final_peak = train_env.peak_portfolio_value
            final_drawdown = ((final_peak - final_portfolio) / final_peak) if final_peak > 0 else 0.0
            total_return = (final_portfolio - train_env.initial_capital) / train_env.initial_capital
            
            logger.info(f"   💰 Final Portfolio: ${final_portfolio:,.2f}")
            logger.info(f"   🏔️ Peak Portfolio: ${final_peak:,.2f}")
            logger.info(f"   📉 Final Drawdown: {final_drawdown:.2%}")
            logger.info(f"   📈 Total Return: {total_return:+.2%}")
            
        except Exception as e:
            logger.warning(f"Could not access final portfolio values: {e}")
        
        # Step 8: Save final model and VecNormalize
        model_path = "models/dual_ticker_portfolio_monitored_50k_final.zip"
        vecnorm_path = "models/dual_ticker_portfolio_monitored_50k_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)  # Save VecNormalize statistics
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 PORTFOLIO MONITORING TRAINING COMPLETED!")
        logger.info(f"⏱️ Duration: {duration}")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        logger.info("✅ PORTFOLIO MONITORING SYSTEM SUCCESSFULLY COMPLETED")
        
    except Exception as e:
        logger.error(f"❌ PORTFOLIO MONITORING Training failed: {e}")
        raise

if __name__ == "__main__":
    main()