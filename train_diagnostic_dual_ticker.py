#!/usr/bin/env python3
"""
🔧 Diagnostic Dual-Ticker Training Script
Short 3K step training run to validate optimizations

Usage:
python train_diagnostic_dual_ticker.py \
    --total-timesteps 3000 \
    --vecnorm true \
    --tc-bp 2.0 --trade-penalty-bp 5.0 --turnover-bp 1.0
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load environment variables
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our dual-ticker components
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diagnostic Dual-Ticker Training')
    parser.add_argument('--total-timesteps', type=int, default=3000, help='Total training steps')
    parser.add_argument('--vecnorm', type=str, default='true', help='Use VecNormalize (true/false)')
    parser.add_argument('--tc-bp', type=float, default=2.0, help='Transaction cost basis points')
    parser.add_argument('--trade-penalty-bp', type=float, default=5.0, help='Trade penalty basis points')
    parser.add_argument('--turnover-bp', type=float, default=1.0, help='Turnover penalty basis points')
    parser.add_argument('--learning-rate', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--clip-range', type=float, default=0.1, help='PPO clip range')
    return parser.parse_args()

def create_training_environment(tc_bp=2.0, trade_penalty_bp=5.0, turnover_bp=1.0):
    """Create the dual-ticker training environment with specified parameters"""
    logger.info(f"🏗️ Creating diagnostic environment (tc_bp={tc_bp}, trade_penalty_bp={trade_penalty_bp}, turnover_bp={turnover_bp})...")
    
    # Try to connect to TimescaleDB first
    try:
        timescaledb_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_data',
            'user': 'postgres',
            'password': os.getenv('TIMESCALE_PASSWORD')
        }
        data_adapter = DualTickerDataAdapter(
            timescaledb_config=timescaledb_config,
            live_trading_mode=False
        )
        
        # Get training data
        nvda_data, msft_data, nvda_prices, msft_prices, trading_days = data_adapter.get_training_data(
            start_date='2024-01-01',
            end_date='2024-01-31',  # Shorter period for diagnostic
            bar_size='1min'
        )
        
        logger.info(f"✅ TimescaleDB data loaded: {len(trading_days)} bars")
        
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB failed ({e}), using mock data...")
        
        # Generate mock data for diagnostic
        n_periods = 5000  # Longer to support 3K training steps
        trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
        
        # Mock feature data
        nvda_data = np.random.randn(n_periods, 12).astype(np.float32)
        msft_data = np.random.randn(n_periods, 12).astype(np.float32)
        
        # Mock price data with some trend
        nvda_prices = pd.Series(
            [170.0 + i * 0.01 + np.random.normal(0, 0.5) for i in range(n_periods)], 
            index=trading_days
        )
        msft_prices = pd.Series(
            [510.0 + i * 0.005 + np.random.normal(0, 1.0) for i in range(n_periods)], 
            index=trading_days
        )
    
    # Create environment with specified parameters
    env = DualTickerTradingEnv(
        nvda_data=nvda_data,
        msft_data=msft_data,
        nvda_prices=nvda_prices,
        msft_prices=msft_prices,
        trading_days=trading_days,
        initial_capital=100000,
        tc_bp=tc_bp,
        trade_penalty_bp=trade_penalty_bp,
        turnover_bp=turnover_bp,
        reward_scaling=0.1,
        log_trades=True
    )
    
    return env

def create_model(vec_env, learning_rate=1.5e-4, clip_range=0.1):
    """Create RecurrentPPO model with optimized parameters"""
    logger.info(f"🤖 Creating RecurrentPPO model (lr={learning_rate}, clip={clip_range})...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="logs/",
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            lstm_kwargs=dict(dropout=0.0, batch_first=True)
        ),
        verbose=1,
        seed=42,
        device="auto"
    )
    
    return model

def main():
    """Main diagnostic training loop"""
    args = parse_args()
    
    logger.info("🔧 Starting Diagnostic Dual-Ticker Training")
    logger.info(f"📊 Parameters: steps={args.total_timesteps}, vecnorm={args.vecnorm}")
    logger.info(f"💰 Friction: tc_bp={args.tc_bp}, trade_penalty_bp={args.trade_penalty_bp}, turnover_bp={args.turnover_bp}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create environment
        train_env = create_training_environment(
            tc_bp=args.tc_bp,
            trade_penalty_bp=args.trade_penalty_bp,
            turnover_bp=args.turnover_bp
        )
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Step 1.5: Add VecNormalize if requested
        if args.vecnorm.lower() == 'true':
            logger.info("🔧 Adding VecNormalize wrapper...")
            vec_env = VecNormalize(
                vec_env, 
                norm_obs=False,
                norm_reward=True,
                clip_reward=10.0
            )
        
        # Step 2: Create model
        model = create_model(
            vec_env, 
            learning_rate=args.learning_rate,
            clip_range=args.clip_range
        )
        
        # Step 3: Start training
        logger.info(f"🎯 Starting {args.total_timesteps} diagnostic steps...")
        logger.info("📊 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 4: Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/diagnostic_dual_ticker_{timestamp}.zip"
        model.save(model_path)
        
        if args.vecnorm.lower() == 'true':
            vecnorm_path = f"models/diagnostic_dual_ticker_{timestamp}_vecnormalize.pkl"
            vec_env.save(vecnorm_path)
            logger.info(f"📊 VecNormalize saved: {vecnorm_path}")
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("✅ Diagnostic training completed!")
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"⏱️ Duration: {training_duration}")
        
        # Step 5: Quick evaluation
        logger.info("🔍 Running quick evaluation...")
        obs = vec_env.reset()
        total_reward = 0
        steps = 0
        trades = 0
        
        for _ in range(100):  # Quick 100-step eval
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if info[0].get('total_trades', 0) > trades:
                trades = info[0]['total_trades']
            
            if done[0]:
                break
        
        logger.info(f"📈 Eval Results: {steps} steps, {trades} trades, {total_reward:.3f} total reward")
        logger.info(f"📊 Trade frequency: {trades/steps:.3f} trades/step")
        
        # Check diagnostic gates
        logger.info("🚦 Checking Diagnostic Gates:")
        trade_rate = trades / steps if steps > 0 else 0
        avg_reward = total_reward / steps if steps > 0 else 0
        
        logger.info(f"   Trades/1000 steps: {trade_rate * 1000:.1f} (target: <100)")
        logger.info(f"   Episode reward mean: {avg_reward:.3f} (target: ≥0)")
        
        if trade_rate * 1000 < 100:
            logger.info("✅ PASS: Trade frequency under control")
        else:
            logger.warning("❌ FAIL: Trade frequency too high")
            
        if avg_reward >= 0:
            logger.info("✅ PASS: Positive average reward")
        else:
            logger.warning("❌ FAIL: Negative average reward")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()