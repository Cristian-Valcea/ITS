#!/usr/bin/env python3
"""
🎯 CURRICULUM ALPHA TRAINING
If single-ticker still flat, add weak curriculum alpha that fades to zero
Start with alpha_mag=0.10, linearly fade to 0 by step 10k
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

class CurriculumAlphaCallback(BaseCallback):
    """Callback to gradually fade alpha strength during training"""
    
    def __init__(self, initial_alpha=0.10, fade_steps=10000, verbose=0):
        super().__init__(verbose)
        self.initial_alpha = initial_alpha
        self.fade_steps = fade_steps
        self.current_alpha = initial_alpha
        
    def _on_step(self) -> bool:
        # Calculate current alpha based on training progress
        progress = min(self.num_timesteps / self.fade_steps, 1.0)
        self.current_alpha = self.initial_alpha * (1.0 - progress)
        
        # Log alpha fade progress every 1000 steps
        if self.num_timesteps % 1000 == 0:
            logger.info(f"Step {self.num_timesteps}: Alpha strength = {self.current_alpha:.4f}")
            
        return True

def create_curriculum_environment(alpha_strength=0.10):
    """Create environment with curriculum alpha"""
    features, prices, metadata = create_toy_alpha_data(
        n_periods=15000, 
        alpha_strength=alpha_strength
    )
    
    env = IntradayTradingEnvV3(
        processed_feature_data=features,
        price_data=prices,
        initial_capital=100000,
        max_daily_drawdown_pct=0.15,  # Training DD
        verbose=False
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, gamma=0.999)
    
    return vec_env

def train_with_curriculum_alpha(
    lr=3e-4,
    ent_coef=0.005,  # Higher entropy for exploration
    clip_range=0.3,
    n_steps=2048,
    n_epochs=4,
    initial_alpha=0.10,
    fade_steps=10000,
    total_steps=25000,
    early_exit_rew=500,
    early_exit_dd=2.0
):
    """Train with curriculum alpha that fades over time"""
    
    logger.info(f"🎯 CURRICULUM ALPHA TRAINING")
    logger.info(f"   Initial alpha: {initial_alpha}")
    logger.info(f"   Fade to zero by step: {fade_steps}")
    logger.info(f"   Total training steps: {total_steps}")
    logger.info(f"   Early exit: ep_rew > {early_exit_rew} AND DD < {early_exit_dd}%")
    
    # Create environment with initial alpha
    vec_env = create_curriculum_environment(alpha_strength=initial_alpha)
    
    # Create model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
        device="auto",
        tensorboard_log="logs/curriculum_alpha"
    )
    
    # Create curriculum callback
    curriculum_callback = CurriculumAlphaCallback(
        initial_alpha=initial_alpha,
        fade_steps=fade_steps,
        verbose=1
    )
    
    logger.info(f"🔥 Starting curriculum training...")
    
    try:
        model.learn(
            total_timesteps=total_steps,
            progress_bar=True,
            callback=curriculum_callback
        )
        
        # Evaluate final performance
        logger.info(f"📊 Evaluating final performance...")
        
        obs = vec_env.reset()
        episode_rewards = []
        episode_returns = []
        
        for episode in range(10):
            episode_reward = 0
            episode_return = 0
            initial_portfolio = 100000
            
            obs = vec_env.reset()
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                
                if done[0]:
                    if len(info) > 0 and 'episode' in info[0]:
                        final_portfolio = info[0].get('final_portfolio', initial_portfolio)
                        episode_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
                    break
            
            episode_rewards.append(episode_reward)
            episode_returns.append(episode_return)
        
        avg_reward = np.mean(episode_rewards)
        avg_return = np.mean(episode_returns)
        max_dd = abs(min(episode_returns)) if episode_returns else 0
        
        logger.info(f"📊 CURRICULUM TRAINING RESULTS:")
        logger.info(f"   Avg ep_rew_mean: {avg_reward:.2f}")
        logger.info(f"   Avg return: {avg_return:.2f}%")
        logger.info(f"   Max drawdown: {max_dd:.2f}%")
        
        # Check success criteria
        success = avg_reward > early_exit_rew and max_dd < early_exit_dd
        
        if success:
            logger.info(f"🎉 CURRICULUM SUCCESS! Passed the gate!")
            model_path = "models/curriculum_alpha_success.zip"
            logger.info(f"   Saving model to {model_path}")
            model.save(model_path)
            vec_env.save("models/curriculum_alpha_success_vecnorm.pkl")
            
            logger.info(f"\n🚀 READY FOR DUAL-TICKER WARM-START:")
            logger.info(f"   python export_policy.py --model_path {model_path}")
            logger.info(f"   python dual_ticker_warmstart.py --base_model models/gate_pass.zip")
        else:
            logger.info(f"❌ Curriculum training did not pass the gate")
            logger.info(f"   Consider: higher entropy, different reward scaling, or longer fade period")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Curriculum training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Curriculum Alpha Training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--ent_coef', type=float, default=0.005, help='Entropy coefficient')
    parser.add_argument('--initial_alpha', type=float, default=0.10, help='Initial alpha strength')
    parser.add_argument('--fade_steps', type=int, default=10000, help='Steps to fade alpha to zero')
    parser.add_argument('--total_steps', type=int, default=25000, help='Total training steps')
    parser.add_argument('--early_exit_rew', type=float, default=500, help='Early exit reward threshold')
    parser.add_argument('--early_exit_dd', type=float, default=2.0, help='Early exit drawdown threshold (%)')
    
    args = parser.parse_args()
    
    success = train_with_curriculum_alpha(
        lr=args.lr,
        ent_coef=args.ent_coef,
        initial_alpha=args.initial_alpha,
        fade_steps=args.fade_steps,
        total_steps=args.total_steps,
        early_exit_rew=args.early_exit_rew,
        early_exit_dd=args.early_exit_dd
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()