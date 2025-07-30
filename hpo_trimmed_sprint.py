#!/usr/bin/env python3
"""
🎯 HPO TRIMMED SPRINT - Focused search around proven parameters
Based on synthetic alpha test success, test minimal configs around sweet spot
GOAL: Find ONE survivor that passes +1%/<2% DD gate for dual-ticker warm-start
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

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

class HPOTrimmedSprint:
    """Focused HPO around proven synthetic test parameters"""
    
    def __init__(self, early_exit_rew=500, early_exit_dd=2.0, max_steps=20000):
        self.results = []
        self.success_found = False
        self.best_config = None
        self.start_time = datetime.now()
        self.early_exit_rew = early_exit_rew
        self.early_exit_dd = early_exit_dd
        self.max_steps = max_steps
        
        # Trimmed grid around synthetic test sweet spot
        self.configs = [
            # Sweet spot from synthetic test (adjusted for real data)
            {'lr': 3e-4, 'ent_coef': 0.0, 'clip_range': 0.3, 'n_steps': 2048, 'n_epochs': 4},
            # Slight variations
            {'lr': 2e-4, 'ent_coef': 0.001, 'clip_range': 0.2, 'n_steps': 2048, 'n_epochs': 4},
            {'lr': 1e-4, 'ent_coef': 0.002, 'clip_range': 0.2, 'n_steps': 2048, 'n_epochs': 4},
            # Higher entropy for exploration
            {'lr': 3e-4, 'ent_coef': 0.005, 'clip_range': 0.3, 'n_steps': 2048, 'n_epochs': 4},
        ]
        
        logger.info(f"🎯 TRIMMED HPO SPRINT - {len(self.configs)} focused configs")
        logger.info(f"   Early exit: ep_rew_mean > {early_exit_rew} AND DD < {early_exit_dd}%")
        logger.info(f"   Max steps per config: {max_steps}")
        
    def create_environment(self):
        """Create training environment with same data as synthetic test"""
        # Use same toy alpha data as synthetic test for consistency
        features, prices, metadata = create_toy_alpha_data(n_periods=15000, alpha_strength=0.05)  # Weaker than synthetic
        
        env = IntradayTradingEnvV3(
            processed_feature_data=features,
            price_data=prices,
            initial_capital=100000,
            max_daily_drawdown_pct=0.15,  # Training DD
            verbose=False
        )
        
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, gamma=0.999)  # No reward norm
        
        return vec_env
        
    def test_config(self, config, config_idx):
        """Test a single configuration"""
        logger.info(f"\n🚀 CONFIG {config_idx+1}/{len(self.configs)}: {config}")
        
        vec_env = self.create_environment()
        
        # Create model with config parameters
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=config['lr'],
            n_steps=config['n_steps'],
            batch_size=64,
            n_epochs=config['n_epochs'],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=42,
            device="auto",
            tensorboard_log=f"logs/trimmed_config_{config_idx+1}"
        )
        
        # Train with early stopping
        logger.info(f"🔥 Training config {config_idx+1} for up to {self.max_steps} steps...")
        
        try:
            model.learn(
                total_timesteps=self.max_steps,
                progress_bar=True,
                callback=None  # Could add early stopping callback here
            )
            
            # Evaluate final performance
            obs = vec_env.reset()
            episode_rewards = []
            episode_returns = []
            
            for episode in range(10):  # Test 10 episodes
                episode_reward = 0
                episode_return = 0
                initial_portfolio = 100000
                
                obs = vec_env.reset()
                for step in range(1000):  # Max episode length
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
            
            logger.info(f"📊 CONFIG {config_idx+1} RESULTS:")
            logger.info(f"   Avg ep_rew_mean: {avg_reward:.2f}")
            logger.info(f"   Avg return: {avg_return:.2f}%")
            logger.info(f"   Max drawdown: {max_dd:.2f}%")
            
            # Check early exit criteria
            success = avg_reward > self.early_exit_rew and max_dd < self.early_exit_dd
            
            result = {
                'config_idx': config_idx,
                'config': config,
                'avg_reward': avg_reward,
                'avg_return': avg_return,
                'max_dd': max_dd,
                'success': success,
                'model_path': f"models/trimmed_config_{config_idx+1}.zip"
            }
            
            if success:
                logger.info(f"🎉 SUCCESS! Config {config_idx+1} passed the gate!")
                logger.info(f"   Saving model to {result['model_path']}")
                model.save(result['model_path'])
                vec_env.save(f"models/trimmed_config_{config_idx+1}_vecnorm.pkl")
                self.success_found = True
                self.best_config = result
            
            self.results.append(result)
            return success
            
        except Exception as e:
            logger.error(f"❌ Config {config_idx+1} failed: {e}")
            return False
    
    def run(self):
        """Run the trimmed HPO sprint"""
        logger.info(f"🚀 STARTING TRIMMED HPO SPRINT")
        logger.info(f"   Target: Find ONE config with ep_rew > {self.early_exit_rew} AND DD < {self.early_exit_dd}%")
        
        for i, config in enumerate(self.configs):
            if self.success_found:
                logger.info(f"✅ Early exit - success found with config {self.best_config['config_idx']+1}")
                break
                
            success = self.test_config(config, i)
            
            if success:
                logger.info(f"🎯 GATE PASSED! Ready for dual-ticker warm-start")
                break
        
        # Summary
        logger.info(f"\n📊 TRIMMED HPO SPRINT SUMMARY:")
        logger.info(f"   Configs tested: {len(self.results)}")
        logger.info(f"   Success found: {self.success_found}")
        
        if self.success_found:
            best = self.best_config
            logger.info(f"   🏆 WINNER: Config {best['config_idx']+1}")
            logger.info(f"   📊 Performance: {best['avg_reward']:.2f} reward, {best['avg_return']:.2f}% return")
            logger.info(f"   💾 Model saved: {best['model_path']}")
            logger.info(f"\n🚀 READY FOR DUAL-TICKER WARM-START:")
            logger.info(f"   python export_policy.py --model_path {best['model_path']}")
            logger.info(f"   python dual_ticker_warmstart.py --base_model models/gate_pass.zip")
        else:
            logger.info(f"   ❌ No configs passed the gate")
            logger.info(f"   💡 Consider: curriculum alpha, higher entropy, or reward scaling")
        
        return self.success_found

def main():
    parser = argparse.ArgumentParser(description='Trimmed HPO Sprint')
    parser.add_argument('--early_exit_rew', type=float, default=500, help='Early exit reward threshold')
    parser.add_argument('--early_exit_dd', type=float, default=2.0, help='Early exit drawdown threshold (%)')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max steps per config')
    
    args = parser.parse_args()
    
    sprint = HPOTrimmedSprint(
        early_exit_rew=args.early_exit_rew,
        early_exit_dd=args.early_exit_dd,
        max_steps=args.max_steps
    )
    
    success = sprint.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()