#!/usr/bin/env python3
"""
🎯 HPO GRID SPRINT - Single-Ticker Alpha Learning
Focused hyperparameter optimization to prove RL can learn alpha extraction
ABORT CRITERIA: Stop immediately when any config hits +1% return & <2% DD

Grid: n_steps{2048,4096}, lr{7e-5,3e-4}, ent_coef{0.002,0.005}
Target: Pass gate within 20K steps to prove learning loop works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import itertools
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

class HPOGridSprint:
    """Focused HPO grid for alpha signal learning"""
    
    def __init__(self, configs=None, n_steps=2048, n_epochs=4, early_exit_rew=500, early_exit_dd=2.0, max_steps=20000):
        self.results = []
        self.success_found = False
        self.best_config = None
        self.start_time = datetime.now()
        self.early_exit_rew = early_exit_rew
        self.early_exit_dd = early_exit_dd
        self.max_steps = max_steps
        
        # Setup fixed parameters and thresholds
        self.setup_fixed_params()
        
        # Parse custom configs if provided
        if configs:
            self.grid = self.parse_configs(configs, n_steps, n_epochs)
        else:
            # Default HPO Grid - focused on key parameters
            self.grid = {
                'n_steps': [n_steps],
                'learning_rate': [7e-5, 3e-4], 
                'ent_coef': [0.002, 0.005],
                'n_epochs': [n_epochs],
            }
    
    def parse_configs(self, configs_str, default_n_steps, default_n_epochs):
        """Parse config string like 'lr=3e-4,ent=0.0,clip=0.3; lr=2e-4,ent=0.001,clip=0.2'"""
        configs = []
        
        # Split by semicolon for multiple configs
        config_strings = [c.strip() for c in configs_str.split(';') if c.strip()]
        
        for config_str in config_strings:
            config = {
                'n_steps': default_n_steps,
                'n_epochs': default_n_epochs,
                'clip_range': 0.2,  # default
            }
            
            # Parse key=value pairs
            pairs = [p.strip() for p in config_str.split(',') if p.strip()]
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'lr':
                        config['learning_rate'] = float(value)
                    elif key == 'ent':
                        config['ent_coef'] = float(value)
                    elif key == 'clip':
                        config['clip_range'] = float(value)
                    elif key == 'n_steps':
                        config['n_steps'] = int(value)
                    elif key == 'n_epochs':
                        config['n_epochs'] = int(value)
            
            configs.append(config)
        
        logger.info(f"🎯 Parsed {len(configs)} custom configurations")
        for i, config in enumerate(configs):
            logger.info(f"   Config {i+1}: {config}")
        
        return {'custom_configs': configs}
    
    def setup_fixed_params(self):
        """Setup fixed parameters and thresholds"""
        # Gate thresholds
        self.target_return = self.early_exit_rew / 100000  # Convert reward to return %
        self.max_drawdown = self.early_exit_dd / 100       # Convert to decimal
        self.max_training_steps = self.max_steps
        
        # Fixed parameters (proven to work)
        self.fixed_params = {
            'batch_size': 64,
            'gamma': 0.999,
            'gae_lambda': 0.95,
            'clip_range': 0.1,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0,
            'seed': 42,
            'device': "auto"
        }
        
        # Gate criteria
        self.target_return = 0.01      # +1% return
        self.max_drawdown = 0.02       # <2% drawdown
        self.max_training_steps = 20000  # Stop training at 20K steps
        
    def create_test_environment(self):
        """Create consistent test environment for all HPO runs"""
        
        # Create strong, learnable alpha data
        enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
            n_periods=3000, 
            seed=42, 
            alpha_strength=0.15  # Strong but not too obvious
        )
        
        # Create V3 environment with calibrated parameters
        env = IntradayTradingEnvV3(
            processed_feature_data=enhanced_features,
            price_data=price_series,
            initial_capital=100000,
            max_daily_drawdown_pct=self.max_drawdown,
            transaction_cost_pct=0.0001,
            log_trades=False,
            base_impact_bp=68.0,       # Calibrated impact
            impact_exponent=0.5,
            verbose=False
        )
        
        # Wrap environment 
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
        
        return vec_env, alpha_metadata
    
    def evaluate_config(self, config_params, vec_env, config_id):
        """Evaluate single HPO configuration"""
        
        logger.info(f"🧪 Config {config_id}: {config_params}")
        
        # Create model with current config
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            **{**self.fixed_params, **config_params}
        )
        
        # Train model with tightened early-exit triggers
        checkpoints = [5000, 8000, 10000, 15000, 20000]
        training_stats = {'entropy_values': [], 'trade_counts': []}
        
        for i, checkpoint in enumerate(checkpoints):
            if i == 0:
                steps_to_train = checkpoint
            else:
                steps_to_train = checkpoint - checkpoints[i-1]
            
            model.learn(total_timesteps=steps_to_train, progress_bar=False, reset_num_timesteps=False)
            
            # Quick evaluation with early abort signals
            result = self.quick_evaluation(model, vec_env)
            total_trades = sum(1 for action in result.get('actions_taken', []) if action != 1)  # Non-HOLD actions
            
            # EARLY ABORT SIGNAL 1: P&L disaster + high DD at 5K steps
            if checkpoint == 5000:
                if result['total_return'] < -0.005 and result['max_drawdown'] > 0.015:  # < -0.5% return, > 1.5% DD
                    logger.info(f"🚫 EARLY ABORT at 5K: P&L {result['total_return']:+.2%}, DD {result['max_drawdown']:.2%} - no recovery likely")
                    result['training_steps'] = checkpoint
                    result['config'] = config_params
                    result['config_id'] = config_id
                    result['abort_reason'] = 'disaster_5k'
                    return result
            
            # EARLY ABORT SIGNAL 2: Stuck in "do nothing" at 8K steps
            if checkpoint == 8000:
                # Estimate entropy from action distribution
                action_dist = result['action_distribution']
                hold_freq = action_dist[1]  # HOLD frequency
                estimated_entropy = -np.sum([p * np.log(p + 1e-8) for p in action_dist if p > 0])
                
                if estimated_entropy < 0.25 and total_trades < 5:
                    logger.info(f"🚫 EARLY ABORT at 8K: Entropy {estimated_entropy:.3f}, Trades {total_trades} - stuck in do-nothing")
                    result['training_steps'] = checkpoint
                    result['config'] = config_params
                    result['config_id'] = config_id
                    result['abort_reason'] = 'do_nothing_8k'
                    return result
            
            # SUCCESS CHECK: Any checkpoint can trigger success
            if result['gate_pass']:
                logger.info(f"✅ EARLY SUCCESS at {checkpoint} steps!")
                result['training_steps'] = checkpoint
                result['config'] = config_params
                result['config_id'] = config_id
                return result
            
            # Progress logging with key metrics
            logger.info(f"   Checkpoint {checkpoint}: Return {result['total_return']:+.2%}, DD {result['max_drawdown']:.2%}, Trades {total_trades}")
            
            training_stats['trade_counts'].append(total_trades)
        
        # Final evaluation
        result = self.quick_evaluation(model, vec_env)
        result['training_steps'] = self.max_training_steps
        result['config'] = config_params
        result['config_id'] = config_id
        
        return result
    
    def quick_evaluation(self, model, vec_env, eval_steps=800):
        """Quick evaluation to check gate criteria"""
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        portfolio_values = []
        actions_taken = []
        initial_capital = 100000
        
        for step in range(eval_steps):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            
            result = vec_env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done, _, info = result
            
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
                actions_taken.append(action[0])
            
            if done[0]:
                episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
            else:
                episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        
        # Calculate metrics
        final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_portfolio - initial_capital) / initial_capital
        peak_portfolio = max(portfolio_values) if portfolio_values else initial_capital
        max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio if portfolio_values else 0
        
        # Action analysis
        action_counts = np.bincount(actions_taken, minlength=3) if actions_taken else [0, 0, 0]
        action_dist = action_counts / len(actions_taken) if actions_taken else [0, 0, 0]
        trading_freq = 1.0 - action_dist[1]  # 1 - HOLD frequency
        
        # Gate criteria
        return_gate = total_return >= self.target_return
        dd_gate = max_drawdown <= self.max_drawdown
        gate_pass = return_gate and dd_gate
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_portfolio': final_portfolio,
            'trading_frequency': trading_freq,
            'action_distribution': action_dist,
            'actions_taken': actions_taken,  # Added for early abort detection
            'return_gate': return_gate,
            'dd_gate': dd_gate,
            'gate_pass': gate_pass
        }
    
    def run_grid_search(self):
        """Run HPO grid search with early abort on success"""
        
        # Handle custom configs vs grid search
        if 'custom_configs' in self.grid:
            configs_to_test = self.grid['custom_configs']
            logger.info(f"🎯 HPO GRID SPRINT - Custom Configurations")
            logger.info(f"   Configs: {len(configs_to_test)} custom configurations")
        else:
            configs_to_test = list(itertools.product(*self.grid.values()))
            param_names = list(self.grid.keys())
            configs_to_test = [dict(zip(param_names, values)) for values in configs_to_test]
            logger.info(f"🎯 HPO GRID SPRINT - Grid Search")
            logger.info(f"   Grid size: {len(configs_to_test)} configurations")
        
        logger.info(f"   Target: ep_rew > {self.early_exit_rew}, DD < {self.early_exit_dd}%")
        logger.info(f"   Max training: {self.max_steps} steps per config")
        logger.info(f"   ABORT on first success!")
        
        # Create test environment once
        vec_env, alpha_metadata = self.create_test_environment()
        logger.info(f"   Alpha signals: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish")
        
        logger.info(f"\\n🚀 Starting configuration testing...")
        
        for i, config_params in enumerate(configs_to_test):
            config_id = f"{i+1}/{len(configs_to_test)}"
            
            try:
                result = self.evaluate_config(config_params, vec_env, config_id)
                self.results.append(result)
                
                # Log result
                status = "✅ PASS" if result['gate_pass'] else "❌ FAIL"
                logger.info(f"   Result: {status} | Return {result['total_return']:+.2%} | DD {result['max_drawdown']:.2%} | Trade {result['trading_frequency']:.1%}")
                
                # ABORT on first success
                if result['gate_pass']:
                    self.success_found = True
                    self.best_config = result
                    logger.info(f"\\n🎉 SUCCESS FOUND! Aborting grid search early.")
                    logger.info(f"   Winning config: {config_params}")
                    logger.info(f"   Performance: {result['total_return']:+.2%} return, {result['max_drawdown']:.2%} DD")
                    logger.info(f"   Training steps: {result['training_steps']}")
                    break
                    
            except Exception as e:
                logger.error(f"   Config {config_id} failed: {e}")
                continue
        
        elapsed = datetime.now() - self.start_time
        
        # Final results
        logger.info(f"\\n📊 HPO GRID SPRINT RESULTS:")
        logger.info(f"   Elapsed time: {elapsed}")
        logger.info(f"   Configurations tested: {len(self.results)}")
        logger.info(f"   Success found: {'YES' if self.success_found else 'NO'}")
        
        if self.success_found:
            logger.info(f"\\n🏆 WINNING CONFIGURATION:")
            logger.info(f"   {self.best_config['config']}")
            logger.info(f"   Performance: {self.best_config['total_return']:+.2%} return, {self.best_config['max_drawdown']:.2%} DD")
            logger.info(f"   Trading frequency: {self.best_config['trading_frequency']:.1%}")
            logger.info(f"   Training steps: {self.best_config['training_steps']}")
            
            # Save winning config
            import json
            with open('winning_hpo_config.json', 'w') as f:
                config_to_save = {
                    'success': True,
                    'config': self.best_config['config'],
                    'performance': {
                        'total_return': float(self.best_config['total_return']),
                        'max_drawdown': float(self.best_config['max_drawdown']),
                        'trading_frequency': float(self.best_config['trading_frequency'])
                    },
                    'training_steps': self.best_config['training_steps'],
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(config_to_save, f, indent=2)
                
            logger.info(f"   ✅ Config saved to winning_hpo_config.json")
            
        else:
            logger.info(f"\\n⚠️ NO SUCCESS FOUND")
            logger.info(f"   Best result: {max(self.results, key=lambda x: x['total_return'])['total_return']:+.2%} return")
            logger.info(f"   All configs learned do-nothing or exceeded DD limit")
            
            # Save failure analysis
            import json
            with open('hpo_failure_analysis.json', 'w') as f:
                analysis = {
                    'success': False,
                    'best_return': float(max(self.results, key=lambda x: x['total_return'])['total_return']),
                    'configs_tested': len(self.results),
                    'all_results': [
                        {
                            'config': r['config'],
                            'return': float(r['total_return']),
                            'drawdown': float(r['max_drawdown']),
                            'trading_freq': float(r['trading_frequency'])
                        } for r in self.results
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(analysis, f, indent=2)
                
            logger.info(f"   📊 Analysis saved to hpo_failure_analysis.json")
        
        return self.success_found, self.best_config

def main():
    """Run HPO grid sprint with command-line arguments"""
    parser = argparse.ArgumentParser(description='HPO Grid Sprint for Alpha Learning')
    parser.add_argument('--configs', type=str, help='Custom configs: "lr=3e-4,ent=0.0,clip=0.3; lr=2e-4,ent=0.001,clip=0.2"')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per rollout')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs per update')
    parser.add_argument('--early_exit_rew', type=float, default=500, help='Early exit reward threshold')
    parser.add_argument('--early_exit_dd', type=float, default=2.0, help='Early exit drawdown threshold (%)')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max training steps per config')
    
    args = parser.parse_args()
    
    sprint = HPOGridSprint(
        configs=args.configs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        early_exit_rew=args.early_exit_rew,
        early_exit_dd=args.early_exit_dd,
        max_steps=args.max_steps
    )
    
    success, best_config = sprint.run_grid_search()
    
    if success:
        print(f"\\n🎉 HPO GRID SPRINT: SUCCESS")
        print(f"Learning loop proven - ready for dual-ticker merge!")
    else:
        print(f"\\n⚠️ HPO GRID SPRINT: NO SUCCESS")
        print(f"RL hyperparameter tuning needs deeper investigation")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)