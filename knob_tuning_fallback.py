#!/usr/bin/env python3
"""
🔧 KNOB TUNING FALLBACK
Quick knob adjustments if entire grid whiffs - test specific parameter changes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from alpha_signal_generator import create_toy_alpha_data

class KnobTuningFallback:
    """Test specific parameter knobs if grid search fails"""
    
    def __init__(self):
        # Base configuration (from original grid)
        self.base_config = {
            'n_steps': 2048,
            'learning_rate': 7e-5,
            'ent_coef': 0.002,
            'n_epochs': 4,
            'batch_size': 64,
            'gamma': 0.999,              # Current
            'gae_lambda': 0.95,          # Current  
            'clip_range': 0.2,           # Current
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0,
            'seed': 42,
            'device': "auto"
        }
        
        # Knob variations to test
        self.knob_tests = [
            {
                'name': 'Shorter Horizon',
                'changes': {'gamma': 0.95},
                'rationale': 'Helps credit assignment in impact-heavy envs'
            },
            {
                'name': 'Lower GAE Lambda', 
                'changes': {'gae_lambda': 0.9},
                'rationale': 'Lowers variance of advantage estimates'
            },
            {
                'name': 'Looser Clips',
                'changes': {'clip_range': 0.3},
                'rationale': 'Helps escape "do nothing" local minima'
            },
            {
                'name': 'Combined Adjustments',
                'changes': {'gamma': 0.95, 'gae_lambda': 0.9, 'clip_range': 0.3},
                'rationale': 'Multiple knobs for stronger effect'
            },
            {
                'name': 'High Exploration',
                'changes': {'ent_coef': 0.01, 'clip_range': 0.3},
                'rationale': 'Force more exploration to find alpha'
            }
        ]
        
        # Test parameters
        self.training_steps = 15000
        self.eval_steps = 800
        self.target_return = 0.01
        self.max_drawdown = 0.02
    
    def create_test_environment(self):
        """Create consistent test environment"""
        
        # Create strong alpha data
        enhanced_features, price_series, alpha_metadata = create_toy_alpha_data(
            n_periods=2500, 
            seed=42, 
            alpha_strength=0.15
        )
        
        # Create V3 environment
        env = IntradayTradingEnvV3(
            processed_feature_data=enhanced_features,
            price_data=price_series,
            initial_capital=100000,
            max_daily_drawdown_pct=self.max_drawdown,
            transaction_cost_pct=0.0001,
            log_trades=False,
            base_impact_bp=68.0,
            impact_exponent=0.5,
            verbose=False
        )
        
        # Wrap environment
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
        
        return vec_env, alpha_metadata
    
    def test_knob_configuration(self, knob_test, vec_env):
        """Test a specific knob configuration"""
        
        logger.info(f"🔧 Testing: {knob_test['name']}")
        logger.info(f"   Changes: {knob_test['changes']}")
        logger.info(f"   Rationale: {knob_test['rationale']}")
        
        # Create config with knob adjustments
        test_config = self.base_config.copy()
        test_config.update(knob_test['changes'])
        
        # Create model
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            **test_config
        )
        
        # Train model
        start_time = datetime.now()
        model.learn(total_timesteps=self.training_steps, progress_bar=False)
        training_time = datetime.now() - start_time
        
        # Evaluate model
        result = self._quick_evaluation(model, vec_env)
        result['training_time'] = training_time
        result['config'] = test_config
        result['knob_test'] = knob_test
        
        # Log results
        status = "✅ PASS" if result['gate_pass'] else "❌ FAIL"
        logger.info(f"   Result: {status} | Return {result['total_return']:+.2%} | DD {result['max_drawdown']:.2%} | Time {training_time}")
        
        return result
    
    def _quick_evaluation(self, model, vec_env):
        """Quick evaluation of model performance"""
        
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        
        portfolio_values = []
        actions_taken = []
        initial_capital = 100000
        
        for step in range(self.eval_steps):
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
        trading_freq = 1.0 - action_dist[1]
        
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
            'actions_taken': actions_taken,
            'return_gate': return_gate,
            'dd_gate': dd_gate,
            'gate_pass': gate_pass
        }
    
    def run_knob_tuning_sweep(self):
        """Run complete knob tuning sweep"""
        
        logger.info("🔧 KNOB TUNING FALLBACK")
        logger.info(f"   Base config: {self.base_config}")
        logger.info(f"   Testing {len(self.knob_tests)} knob variations")
        logger.info(f"   Target: {self.target_return:.1%} return, <{self.max_drawdown:.1%} DD")
        
        start_time = datetime.now()
        
        # Create test environment
        vec_env, alpha_metadata = self.create_test_environment()
        logger.info(f"   Alpha signals: {alpha_metadata['bullish_signals']} bullish, {alpha_metadata['bearish_signals']} bearish")
        
        # Test each knob configuration
        results = []
        successful_configs = []
        
        for i, knob_test in enumerate(self.knob_tests):
            logger.info(f"\\n🔧 Knob Test {i+1}/{len(self.knob_tests)}")
            
            try:
                result = self.test_knob_configuration(knob_test, vec_env)
                results.append(result)
                
                if result['gate_pass']:
                    successful_configs.append(result)
                    logger.info(f"✅ SUCCESS FOUND! {knob_test['name']}")
                    
            except Exception as e:
                logger.error(f"❌ Knob test failed: {e}")
                continue
        
        elapsed = datetime.now() - start_time
        
        # Summary results
        logger.info(f"\\n📊 KNOB TUNING RESULTS:")
        logger.info(f"   Total time: {elapsed}")
        logger.info(f"   Configurations tested: {len(results)}")
        logger.info(f"   Successful configs: {len(successful_configs)}")
        
        if successful_configs:
            logger.info(f"\\n🏆 SUCCESSFUL CONFIGURATIONS:")
            for i, config in enumerate(successful_configs):
                knob_info = config['knob_test']
                logger.info(f"   {i+1}. {knob_info['name']}")
                logger.info(f"      Changes: {knob_info['changes']}")
                logger.info(f"      Performance: {config['total_return']:+.2%} return, {config['max_drawdown']:.2%} DD")
                logger.info(f"      Trading: {config['trading_frequency']:.1%}")
            
            # Save best config for potential use
            best_config = max(successful_configs, key=lambda x: x['total_return'])
            
            import json
            with open('knob_tuning_success.json', 'w') as f:
                config_to_save = {
                    'success': True,
                    'best_knob_test': best_config['knob_test'],
                    'best_config': {k: v for k, v in best_config['config'].items() if k != 'device'},
                    'performance': {
                        'total_return': float(best_config['total_return']),
                        'max_drawdown': float(best_config['max_drawdown']),
                        'trading_frequency': float(best_config['trading_frequency'])
                    },
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(config_to_save, f, indent=2)
            
            logger.info(f"\\n✅ Best config saved to knob_tuning_success.json")
            logger.info(f"   Ready to proceed with: {best_config['knob_test']['name']}")
            
        else:
            logger.info(f"\\n⚠️ NO SUCCESSFUL CONFIGURATIONS")
            logger.info("   All knob adjustments failed to achieve gate criteria")
            logger.info("   May need more aggressive parameter changes or different approach")
            
            # Save analysis for debugging
            import json
            with open('knob_tuning_analysis.json', 'w') as f:
                analysis = {
                    'success': False,
                    'results': [
                        {
                            'knob_test': r['knob_test'],
                            'return': float(r['total_return']),
                            'drawdown': float(r['max_drawdown']),
                            'trading_freq': float(r['trading_frequency'])
                        } for r in results
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(analysis, f, indent=2)
            
            logger.info(f"   Analysis saved to knob_tuning_analysis.json")
        
        return len(successful_configs) > 0, successful_configs

def main():
    """Main knob tuning function"""
    
    logger.info("🔧 KNOB TUNING FALLBACK - Parameter Isolation")
    
    tuner = KnobTuningFallback()
    success, configs = tuner.run_knob_tuning_sweep()
    
    if success:
        logger.info("\\n🎉 KNOB TUNING SUCCESS!")
        logger.info("   Found working parameter configuration")
        logger.info("   Can proceed with these settings to dual-ticker")
    else:
        logger.info("\\n🔬 KNOB TUNING INSIGHTS:")
        logger.info("   Issue appears deeper than simple parameter adjustments")
        logger.info("   May need architectural changes or different RL approach")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)