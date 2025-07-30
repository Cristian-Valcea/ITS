#!/usr/bin/env python3
"""
🎯 DUAL-TICKER WARM-START
Load successful single-ticker weights and fine-tune for dual-ticker environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import zipfile
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.gym_env.dual_ticker_trading_env_v3 import DualTickerTradingEnvV3, create_dual_ticker_alpha_data

class DualTickerWarmStart:
    """Warm-start dual-ticker training from successful single-ticker policy"""
    
    def __init__(self, exported_weights_path: str):
        self.exported_weights_path = exported_weights_path
        self.export_data = None
        self.config_info = None
        
        # Load exported weights
        self._load_exported_weights()
        
        # Training parameters (conservative for fine-tuning)
        self.fine_tune_steps = 25000
        self.fine_tune_lr = 2e-5  # Very low LR to avoid unlearning
        
        # Gate criteria (same as single-ticker)
        self.target_return = 0.01   # +1% return
        self.max_drawdown = 0.02    # <2% drawdown
    
    def _load_exported_weights(self):
        """Load exported weights from archive"""
        
        logger.info(f"🔍 Loading exported weights from {self.exported_weights_path}")
        
        try:
            with zipfile.ZipFile(self.exported_weights_path, 'r') as zf:
                # Load main export data
                export_data_bytes = zf.read('export_data.pkl')
                self.export_data = pickle.loads(export_data_bytes)
                
                # Load metadata
                metadata_bytes = zf.read('metadata.json')
                self.config_info = json.loads(metadata_bytes.decode('utf-8'))
                
            logger.info(f"✅ Exported weights loaded successfully")
            logger.info(f"   Source: {self.config_info['source_model_path']}")
            logger.info(f"   Export date: {self.config_info['timestamp']}")
            logger.info(f"   Shared weights: {len(self.export_data['shared_weights'])}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load exported weights: {e}")
            raise
    
    def create_dual_ticker_environment(self):
        """Create dual-ticker environment for fine-tuning"""
        
        logger.info("🏗️ Creating dual-ticker environment...")
        
        # Create dual-ticker data with alpha signals
        features, prices = create_dual_ticker_alpha_data(
            n_periods=3000, 
            seed=42, 
            alpha_strength=0.15  # Same strength as successful single-ticker
        )
        
        # Create dual-ticker V3 environment
        env = DualTickerTradingEnvV3(
            processed_feature_data=features,
            price_data=prices,
            initial_capital=100000,
            max_daily_drawdown_pct=self.max_drawdown,
            transaction_cost_pct=0.0001,
            log_trades=False,
            base_impact_bp=68.0,       # Same calibrated impact
            impact_exponent=0.5,
            verbose=False
        )
        
        # Wrap environment
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
        
        logger.info(f"✅ Dual-ticker environment created")
        logger.info(f"   Observation space: {env.observation_space.shape}")
        logger.info(f"   Action space: {env.action_space.n}")
        
        return vec_env
    
    def create_warmstart_model(self, vec_env):
        """Create dual-ticker model with warm-started weights"""
        
        logger.info("🔥 Creating warm-start model...")
        
        # Get successful single-ticker hyperparameters
        source_config = self.export_data.get('config_info', {}).get('config', {})
        
        # Create new dual-ticker model with same hyperparameters
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=self.fine_tune_lr,  # Lower LR for fine-tuning
            n_steps=source_config.get('n_steps', 2048),
            batch_size=64,
            n_epochs=source_config.get('n_epochs', 4),
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=source_config.get('ent_coef', 0.002),
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=42,
            device="auto"
        )
        
        logger.info(f"✅ Base dual-ticker model created")
        logger.info(f"   Using config: {source_config}")
        logger.info(f"   Fine-tune LR: {self.fine_tune_lr}")
        
        # Load shared weights (feature extraction, LSTM)
        self._load_shared_weights(model)
        
        # Initialize action head for 3→9 action expansion
        self._initialize_action_head(model)
        
        return model
    
    def _load_shared_weights(self, model):
        """Load shared weights (feature extraction, LSTM) from single-ticker"""
        
        logger.info("🔄 Loading shared weights...")
        
        model_state_dict = model.policy.state_dict()
        shared_weights = self.export_data['shared_weights']
        
        loaded_count = 0
        for key, tensor in shared_weights.items():
            if key in model_state_dict:
                # Check dimension compatibility
                if model_state_dict[key].shape == tensor.shape:
                    model_state_dict[key].copy_(tensor)
                    loaded_count += 1
                else:
                    logger.warning(f"   Dimension mismatch for {key}: {model_state_dict[key].shape} vs {tensor.shape}")
            else:
                logger.warning(f"   Key not found in dual-ticker model: {key}")
        
        logger.info(f"✅ Loaded {loaded_count} shared weight tensors")
    
    def _initialize_action_head(self, model):
        """Initialize action head for 3→9 action expansion"""
        
        logger.info("🎯 Initializing 3→9 action head expansion...")
        
        # Get single-ticker action head weights
        action_head_weights = self.export_data['action_head_weights']
        
        # Find action network layer
        model_state_dict = model.policy.state_dict()
        
        for key, tensor in model_state_dict.items():
            if 'action_net' in key and 'weight' in key:
                # Expand 3-action weights to 9-action weights
                if key in action_head_weights:
                    source_weights = action_head_weights[key]  # Shape: [3, hidden_dim]
                    
                    if len(source_weights.shape) == 2 and source_weights.shape[0] == 3:
                        # Replicate each action 3 times for dual-ticker matrix
                        # [SELL, HOLD, BUY] → [SELL_BOTH, SELL_NVDA_HOLD_MSFT, ..., BUY_BOTH]
                        expanded_weights = torch.zeros_like(tensor)  # [9, hidden_dim]
                        
                        for i in range(3):  # For each NVDA action
                            for j in range(3):  # For each MSFT action
                                dual_action_idx = i * 3 + j
                                # Initialize with average of NVDA and MSFT actions
                                expanded_weights[dual_action_idx] = (source_weights[i] + source_weights[j]) / 2
                        
                        model_state_dict[key].copy_(expanded_weights)
                        logger.info(f"   Expanded {key}: {source_weights.shape} → {expanded_weights.shape}")
            
            elif 'action_net' in key and 'bias' in key:
                # Expand action biases similarly
                if key in action_head_weights:
                    source_bias = action_head_weights[key]  # Shape: [3]
                    
                    if len(source_bias.shape) == 1 and source_bias.shape[0] == 3:
                        expanded_bias = torch.zeros_like(tensor)  # [9]
                        
                        for i in range(3):
                            for j in range(3):
                                dual_action_idx = i * 3 + j
                                expanded_bias[dual_action_idx] = (source_bias[i] + source_bias[j]) / 2
                        
                        model_state_dict[key].copy_(expanded_bias)
                        logger.info(f"   Expanded {key}: {source_bias.shape} → {expanded_bias.shape}")
        
        logger.info("✅ Action head expansion completed")
    
    def evaluate_model(self, model, vec_env, eval_steps=1000):
        """Evaluate dual-ticker model performance"""
        
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
        action_counts = np.bincount(actions_taken, minlength=9) if actions_taken else [0] * 9
        action_dist = action_counts / len(actions_taken) if actions_taken else [0] * 9
        
        # Gate criteria
        return_gate = total_return >= self.target_return
        dd_gate = max_drawdown <= self.max_drawdown
        gate_pass = return_gate and dd_gate
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_portfolio': final_portfolio,
            'action_distribution': action_dist,
            'return_gate': return_gate,
            'dd_gate': dd_gate,
            'gate_pass': gate_pass
        }
    
    def run_warmstart_training(self):
        """Run complete warm-start training process"""
        
        logger.info(f"🎯 DUAL-TICKER WARM-START TRAINING")
        logger.info(f"   Source weights: {self.exported_weights_path}")
        logger.info(f"   Fine-tune steps: {self.fine_tune_steps}")
        logger.info(f"   Target: {self.target_return:.1%} return, <{self.max_drawdown:.1%} DD")
        
        start_time = datetime.now()
        
        # Create environment
        vec_env = self.create_dual_ticker_environment()
        
        # Create warm-start model
        model = self.create_warmstart_model(vec_env)
        
        # Evaluate pre-training performance
        logger.info("🔍 Pre-training evaluation...")
        pre_result = self.evaluate_model(model, vec_env, eval_steps=500)
        logger.info(f"   Pre-training: Return {pre_result['total_return']:+.2%}, DD {pre_result['max_drawdown']:.2%}")
        
        # Fine-tune training
        logger.info(f"🔥 Fine-tuning for {self.fine_tune_steps} steps...")
        model.learn(total_timesteps=self.fine_tune_steps, progress_bar=True)
        
        # Final evaluation
        logger.info("🔍 Post-training evaluation...")
        post_result = self.evaluate_model(model, vec_env, eval_steps=1000)
        
        elapsed = datetime.now() - start_time
        
        # Results
        logger.info(f"\\n📊 DUAL-TICKER WARM-START RESULTS:")
        logger.info(f"   Training time: {elapsed}")
        logger.info(f"   Fine-tune steps: {self.fine_tune_steps}")
        
        logger.info(f"\\n💰 PERFORMANCE:")
        logger.info(f"   Final portfolio: ${post_result['final_portfolio']:,.0f}")
        logger.info(f"   Total return: {post_result['total_return']:+.2%}")
        logger.info(f"   Max drawdown: {post_result['max_drawdown']:.2%}")
        
        logger.info(f"\\n🎯 DUAL-TICKER ACTION DISTRIBUTION:")
        action_names = [
            "SELL_BOTH", "SELL_NVDA_HOLD_MSFT", "SELL_NVDA_BUY_MSFT",
            "HOLD_NVDA_SELL_MSFT", "HOLD_BOTH", "HOLD_NVDA_BUY_MSFT",
            "BUY_NVDA_SELL_MSFT", "BUY_NVDA_HOLD_MSFT", "BUY_BOTH"
        ]
        for i, (name, freq) in enumerate(zip(action_names, post_result['action_distribution'])):
            if freq > 0.01:  # Only show actions used >1%
                logger.info(f"   {name}: {freq:.1%}")
        
        logger.info(f"\\n🚨 GATE CRITERIA:")
        logger.info(f"   Return Gate (≥{self.target_return:.1%}): {'✅ PASS' if post_result['return_gate'] else '❌ FAIL'} ({post_result['total_return']:+.2%})")
        logger.info(f"   DD Gate (≤{self.max_drawdown:.1%}): {'✅ PASS' if post_result['dd_gate'] else '❌ FAIL'} ({post_result['max_drawdown']:.2%})")
        logger.info(f"   Overall: {'✅ PASS' if post_result['gate_pass'] else '❌ FAIL'}")
        
        if post_result['gate_pass']:
            logger.info("🎉 ✅ DUAL-TICKER WARM-START SUCCESS!")
            logger.info("   Cross-asset model learned incremental alpha with safety rails intact")
            
            # Save successful dual-ticker model
            model.save("models/dual_ticker_success.zip")
            logger.info("   Model saved to models/dual_ticker_success.zip")
            
        else:
            logger.info("⚠️ ❌ DUAL-TICKER WARM-START INCOMPLETE")
            if not post_result['return_gate']:
                logger.info("   📈 Insufficient return extraction from dual-ticker alpha")
            if not post_result['dd_gate']:
                logger.info("   🛡️ Drawdown exceeded safety limit")
        
        return post_result['gate_pass'], post_result

def main():
    """Main warm-start function"""
    
    # Check if we have exported weights
    weights_path = "models/singleticker_gatepass.zip"
    if not Path(weights_path).exists():
        logger.error(f"❌ No exported weights found at {weights_path}")
        logger.error("   Run HPO grid search first to get successful single-ticker policy")
        return False
    
    # Run warm-start training
    warmstart = DualTickerWarmStart(weights_path)
    success, results = warmstart.run_warmstart_training()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)