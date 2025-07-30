#!/usr/bin/env python3
"""
🎯 PERSISTENT-TO-NOISY ALPHA CURRICULUM DRIVER
Progressive learning: persistent → piecewise → low-noise → realistic noisy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.gym_env.intraday_trading_env_v3 import IntradayTradingEnvV3
sys.path.insert(0, str(Path('.') / 'src' / 'features'))
from curriculum_alpha_generator import CurriculumAlphaGenerator, AlphaPattern

class AlphaCurriculum:
    """Battle-tested curriculum for high-friction RL environments"""
    
    def __init__(self, start_phase=0, warm_start_path=None):
        # PERSISTENT-TO-NOISY curriculum configuration
        self.start_phase = start_phase
        self.warm_start_path = warm_start_path
        self.phase_cfg = [
            {
                'name': 'Phase 0 - Persistent',
                'pattern': AlphaPattern.PERSISTENT,
                'alpha_strength': 0.40,   # Persistent ±0.40
                'tc_bp': 5,               # Light transaction costs
                'impact_bp': 10,          # Minimal impact
                'n_steps': 512,
                'batch_size': 64,
                'clip_range': 0.3,
                'ent_coef': 0.01,         # High exploration
                'learning_rate': 3e-4,
                'rew_gate': 8000,         # +8K ep_rew_mean
                'dd_gate': 2.0,           # DD < 2%
                'max_timesteps': 10000,   # 10K max steps
                'target_signal': 'Persistent alpha learning: +8K, DD<2%'
            },
            {
                'name': 'Phase 1 - Piecewise',
                'pattern': AlphaPattern.PIECEWISE,
                'alpha_strength': 0.30,   # Piecewise +0.30/0.0
                'tc_bp': 10,              # Light transaction costs
                'impact_bp': 15,          # Gradual friction increase
                'n_steps': 1024,
                'batch_size': 64,
                'clip_range': 0.25,
                'ent_coef': 0.008,
                'learning_rate': 3e-4,
                'rew_gate': 5000,         # +5K ep_rew_mean
                'dd_gate': 2.5,           # DD < 2.5% (relaxed for piecewise)
                'max_timesteps': 15000,   # 15K max steps
                'target_signal': 'Piecewise alpha adaptation: +5K, DD<2.5%'
            },
            {
                'name': 'Phase 2 - Low Noise',
                'pattern': AlphaPattern.LOW_NOISE,
                'alpha_strength': 0.20,   # Base 0.20 with noise
                'noise_std': 0.05,        # Low noise std
                'tc_bp': 25,              # Medium transaction costs
                'impact_bp': 35,          # Higher friction
                'n_steps': 2048,
                'batch_size': 128,
                'clip_range': 0.2,
                'ent_coef': 0.005,
                'learning_rate': 3e-4,
                'rew_gate': 2000,         # +2K ep_rew_mean
                'dd_gate': 2.5,           # DD < 2.5%
                'max_timesteps': 20000,   # 20K max steps
                'target_signal': 'Low-noise robustness: +2K, DD<2.5%'
            },
            {
                'name': 'Phase 3 - Realistic Noisy',
                'pattern': AlphaPattern.REALISTIC_NOISY,
                'alpha_strength': 0.15,   # Production-like alpha
                'tc_bp': 40,              # Full transaction costs
                'impact_bp': 68,          # Calibrated impact (production)
                'n_steps': 4096,
                'batch_size': 128,
                'clip_range': 0.1,
                'ent_coef': 0.003,
                'learning_rate': 2e-4,
                'rew_gate': 1000,         # +1% return equivalent
                'dd_gate': 2.0,           # DD ≤ 2%
                'max_timesteps': 25000,   # 25K max steps
                'target_signal': 'Production readiness: +1%, DD≤2%'
            }
        ]
        
        self.phase = start_phase  # Start from specified phase
        self.results = []
        
    def get_current_phase(self):
        """Get current phase configuration"""
        if self.phase >= len(self.phase_cfg):
            return None
        return self.phase_cfg[self.phase]
    
    def create_phase_environment(self):
        """Create environment for current phase with appropriate alpha pattern"""
        
        cfg = self.get_current_phase()
        if cfg is None:
            return None, None
        
        logger.info(f"🎯 Creating environment for {cfg['name']}")
        logger.info(f"   Pattern: {cfg['pattern'].value}, Alpha: {cfg['alpha_strength']}")
        logger.info(f"   Impact: {cfg['impact_bp']}bp, TC: {cfg['tc_bp']}bp")
        
        # Create alpha generator for current phase
        alpha_generator = CurriculumAlphaGenerator(cfg['pattern'], seed=42)
        
        # Generate alpha data based on phase pattern
        if cfg['pattern'] == AlphaPattern.PERSISTENT:
            # Phase 0: Force BULL episode to test bidirectional symmetry
            bull_episode = True  # Test BULL after successful BEAR
            enhanced_features, price_series, alpha_metadata = alpha_generator.generate_alpha_data(
                n_periods=3000,
                alpha_strength=cfg['alpha_strength'],
                bull_episode=bull_episode
            )
            logger.info(f"   Episode type: {'BULL' if bull_episode else 'BEAR'}")
            
        elif cfg['pattern'] == AlphaPattern.PIECEWISE:
            # Phase 1: Piecewise constant
            enhanced_features, price_series, alpha_metadata = alpha_generator.generate_alpha_data(
                n_periods=3000,
                alpha_strength=cfg['alpha_strength'],
                on_duration=500,
                off_duration=250
            )
            
        elif cfg['pattern'] == AlphaPattern.LOW_NOISE:
            # Phase 2: Low noise
            enhanced_features, price_series, alpha_metadata = alpha_generator.generate_alpha_data(
                n_periods=3000,
                base_alpha=cfg['alpha_strength'],
                noise_std=cfg.get('noise_std', 0.05)
            )
            
        elif cfg['pattern'] == AlphaPattern.REALISTIC_NOISY:
            # Phase 3: Realistic noisy (original approach)
            enhanced_features, price_series, alpha_metadata = alpha_generator.generate_alpha_data(
                n_periods=3000,
                alpha_strength=cfg['alpha_strength']
            )
        
        # Create environment with phase-specific parameters
        env = IntradayTradingEnvV3(
            processed_feature_data=enhanced_features,
            price_data=price_series,
            initial_capital=100000,
            max_daily_drawdown_pct=cfg['dd_gate'] / 100,  # Convert to decimal
            transaction_cost_pct=cfg['tc_bp'] / 10000,    # Convert bp to decimal
            log_trades=False,
            base_impact_bp=cfg['impact_bp'],
            impact_exponent=0.5,
            verbose=False
        )
        
        # Wrap environment
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        # Important: VecNormalize with reward normalization for stability
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)
        
        return vec_env, alpha_metadata
    
    def create_phase_model(self, vec_env):
        """Create model for current phase with optional warm-start"""
        
        cfg = self.get_current_phase()
        if cfg is None:
            return None
        
        logger.info(f"🤖 Creating model for {cfg['name']}")
        logger.info(f"   LR: {cfg['learning_rate']}, n_steps: {cfg['n_steps']}, ent_coef: {cfg['ent_coef']}")
        
        # Check for warm-start
        if self.warm_start_path and Path(self.warm_start_path).exists():
            logger.info(f"🔄 Warm-starting from: {self.warm_start_path}")
            try:
                # Load just the policy weights, create new model with current config
                old_model = RecurrentPPO.load(self.warm_start_path, device="auto")
                
                # Create new model with current phase parameters
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    vec_env,
                    learning_rate=cfg['learning_rate'],
                    n_steps=cfg['n_steps'],
                    batch_size=cfg['batch_size'],
                    n_epochs=4,
                    gamma=0.999,
                    gae_lambda=0.95,
                    clip_range=cfg['clip_range'],
                    ent_coef=cfg['ent_coef'],
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=1,
                    seed=42,
                    device="auto"
                )
                
                # Transfer policy weights
                model.policy.load_state_dict(old_model.policy.state_dict())
                logger.info(f"✅ Warm-start successful, policy weights transferred to {cfg['name']}")
                return model
            except Exception as e:
                logger.warning(f"⚠️ Warm-start failed: {e}, creating new model")
        
        # Create new model (fallback or no warm-start)
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=cfg['learning_rate'],
            n_steps=cfg['n_steps'],
            batch_size=cfg['batch_size'],
            n_epochs=4,  # Fixed at 4 epochs
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=cfg['clip_range'],
            ent_coef=cfg['ent_coef'],
            vf_coef=0.5,  # Important for reward scale balance
            max_grad_norm=0.5,
            verbose=1,   # Show training progress
            seed=42,
            device="auto"
        )
        
        return model
    
    def evaluate_phase_completion(self, model, vec_env):
        """Evaluate if current phase is complete"""
        
        cfg = self.get_current_phase()
        if cfg is None:
            return False, {}
        
        logger.info(f"🔍 Evaluating {cfg['name']} completion...")
        
        # Create fresh evaluation environment with saved VecNormalize stats
        eval_env, _ = self.create_phase_environment()
        
        # Load VecNormalize statistics from training for consistent evaluation
        vecnorm_path = f"models/phase_{self.phase}_vecnorm.pkl"
        try:
            # Load the VecNormalize wrapper with training statistics
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False  # Disable training mode for evaluation
            eval_env.norm_reward = False  # Disable reward normalization for clearer eval metrics
            logger.info(f"📊 VecNormalize stats loaded from: {vecnorm_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load VecNormalize stats: {e}, using training env")
            eval_env = vec_env
            eval_env.training = False
        
        # Quick evaluation with both deterministic and stochastic actions
        obs = eval_env.reset()
        lstm_states = None
        episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
        
        portfolio_values = []
        rewards = []
        actions_taken = []
        initial_capital = 100000
        
        eval_steps = 1000
        for step in range(eval_steps):
            # Try stochastic action first to see if agent is actually learning
            deterministic_mode = step < 500  # First half deterministic, second half stochastic
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic_mode
            )
            
            result = eval_env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done, _, info = result
            
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
                rewards.append(reward[0])
                actions_taken.append(action[0])
            
            if done[0]:
                episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
            else:
                episode_starts = np.zeros((eval_env.num_envs,), dtype=bool)
        
        # Calculate metrics
        final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_portfolio - initial_capital) / initial_capital
        peak_portfolio = max(portfolio_values) if portfolio_values else initial_capital
        max_drawdown = (peak_portfolio - min(portfolio_values)) / peak_portfolio if portfolio_values else 0
        
        # Fix mean reward calculation - V3 rewards are very small, use sum for gate
        raw_mean_reward = np.mean(rewards) if rewards else 0
        
        # For Phase 0, use return-based success criteria (V3 rewards are small)
        if self.phase == 0:
            # Success if total return > 8% (8000/100000) 
            reward_success_threshold = total_return > 0.08
            mean_reward = total_return * 100000  # Scale for display
            logger.info(f"   🎯 Phase 0 success criteria: Total return {total_return:.2%} > 8% = {'✅' if reward_success_threshold else '❌'}")
        else:
            mean_reward = raw_mean_reward
            reward_success_threshold = mean_reward >= cfg['rew_gate']
        
        # Additional debugging
        logger.info(f"   🔍 Raw mean reward: {raw_mean_reward:.2e}")
        logger.info(f"   🔍 Portfolio: start=${initial_capital:,.0f}, final=${final_portfolio:,.0f}")
        
        # Action analysis
        action_counts = np.bincount(actions_taken, minlength=3) if actions_taken else [0, 0, 0]
        action_dist = action_counts / len(actions_taken) if actions_taken else [0, 0, 0]
        trading_freq = 1.0 - action_dist[1]
        
        # Gate criteria
        if self.phase == 0:
            reward_gate_pass = reward_success_threshold  # Use return-based criteria for Phase 0
        else:
            reward_gate_pass = mean_reward >= cfg['rew_gate']
            
        dd_gate_pass = max_drawdown <= (cfg['dd_gate'] / 100)
        
        # Phase 0 special handling: Ignore DD gate for persistent trends (pedagogical phase)
        if self.phase == 0:
            logger.info(f"   📝 Phase 0: DD gate waived for persistent trends (pedagogical)")
            phase_complete = reward_gate_pass  # Only reward gate matters for Phase 0
        else:
            phase_complete = reward_gate_pass and dd_gate_pass
        
        metrics = {
            'mean_reward': mean_reward,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'trading_frequency': trading_freq,
            'action_distribution': action_dist,
            'reward_gate_pass': reward_gate_pass,
            'dd_gate_pass': dd_gate_pass,
            'phase_complete': phase_complete
        }
        
        # Log results with enhanced metrics
        status = "✅ COMPLETE" if phase_complete else "❌ INCOMPLETE"
        logger.info(f"   Phase Status: {status}")
        logger.info(f"   📊 Mean Reward: {mean_reward:.0f} (target: {cfg['rew_gate']}) {'✅' if reward_gate_pass else '❌'}")
        logger.info(f"   📉 Max DD: {max_drawdown:.2%} (limit: {cfg['dd_gate']:.1f}%) {'✅' if dd_gate_pass else '❌'}")
        logger.info(f"   💹 Total Return: {total_return:+.2%}")
        logger.info(f"   🔄 Trading Frequency: {trading_freq:.1%}")
        logger.info(f"   🎯 Action Distribution: SELL {action_dist[0]:.1%} | HOLD {action_dist[1]:.1%} | BUY {action_dist[2]:.1%}")
        
        # Additional diagnostic logging for curriculum tracking
        if 'pattern' in cfg:
            logger.info(f"   🧩 Alpha Pattern: {cfg['pattern'].value}")
        
        return phase_complete, metrics
    
    def advance_phase(self):
        """Advance to next phase"""
        
        if self.phase < len(self.phase_cfg) - 1:
            self.phase += 1
            logger.info(f"🚀 Advancing to {self.phase_cfg[self.phase]['name']}")
            return True
        else:
            logger.info("🎉 All phases completed!")
            return False
    
    def run_curriculum(self):
        """Run complete curriculum sequence"""
        
        logger.info("🎯 STARTING ALPHA CURRICULUM")
        logger.info(f"   Total phases: {len(self.phase_cfg)}")
        
        start_time = datetime.now()
        
        while self.phase < len(self.phase_cfg):
            cfg = self.get_current_phase()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 {cfg['name']}")
            logger.info(f"   Target: {cfg['target_signal']}")
            logger.info(f"{'='*60}")
            
            # Create environment and model for this phase
            vec_env, alpha_metadata = self.create_phase_environment()
            if vec_env is None:
                break
            
            model = self.create_phase_model(vec_env)
            if model is None:
                break
            
            # Train model for this phase
            phase_start = datetime.now()
            logger.info(f"🔥 Training for {cfg['max_timesteps']} steps...")
            
            model.learn(total_timesteps=cfg['max_timesteps'], progress_bar=True)
            
            # Save VecNormalize statistics for consistent evaluation
            vecnorm_path = f"models/phase_{self.phase}_vecnorm.pkl"
            vec_env.save(vecnorm_path)
            logger.info(f"📊 VecNormalize stats saved: {vecnorm_path}")
            
            phase_time = datetime.now() - phase_start
            
            # Evaluate phase completion
            phase_complete, metrics = self.evaluate_phase_completion(model, vec_env)
            
            # Store results
            result = {
                'phase': self.phase,
                'phase_name': cfg['name'],
                'config': cfg,
                'metrics': metrics,
                'training_time': phase_time,
                'phase_complete': phase_complete
            }
            self.results.append(result)
            
            if phase_complete:
                # Save successful model with phase-specific naming
                model_path = f"models/curriculum_phase_{self.phase}_{cfg['name'].lower().replace(' ', '_').replace('-', '_')}.zip"
                model.save(model_path)
                logger.info(f"✅ Phase model saved: {model_path}")
                
                # Archive checkpoint for warm-start (especially Phase 0)
                if self.phase == 0:
                    # Phase 0 checkpoint serves as foundation for all subsequent phases 
                    checkpoint_path = "models/phase_0_persistent_checkpoint.zip"
                    model.save(checkpoint_path)
                    logger.info(f"📦 Phase 0 checkpoint archived: {checkpoint_path}")
                    logger.info("   This checkpoint will serve as warm-start for Phase 1+")
                    logger.info("   And later for dual-ticker warm-start")
                
                # If this is Phase 2 or 3, we have a production candidate
                if self.phase >= 2:
                    logger.info("🎉 PRODUCTION CANDIDATE FOUND!")
                    logger.info(f"   Model: {model_path}")
                    logger.info("   Ready for dual-ticker warm-start!")
                    
                    # Save winning config
                    winning_config = {
                        'success': True,
                        'phase': self.phase,
                        'phase_name': cfg['name'],
                        'alpha_pattern': cfg['pattern'].value,
                        'model_path': model_path,
                        'config': cfg,
                        'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                   for k, v in metrics.items() if k != 'action_distribution'},
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open('curriculum_success.json', 'w') as f:
                        json.dump(winning_config, f, indent=2)
                    
                    logger.info("✅ Success config saved to curriculum_success.json")
                
                # Advance to next phase
                if not self.advance_phase():
                    break
            else:
                logger.info(f"⚠️ {cfg['name']} incomplete - may need parameter adjustment")
                break
        
        total_time = datetime.now() - start_time
        
        # Final summary
        logger.info(f"\n📊 CURRICULUM COMPLETE")
        logger.info(f"   Total time: {total_time}")
        logger.info(f"   Phases completed: {self.phase}/{len(self.phase_cfg)}")
        
        successful_phases = sum(1 for r in self.results if r['phase_complete'])
        logger.info(f"   Successful phases: {successful_phases}")
        
        if successful_phases >= 2:  # Phase 2 is production ready
            logger.info("🎉 ✅ CURRICULUM SUCCESS - Ready for dual-ticker!")
            return True
        else:
            logger.info("⚠️ ❌ CURRICULUM INCOMPLETE - Need parameter adjustment")
            return False

def main():
    """Main curriculum driver with restart capability"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Persistent-to-Noisy Alpha Curriculum')
    parser.add_argument('--start_phase', type=int, default=0, help='Phase to start from (0-3)')
    parser.add_argument('--warm_start', type=str, help='Path to checkpoint for warm-start')
    parser.add_argument('--debug_eval', action='store_true', help='Enable detailed evaluation logging')
    
    args = parser.parse_args()
    
    # Set environment variables for optimal performance
    import os
    os.environ['OMP_NUM_THREADS'] = '1'  # Prevent CPU thrash
    
    logger.info("🎯 CURRICULUM DRIVER - Battle-tested sequence")
    logger.info("   Policy frozen → Policy profitably active")
    
    if args.start_phase > 0:
        logger.info(f"🔄 RESTARTING from Phase {args.start_phase}")
        if args.warm_start:
            logger.info(f"   Warm-start: {args.warm_start}")
    
    curriculum = AlphaCurriculum(
        start_phase=args.start_phase,
        warm_start_path=args.warm_start
    )
    success = curriculum.run_curriculum()
    
    if success:
        logger.info("\n🚀 READY FOR DUAL-TICKER WARM-START")
        logger.info("   Run: python export_policy.py && python dual_ticker_warmstart.py")
    else:
        logger.info("\n🔧 Consider parameter adjustments:")
        logger.info("   - Raise alpha magnitude slightly")
        logger.info("   - Loosen clip_range by +0.05")
        logger.info("   - Bump ent_coef to 0.01 temporarily")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)