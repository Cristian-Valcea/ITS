#!/usr/bin/env python3
"""
🚀 SIMPLE SINGLE-TICKER PROTOTYPE - AM SESSION
Quick validation with standard RecurrentPPO + v2 reward system
Test new reward components before full transformer implementation
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

# Import our components
from src.gym_env.intraday_trading_env import IntradayTradingEnv
from src.gym_env.dual_reward_v2 import DualTickerRewardV2

class SimpleTransformerEnv(IntradayTradingEnv):
    """Simple environment with v2 reward system for AM session validation"""
    
    def __init__(self, *args, **kwargs):
        # Initialize reward system
        self.reward_calculator = DualTickerRewardV2(
            tc_bp=1.0,                  # REALISTIC: 1bp transaction cost
            market_impact_bp=0.5,       # Market impact
            lambda_turnover=0.001,      # Turnover penalty
            target_turnover=2.0,        # Annual target
            beta_volatility=0.01,       # Volatility penalty
            verbose=False
        )
        
        super().__init__(*args, **kwargs)
        
        logger.info("🧠 SimpleTransformerEnv initialized:")
        logger.info(f"   💰 Transaction costs: 1.0bp + 0.5bp impact")
        logger.info(f"   🎯 Reward: Risk-adjusted P&L with turnover penalty")
    
    def step(self, action):
        """Enhanced step with v2 reward calculation"""
        # Store previous values
        prev_portfolio = self.portfolio_value
        prev_position = getattr(self, '_prev_position_quantity', 0.0)
        
        # Execute parent step
        obs, reward, done, truncated, info = super().step(action)
        
        # Calculate trade value
        curr_position = getattr(self, 'position_quantity', 0.0)
        trade_value = abs(curr_position - prev_position) * self.price_data.iloc[self.current_step-1]
        self._prev_position_quantity = curr_position
        
        # Calculate v2 reward
        v2_reward, reward_components = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio,
            curr_portfolio_value=self.portfolio_value,
            nvda_trade_value=trade_value,
            msft_trade_value=0.0,  # Single ticker
            nvda_position=curr_position,
            msft_position=0.0,  # Single ticker
            nvda_price=self.price_data.iloc[self.current_step-1],
            msft_price=510.0,  # Dummy price
            step=self.current_step
        )
        
        # Use v2 reward
        reward = v2_reward
        
        # Add reward breakdown to info
        info.update({
            'reward_breakdown': reward_components.to_dict(),
            'v2_reward': v2_reward
        })
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset with reward calculator reset"""
        self.reward_calculator.reset()
        self._prev_position_quantity = 0.0
        return super().reset(**kwargs)

def create_training_environment(seed=0):
    """Create simple training environment with realistic friction"""
    logger.info("🏗️ Creating simple transformer training environment...")
    
    # Set seed
    np.random.seed(seed)
    
    # Generate training data (NVDA-like)
    n_periods = 15000  # 15K periods for training
    trading_days = pd.date_range('2024-01-01', periods=n_periods, freq='1min')
    
    # Generate realistic NVDA price series
    base_price = 170.0
    returns = np.random.normal(0.0001, 0.02, n_periods)  # 2% daily vol
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create price series
    price_series = pd.Series(prices, index=trading_days)
    
    # Create mock feature data (12 features standard)
    feature_data = np.random.randn(n_periods, 12).astype(np.float32)
    
    # Create environment with REALISTIC FRICTION
    env = SimpleTransformerEnv(
        processed_feature_data=feature_data,
        price_data=price_series,
        initial_capital=100000,
        transaction_cost_pct=0.0001,    # 1bp base cost (enhanced by v2 reward)
        max_daily_drawdown_pct=0.15,    # Training drawdown
        hourly_turnover_cap=10.0,       # High cap, using v2 turnover penalty
        reward_scaling=1.0,             # No scaling, using raw v2 rewards
        log_trades=False
    )
    
    # Wrap with Monitor
    env = Monitor(env)
    
    logger.info("✅ Simple transformer environment created")
    logger.info(f"🌱 Seed: {seed}")
    logger.info(f"📊 Training periods: {n_periods:,}")
    logger.info(f"💰 Initial capital: ${env.env.initial_capital:,.2f}")
    logger.info(f"🎯 REALISTIC FRICTION: 1.0bp TC + 0.5bp impact")
    logger.info(f"📈 Price range: ${price_series.min():.2f} - ${price_series.max():.2f}")
    
    return env

def create_simple_model(env, config):
    """Create RecurrentPPO model with standard MlpLstmPolicy"""
    logger.info("🧠 Creating RecurrentPPO with standard policy...")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=10,
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=config['max_grad_norm'],
        tensorboard_log="logs/",
        verbose=1,
        seed=0,
        device="auto"
    )
    
    logger.info("✅ Standard RecurrentPPO model created")
    logger.info(f"🎯 Policy: MlpLstmPolicy (standard)")
    logger.info(f"📊 Learning rate: {config['learning_rate']}")
    logger.info(f"🔄 Rollout steps: {config['n_steps']}")
    
    return model

def run_strict_evaluation(model_path, vecnorm_path, seed=0):
    """Run strict 2% drawdown evaluation"""
    logger.info("🔍 STRICT EVALUATION: 5K steps, 2% DD gate")
    
    # Create evaluation environment
    eval_env = create_training_environment(seed=seed)
    eval_env.env.max_daily_drawdown_pct = 0.02  # STRICT 2% DD
    
    vec_env = DummyVecEnv([lambda: eval_env])
    
    # Load VecNormalize if exists
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model = RecurrentPPO.load(model_path, env=vec_env)
    
    # Run evaluation
    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    
    total_steps = 0
    total_reward = 0
    portfolio_values = []
    initial_capital = 100000
    peak_portfolio = initial_capital
    max_drawdown = 0
    
    logger.info("🚀 Running 5K evaluation steps...")
    
    for step in range(5000):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        
        obs, reward, done, info = vec_env.step(action)
        
        total_reward += reward[0]
        total_steps += 1
        
        # Extract portfolio info
        if 'portfolio_value' in info[0]:
            portfolio_value = info[0]['portfolio_value']
            portfolio_values.append(portfolio_value)
            
            # Update peak and drawdown
            if portfolio_value > peak_portfolio:
                peak_portfolio = portfolio_value
            
            current_drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Progress logging
        if step % 1000 == 0 and step > 0:
            current_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
            current_return = (current_portfolio - initial_capital) / initial_capital
            logger.info(f"Step {step:4d}: Portfolio ${current_portfolio:8,.2f} ({current_return:+.2%}), DD {max_drawdown:.2%}")
        
        # Check episode termination
        if done[0]:
            episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
            logger.info(f"📊 Episode ended at step {step}")
            break
        else:
            episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
    
    # Final results
    final_portfolio = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_portfolio - initial_capital) / initial_capital
    
    logger.info("🏁 STRICT EVALUATION COMPLETED!")
    logger.info(f"📊 RESULTS (Seed {seed}):")
    logger.info(f"   Steps completed: {total_steps:,}")
    logger.info(f"   Final portfolio: ${final_portfolio:,.2f}")
    logger.info(f"   Total return: {total_return:+.2%}")
    logger.info(f"   Peak portfolio: ${peak_portfolio:,.2f}")
    logger.info(f"   Maximum drawdown: {max_drawdown:.2%}")
    
    # Gate evaluation
    return_pass = total_return >= 0.01  # ≥+1%
    drawdown_pass = max_drawdown <= 0.02  # ≤2%
    stability_pass = total_steps >= 4000  # No early termination
    
    logger.info("🎯 GATE EVALUATION:")
    logger.info(f"   📈 Return ≥ +1.0%: {'✅ PASS' if return_pass else '❌ FAIL'} ({total_return:+.2%})")
    logger.info(f"   📉 Max DD ≤ 2.0%: {'✅ PASS' if drawdown_pass else '❌ FAIL'} ({max_drawdown:.2%})")
    logger.info(f"   🛡️ Stability: {'✅ PASS' if stability_pass else '❌ FAIL'} ({total_steps:,} steps)")
    
    gate_pass = return_pass and drawdown_pass and stability_pass
    
    if gate_pass:
        logger.info("🎉 ✅ GREEN LIGHT: Simple prototype PASSED gate!")
        logger.info("✅ V2 reward system validated - ready for transformer integration")
    else:
        logger.warning("⚠️ ❌ RED LIGHT: Simple prototype FAILED gate!")
        logger.warning("🛑 ABORT: V2 reward system needs adjustment")
    
    return gate_pass, {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_portfolio': final_portfolio,
        'steps': total_steps,
        'seed': seed
    }

def main():
    """Main AM session: 10K training + 5K evaluation with v2 reward validation"""
    logger.info("🚀 SIMPLE PROTOTYPE - AM SESSION")
    logger.info("🎯 Day 1 AM: Validate v2 reward system before transformer")
    
    start_time = datetime.now()
    
    # Configuration (adjusted for simple model)
    config = {
        'learning_rate': 1e-4,      # Standard learning rate
        'n_steps': 2048,            # Standard rollout length
        'batch_size': 256,          # Good balance
        'gamma': 0.999,             # Discount factor
        'gae_lambda': 0.95,         # GAE parameter
        'clip_range': 0.1,          # Conservative clipping
        'ent_coef': 0.01,           # Standard exploration
        'max_grad_norm': 0.5        # Gradient clipping
    }
    
    try:
        # Step 1: Create training environment (seed=0)
        train_env = create_training_environment(seed=0)
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Add VecNormalize for stability
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,      # Don't normalize observations
            norm_reward=True,    # Normalize rewards for stability
            clip_reward=10.0,    # Clip extreme rewards
            gamma=config['gamma']
        )
        
        # Step 2: Create simple model
        model = create_simple_model(vec_env, config)
        
        # Step 3: Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path="./models/checkpoints/",
            name_prefix="simple_transformer_v2"
        )
        
        # Step 4: Train 10K steps
        logger.info("🎯 Starting 10K simple training...")
        logger.info("📈 Monitor with: tensorboard --logdir logs/")
        
        model.learn(
            total_timesteps=10000,
            callback=[checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Step 5: Save model
        model_path = "models/simple_transformer_v2_10k.zip"
        vecnorm_path = "models/simple_transformer_v2_vecnorm.pkl"
        
        model.save(model_path)
        vec_env.save(vecnorm_path)
        
        training_time = datetime.now() - start_time
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info(f"⏱️ Training duration: {training_time}")
        logger.info(f"💾 Model saved: {model_path}")
        
        # Step 6: Strict evaluation
        logger.info("🔍 Starting strict 2% DD gate evaluation...")
        
        gate_pass, results = run_strict_evaluation(model_path, vecnorm_path, seed=0)
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        logger.info("🏁 AM SESSION COMPLETED!")
        logger.info(f"⏱️ Total duration: {total_duration}")
        logger.info(f"🎯 Gate result: {'✅ PASS' if gate_pass else '❌ FAIL'}")
        
        # Save results for mid-day session
        results_file = "simple_transformer_v2_am_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'gate_pass': gate_pass,
                'results': results,
                'config': config,
                'training_duration': str(training_time),
                'total_duration': str(total_duration)
            }, f, indent=2)
        logger.info(f"📄 Results saved: {results_file}")
        
        return gate_pass
        
    except Exception as e:
        logger.error(f"❌ AM session failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 ✅ AM SESSION SUCCESS - V2 reward system validated")
        print("📋 Next: Full transformer implementation ready")
    else:
        print("🛑 ❌ AM SESSION FAILED - V2 reward system needs adjustment")