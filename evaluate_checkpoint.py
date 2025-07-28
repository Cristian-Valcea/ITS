#!/usr/bin/env python3
"""
Quick evaluation of the 10K checkpoint to see training progress
"""

import os
import sys
import numpy as np
import pandas as pd
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

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.gym_env.dual_ticker_trading_env import DualTickerTradingEnv

print("🔍 Evaluating 10K Checkpoint")
print("=" * 50)

# Load the checkpoint
checkpoint_path = "checkpoints/dual_ticker_50k_10000_steps.zip"

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

# Create test environment
print("🏗️ Creating test environment...")
n_periods = 1000
trading_days = pd.date_range('2025-01-01', periods=n_periods, freq='1min')

nvda_prices = pd.Series(
    170 + np.cumsum(np.random.randn(n_periods) * 0.01), 
    index=trading_days
)
nvda_data = np.random.randn(n_periods, 12).astype(np.float32)

msft_prices = pd.Series(
    510 + np.cumsum(np.random.randn(n_periods) * 0.01),
    index=trading_days
)
msft_data = np.random.randn(n_periods, 12).astype(np.float32)

env = DualTickerTradingEnv(
    nvda_data=nvda_data,
    msft_data=msft_data,
    nvda_prices=nvda_prices,
    msft_prices=msft_prices,
    trading_days=trading_days,
    initial_capital=10000,
    tc_bp=1.0,
    reward_scaling=0.01,
    max_daily_drawdown_pct=0.25
)

vec_env = DummyVecEnv([lambda: env])

# Load model from checkpoint
print(f"📦 Loading model from: {checkpoint_path}")
try:
    model = RecurrentPPO.load(checkpoint_path, env=vec_env)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Run evaluation
print("🎯 Running 500-step evaluation...")
obs = vec_env.reset()
total_reward = 0
episode_rewards = []
portfolio_values = []
actions_taken = []
steps = 0

for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
    total_reward += reward[0]
    steps += 1
    
    # Track metrics
    portfolio_values.append(info[0]['portfolio_value'])
    actions_taken.append(action[0])
    
    # Check for episode completion
    if done[0]:
        episode_rewards.append(total_reward)
        total_reward = 0
        print(f"📊 Episode completed at step {i+1}")
        obs = vec_env.reset()

# Calculate performance metrics
final_portfolio = portfolio_values[-1] if portfolio_values else 10000
total_return = (final_portfolio - 10000) / 10000 * 100
avg_reward_per_step = total_reward / steps if steps > 0 else 0

print("\n📈 EVALUATION RESULTS")
print("=" * 30)
print(f"Total steps evaluated: {steps}")
print(f"Final portfolio value: ${final_portfolio:.2f}")
print(f"Total return: {total_return:.2f}%")
print(f"Average reward per step: {avg_reward_per_step:.4f}")
print(f"Episodes completed: {len(episode_rewards)}")

if episode_rewards:
    print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
    print(f"Best episode reward: {max(episode_rewards):.4f}")

# Action distribution
action_counts = np.bincount(actions_taken, minlength=9)
action_names = ["SELL_BOTH", "SELL_NVDA_HOLD_MSFT", "SELL_NVDA_BUY_MSFT", 
                "HOLD_NVDA_SELL_MSFT", "HOLD_BOTH", "HOLD_NVDA_BUY_MSFT",
                "BUY_NVDA_SELL_MSFT", "BUY_NVDA_HOLD_MSFT", "BUY_BOTH"]

print(f"\n🎯 ACTION DISTRIBUTION")
print("=" * 30)
for i, (count, name) in enumerate(zip(action_counts, action_names)):
    pct = count / len(actions_taken) * 100 if actions_taken else 0
    print(f"Action {i} ({name}): {count} ({pct:.1f}%)")

# Portfolio trajectory analysis
if len(portfolio_values) > 10:
    print(f"\n💼 PORTFOLIO TRAJECTORY")
    print("=" * 30)
    print(f"Starting value: ${portfolio_values[0]:.2f}")
    print(f"Peak value: ${max(portfolio_values):.2f}")
    print(f"Final value: ${portfolio_values[-1]:.2f}")
    
    # Simple drawdown calculation
    peak_value = 10000
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak_value:
            peak_value = value
        drawdown = (peak_value - value) / peak_value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print(f"Maximum drawdown: {max_drawdown:.2f}% ({max_drawdown*100:.1f}%)")

print(f"\n✅ 10K Checkpoint Evaluation Complete!")
print(f"🚀 Next: Resume full 50K training or continue to 200K")