#!/usr/bin/env python3
"""
Integration guide for Sharpe-based early stopping in IntradayJules.

Replace reward-based early stopping with rolling Sharpe early stopping.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rolling_sharpe_tracker import RollingSharpeTracker, SharpeEarlyStoppingCallback

class IntradayJulesEarlyStoppingIntegration:
    """
    Integration wrapper for IntradayJules training with Sharpe-based early stopping.
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 patience_episodes: int = 50,
                 min_improvement_threshold: float = 0.05):
        """
        Initialize Sharpe-based early stopping for IntradayJules.
        
        Args:
            window_size: Rolling window for Sharpe calculation (20 episodes)
            patience_episodes: Episodes to wait without improvement (50)
            min_improvement_threshold: Minimum Sharpe improvement (0.05)
        """
        self.sharpe_tracker = RollingSharpeTracker(
            window_size=window_size,
            patience_episodes=patience_episodes,
            min_improvement_threshold=min_improvement_threshold,
            output_file="logs/sharpe_early_stopping.csv"
        )
        
        self.early_stopping = SharpeEarlyStoppingCallback(self.sharpe_tracker)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"🎯 Sharpe-based early stopping initialized:")
        self.logger.info(f"   Window: {window_size} episodes")
        self.logger.info(f"   Patience: {patience_episodes} episodes")
        self.logger.info(f"   Min improvement: {min_improvement_threshold}")
    
    def check_early_stopping(self, episode_summary: dict, total_reward: float) -> bool:
        """
        Check if training should stop based on rolling Sharpe.
        
        Args:
            episode_summary: Episode summary from IntradayTradingEnv
            total_reward: Cumulative reward for the episode
            
        Returns:
            bool: True if training should stop
        """
        return self.early_stopping.on_episode_end(episode_summary, total_reward)
    
    def get_status(self) -> dict:
        """Get current early stopping status."""
        return self.sharpe_tracker.get_early_stop_status()
    
    def plot_progress(self, save_path: str = "logs/sharpe_progress.png"):
        """Plot Sharpe evolution."""
        self.sharpe_tracker.plot_sharpe_evolution(save_path)

# Example integration with existing training code
def example_stable_baselines3_integration():
    """
    Example of integrating Sharpe early stopping with Stable-Baselines3 training.
    """
    print("🎯 STABLE-BASELINES3 INTEGRATION EXAMPLE")
    print("=" * 50)
    
    # Initialize Sharpe-based early stopping
    sharpe_early_stop = IntradayJulesEarlyStoppingIntegration(
        window_size=20,
        patience_episodes=50,
        min_improvement_threshold=0.05
    )
    
    print("""
# In your training script, replace this:
# ❌ OLD WAY (reward-based early stopping)
from stable_baselines3.common.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='rollout/ep_rew_mean',  # ❌ Monitoring raw reward
    min_delta=0.01,
    patience=50,
    verbose=1
)

# ✅ NEW WAY (Sharpe-based early stopping)
from sharpe_early_stopping_integration import IntradayJulesEarlyStoppingIntegration

sharpe_early_stop = IntradayJulesEarlyStoppingIntegration()

# Custom callback for SB3
class SharpeEarlyStoppingCallback(BaseCallback):
    def __init__(self, sharpe_tracker):
        super().__init__()
        self.sharpe_tracker = sharpe_tracker
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def _on_step(self) -> bool:
        # Accumulate rewards
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if self.locals['dones'][0]:
            # Get episode summary from environment
            if 'episode_summary' in self.locals['infos'][0]:
                episode_summary = self.locals['infos'][0]['episode_summary']
                
                # Check early stopping
                should_stop = self.sharpe_tracker.check_early_stopping(
                    episode_summary, self.current_episode_reward
                )
                
                if should_stop:
                    print("🛑 Early stopping triggered by Sharpe stagnation")
                    return False  # Stop training
                
                self.current_episode_reward = 0
        
        return True

# Use in model training
model.learn(
    total_timesteps=1000000,
    callback=SharpeEarlyStoppingCallback(sharpe_early_stop)
)
""")

def example_custom_training_loop():
    """
    Example of integrating with custom training loop.
    """
    print("\n🎯 CUSTOM TRAINING LOOP INTEGRATION")
    print("=" * 50)
    
    print("""
# In your custom training loop:

from sharpe_early_stopping_integration import IntradayJulesEarlyStoppingIntegration

# Initialize
sharpe_early_stop = IntradayJulesEarlyStoppingIntegration()

for episode in range(max_episodes):
    # Run episode
    obs = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    # ✅ NEW: Check Sharpe-based early stopping
    if 'episode_summary' in info:
        should_stop = sharpe_early_stop.check_early_stopping(
            info['episode_summary'], total_reward
        )
        
        if should_stop:
            print(f"🛑 Training stopped at episode {episode} due to Sharpe stagnation")
            break
    
    # Periodic status logging
    if episode % 20 == 0:
        status = sharpe_early_stop.get_status()
        print(f"Episode {episode}: Sharpe={status['current_sharpe']:.3f}, "
              f"Patience={status['patience_remaining']}")

# Plot final results
sharpe_early_stop.plot_progress("final_sharpe_evolution.png")
""")

def example_ray_rllib_integration():
    """
    Example of integrating with Ray RLLib.
    """
    print("\n🎯 RAY RLLIB INTEGRATION")
    print("=" * 50)
    
    print("""
# For Ray RLLib integration:

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Custom callback for RLLib
class SharpeEarlyStoppingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.sharpe_tracker = IntradayJulesEarlyStoppingIntegration()
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Get episode info
        info = episode.last_info_for()
        
        if 'episode_summary' in info:
            total_reward = episode.total_reward
            should_stop = self.sharpe_tracker.check_early_stopping(
                info['episode_summary'], total_reward
            )
            
            if should_stop:
                # Signal to stop training
                episode.custom_metrics['should_stop'] = 1.0

# Use in tune.run
tune.run(
    PPO,
    config={
        "env": "IntradayTradingEnv",
        "callbacks": SharpeEarlyStoppingCallback,
        # ... other config
    },
    stop={"custom_metrics/should_stop_mean": 0.5}  # Stop when signal received
)
""")

def main():
    """Show integration examples."""
    print("🎯 SHARPE-BASED EARLY STOPPING INTEGRATION GUIDE")
    print("=" * 60)
    
    print("""
🔍 PROBLEM:
   Training plateaus after ~100 episodes because early-stopping 
   triggers on stagnant reward, not risk-adjusted performance.

✅ SOLUTION:
   Track 20-episode rolling Sharpe ratio and stop when THAT stalls,
   not raw reward. This prevents premature stopping when the agent
   is still learning better risk-adjusted strategies.

📊 KEY BENEFITS:
   • Prevents premature training termination
   • Focuses on risk-adjusted performance
   • Allows continued learning even if raw rewards plateau
   • Better final model performance
""")
    
    # Show integration examples
    example_stable_baselines3_integration()
    example_custom_training_loop()
    example_ray_rllib_integration()
    
    print(f"\n💡 QUICK START:")
    print(f"1. Replace reward-based early stopping with SharpeEarlyStoppingIntegration")
    print(f"2. Pass episode_summary and total_reward to check_early_stopping()")
    print(f"3. Stop training when it returns True")
    print(f"4. Monitor logs/sharpe_early_stopping.csv for progress")
    
    print(f"\n🔧 TUNING PARAMETERS:")
    print(f"• window_size=20: Rolling window for Sharpe calculation")
    print(f"• patience_episodes=50: Episodes to wait without improvement")
    print(f"• min_improvement_threshold=0.05: Minimum Sharpe improvement")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())