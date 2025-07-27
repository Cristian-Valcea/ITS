#!/usr/bin/env python3
"""
Rolling Sharpe ratio tracker for proper early-stopping in RL trading.

Problem: Early-stopping callback triggers on stagnant reward, not P&L
Solution: Track 20-episode rolling Sharpe and stop when that stalls, not raw reward
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

class RollingSharpeTracker:
    """
    Track rolling Sharpe ratio for intelligent early-stopping.
    
    Stops training when risk-adjusted performance stalls, not raw rewards.
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 min_episodes_for_sharpe: int = 10,
                 patience_episodes: int = 50,
                 min_improvement_threshold: float = 0.05,
                 risk_free_rate: float = 0.02,  # 2% annual risk-free rate
                 output_file: str = "logs/rolling_sharpe_tracking.csv"):
        """
        Initialize rolling Sharpe tracker.
        
        Args:
            window_size: Rolling window for Sharpe calculation (default: 20 episodes)
            min_episodes_for_sharpe: Minimum episodes before calculating Sharpe
            patience_episodes: Episodes to wait for improvement before early stop
            min_improvement_threshold: Minimum Sharpe improvement to reset patience
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            output_file: CSV file to save tracking data
        """
        self.window_size = window_size
        self.min_episodes_for_sharpe = min_episodes_for_sharpe
        self.patience_episodes = patience_episodes
        self.min_improvement_threshold = min_improvement_threshold
        self.risk_free_rate = risk_free_rate
        self.output_file = output_file
        
        # Rolling data storage
        self.episode_returns = deque(maxlen=window_size)
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_durations = deque(maxlen=window_size)
        
        # Tracking variables
        self.episode_count = 0
        self.best_sharpe = -np.inf
        self.episodes_without_improvement = 0
        self.should_stop = False
        
        # History for analysis
        self.sharpe_history = []
        self.return_history = []
        self.reward_history = []
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"🎯 RollingSharpeTracker initialized: window={window_size}, patience={patience_episodes}")
    
    def add_episode(self, episode_summary: Dict, total_reward: float) -> Dict:
        """
        Add episode data and check for early stopping.
        
        Args:
            episode_summary: Episode summary from trading environment
            total_reward: Cumulative reward for the episode
            
        Returns:
            Dict with early stopping decision and metrics
        """
        self.episode_count += 1
        
        # Extract key metrics
        return_pct = episode_summary.get('total_return_pct', 0.0)
        duration_hours = episode_summary.get('episode_duration_hours', 1.0)
        
        # Add to rolling windows
        self.episode_returns.append(return_pct)
        self.episode_rewards.append(total_reward)
        self.episode_durations.append(duration_hours)
        
        # Add to history
        self.return_history.append(return_pct)
        self.reward_history.append(total_reward)
        
        # Calculate rolling Sharpe if we have enough data
        current_sharpe = None
        if len(self.episode_returns) >= self.min_episodes_for_sharpe:
            current_sharpe = self._calculate_rolling_sharpe()
            self.sharpe_history.append(current_sharpe)
            
            # Check for improvement
            if current_sharpe > self.best_sharpe + self.min_improvement_threshold:
                self.best_sharpe = current_sharpe
                self.episodes_without_improvement = 0
                self.logger.info(f"🎯 NEW BEST SHARPE: {current_sharpe:.3f} (Episode {self.episode_count})")
            else:
                self.episodes_without_improvement += 1
            
            # Check early stopping condition
            if self.episodes_without_improvement >= self.patience_episodes:
                self.should_stop = True
                self.logger.warning(f"🛑 EARLY STOP TRIGGERED: {self.episodes_without_improvement} episodes without Sharpe improvement")
        
        # Prepare result
        result = {
            'episode': self.episode_count,
            'current_sharpe': current_sharpe,
            'best_sharpe': self.best_sharpe,
            'episodes_without_improvement': self.episodes_without_improvement,
            'should_stop': self.should_stop,
            'return_pct': return_pct,
            'total_reward': total_reward,
            'rolling_mean_return': np.mean(self.episode_returns) if self.episode_returns else 0.0,
            'rolling_std_return': np.std(self.episode_returns) if len(self.episode_returns) > 1 else 0.0
        }
        
        # Save data periodically
        if self.episode_count % 10 == 0:
            self._save_tracking_data()
        
        # Log progress
        if current_sharpe is not None:
            self.logger.log(logging.DEBUG if self.episode_count % 10 != 0 else logging.INFO,
                          f"📊 Episode {self.episode_count}: Sharpe={current_sharpe:.3f}, "
                          f"Return={return_pct:+.2f}%, Patience={self.episodes_without_improvement}/{self.patience_episodes}")
        
        return result
    
    def _calculate_rolling_sharpe(self) -> float:
        """Calculate Sharpe ratio for current rolling window."""
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        durations = np.array(self.episode_durations)
        
        # Annualize returns (assuming episodes are intraday)
        # Convert episode returns to daily, then annualize
        daily_returns = returns  # Already in percentage
        
        # Calculate excess returns (subtract risk-free rate)
        risk_free_daily = (self.risk_free_rate / 252) * 100  # Daily risk-free rate in %
        excess_returns = daily_returns - risk_free_daily
        
        # Calculate Sharpe ratio
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _save_tracking_data(self):
        """Save tracking data to CSV."""
        if not self.sharpe_history:
            return
        
        # Create DataFrame with all tracking data
        data = []
        for i, (sharpe, return_pct, reward) in enumerate(zip(
            self.sharpe_history, 
            self.return_history[-len(self.sharpe_history):], 
            self.reward_history[-len(self.sharpe_history):]
        )):
            data.append({
                'episode': i + self.min_episodes_for_sharpe,
                'rolling_sharpe': sharpe,
                'return_pct': return_pct,
                'total_reward': reward,
                'best_sharpe': self.best_sharpe,
                'episodes_without_improvement': self.episodes_without_improvement
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        self.logger.debug(f"💾 Rolling Sharpe tracking data saved: {self.output_file}")
    
    def get_early_stop_status(self) -> Dict:
        """Get current early stopping status."""
        return {
            'should_stop': self.should_stop,
            'current_sharpe': self.sharpe_history[-1] if self.sharpe_history else None,
            'best_sharpe': self.best_sharpe,
            'episodes_without_improvement': self.episodes_without_improvement,
            'patience_remaining': max(0, self.patience_episodes - self.episodes_without_improvement),
            'total_episodes': self.episode_count
        }
    
    def reset_patience(self):
        """Reset patience counter (useful for manual intervention)."""
        self.episodes_without_improvement = 0
        self.should_stop = False
        self.logger.info("🔄 Early stopping patience reset")
    
    def plot_sharpe_evolution(self, save_path: str = None):
        """Plot rolling Sharpe evolution for analysis."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib not available for plotting")
            return
        
        if not self.sharpe_history:
            self.logger.warning("No Sharpe history to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('🎯 Rolling Sharpe Early-Stopping Analysis', fontsize=16, fontweight='bold')
        
        episodes = range(self.min_episodes_for_sharpe, self.min_episodes_for_sharpe + len(self.sharpe_history))
        
        # 1. Rolling Sharpe ratio
        ax1.plot(episodes, self.sharpe_history, 'b-', linewidth=2, label='Rolling Sharpe')
        ax1.axhline(self.best_sharpe, color='green', linestyle='--', alpha=0.7, label=f'Best Sharpe: {self.best_sharpe:.3f}')
        ax1.set_ylabel('Rolling Sharpe Ratio')
        ax1.set_title(f'Rolling Sharpe ({self.window_size}-episode window)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark early stop point if triggered
        if self.should_stop:
            stop_episode = len(self.sharpe_history) + self.min_episodes_for_sharpe - 1
            ax1.axvline(stop_episode, color='red', linestyle=':', alpha=0.8, label='Early Stop')
        
        # 2. Episode returns
        ax2.plot(range(1, len(self.return_history) + 1), self.return_history, 'g-', alpha=0.7, label='Episode Returns')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Return %')
        ax2.set_title('Episode Returns Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Patience counter
        patience_history = []
        current_patience = 0
        for i, sharpe in enumerate(self.sharpe_history):
            if i == 0 or sharpe > max(self.sharpe_history[:i]) + self.min_improvement_threshold:
                current_patience = 0
            else:
                current_patience += 1
            patience_history.append(current_patience)
        
        ax3.plot(episodes, patience_history, 'r-', linewidth=2, label='Episodes w/o Improvement')
        ax3.axhline(self.patience_episodes, color='red', linestyle='--', alpha=0.7, label=f'Patience Limit: {self.patience_episodes}')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episodes w/o Improvement')
        ax3.set_title('Early Stopping Patience Counter')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Sharpe evolution plot saved: {save_path}")
        
        plt.show()

class SharpeEarlyStoppingCallback:
    """
    Early stopping callback for RL training based on rolling Sharpe ratio.
    
    Use this instead of reward-based early stopping for trading agents.
    """
    
    def __init__(self, sharpe_tracker: RollingSharpeTracker):
        self.sharpe_tracker = sharpe_tracker
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_episode_end(self, episode_summary: Dict, total_reward: float) -> bool:
        """
        Called at the end of each episode.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        result = self.sharpe_tracker.add_episode(episode_summary, total_reward)
        
        if result['should_stop']:
            self.logger.warning(f"🛑 EARLY STOPPING: Rolling Sharpe stalled at {result['current_sharpe']:.3f}")
            self.logger.info(f"   Best Sharpe: {result['best_sharpe']:.3f}")
            self.logger.info(f"   Episodes without improvement: {result['episodes_without_improvement']}")
            return True
        
        return False

# Example integration
def example_training_with_sharpe_early_stopping():
    """Example of how to integrate rolling Sharpe early stopping."""
    
    # Initialize tracker
    sharpe_tracker = RollingSharpeTracker(
        window_size=20,
        patience_episodes=50,
        min_improvement_threshold=0.05
    )
    
    early_stopping = SharpeEarlyStoppingCallback(sharpe_tracker)
    
    # Training loop (pseudo-code)
    for episode in range(1000):
        # Run episode
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # Check early stopping based on Sharpe, not reward
        if 'episode_summary' in info:
            should_stop = early_stopping.on_episode_end(info['episode_summary'], total_reward)
            if should_stop:
                print(f"🛑 Training stopped early at episode {episode}")
                break
        
        # Log progress
        if episode % 20 == 0:
            status = sharpe_tracker.get_early_stop_status()
            print(f"Episode {episode}: Sharpe={status['current_sharpe']:.3f}, "
                  f"Patience={status['patience_remaining']}/{sharpe_tracker.patience_episodes}")

if __name__ == "__main__":
    print("🎯 Rolling Sharpe Early-Stopping Tracker")
    print("Use this instead of reward-based early stopping for trading agents")
    example_training_with_sharpe_early_stopping()