#!/usr/bin/env python3
"""
Simple reward tracker for reward-P&L correlation monitoring.

Add this to your training loop to track cumulative rewards per episode.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

class RewardPnLTracker:
    """Track rewards and P&L for correlation analysis."""
    
    def __init__(self, output_file: str = "logs/reward_pnl_tracking.csv"):
        self.output_file = output_file
        self.episode_data = []
        self.current_episode_reward = 0.0
        self.episode_count = 0
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def reset_episode(self):
        """Call at the start of each episode."""
        self.current_episode_reward = 0.0
        self.episode_count += 1
    
    def add_step_reward(self, reward: float):
        """Add reward from current step."""
        self.current_episode_reward += reward
    
    def end_episode(self, episode_summary: Dict):
        """Call at the end of each episode with episode summary."""
        episode_record = {
            'episode': self.episode_count,
            'total_reward': self.current_episode_reward,
            'net_pnl': episode_summary.get('net_pnl_after_fees', 0.0),
            'total_return_pct': episode_summary.get('total_return_pct', 0.0),
            'max_drawdown_pct': episode_summary.get('max_drawdown_pct', 0.0),
            'total_trades': episode_summary.get('total_trades', 0),
            'total_fees': episode_summary.get('total_fees', 0.0),
            'episode_duration_hours': episode_summary.get('episode_duration_hours', 0.0)
        }
        
        self.episode_data.append(episode_record)
        
        # Save every 10 episodes
        if len(self.episode_data) % 10 == 0:
            self.save_data()
    
    def save_data(self):
        """Save tracking data to CSV."""
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        df.to_csv(self.output_file, index=False)
        self.logger.info(f"💾 Reward-P&L tracking data saved: {self.output_file}")
    
    def get_correlation_check(self, min_episodes: int = 20):
        """Quick correlation check."""
        if len(self.episode_data) < min_episodes:
            return {
                'status': 'INSUFFICIENT_DATA',
                'episodes': len(self.episode_data),
                'min_required': min_episodes
            }
        
        df = pd.DataFrame(self.episode_data)
        correlation = df['total_reward'].corr(df['net_pnl'])
        
        return {
            'status': 'HEALTHY' if correlation >= 0.6 else 'REWARD_PNL_MISMATCH',
            'correlation': correlation,
            'episodes': len(self.episode_data),
            'avg_reward': df['total_reward'].mean(),
            'avg_pnl': df['net_pnl'].mean()
        }

# Example integration with training loop
def example_training_integration():
    """
    Example of how to integrate RewardPnLTracker into your training loop.
    """
    
    # Initialize tracker
    tracker = RewardPnLTracker("logs/reward_pnl_tracking.csv")
    
    # Training loop (pseudo-code)
    for episode in range(1000):
        # Start episode
        tracker.reset_episode()
        obs = env.reset()
        done = False
        
        # Episode loop
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            
            # Track reward
            tracker.add_step_reward(reward)
        
        # End episode
        if 'episode_summary' in info:
            tracker.end_episode(info['episode_summary'])
        
        # Periodic correlation check
        if episode % 50 == 0:
            check = tracker.get_correlation_check()
            if check['status'] == 'REWARD_PNL_MISMATCH':
                print(f"⚠️  Episode {episode}: Reward-P&L correlation = {check['correlation']:.3f}")
                print("   Consider reviewing reward function!")

if __name__ == "__main__":
    print("📊 RewardPnLTracker - Integration Example")
    print("Add this to your training loop to monitor reward-P&L correlation")
    example_training_integration()