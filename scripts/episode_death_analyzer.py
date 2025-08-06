#!/usr/bin/env python3
"""
🔍 EPISODE DEATH ANALYZER
Quantify exactly where and why episodes terminate early
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import sys
from datetime import datetime
import json

# Add src to path
sys.path.append("src")

from stable_baselines3 import PPO
import gymnasium as gym

# Import with proper path handling
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_env.dual_ticker_trading_env_v3_enhanced import DualTickerTradingEnvV3Enhanced
from gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from gym_env.refined_reward_system import RefinedRewardSystem

class EpisodeDeathAnalyzer:
    def __init__(self, config_path, checkpoint_path, output_dir):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.termination_data = []
        
    def create_diagnostic_environment(self):
        """Create environment with detailed logging"""
        # Create data adapter
        data_adapter = DualTickerDataAdapter(
            timescaledb_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_data',
                'user': 'trading_user',
                'password': 'your_password'
            },
            live_trading_mode=False
        )
        
        # Load evaluation data (Feb 2024)
        market_data = data_adapter.load_training_data(
            start_date='2024-02-01',
            end_date='2024-02-29',
            symbols=['NVDA', 'MSFT'],
            bar_size='1min',
            data_split='eval'
        )
        
        # Prepare data
        nvda_features = market_data['nvda_features']
        nvda_prices = market_data['nvda_prices']
        msft_features = market_data['msft_features']
        msft_prices = market_data['msft_prices']
        trading_days = market_data['trading_days']
        
        # Combine features
        combined_features = np.concatenate([nvda_features, msft_features], axis=1)
        position_features = np.zeros((combined_features.shape[0], 2))
        combined_features = np.concatenate([combined_features, position_features], axis=1)
        
        # Create price data
        nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
        msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
        combined_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
        
        # Create base environment
        base_env = DualTickerTradingEnvV3Enhanced(
            processed_feature_data=combined_features,
            processed_price_data=combined_prices,
            trading_days=trading_days,
            initial_capital=self.config['environment']['initial_capital'],
            lookback_window=self.config['environment']['lookback_window'],
            max_episode_steps=self.config['environment']['max_episode_steps'],
            max_daily_drawdown_pct=self.config['environment']['max_drawdown_pct'],
            transaction_cost_pct=self.config['environment']['transaction_cost_pct']
        )
        
        # Add RefinedRewardSystem wrapper with detailed logging
        reward_params = self.config['reward_system']['parameters']
        refined_reward_system = RefinedRewardSystem(
            initial_capital=self.config['environment']['initial_capital'],
            pnl_epsilon=reward_params['pnl_epsilon'],
            holding_alpha=reward_params['holding_alpha'],
            penalty_beta=reward_params['penalty_beta'],
            exploration_coef=reward_params['exploration_coef']
        )
        
        class DiagnosticWrapper(gym.Wrapper):
            def __init__(self, env, reward_system, analyzer):
                super().__init__(env)
                self.reward_system = reward_system
                self.analyzer = analyzer
                self.episode_data = []
                self.step_count = 0
                
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.reward_system.reset_episode()
                self.episode_data = []
                self.step_count = 0
                return obs, info
                
            def step(self, action):
                obs, original_reward, terminated, truncated, info = self.env.step(action)
                self.step_count += 1
                
                # Calculate refined reward
                refined_reward = self.reward_system.calculate_reward(
                    portfolio_value=info.get('portfolio_value', 10000.0),
                    previous_portfolio_value=info.get('previous_portfolio_value', 10000.0),
                    nvda_position=info.get('positions', [0.0, 0.0])[0],
                    msft_position=info.get('positions', [0.0, 0.0])[1],
                    action=action if isinstance(action, int) else int(action),
                    drawdown_pct=info.get('drawdown_pct', 0.0)
                )
                
                if hasattr(refined_reward, 'to_dict'):
                    info['reward_breakdown'] = refined_reward.to_dict()
                    final_reward = refined_reward.total_reward
                else:
                    final_reward = refined_reward
                
                # Record step data
                step_data = {
                    'step': self.step_count,
                    'action': int(action),
                    'portfolio_value': info.get('portfolio_value', 10000.0),
                    'cash': info.get('cash', 10000.0),
                    'unrealized_pnl': info.get('unrealized_pnl', 0.0),
                    'drawdown_pct': info.get('drawdown_pct', 0.0),
                    'positions': info.get('positions', [0.0, 0.0]).copy(),
                    'reward': final_reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'termination_reason': info.get('termination_reason', None)
                }
                self.episode_data.append(step_data)
                
                # If episode ends, record termination data
                if terminated or truncated:
                    termination_record = {
                        'episode_length': self.step_count,
                        'final_portfolio_value': info.get('portfolio_value', 10000.0),
                        'final_drawdown_pct': info.get('drawdown_pct', 0.0),
                        'termination_reason': info.get('termination_reason', 'unknown'),
                        'terminated': terminated,
                        'truncated': truncated,
                        'step_data': self.episode_data.copy()
                    }
                    self.analyzer.termination_data.append(termination_record)
                
                return obs, final_reward, terminated, truncated, info
        
        wrapped_env = DiagnosticWrapper(base_env, refined_reward_system, self)
        return wrapped_env
    
    def analyze_episodes(self, num_episodes=100):
        """Run episodes and collect termination data"""
        print(f"🔍 Analyzing {num_episodes} episodes for termination patterns...")
        
        # Load model
        model = PPO.load(self.checkpoint_path)
        
        # Create environment
        env = self.create_diagnostic_environment()
        
        for episode in range(num_episodes):
            if episode % 10 == 0:
                print(f"   Episode {episode}/{num_episodes}")
                
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        
        print(f"✅ Collected data from {len(self.termination_data)} episodes")
        
    def generate_analysis(self):
        """Generate comprehensive termination analysis"""
        if not self.termination_data:
            print("❌ No termination data collected")
            return
            
        print(f"📊 Generating analysis from {len(self.termination_data)} episodes...")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'episode_length': ep['episode_length'],
                'final_portfolio_value': ep['final_portfolio_value'],
                'final_drawdown_pct': ep['final_drawdown_pct'],
                'termination_reason': ep['termination_reason'],
                'terminated': ep['terminated'],
                'truncated': ep['truncated']
            }
            for ep in self.termination_data
        ])
        
        # Basic statistics
        print(f"\n📈 EPISODE TERMINATION ANALYSIS")
        print("=" * 50)
        print(f"Total episodes analyzed: {len(df)}")
        print(f"Mean episode length: {df['episode_length'].mean():.1f} steps")
        print(f"Median episode length: {df['episode_length'].median():.1f} steps")
        print(f"Max episode length: {df['episode_length'].max()} steps")
        print(f"Min episode length: {df['episode_length'].min()} steps")
        
        print(f"\n🚨 TERMINATION REASONS:")
        termination_counts = df['termination_reason'].value_counts()
        for reason, count in termination_counts.items():
            pct = count / len(df) * 100
            print(f"   {reason}: {count} episodes ({pct:.1f}%)")
        
        print(f"\n💰 PORTFOLIO VALUES AT TERMINATION:")
        print(f"Mean final value: ${df['final_portfolio_value'].mean():.2f}")
        print(f"Median final value: ${df['final_portfolio_value'].median():.2f}")
        print(f"Min final value: ${df['final_portfolio_value'].min():.2f}")
        print(f"Max final value: ${df['final_portfolio_value'].max():.2f}")
        
        print(f"\n📉 DRAWDOWN AT TERMINATION:")
        print(f"Mean drawdown: {df['final_drawdown_pct'].mean():.2f}%")
        print(f"Median drawdown: {df['final_drawdown_pct'].median():.2f}%")
        print(f"Max drawdown: {df['final_drawdown_pct'].max():.2f}%")
        
        # Save detailed data
        df.to_csv(self.output_dir / 'episode_termination_summary.csv', index=False)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Analyze step-by-step patterns
        self.analyze_step_patterns()
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
    
    def create_visualizations(self, df):
        """Create visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Episode length distribution
        axes[0, 0].hist(df['episode_length'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].set_xlabel('Episode Length (steps)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['episode_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["episode_length"].mean():.1f}')
        axes[0, 0].legend()
        
        # Drawdown at termination
        axes[0, 1].hist(df['final_drawdown_pct'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Drawdown at Termination')
        axes[0, 1].set_xlabel('Drawdown (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(30, color='red', linestyle='--', label='30% Limit')
        axes[0, 1].legend()
        
        # Portfolio value at termination
        axes[1, 0].hist(df['final_portfolio_value'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Portfolio Value at Termination')
        axes[1, 0].set_xlabel('Portfolio Value ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(10000, color='red', linestyle='--', label='Initial: $10,000')
        axes[1, 0].legend()
        
        # Termination reasons
        termination_counts = df['termination_reason'].value_counts()
        axes[1, 1].pie(termination_counts.values, labels=termination_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Termination Reasons')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'termination_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_step_patterns(self):
        """Analyze step-by-step patterns leading to termination"""
        print(f"\n🔍 Analyzing step-by-step patterns...")
        
        # Collect all step data
        all_steps = []
        for ep in self.termination_data:
            for step_data in ep['step_data']:
                step_data['episode_id'] = len(all_steps) // len(ep['step_data'])
                all_steps.append(step_data)
        
        if not all_steps:
            print("❌ No step data available")
            return
            
        step_df = pd.DataFrame(all_steps)
        
        # Analyze drawdown progression
        print(f"\n📉 DRAWDOWN PROGRESSION ANALYSIS:")
        
        # Group by step number and analyze average drawdown
        step_stats = step_df.groupby('step').agg({
            'drawdown_pct': ['mean', 'std', 'max'],
            'portfolio_value': ['mean', 'std'],
            'reward': ['mean', 'std']
        }).round(3)
        
        # Save step statistics
        step_stats.to_csv(self.output_dir / 'step_by_step_stats.csv')
        
        # Find critical steps where drawdown spikes
        drawdown_spikes = step_df[step_df['drawdown_pct'] > 25.0]
        if len(drawdown_spikes) > 0:
            print(f"   Found {len(drawdown_spikes)} steps with >25% drawdown")
            spike_steps = drawdown_spikes['step'].value_counts().head(10)
            print(f"   Most common spike steps: {spike_steps.to_dict()}")
        
        # Analyze action patterns before termination
        print(f"\n🎯 ACTION PATTERNS BEFORE TERMINATION:")
        
        # Look at last 5 steps before termination
        last_steps = []
        for ep in self.termination_data:
            if len(ep['step_data']) >= 5:
                last_steps.extend(ep['step_data'][-5:])
        
        if last_steps:
            last_step_df = pd.DataFrame(last_steps)
            action_counts = last_step_df['action'].value_counts()
            print(f"   Action distribution in last 5 steps:")
            action_names = ['SELL_BOTH', 'SELL_NVDA_HOLD_MSFT', 'SELL_NVDA_BUY_MSFT', 
                           'HOLD_NVDA_SELL_MSFT', 'HOLD_BOTH', 'HOLD_NVDA_BUY_MSFT',
                           'BUY_NVDA_SELL_MSFT', 'BUY_NVDA_HOLD_MSFT', 'BUY_BOTH']
            
            for action, count in action_counts.items():
                pct = count / len(last_steps) * 100
                name = action_names[action] if action < len(action_names) else f"Action_{action}"
                print(f"     {name}: {count} ({pct:.1f}%)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze episode termination patterns')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to analyze')
    
    args = parser.parse_args()
    
    analyzer = EpisodeDeathAnalyzer(args.config, args.checkpoint, args.output_dir)
    analyzer.analyze_episodes(args.episodes)
    analyzer.generate_analysis()

if __name__ == "__main__":
    main()