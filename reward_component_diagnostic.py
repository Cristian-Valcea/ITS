#!/usr/bin/env python3
"""
🔍 REWARD COMPONENT DIAGNOSTIC ANALYSIS
Analyzes which reward components are driving the -0.877 mean reward
by running evaluation episodes with detailed reward breakdown logging.
"""

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from secrets_helper import SecretsHelper
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter
from src.gym_env.dual_ticker_trading_env_v3_enhanced_5action import DualTickerTradingEnvV3Enhanced
from src.gym_env.refined_reward_system import RefinedRewardSystem

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except ImportError:
    import gym
    from gym import Wrapper

class DiagnosticRewardWrapper(Wrapper):
    """Wrapper to capture detailed reward component data for analysis"""
    
    def __init__(self, env, refined_reward_system):
        super().__init__(env)
        self.refined_reward = refined_reward_system
        self.step_data = []  # Store detailed step-by-step data
        self.episode_data = []  # Store episode-level summaries
        self.current_episode_steps = []
        
    def step(self, action):
        """Step with detailed reward component logging"""
        obs, original_reward, done, truncated, info = self.env.step(action)
        
        # Extract state information for refined reward calculation
        portfolio_value = info.get('portfolio_value', 10000.0)
        previous_portfolio_value = getattr(self, '_prev_portfolio_value', 10000.0)
        nvda_position = info.get('nvda_position', 0.0)
        msft_position = info.get('msft_position', 0.0)
        
        # Calculate drawdown percentage
        initial_value = 10000.0
        drawdown_pct = max(0, (initial_value - portfolio_value) / initial_value)
        
        # Calculate refined reward with detailed breakdown
        reward_components = self.refined_reward.calculate_reward(
            portfolio_value=portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            nvda_position=nvda_position,
            msft_position=msft_position,
            action=action,
            drawdown_pct=drawdown_pct
        )
        
        # Store detailed step data
        step_info = {
            'step': len(self.step_data),
            'episode': len(self.episode_data),
            'action': action,
            'portfolio_value': portfolio_value,
            'previous_portfolio_value': previous_portfolio_value,
            'pnl_change': portfolio_value - previous_portfolio_value,
            'nvda_position': nvda_position,
            'msft_position': msft_position,
            'drawdown_pct': drawdown_pct,
            'original_reward': original_reward,
            **reward_components.to_dict()
        }
        
        self.step_data.append(step_info)
        self.current_episode_steps.append(step_info)
        
        # Update info with reward breakdown
        info['refined_reward_components'] = reward_components.to_dict()
        info['original_reward'] = original_reward
        
        # Store for next step
        self._prev_portfolio_value = portfolio_value
        
        # If episode ended, summarize episode data
        if done or truncated:
            episode_summary = self._summarize_episode()
            self.episode_data.append(episode_summary)
            self.current_episode_steps = []
        
        return obs, reward_components.total_reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and refined reward system"""
        obs, info = self.env.reset(**kwargs)
        self.refined_reward.reset_episode()
        self._prev_portfolio_value = 10000.0
        return obs, info
    
    def _summarize_episode(self) -> Dict[str, Any]:
        """Summarize current episode reward components"""
        if not self.current_episode_steps:
            return {}
        
        df = pd.DataFrame(self.current_episode_steps)
        
        return {
            'episode': len(self.episode_data),
            'length': len(self.current_episode_steps),
            'total_reward': df['total_reward'].sum(),
            'mean_reward': df['total_reward'].mean(),
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'total_pnl': df['portfolio_value'].iloc[-1] - 10000.0,
            'max_drawdown_pct': df['drawdown_pct'].max(),
            
            # Component averages
            'mean_normalized_pnl': df['normalized_pnl'].mean(),
            'mean_holding_bonus': df['holding_bonus'].mean(),
            'mean_smoothed_penalty': df['smoothed_penalty'].mean(),
            'mean_exploration_bonus': df['exploration_bonus'].mean(),
            'mean_directional_bonus': df['directional_bonus'].mean(),
            
            # Component sums
            'sum_normalized_pnl': df['normalized_pnl'].sum(),
            'sum_holding_bonus': df['holding_bonus'].sum(),
            'sum_smoothed_penalty': df['smoothed_penalty'].sum(),
            'sum_exploration_bonus': df['exploration_bonus'].sum(),
            'sum_directional_bonus': df['directional_bonus'].sum(),
            
            # Action distribution
            'hold_rate': (df['action'] == 4).mean(),
            'buy_nvda_rate': (df['action'] == 0).mean(),
            'sell_nvda_rate': (df['action'] == 1).mean(),
            'buy_msft_rate': (df['action'] == 2).mean(),
            'sell_msft_rate': (df['action'] == 3).mean(),
        }

def load_model_and_create_env(model_path: str, num_episodes: int = 50):
    """Load the trained model and create diagnostic environment"""
    
    logger.info(f"🔍 Loading model from: {model_path}")
    
    # Load real market data (same as training)
    logger.info("📈 Loading real market data...")
    db_password = SecretsHelper.get_timescaledb_password()
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading_data', 
        'user': 'postgres',
        'password': db_password
    }
    
    adapter = DualTickerDataAdapter(db_config, live_trading_mode=False)
    market_data = adapter.load_training_data(
        start_date='2022-01-03',
        end_date='2024-12-31',
        symbols=['NVDA', 'MSFT'],
        bar_size='1min',
        data_split='train'
    )
    
    # Prepare data for environment (same as training)
    nvda_features = market_data['nvda_features']
    nvda_prices = market_data['nvda_prices']
    msft_features = market_data['msft_features']
    msft_prices = market_data['msft_prices']
    trading_days = market_data['trading_days']
    
    # Combine features (26-dim observation)
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    position_features = np.zeros((combined_features.shape[0], 2))
    combined_features = np.concatenate([combined_features, position_features], axis=1)
    
    # Create 4-column price data
    nvda_returns = np.concatenate([[0], np.diff(nvda_prices) / nvda_prices[:-1]])
    msft_returns = np.concatenate([[0], np.diff(msft_prices) / msft_prices[:-1]])
    combined_prices = np.column_stack([nvda_prices, nvda_returns, msft_prices, msft_returns])
    
    # Create base environment (same settings as training)
    base_env = DualTickerTradingEnvV3Enhanced(
        processed_feature_data=combined_features,
        processed_price_data=combined_prices,
        trading_days=trading_days,
        initial_capital=10000.0,
        lookback_window=50,
        max_episode_steps=390,
        max_daily_drawdown_pct=0.20,
        transaction_cost_pct=0.001
    )
    
    # Create refined reward system (same parameters as training)
    refined_reward_system = RefinedRewardSystem(
        initial_capital=10000.0,
        pnl_epsilon=1000.0,
        holding_alpha=0.01,
        penalty_beta=0.5,
        exploration_coef=0.05,
        exploration_decay=0.9999,
        verbose=False  # Reduce logging for diagnostic
    )
    
    # Wrap with diagnostic wrapper
    env = DiagnosticRewardWrapper(base_env, refined_reward_system)
    
    # Load trained model
    model = PPO.load(model_path)
    
    logger.info(f"✅ Model and environment loaded for {num_episodes} diagnostic episodes")
    
    return model, env

def run_diagnostic_episodes(model, env, num_episodes: int = 50):
    """Run diagnostic episodes to collect reward component data"""
    
    logger.info(f"🧪 Running {num_episodes} diagnostic episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 500:  # Safety limit
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            done = done or truncated
        
        if (episode + 1) % 10 == 0:
            logger.info(f"  Completed episode {episode + 1}/{num_episodes}")
    
    logger.info(f"✅ Diagnostic episodes complete. Collected {len(env.step_data)} steps across {len(env.episode_data)} episodes")
    
    return env.step_data, env.episode_data

def analyze_reward_components(step_data: List[Dict], episode_data: List[Dict], save_path: Path):
    """Comprehensive analysis of reward components"""
    
    logger.info("📊 Analyzing reward components...")
    
    # Convert to DataFrames
    steps_df = pd.DataFrame(step_data)
    episodes_df = pd.DataFrame(episode_data)
    
    # Create analysis directory
    analysis_dir = save_path / "reward_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # 1. OVERALL STATISTICS
    logger.info("📈 Computing overall statistics...")
    
    reward_components = ['normalized_pnl', 'holding_bonus', 'smoothed_penalty', 'exploration_bonus', 'directional_bonus']
    
    overall_stats = {
        'total_steps': len(steps_df),
        'total_episodes': len(episodes_df),
        'mean_episode_length': episodes_df['length'].mean(),
        'mean_total_reward': steps_df['total_reward'].mean(),
        'mean_episode_reward': episodes_df['total_reward'].mean(),
    }
    
    # Component statistics
    for component in reward_components:
        overall_stats[f'mean_{component}'] = steps_df[component].mean()
        overall_stats[f'sum_{component}'] = steps_df[component].sum()
        overall_stats[f'std_{component}'] = steps_df[component].std()
        overall_stats[f'min_{component}'] = steps_df[component].min()
        overall_stats[f'max_{component}'] = steps_df[component].max()
    
    # Save overall statistics
    stats_df = pd.DataFrame([overall_stats]).T
    stats_df.columns = ['Value']
    stats_df.to_csv(analysis_dir / "overall_statistics.csv")
    
    logger.info(f"📊 OVERALL REWARD BREAKDOWN:")
    logger.info(f"  Mean Total Reward: {overall_stats['mean_total_reward']:.6f}")
    for component in reward_components:
        mean_val = overall_stats[f'mean_{component}']
        logger.info(f"  Mean {component}: {mean_val:.6f}")
    
    # 2. TIME SERIES PLOTS
    logger.info("📈 Creating time series plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Reward Components Over Time', fontsize=16)
    
    # Plot each component
    for i, component in enumerate(reward_components):
        ax = axes[i//2, i%2]
        ax.plot(steps_df[component], alpha=0.7, linewidth=0.5)
        ax.set_title(f'{component.replace("_", " ").title()}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Value')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Total reward in the last subplot
    axes[2, 1].plot(steps_df['total_reward'], alpha=0.7, linewidth=0.5, color='black')
    axes[2, 1].set_title('Total Reward')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Total Reward')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "reward_components_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. EPISODE-AVERAGED DISTRIBUTIONS (BOX PLOTS)
    logger.info("📊 Creating distribution box plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Episode-Averaged Reward Component Distributions', fontsize=16)
    
    for i, component in enumerate(reward_components):
        ax = axes[i//3, i%3]
        episodes_df.boxplot(column=f'mean_{component}', ax=ax)
        ax.set_title(f'{component.replace("_", " ").title()}')
        ax.set_ylabel('Mean Episode Value')
        ax.grid(True, alpha=0.3)
    
    # Total reward distribution
    axes[1, 2].boxplot(episodes_df['mean_reward'])
    axes[1, 2].set_title('Total Reward')
    axes[1, 2].set_ylabel('Mean Episode Reward')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "reward_components_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. CUMULATIVE CONTRIBUTION ANALYSIS
    logger.info("📈 Creating cumulative contribution analysis...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate cumulative sums
    cumulative_data = {}
    for component in reward_components:
        cumulative_data[component] = steps_df[component].cumsum()
    
    # Create stacked area plot
    ax.stackplot(range(len(steps_df)), 
                 *[cumulative_data[comp] for comp in reward_components],
                 labels=reward_components, alpha=0.7)
    
    ax.plot(steps_df['total_reward'].cumsum(), color='black', linewidth=2, label='Total Reward')
    ax.set_title('Cumulative Reward Component Contributions')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "cumulative_contributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. CORRELATION ANALYSIS
    logger.info("🔗 Creating correlation analysis...")
    
    # Behavioral correlations
    behavioral_cols = ['hold_rate', 'buy_nvda_rate', 'sell_nvda_rate', 'buy_msft_rate', 'sell_msft_rate']
    reward_cols = [f'mean_{comp}' for comp in reward_components] + ['mean_reward']
    
    behavioral_corr = episodes_df[behavioral_cols + reward_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(behavioral_corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Behavioral vs Reward Component Correlations')
    plt.tight_layout()
    plt.savefig(analysis_dir / "behavioral_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance correlations
    performance_cols = ['length', 'total_pnl', 'max_drawdown_pct', 'final_portfolio_value']
    performance_corr = episodes_df[performance_cols + reward_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(performance_corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Performance vs Reward Component Correlations')
    plt.tight_layout()
    plt.savefig(analysis_dir / "performance_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. SAVE RAW DATA
    logger.info("💾 Saving raw data...")
    steps_df.to_csv(analysis_dir / "step_data.csv", index=False)
    episodes_df.to_csv(analysis_dir / "episode_data.csv", index=False)
    
    # 7. GENERATE SUMMARY REPORT
    logger.info("📝 Generating summary report...")
    
    report_lines = [
        "# REWARD COMPONENT DIAGNOSTIC REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## OVERALL STATISTICS",
        f"- Total Steps: {overall_stats['total_steps']:,}",
        f"- Total Episodes: {overall_stats['total_episodes']}",
        f"- Mean Episode Length: {overall_stats['mean_episode_length']:.1f}",
        f"- **Mean Total Reward: {overall_stats['mean_total_reward']:.6f}**",
        "",
        "## REWARD COMPONENT BREAKDOWN",
    ]
    
    # Component analysis
    for component in reward_components:
        mean_val = overall_stats[f'mean_{component}']
        sum_val = overall_stats[f'sum_{component}']
        contribution_pct = (sum_val / steps_df['total_reward'].sum()) * 100 if steps_df['total_reward'].sum() != 0 else 0
        
        report_lines.extend([
            f"### {component.replace('_', ' ').title()}",
            f"- Mean: {mean_val:.6f}",
            f"- Sum: {sum_val:.3f}",
            f"- Contribution: {contribution_pct:.1f}%",
            f"- Std: {overall_stats[f'std_{component}']:.6f}",
            f"- Range: [{overall_stats[f'min_{component}']:.6f}, {overall_stats[f'max_{component}']:.6f}]",
            ""
        ])
    
    # Identify the dominant negative component
    component_sums = {comp: overall_stats[f'sum_{comp}'] for comp in reward_components}
    most_negative = min(component_sums.items(), key=lambda x: x[1])
    most_positive = max(component_sums.items(), key=lambda x: x[1])
    
    report_lines.extend([
        "## KEY FINDINGS",
        f"- **Most Negative Component**: {most_negative[0]} (sum: {most_negative[1]:.3f})",
        f"- **Most Positive Component**: {most_positive[0]} (sum: {most_positive[1]:.3f})",
        f"- **Primary Bottleneck**: {most_negative[0]} is dragging down total reward",
        "",
        "## RECOMMENDATIONS",
        f"1. **Target {most_negative[0]}** for immediate tuning",
        f"2. Consider adjusting parameters related to {most_negative[0]}",
        f"3. Boost {most_positive[0]} if it's providing good signal",
        "",
        "## FILES GENERATED",
        "- step_data.csv: Raw step-by-step data",
        "- episode_data.csv: Episode-level summaries", 
        "- overall_statistics.csv: Summary statistics",
        "- reward_components_timeseries.png: Time series plots",
        "- reward_components_distributions.png: Distribution box plots",
        "- cumulative_contributions.png: Cumulative contribution analysis",
        "- behavioral_correlations.png: Behavioral correlation heatmap",
        "- performance_correlations.png: Performance correlation heatmap"
    ])
    
    with open(analysis_dir / "diagnostic_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✅ Analysis complete! Results saved to: {analysis_dir}")
    logger.info(f"📊 KEY FINDING: {most_negative[0]} is the primary bottleneck (sum: {most_negative[1]:.3f})")
    
    return overall_stats, most_negative, most_positive

def main():
    """Main diagnostic execution"""
    
    logger.info("🔍 REWARD COMPONENT DIAGNOSTIC ANALYSIS")
    logger.info("=" * 60)
    
    # Configuration
    model_path = "/home/cristian/IntradayTrading/ITS/train_runs/warmup_refined_50k_20250804_111836/warmup_refined_model.zip"
    num_episodes = 50  # Diagnostic episodes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f"diagnostic_runs/reward_analysis_{timestamp}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎯 Model: {model_path}")
    logger.info(f"🧪 Episodes: {num_episodes}")
    logger.info(f"📁 Output: {save_path}")
    
    try:
        # Load model and create environment
        model, env = load_model_and_create_env(model_path, num_episodes)
        
        # Run diagnostic episodes
        step_data, episode_data = run_diagnostic_episodes(model, env, num_episodes)
        
        # Analyze reward components
        overall_stats, most_negative, most_positive = analyze_reward_components(step_data, episode_data, save_path)
        
        logger.info("🎉 DIAGNOSTIC COMPLETE!")
        logger.info(f"📁 Results: {save_path}/reward_analysis/")
        logger.info(f"📊 Primary Bottleneck: {most_negative[0]} (sum: {most_negative[1]:.3f})")
        logger.info(f"🚀 Next Step: Tune {most_negative[0]} parameters to improve reward")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ REWARD COMPONENT DIAGNOSTIC: SUCCESS")
        print("📊 Check diagnostic_runs/ for detailed analysis")
    else:
        print("❌ REWARD COMPONENT DIAGNOSTIC: FAILED")
    
    sys.exit(0 if success else 1)