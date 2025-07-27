# src/training/reward_pnl_audit.py
"""
Reward-P&L Audit System for IntradayJules

This module provides comprehensive auditing to ensure the agent's reward signal
aligns tightly with realized P&L, preventing "looks-good-in-training, bad-in-cash" behavior.

Key Features:
- Step-wise correlation tracking between reward and P&L
- Episode-level correlation analysis
- Visual diagnostics and alerts
- Fail-fast mechanisms for misaligned rewards
- Integration with TensorBoard/W&B for live monitoring
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings

# Stable Baselines3 imports
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logger = logging.getLogger(__name__)


class RewardPnLAudit(BaseCallback):
    """
    Comprehensive callback for auditing reward-P&L alignment during training.
    
    This callback tracks the correlation between the agent's reward signal and
    actual realized P&L to ensure the agent optimizes for real profitability.
    
    Features:
    - Step-wise correlation tracking
    - Episode-level analysis
    - Automatic alerts for misalignment
    - Visual diagnostics
    - CSV export for post-training analysis
    - Integration with TensorBoard/W&B
    """
    
    def __init__(
        self,
        output_dir: str = "reward_pnl_audit",
        min_correlation_threshold: float = 0.5,
        alert_episodes: int = 10,
        save_plots: bool = True,
        verbose: bool = True,
        fail_fast: bool = False,
        memory_dump_threshold: int = 10000  # Dump to disk after N steps to prevent RAM issues
    ):
        """
        Initialize the Reward-P&L Audit callback.
        
        Args:
            output_dir: Directory to save audit results
            min_correlation_threshold: Minimum acceptable correlation
            alert_episodes: Number of episodes to check for alerts
            save_plots: Whether to save diagnostic plots
            verbose: Whether to print detailed logs
            fail_fast: Whether to raise exception on low correlation
            memory_dump_threshold: Dump episode data to disk after N steps to prevent RAM issues
        """
        super().__init__(verbose=verbose)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_correlation_threshold = min_correlation_threshold
        self.alert_episodes = alert_episodes
        self.save_plots = save_plots
        self.fail_fast = fail_fast
        self.memory_dump_threshold = memory_dump_threshold
        
        # Episode-level tracking (now supports VecEnv with dict of env_idx -> data)
        self.episode_logs: List[Dict[str, Any]] = []
        self.current_episode_data: Dict[int, Dict[str, List]] = {}
        self.dumped_episodes: List[Dict[str, Any]] = []  # Episodes dumped to disk
        
        # Global statistics
        self.total_episodes = 0
        self.correlation_history = []
        self.alert_count = 0
        
        # Initialize audit log path
        self.audit_log_path = self.output_dir / "audit_log.txt"
        
        logger.info(f"🎯 RewardPnLAudit initialized - Output: {self.output_dir}")
        logger.info(f"📊 Correlation threshold: {min_correlation_threshold:.3f}")
        logger.info(f"🚨 Alert after {alert_episodes} episodes with low correlation")
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        logger.info("🚀 Starting Reward-P&L audit monitoring...")
        
        # Create audit log file
        self.audit_log_path = self.output_dir / "audit_log.txt"
        with open(self.audit_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Reward-P&L Audit Log - Started: {datetime.now()}\n")
            f.write(f"Correlation Threshold: {self.min_correlation_threshold:.3f}\n")
            f.write("=" * 60 + "\n\n")
    
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Extract data from the current step
        rewards = self.locals.get("rewards", [])
        if len(rewards) == 0:
            return True
        
        # Handle VecEnv: process all parallel environments
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [0])
        
        # Process each parallel environment
        for env_idx in range(len(rewards)):
            reward = rewards[env_idx] if env_idx < len(rewards) else 0.0
            info = infos[env_idx] if env_idx < len(infos) else {}
            action = actions[env_idx] if env_idx < len(actions) else 0
            
            # Store step data for this environment
            self._store_step_data(reward, info, action, env_idx)
        
        # Check memory pressure and dump to disk if needed
        self._check_memory_pressure()
        
        return True
    
    def _store_step_data(self, reward: float, info: Dict[str, Any], action: int, env_idx: int = 0) -> None:
        """Store step data for a single environment."""
        # Ensure we have environment-specific storage
        if env_idx not in self.current_episode_data:
            self.current_episode_data[env_idx] = self._create_empty_episode_data()
        
        episode_data = self.current_episode_data[env_idx]
        
        # Store step data
        episode_data['rewards'].append(reward)
        episode_data['realized_pnl'].append(info.get('realized_pnl_step', 0.0))
        episode_data['unrealized_pnl'].append(info.get('unrealized_pnl_step', 0.0))
        episode_data['total_pnl'].append(info.get('total_pnl_step', 0.0))
        episode_data['fees'].append(info.get('fees_step', 0.0))
        episode_data['net_pnl'].append(info.get('net_pnl_step', 0.0))
        episode_data['raw_rewards'].append(info.get('raw_reward', reward))
        episode_data['scaled_rewards'].append(info.get('scaled_reward', reward))
        episode_data['actions'].append(action)
        episode_data['portfolio_values'].append(info.get('portfolio_value', 0.0))
        episode_data['timestamps'].append(info.get('timestamp', datetime.now()))
    
    def _create_empty_episode_data(self) -> Dict[str, List]:
        """Create empty episode data structure."""
        return {
            'rewards': [],
            'realized_pnl': [],
            'unrealized_pnl': [],
            'total_pnl': [],
            'fees': [],
            'net_pnl': [],
            'raw_rewards': [],
            'scaled_rewards': [],
            'actions': [],
            'portfolio_values': [],
            'timestamps': []
        }
    
    def _check_memory_pressure(self) -> None:
        """Check memory pressure and dump to disk if needed."""
        # Count total steps across all environments
        total_steps = 0
        for env_data in self.current_episode_data.values():
            if isinstance(env_data, dict):
                total_steps += len(env_data.get('rewards', []))
        
        # Dump to disk if we exceed memory threshold
        if total_steps > self.memory_dump_threshold:
            self._dump_episode_data_to_disk()
            logger.info(f"💾 Memory pressure relief: Dumped {total_steps} steps to disk")
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each episode/rollout."""
        # Process completed episodes from all environments
        for env_idx, episode_data in self.current_episode_data.items():
            if isinstance(episode_data, dict) and len(episode_data.get('rewards', [])) > 0:
                # Calculate episode-level metrics for this environment
                episode_metrics = self._calculate_episode_metrics(episode_data)
                episode_metrics['env_idx'] = env_idx
                
                self.episode_logs.append(episode_metrics)
                self.total_episodes += 1
                
                # Track correlation history (filter out NaN values)
                step_corr = episode_metrics.get('step_correlation', np.nan)
                if not np.isnan(step_corr):
                    self.correlation_history.append(step_corr)
                
                # Log episode summary
                if self.verbose:
                    self._log_episode_summary(episode_metrics)
                
                # Check for alerts
                self._check_correlation_alerts(episode_metrics)
                
                # Log to TensorBoard if available
                self._log_to_tensorboard(episode_metrics)
        
        # Reset episode data
        self._reset_episode_data()
    
    def _dump_episode_data_to_disk(self) -> None:
        """Dump current episode data to disk to free memory."""
        dump_file = self.output_dir / f"episode_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Convert current episode data to serializable format
        dump_data = []
        for env_idx, episode_data in self.current_episode_data.items():
            if isinstance(episode_data, dict) and len(episode_data.get('rewards', [])) > 0:
                # Calculate metrics before dumping
                metrics = self._calculate_episode_metrics(episode_data)
                metrics['env_idx'] = env_idx
                dump_data.append(metrics)
        
        # Save to disk
        import pickle
        with open(dump_file, 'wb') as f:
            pickle.dump(dump_data, f)
        
        # Add to dumped episodes list
        self.dumped_episodes.extend(dump_data)
        
        # Clear current episode data to free memory
        self.current_episode_data.clear()
        
        logger.info(f"💾 Dumped {len(dump_data)} episodes to {dump_file}")
    
    def _calculate_episode_metrics(self, data: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """Calculate comprehensive episode-level metrics."""
        if data is None:
            # Fallback for backward compatibility - use first environment
            data = self.current_episode_data.get(0, self._create_empty_episode_data())
        
        # Convert to numpy arrays for calculations
        rewards = np.array(data['rewards'])
        realized_pnl = np.array(data['realized_pnl'])
        total_pnl = np.array(data['total_pnl'])
        net_pnl = np.array(data['net_pnl'])
        raw_rewards = np.array(data['raw_rewards'])
        
        # Calculate correlations (handle edge cases)
        def safe_correlation(x, y):
            if len(x) < 2:
                return np.nan  # Not enough data points
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan  # Zero variance - degenerate case, make visible
            corr = np.corrcoef(x, y)[0, 1]
            return corr if not np.isnan(corr) else np.nan
        
        step_correlation = safe_correlation(rewards, realized_pnl)
        total_pnl_correlation = safe_correlation(rewards, total_pnl)
        net_pnl_correlation = safe_correlation(rewards, net_pnl)
        raw_reward_correlation = safe_correlation(raw_rewards, realized_pnl)
        
        # Episode totals
        total_reward = float(np.sum(rewards))
        total_realized_pnl = float(np.sum(realized_pnl))
        total_net_pnl = float(np.sum(net_pnl))
        total_fees = float(np.sum(data['fees']))
        
        # Portfolio metrics
        initial_portfolio = data['portfolio_values'][0] if data['portfolio_values'] else 0.0
        final_portfolio = data['portfolio_values'][-1] if data['portfolio_values'] else 0.0
        portfolio_return = final_portfolio - initial_portfolio
        
        # Action distribution (handle empty arrays)
        actions = np.array(data['actions'])
        if len(actions) > 0:
            action_counts = np.bincount(actions.astype(int), minlength=3)  # Assuming 3 actions: 0, 1, 2
        else:
            action_counts = np.array([0, 0, 0])
        
        return {
            'episode': self.total_episodes,
            'timestamp': datetime.now(),
            'episode_length': len(rewards),
            
            # Correlation metrics (key indicators)
            'step_correlation': step_correlation,
            'total_pnl_correlation': total_pnl_correlation,
            'net_pnl_correlation': net_pnl_correlation,
            'raw_reward_correlation': raw_reward_correlation,
            
            # Episode totals
            'total_reward': total_reward,
            'total_realized_pnl': total_realized_pnl,
            'total_net_pnl': total_net_pnl,
            'total_fees': total_fees,
            'portfolio_return': portfolio_return,
            
            # Portfolio metrics
            'initial_portfolio_value': initial_portfolio,
            'final_portfolio_value': final_portfolio,
            'return_pct': (portfolio_return / initial_portfolio * 100) if initial_portfolio > 0 else 0.0,
            
            # Action distribution
            'action_sell_count': int(action_counts[0]),
            'action_hold_count': int(action_counts[1]),
            'action_buy_count': int(action_counts[2]),
            
            # Statistical measures
            'reward_std': float(np.std(rewards)),
            'pnl_std': float(np.std(realized_pnl)),
            'reward_mean': float(np.mean(rewards)),
            'pnl_mean': float(np.mean(realized_pnl))
        }
    
    def _log_episode_summary(self, metrics: Dict[str, Any]) -> None:
        """Log episode summary with key metrics."""
        ep = metrics['episode']
        corr = metrics['step_correlation']
        total_reward = metrics['total_reward']
        total_pnl = metrics['total_realized_pnl']
        
        status = "✅" if corr >= self.min_correlation_threshold else "⚠️"
        
        logger.info(
            f"{status} Episode {ep}: Reward↔P&L correlation: {corr:.3f} | "
            f"Total Reward: {total_reward:+.4f} | Total P&L: ${total_pnl:+.2f}"
        )
        
        # Log to audit file
        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(f"Episode {ep}: Correlation={corr:.3f}, Reward={total_reward:+.4f}, P&L=${total_pnl:+.2f}\n")
    
    def _check_correlation_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check for correlation alerts and take action."""
        corr = metrics['step_correlation']
        
        # Handle NaN correlations (degenerate cases)
        if np.isnan(corr):
            logger.warning(f"⚠️ Episode {metrics['episode']}: NaN correlation detected - "
                          f"likely zero variance in rewards or P&L (degenerate case)")
            self.alert_count += 1
        elif corr < self.min_correlation_threshold:
            self.alert_count += 1
        else:
            # Reset alert count on good correlation
            self.alert_count = 0
            return
            
        # Check if we need to raise an alert
        if self.alert_count >= self.alert_episodes:
            # Filter out NaN values for average calculation
            recent_correlations = [c for c in self.correlation_history[-self.alert_episodes:] if not np.isnan(c)]
            
            if len(recent_correlations) == 0:
                avg_recent_corr = np.nan
                alert_msg = (
                    f"🚨 CRITICAL: ALL RECENT CORRELATIONS ARE NaN! 🚨\n"
                    f"Last {self.alert_episodes} episodes had degenerate correlations.\n"
                    f"This indicates zero variance in rewards or P&L - check reward function!"
                )
            else:
                avg_recent_corr = np.mean(recent_correlations)
                alert_msg = (
                    f"🚨 REWARD-P&L MISALIGNMENT ALERT! 🚨\n"
                    f"Average correlation over last {self.alert_episodes} episodes: {avg_recent_corr:.3f}\n"
                    f"Threshold: {self.min_correlation_threshold:.3f}\n"
                    f"This suggests the reward signal is not aligned with actual profitability!\n"
                    f"Consider revising the reward function before continuing training."
                )
            
            logger.error(alert_msg)
            
            # Write to audit log
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{alert_msg}\n\n")
            
            # Fail-fast if enabled
            if self.fail_fast:
                if np.isnan(avg_recent_corr):
                    raise ValueError(
                        f"All recent correlations are NaN (degenerate case) for {self.alert_episodes} episodes. "
                        f"Training stopped - check reward function for zero variance issues."
                    )
                else:
                    raise ValueError(
                        f"Reward-P&L correlation ({avg_recent_corr:.3f}) below threshold "
                        f"({self.min_correlation_threshold:.3f}) for {self.alert_episodes} episodes. "
                        f"Training stopped to prevent learning misaligned behavior."
                    )
            
            # Reset alert count after triggering
            self.alert_count = 0
    
    def _log_to_tensorboard(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to TensorBoard if available."""
        try:
            if hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'logger'):
                model_logger = self.model.logger
                if hasattr(model_logger, 'record'):
                    model_logger.record("reward_pnl_audit/step_correlation", metrics['step_correlation'])
                    model_logger.record("reward_pnl_audit/total_pnl_correlation", metrics['total_pnl_correlation'])
                    model_logger.record("reward_pnl_audit/episode_reward", metrics['total_reward'])
                    model_logger.record("reward_pnl_audit/episode_pnl", metrics['total_realized_pnl'])
                    model_logger.record("reward_pnl_audit/portfolio_return_pct", metrics['return_pct'])
        except Exception:
            # Silently skip TensorBoard logging if not available
            pass
    
    def _reset_episode_data(self) -> None:
        """Reset episode data for next episode."""
        # Clear all environment data
        self.current_episode_data.clear()
    
    def _on_training_end(self) -> None:
        """Called when training ends - generate final report."""
        logger.info("🏁 Training completed - Generating Reward-P&L audit report...")
        
        # Combine in-memory and dumped episodes
        all_episodes = self.episode_logs + self.dumped_episodes
        
        if not all_episodes:
            logger.warning("No episode data collected for audit report")
            return
        
        # Save episode logs to CSV
        df = pd.DataFrame(all_episodes)
        csv_path = self.output_dir / "reward_pnl_audit.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 Episode logs saved to: {csv_path} ({len(all_episodes)} episodes total)")
        
        if self.dumped_episodes:
            logger.info(f"💾 Included {len(self.dumped_episodes)} episodes from disk dumps")
        
        # Generate summary statistics
        self._generate_summary_report(df)
        
        # Generate diagnostic plots
        if self.save_plots:
            self._generate_diagnostic_plots(df)
        
        # Final correlation check
        self._final_correlation_check(df)
    
    def _generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive summary report."""
        report_path = self.output_dir / "audit_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REWARD-P&L AUDIT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Episodes: {len(df)}\n\n")
            
            # Correlation statistics (handle NaN values)
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            # Filter out NaN values for statistics
            valid_correlations = df['step_correlation'].dropna()
            nan_count = df['step_correlation'].isna().sum()
            
            if len(valid_correlations) > 0:
                f.write(f"Mean Step Correlation: {valid_correlations.mean():.3f}\n")
                f.write(f"Median Step Correlation: {valid_correlations.median():.3f}\n")
                f.write(f"Std Step Correlation: {valid_correlations.std():.3f}\n")
                f.write(f"Min Step Correlation: {valid_correlations.min():.3f}\n")
                f.write(f"Max Step Correlation: {valid_correlations.max():.3f}\n")
                
                # Episodes below threshold (excluding NaN)
                below_threshold = (valid_correlations < self.min_correlation_threshold).sum()
                f.write(f"Episodes below threshold ({self.min_correlation_threshold:.3f}): {below_threshold}/{len(valid_correlations)} ({below_threshold/len(valid_correlations)*100:.1f}%)\n")
            else:
                f.write("No valid correlations found (all NaN)\n")
            
            if nan_count > 0:
                f.write(f"⚠️ Episodes with NaN correlation (degenerate cases): {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)\n")
            f.write("\n")
            
            # Episode-level correlation (handle NaN)
            try:
                episode_corr = df[['total_reward', 'total_realized_pnl']].corr().iloc[0, 1]
                if np.isnan(episode_corr):
                    f.write("Episode-level Reward↔P&L correlation: NaN (degenerate case)\n\n")
                else:
                    f.write(f"Episode-level Reward↔P&L correlation: {episode_corr:.3f}\n\n")
            except Exception:
                f.write("Episode-level Reward↔P&L correlation: Could not calculate\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Reward: {df['total_reward'].sum():.4f}\n")
            f.write(f"Total Realized P&L: ${df['total_realized_pnl'].sum():.2f}\n")
            f.write(f"Total Portfolio Return: ${df['portfolio_return'].sum():.2f}\n")
            f.write(f"Average Episode Return: {df['return_pct'].mean():.2f}%\n")
            f.write(f"Total Fees Paid: ${df['total_fees'].sum():.2f}\n\n")
            
            # Action distribution
            f.write("ACTION DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Sell Actions: {df['action_sell_count'].sum()}\n")
            f.write(f"Total Hold Actions: {df['action_hold_count'].sum()}\n")
            f.write(f"Total Buy Actions: {df['action_buy_count'].sum()}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            mean_corr = df['step_correlation'].mean()
            if mean_corr >= 0.8:
                f.write("✅ EXCELLENT: Reward signal is highly aligned with P&L\n")
            elif mean_corr >= 0.6:
                f.write("✅ GOOD: Reward signal is well aligned with P&L\n")
            elif mean_corr >= 0.4:
                f.write("⚠️ MODERATE: Consider improving reward function alignment\n")
            else:
                f.write("❌ POOR: Reward function needs significant revision\n")
        
        logger.info(f"📋 Summary report saved to: {report_path}")
    
    def _generate_diagnostic_plots(self, df: pd.DataFrame) -> None:
        """Generate diagnostic plots for visual analysis."""
        logger.info("📈 Generating diagnostic plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Reward-P&L Audit Diagnostic Plots', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode-level Reward vs P&L scatter
        ax1 = axes[0, 0]
        ax1.scatter(df['total_reward'], df['total_realized_pnl'], alpha=0.6, s=50)
        ax1.axline((0, 0), slope=1, linestyle='--', color='red', alpha=0.7, label='Perfect Alignment')
        ax1.set_xlabel('Cumulative Episode Reward')
        ax1.set_ylabel('Realized P&L ($)')
        ax1.set_title('Episode Reward vs Realized P&L')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation annotation
        episode_corr = df[['total_reward', 'total_realized_pnl']].corr().iloc[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {episode_corr:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Plot 2: Step correlation over episodes (handle NaN values)
        ax2 = axes[0, 1]
        
        # Separate valid and NaN values for plotting
        valid_mask = ~df['step_correlation'].isna()
        valid_episodes = df.loc[valid_mask, 'episode']
        valid_correlations = df.loc[valid_mask, 'step_correlation']
        nan_episodes = df.loc[~valid_mask, 'episode']
        
        # Plot valid correlations
        if len(valid_correlations) > 0:
            ax2.plot(valid_episodes, valid_correlations, marker='o', markersize=3, alpha=0.7, label='Valid')
        
        # Mark NaN episodes
        if len(nan_episodes) > 0:
            ax2.scatter(nan_episodes, [0] * len(nan_episodes), marker='x', s=50, 
                       color='red', alpha=0.7, label='NaN (degenerate)')
        
        ax2.axhline(y=self.min_correlation_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.min_correlation_threshold:.3f})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Step-wise Correlation')
        ax2.set_title('Correlation Trend Over Episodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.1, 1.1)
        
        # Plot 3: Correlation distribution (filter out NaN values)
        ax3 = axes[1, 0]
        valid_correlations = df['step_correlation'].dropna()
        
        if len(valid_correlations) > 0:
            ax3.hist(valid_correlations, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=self.min_correlation_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.min_correlation_threshold:.3f})')
            ax3.axvline(x=valid_correlations.mean(), color='green', linestyle='-', 
                       label=f'Mean ({valid_correlations.mean():.3f})')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No valid correlations\n(all NaN)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        ax3.set_xlabel('Step-wise Correlation')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Step Correlations')
        
        # Add NaN count info
        nan_count = df['step_correlation'].isna().sum()
        if nan_count > 0:
            ax3.text(0.02, 0.98, f'NaN values: {nan_count}', 
                    transform=ax3.transAxes, va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Portfolio return vs reward
        ax4 = axes[1, 1]
        ax4.scatter(df['total_reward'], df['return_pct'], alpha=0.6, s=50, c=df['episode'], cmap='viridis')
        ax4.set_xlabel('Total Episode Reward')
        ax4.set_ylabel('Portfolio Return (%)')
        ax4.set_title('Reward vs Portfolio Return %')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for episode progression
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Episode')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Diagnostic plots saved to: {plot_path}")
        
        # Generate additional detailed plots
        self._generate_detailed_plots(df)
    
    def _generate_detailed_plots(self, df: pd.DataFrame) -> None:
        """Generate additional detailed analysis plots."""
        # Rolling correlation plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if len(df) >= 10:
            rolling_corr = df['step_correlation'].rolling(window=10, min_periods=5).mean()
            ax.plot(df['episode'], rolling_corr, label='10-Episode Rolling Average', linewidth=2)
        
        ax.plot(df['episode'], df['step_correlation'], alpha=0.5, label='Episode Correlation')
        ax.axhline(y=self.min_correlation_threshold, color='red', linestyle='--', 
                  label=f'Threshold ({self.min_correlation_threshold:.3f})')
        ax.fill_between(df['episode'], self.min_correlation_threshold, 1.0, alpha=0.2, color='green', label='Good Zone')
        ax.fill_between(df['episode'], -1.0, self.min_correlation_threshold, alpha=0.2, color='red', label='Alert Zone')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Step-wise Correlation')
        ax.set_title('Reward-P&L Correlation Trend Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _final_correlation_check(self, df: pd.DataFrame) -> None:
        """Perform final correlation check and provide recommendations."""
        # Handle NaN values in correlation calculations
        valid_correlations = df['step_correlation'].dropna()
        nan_count = df['step_correlation'].isna().sum()
        
        if len(valid_correlations) > 0:
            mean_corr = valid_correlations.mean()
        else:
            mean_corr = np.nan
        
        try:
            episode_corr = df[['total_reward', 'total_realized_pnl']].corr().iloc[0, 1]
        except Exception:
            episode_corr = np.nan
        
        logger.info("\n" + "="*60)
        logger.info("🎯 FINAL REWARD-P&L AUDIT RESULTS")
        logger.info("="*60)
        
        if np.isnan(mean_corr):
            logger.info("📊 Mean step-wise correlation: NaN (all degenerate cases)")
        else:
            logger.info(f"📊 Mean step-wise correlation: {mean_corr:.3f}")
            
        if np.isnan(episode_corr):
            logger.info("📈 Episode-level correlation: NaN (degenerate case)")
        else:
            logger.info(f"📈 Episode-level correlation: {episode_corr:.3f}")
            
        logger.info(f"🎯 Correlation threshold: {self.min_correlation_threshold:.3f}")
        
        if nan_count > 0:
            logger.info(f"⚠️ Degenerate episodes (NaN correlation): {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)")
        
        # Provide recommendations
        if mean_corr >= 0.8 and episode_corr >= 0.8:
            logger.info("✅ EXCELLENT: Reward signal is highly aligned with P&L - Ready for deployment!")
        elif mean_corr >= 0.6 and episode_corr >= 0.6:
            logger.info("✅ GOOD: Reward signal is well aligned with P&L - Consider minor optimizations")
        elif mean_corr >= 0.4 or episode_corr >= 0.4:
            logger.warning("⚠️ MODERATE: Consider improving reward function alignment before deployment")
        else:
            logger.error("❌ POOR: Reward function needs significant revision - DO NOT DEPLOY")
        
        logger.info("="*60)
        
        # Save final metrics for easy access
        final_metrics = {
            'mean_step_correlation': float(mean_corr),
            'episode_level_correlation': float(episode_corr),
            'total_episodes': int(len(df)),
            'episodes_below_threshold': int((df['step_correlation'] < self.min_correlation_threshold).sum()),
            'recommendation': self._get_recommendation(mean_corr, episode_corr)
        }
        
        import json
        with open(self.output_dir / "final_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2)
    
    def _get_recommendation(self, mean_corr: float, episode_corr: float) -> str:
        """Get recommendation based on correlation metrics."""
        if mean_corr >= 0.8 and episode_corr >= 0.8:
            return "EXCELLENT - Ready for deployment"
        elif mean_corr >= 0.6 and episode_corr >= 0.6:
            return "GOOD - Consider minor optimizations"
        elif mean_corr >= 0.4 or episode_corr >= 0.4:
            return "MODERATE - Improve reward function alignment"
        else:
            return "POOR - Significant revision needed"


def quick_audit_check(csv_path: str) -> None:
    """
    Quick post-training audit check from saved CSV.
    
    Args:
        csv_path: Path to the reward_pnl_audit.csv file
    """
    try:
        df = pd.read_csv(csv_path)
        
        print("\n" + "="*50)
        print("🔍 QUICK REWARD-P&L AUDIT CHECK")
        print("="*50)
        
        # Basic statistics
        mean_corr = df['step_correlation'].mean()
        episode_corr = df[['total_reward', 'total_realized_pnl']].corr().iloc[0, 1]
        
        print(f"📊 Mean step-wise correlation: {mean_corr:.3f}")
        print(f"📈 Episode-level correlation: {episode_corr:.3f}")
        print(f"📋 Total episodes: {len(df)}")
        
        # Quick visualization
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['total_reward'], df['total_realized_pnl'], alpha=0.6)
        plt.axline((0, 0), slope=1, linestyle='--', color='red', alpha=0.7)
        plt.xlabel('Cumulative Episode Reward')
        plt.ylabel('Realized P&L ($)')
        plt.title(f'Reward vs P&L (r={episode_corr:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(df['episode'], df['step_correlation'], marker='o', markersize=3)
        plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Episode')
        plt.ylabel('Step Correlation')
        plt.title('Correlation Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Recommendation
        if mean_corr >= 0.8 and episode_corr >= 0.8:
            print("✅ EXCELLENT: High reward-P&L alignment!")
        elif mean_corr >= 0.6 and episode_corr >= 0.6:
            print("✅ GOOD: Acceptable reward-P&L alignment")
        elif mean_corr >= 0.4 or episode_corr >= 0.4:
            print("⚠️ MODERATE: Consider improving reward function")
        else:
            print("❌ POOR: Reward function needs revision")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error in quick audit check: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the audit callback
    print("🎯 Reward-P&L Audit System")
    print("This module provides comprehensive auditing for reward-P&L alignment")
    print("\nUsage example:")
    print("""
    from src.training.reward_pnl_audit import RewardPnLAudit
    
    # Create audit callback
    audit_callback = RewardPnLAudit(
        output_dir="reward_audit_results",
        min_correlation_threshold=0.6,
        fail_fast=False
    )
    
    # Use with training
    model.learn(total_timesteps=100000, callback=audit_callback)
    
    # Quick post-training check
    quick_audit_check("reward_audit_results/reward_pnl_audit.csv")
    """)