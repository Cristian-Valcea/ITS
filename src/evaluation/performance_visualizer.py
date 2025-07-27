"""
Performance Visualizer for IntradayJules Trading System

Creates comprehensive performance visualizations after training completion.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from datetime import datetime

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Creates comprehensive performance visualizations for trading system evaluation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up directories
        self.reports_dir = config.get('reports_dir', 'reports/')
        self.plots_dir = os.path.join(self.reports_dir, 'plots')
        self.logs_dir = config.get('logs_dir', 'logs/')
        
        # Create directories
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.logger.info(f"PerformanceVisualizer initialized. Plots will be saved to: {self.plots_dir}")
    
    def create_performance_plots(
        self, 
        metrics: Dict[str, Any],
        trade_log_df: Optional[pd.DataFrame] = None,
        model_name: str = "model"
    ) -> List[str]:
        """Create performance visualization plots."""
        self.logger.info(f"🎨 Creating performance visualizations for {model_name}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_files = []
        
        try:
            # 1. Summary Dashboard
            summary_plot = self._create_summary_plot(metrics, model_name, timestamp)
            if summary_plot:
                plot_files.append(summary_plot)
            
            # 2. Training Progress
            training_plot = self._create_training_plot(model_name, timestamp)
            if training_plot:
                plot_files.append(training_plot)
            
            # 3. Portfolio Performance
            if trade_log_df is not None and not trade_log_df.empty:
                portfolio_plot = self._create_portfolio_plot(trade_log_df, model_name, timestamp)
                if portfolio_plot:
                    plot_files.append(portfolio_plot)
            
            self.logger.info(f"✅ Created {len(plot_files)} performance visualization files")
            return plot_files
            
        except Exception as e:
            self.logger.error(f"❌ Error creating performance visualizations: {e}", exc_info=True)
            return plot_files
    
    def _create_summary_plot(self, metrics: Dict[str, Any], model_name: str, timestamp: str) -> Optional[str]:
        """Create summary performance plot."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'📊 Performance Summary - {model_name}', fontsize=16, fontweight='bold')
            
            # Key metrics text
            ax1.axis('off')
            metrics_text = f"""
KEY METRICS
{'='*20}
📈 Total Return: {metrics.get('total_return', 0)*100:.2f}%
📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
📉 Max Drawdown: {abs(metrics.get('max_drawdown', 0))*100:.2f}%
🔄 Avg Turnover: {metrics.get('avg_turnover', 0):.2f}
💰 Total Trades: {metrics.get('num_trades', 0)}
🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
            """
            ax1.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Performance grade
            grade = self._calculate_grade(metrics)
            colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'F': 'darkred'}
            ax2.text(0.5, 0.5, grade, fontsize=48, fontweight='bold', 
                    ha='center', va='center', color=colors.get(grade, 'black'))
            ax2.set_title('🏆 Performance Grade')
            ax2.axis('off')
            
            # Risk metrics bar chart
            risk_metrics = {
                'Sharpe': metrics.get('sharpe_ratio', 0),
                'Sortino': metrics.get('sortino_ratio', 0),
                'Calmar': metrics.get('calmar_ratio', 0)
            }
            ax3.bar(risk_metrics.keys(), risk_metrics.values(), color=['green', 'blue', 'purple'], alpha=0.7)
            ax3.set_title('📊 Risk-Adjusted Returns')
            ax3.set_ylabel('Ratio')
            ax3.grid(True, alpha=0.3)
            
            # Risk-return scatter
            returns = metrics.get('total_return', 0) * 100
            risk = metrics.get('volatility', 0) * 100
            ax4.scatter([risk], [returns], s=200, color='blue', alpha=0.7)
            ax4.set_title('📊 Risk-Return Profile')
            ax4.set_xlabel('Risk (Volatility %)')
            ax4.set_ylabel('Return (%)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'{model_name}_summary_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Summary plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating summary plot: {e}")
            return None
    
    def _create_training_plot(self, model_name: str, timestamp: str) -> Optional[str]:
        """Create training progress plot."""
        try:
            log_file = os.path.join(self.logs_dir, 'orchestrator_gpu_fixed.log')
            
            if not os.path.exists(log_file):
                self.logger.warning(f"Training log file not found: {log_file}")
                return None
            
            # Parse training data
            episodes, rewards = self._parse_training_logs(log_file)
            
            if not episodes:
                self.logger.warning("No training data found in logs")
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'🚀 Training Progress - {model_name}', fontsize=16, fontweight='bold')
            
            # Episode rewards
            ax1.plot(episodes, rewards, color='green', alpha=0.7, linewidth=1)
            if len(rewards) > 10:
                rolling_mean = pd.Series(rewards).rolling(min(50, len(rewards)//4), min_periods=1).mean()
                ax1.plot(episodes, rolling_mean, color='darkgreen', linewidth=2, label='Moving Average')
            ax1.set_title('📈 Episode Rewards Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Reward distribution
            ax2.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('📊 Reward Distribution')
            ax2.set_xlabel('Reward')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'{model_name}_training_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📈 Training plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating training plot: {e}")
            return None
    
    def _create_portfolio_plot(self, trade_log_df: pd.DataFrame, model_name: str, timestamp: str) -> Optional[str]:
        """Create portfolio performance plot."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'💰 Portfolio Performance - {model_name}', fontsize=16, fontweight='bold')
            
            # Portfolio value over time
            if 'portfolio_value' in trade_log_df.columns:
                trade_log_df['portfolio_value'].plot(ax=ax1, color='blue', linewidth=2)
                ax1.set_title('📈 Portfolio Value Over Time')
                ax1.set_ylabel('Portfolio Value ($)')
                ax1.grid(True, alpha=0.3)
            
            # PnL distribution
            if 'pnl' in trade_log_df.columns:
                trade_log_df['pnl'].hist(bins=30, ax=ax2, alpha=0.7, color='purple', edgecolor='black')
                ax2.set_title('📊 PnL Distribution')
                ax2.set_xlabel('PnL ($)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            
            # Cumulative PnL
            if 'pnl' in trade_log_df.columns:
                cumulative_pnl = trade_log_df['pnl'].cumsum()
                ax3.plot(cumulative_pnl.index, cumulative_pnl.values, color='green', linewidth=2)
                ax3.set_title('📈 Cumulative PnL')
                ax3.set_ylabel('Cumulative PnL ($)')
                ax3.grid(True, alpha=0.3)
            
            # Win/Loss pie chart
            if 'pnl' in trade_log_df.columns:
                wins = (trade_log_df['pnl'] > 0).sum()
                losses = (trade_log_df['pnl'] < 0).sum()
                if wins + losses > 0:
                    ax4.pie([wins, losses], labels=['Wins', 'Losses'], colors=['green', 'red'], 
                           autopct='%1.1f%%', startangle=90)
                    ax4.set_title('🎯 Win/Loss Ratio')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'{model_name}_portfolio_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"💰 Portfolio plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio plot: {e}")
            return None
    
    def _parse_training_logs(self, log_file: str) -> tuple:
        """Parse training progress from log files."""
        episodes = []
        rewards = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Episode' in line and 'reward' in line:
                        import re
                        episode_match = re.search(r'Episode (\d+)', line)
                        reward_match = re.search(r'reward[:\s]+([+-]?\d*\.?\d+)', line)
                        
                        if episode_match and reward_match:
                            episodes.append(int(episode_match.group(1)))
                            rewards.append(float(reward_match.group(1)))
        
        except Exception as e:
            self.logger.error(f"Error parsing training logs: {e}")
        
        return episodes, rewards
    
    def _calculate_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate performance grade."""
        score = 0
        
        # Sharpe ratio (40% weight)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            score += 40
        elif sharpe >= 1.5:
            score += 32
        elif sharpe >= 1.0:
            score += 24
        elif sharpe >= 0.5:
            score += 16
        
        # Max drawdown (30% weight)
        drawdown = abs(metrics.get('max_drawdown', 1))
        if drawdown <= 0.02:
            score += 30
        elif drawdown <= 0.05:
            score += 24
        elif drawdown <= 0.10:
            score += 18
        elif drawdown <= 0.15:
            score += 12
        
        # Win rate (20% weight)
        win_rate = metrics.get('win_rate', 0)
        if win_rate >= 0.6:
            score += 20
        elif win_rate >= 0.55:
            score += 16
        elif win_rate >= 0.5:
            score += 12
        elif win_rate >= 0.45:
            score += 8
        
        # Total return (10% weight)
        total_return = metrics.get('total_return', 0)
        if total_return >= 0.2:
            score += 10
        elif total_return >= 0.1:
            score += 8
        elif total_return >= 0.05:
            score += 6
        elif total_return >= 0:
            score += 4
        
        # Convert to letter grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'