#!/usr/bin/env python3
"""
Enhanced Callbacks for 200K Dual-Ticker Training
Includes P&L tracking, executive dashboard metrics, and TensorBoard integration
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter


class TensorBoardCallback(BaseCallback):
    """Enhanced TensorBoard callback with P&L and drawdown tracking"""
    
    def __init__(self, log_dir: str, track_pnl: bool = True, track_drawdown: bool = True):
        super().__init__()
        self.log_dir = log_dir
        self.track_pnl = track_pnl
        self.track_drawdown = track_drawdown
        self.writer = None
        self.episode_rewards = []
        self.episode_pnls = []
        self.max_portfolio_value = 10000  # Initial balance
        
    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer"""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
    def _on_step(self) -> bool:
        """Log metrics every step"""
        if self.writer is None:
            return True
            
        # Log basic training metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Track episode completion
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                
                # Log to TensorBoard
                self.writer.add_scalar('Episode/Reward', episode_reward, self.num_timesteps)
                self.writer.add_scalar('Episode/Length', episode_length, self.num_timesteps)
                
                # Track P&L if available
                if self.track_pnl and 'portfolio_value' in info:
                    portfolio_value = info['portfolio_value']
                    pnl = portfolio_value - 10000  # vs initial balance
                    self.episode_pnls.append(pnl)
                    
                    self.writer.add_scalar('Trading/PnL', pnl, self.num_timesteps)
                    self.writer.add_scalar('Trading/Portfolio_Value', portfolio_value, self.num_timesteps)
                    
                    # Update max portfolio value for drawdown calculation
                    self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
                    
                    # Calculate drawdown
                    if self.track_drawdown:
                        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
                        self.writer.add_scalar('Risk/Drawdown', drawdown, self.num_timesteps)
                
                # Log recent performance
                if len(self.episode_rewards) >= 100:
                    recent_mean = np.mean(self.episode_rewards[-100:])
                    self.writer.add_scalar('Performance/Recent_100_Mean', recent_mean, self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class PnLTrackingCallback(BaseCallback):
    """Track P&L progress toward demo targets"""
    
    def __init__(self, target_pnl: float = 1000, max_drawdown: float = 0.02):
        super().__init__()
        self.target_pnl = target_pnl
        self.max_drawdown = max_drawdown
        self.best_pnl = -float('inf')
        self.worst_drawdown = 0
        self.episodes_profitable = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        """Track P&L metrics"""
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'episode' in info and 'portfolio_value' in info:
                self.total_episodes += 1
                portfolio_value = info['portfolio_value']
                pnl = portfolio_value - 10000
                
                # Track best P&L
                if pnl > self.best_pnl:
                    self.best_pnl = pnl
                    self.logger.record('PnL/Best_PnL', self.best_pnl)
                
                # Track profitability
                if pnl > 0:
                    self.episodes_profitable += 1
                
                # Calculate win rate
                win_rate = self.episodes_profitable / self.total_episodes
                self.logger.record('PnL/Win_Rate', win_rate)
                
                # Check target progress
                target_progress = min(pnl / self.target_pnl, 1.0) if self.target_pnl > 0 else 0
                self.logger.record('Demo/Target_Progress', target_progress)
                
                # Log milestone achievements
                if pnl >= self.target_pnl * 0.25:
                    self.logger.record('Demo/Milestone_25pct', 1)
                if pnl >= self.target_pnl * 0.5:
                    self.logger.record('Demo/Milestone_50pct', 1)
                if pnl >= self.target_pnl * 0.75:
                    self.logger.record('Demo/Milestone_75pct', 1)
                if pnl >= self.target_pnl:
                    self.logger.record('Demo/Target_Achieved', 1)
        
        return True


class ExecutiveDashboardCallback(BaseCallback):
    """Generate executive dashboard metrics for demo preparation"""
    
    def __init__(self, eval_freq: int = 5000, log_path: str = "reports/executive_dashboard.json"):
        super().__init__()
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.dashboard_data = {
            'training_progress': {},
            'performance_metrics': {},
            'risk_metrics': {},
            'demo_readiness': {}
        }
        
    def _on_step(self) -> bool:
        """Update dashboard metrics"""
        if self.n_calls % self.eval_freq == 0:
            self._update_dashboard()
            self._save_dashboard()
        
        return True
    
    def _update_dashboard(self):
        """Update all dashboard metrics"""
        timestamp = datetime.now().isoformat()
        
        # Training progress
        self.dashboard_data['training_progress'] = {
            'timestamp': timestamp,
            'total_timesteps': self.num_timesteps,
            'progress_pct': (self.num_timesteps / 200000) * 100,
            'estimated_completion': self._estimate_completion()
        }
        
        # Performance metrics (if available)
        if hasattr(self.training_env, 'get_attr'):
            try:
                # Quick evaluation
                mean_reward, std_reward = evaluate_policy(
                    self.model, self.training_env, n_eval_episodes=5, deterministic=True
                )
                
                self.dashboard_data['performance_metrics'] = {
                    'mean_episode_reward': float(mean_reward),
                    'reward_std': float(std_reward),
                    'evaluation_episodes': 5,
                    'timestamp': timestamp
                }
            except Exception as e:
                self.logger.warning(f"Could not evaluate model: {e}")
        
        # Demo readiness assessment
        self._assess_demo_readiness()
    
    def _estimate_completion(self) -> str:
        """Estimate training completion time"""
        if self.num_timesteps > 1000:  # Need some data points
            # Rough estimate based on current progress
            steps_per_hour = 25000  # Approximate for RTX 3060
            remaining_steps = 200000 - self.num_timesteps
            hours_remaining = remaining_steps / steps_per_hour
            
            return f"~{hours_remaining:.1f} hours remaining"
        return "Calculating..."
    
    def _assess_demo_readiness(self):
        """Assess readiness for management demo"""
        progress_pct = (self.num_timesteps / 200000) * 100
        
        readiness_score = 0
        readiness_factors = []
        
        # Training progress factor
        if progress_pct >= 100:
            readiness_score += 40
            readiness_factors.append("✅ Training complete")
        elif progress_pct >= 75:
            readiness_score += 30
            readiness_factors.append("🔄 Training 75%+ complete")
        elif progress_pct >= 50:
            readiness_score += 20
            readiness_factors.append("🔄 Training 50%+ complete")
        
        # Performance factor (placeholder - would need actual metrics)
        if 'mean_episode_reward' in self.dashboard_data.get('performance_metrics', {}):
            mean_reward = self.dashboard_data['performance_metrics']['mean_episode_reward']
            if mean_reward > 0:
                readiness_score += 30
                readiness_factors.append("✅ Positive mean reward")
            elif mean_reward > -100:
                readiness_score += 15
                readiness_factors.append("⚠️ Moderate performance")
        
        # Infrastructure factor (always ready based on Week 2 completion)
        readiness_score += 30
        readiness_factors.append("✅ Infrastructure operational")
        
        self.dashboard_data['demo_readiness'] = {
            'readiness_score': readiness_score,
            'readiness_pct': min(readiness_score, 100),
            'factors': readiness_factors,
            'status': self._get_readiness_status(readiness_score)
        }
    
    def _get_readiness_status(self, score: int) -> str:
        """Get readiness status based on score"""
        if score >= 90:
            return "🎯 DEMO READY"
        elif score >= 70:
            return "🔄 NEARLY READY"
        elif score >= 50:
            return "⚠️ IN PROGRESS"
        else:
            return "🔴 NOT READY"
    
    def _save_dashboard(self):
        """Save dashboard data to JSON file"""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        with open(self.log_path, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2)


class PerformanceMonitorCallback(BaseCallback):
    """Monitor key performance indicators during training"""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        super().__init__()
        self.alert_thresholds = alert_thresholds or {
            'min_reward': -1000,
            'max_drawdown': 0.05,
            'min_win_rate': 0.3
        }
        self.performance_history = []
        
    def _on_step(self) -> bool:
        """Monitor performance and trigger alerts if needed"""
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                episode_reward = info['episode']['r']
                
                # Check alert thresholds
                if episode_reward < self.alert_thresholds['min_reward']:
                    self.logger.warning(f"⚠️ Low episode reward: {episode_reward}")
                
                # Track performance trend
                self.performance_history.append(episode_reward)
                
                # Keep only recent history
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                # Log performance trend
                if len(self.performance_history) >= 10:
                    recent_trend = np.mean(self.performance_history[-10:])
                    self.logger.record('Performance/Recent_Trend', recent_trend)
        
        return True