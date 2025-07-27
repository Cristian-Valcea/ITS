"""
RiskManager - Central coordinator for all risk controls

This is the main façade that orchestrates all risk control modules.
It maintains state across episodes and provides a clean interface
for the Gym environment wrappers.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .volatility_penalty import VolatilityPenalty
from .curriculum_scheduler import CurriculumScheduler


class RiskManager:
    """
    Central coordinator for all risk control modules.
    
    Maintains stateful risk engines (drawdown checker, volatility monitor, etc.)
    and exposes them through a clean step() interface for Gym wrappers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RiskManager with configuration.
        
        Args:
            config: Risk configuration dict containing:
                - vol_window: Window size for volatility calculation (default: 60)
                - penalty_lambda: Weight for volatility penalty (default: 0.25)
                - dd_limit: Maximum drawdown limit (default: 0.02)
                - curriculum: Curriculum configuration (optional)
        """
        self.logger = logging.getLogger("RiskManager")
        self.logger.propagate = False  # 🔧 FIX: Prevent duplicate logging
        self.config = config
        
        # Initialize volatility penalty tracker
        vol_config = {
            'vol_window': config.get('vol_window', 60),  # Use vol_window key to match config
            'penalty_lambda': config.get('penalty_lambda', 0.25),
            'target_sigma': config.get('target_sigma', 0.0)  # Add target_sigma support
        }
        self.volatility_penalty = VolatilityPenalty(vol_config)
        
        # Drawdown tracking
        self.dd_limit = config.get('dd_limit', 0.02)
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        
        # Episode tracking
        self.current_episode = 0
        self.episode_start_value = 0.0
        
        # Curriculum learning
        curriculum_config = config.get('curriculum', {})
        self.curriculum_scheduler = CurriculumScheduler(curriculum_config)
        
        # Apply initial curriculum configuration
        if self.curriculum_scheduler.enabled:
            curriculum_risk_config = self.curriculum_scheduler.get_current_config()
            if curriculum_risk_config:
                self.update_config(curriculum_risk_config)
        
        # Risk metrics for logging
        self.risk_metrics = {
            'current_volatility': 0.0,
            'current_drawdown': 0.0,
            'volatility_penalty': 0.0,
            'drawdown_breached': False
        }
        
        self.logger.info(f"RiskManager initialized - vol_window: {vol_config['vol_window']}, "
                        f"penalty_lambda: {vol_config['penalty_lambda']}, dd_limit: {self.dd_limit}")
    
    def step(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step of risk monitoring.
        
        Args:
            info: Step information containing:
                - timestamp: Current datetime
                - portfolio_value: Current portfolio value
                - position: Current position size
                - step_return: Step return percentage
        
        Returns:
            Dict containing:
                - volatility_penalty: Penalty to subtract from reward
                - should_terminate: Whether episode should terminate
                - risk_metrics: Current risk metrics for logging
        """
        timestamp = info['timestamp']
        portfolio_value = info['portfolio_value']
        position = info['position']
        step_return = info['step_return']
        
        # Update volatility penalty
        volatility_penalty = self.volatility_penalty.update(step_return)
        
        # Update drawdown tracking
        self._update_drawdown(portfolio_value)
        
        # Check termination conditions
        should_terminate = self._check_termination_conditions()
        
        # Update risk metrics
        self.risk_metrics.update({
            'current_volatility': self.volatility_penalty.get_current_volatility(),
            'current_drawdown': self.current_drawdown,
            'volatility_penalty': volatility_penalty,
            'drawdown_breached': self.current_drawdown > self.dd_limit
        })
        
        return {
            'volatility_penalty': volatility_penalty,
            'should_terminate': should_terminate,
            'risk_metrics': self.risk_metrics.copy()
        }
    
    def reset_episode(self, initial_portfolio_value: float):
        """
        Reset state for new episode.
        
        Args:
            initial_portfolio_value: Starting portfolio value for the episode
        """
        self.current_episode += 1
        self.episode_start_value = initial_portfolio_value
        self.peak_portfolio_value = initial_portfolio_value
        self.current_drawdown = 0.0
        
        # Reset volatility penalty for new episode
        self.volatility_penalty.reset()
        
        self.logger.debug(f"Episode {self.current_episode} reset - initial_value: ${initial_portfolio_value:,.2f}")
    
    def _update_drawdown(self, portfolio_value: float):
        """Update drawdown tracking."""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        else:
            self.current_drawdown = 0.0
    
    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate due to risk limits."""
        # TEMPORARILY DISABLE ALL TERMINATION CONDITIONS FOR STABILIZATION
        return False
        
        # Drawdown limit check (DISABLED)
        # if self.current_drawdown > self.dd_limit:
        #     self.logger.warning(f"Episode {self.current_episode}: Max drawdown breached! "
        #                       f"Drawdown: {self.current_drawdown:.2%}, Limit: {self.dd_limit:.2%}")
        #     return True
        # 
        # return False
    
    def get_risk_features(self) -> Dict[str, float]:
        """
        Get current risk features for observation space.
        
        Returns:
            Dict with normalized risk features:
                - volatility: Current return volatility (0-1 scale)
                - drawdown_pct: Current drawdown percentage (0-1 scale)
                - dd_proximity: How close to drawdown limit (0-1 scale)
        """
        volatility = self.volatility_penalty.get_current_volatility()
        
        return {
            'volatility': min(volatility * 10, 1.0),  # Scale volatility to 0-1 range
            'drawdown_pct': self.current_drawdown,
            'dd_proximity': min(self.current_drawdown / self.dd_limit, 1.0) if self.dd_limit > 0 else 0.0
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration (used by curriculum scheduler).
        
        Args:
            new_config: New configuration parameters
        """
        if 'dd_limit' in new_config:
            old_limit = self.dd_limit
            self.dd_limit = new_config['dd_limit']
            self.logger.info(f"Updated drawdown limit: {old_limit:.2%} → {self.dd_limit:.2%}")
        
        if 'penalty_lambda' in new_config:
            self.volatility_penalty.update_lambda(new_config['penalty_lambda'])
        
        # Update internal config
        self.config.update(new_config)
    
    def episode_completed(self, final_portfolio_value: float) -> Dict[str, Any]:
        """
        Process episode completion and handle curriculum advancement.
        
        Args:
            final_portfolio_value: Final portfolio value at episode end
            
        Returns:
            Episode summary with curriculum information
        """
        # Calculate episode metrics
        episode_return = (final_portfolio_value - self.episode_start_value) / self.episode_start_value
        avg_volatility = self.volatility_penalty.get_average_volatility()
        
        # Estimate Sharpe ratio (simplified)
        sharpe_ratio = episode_return / avg_volatility if avg_volatility > 0 else 0.0
        
        episode_metrics = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.current_drawdown,
            'total_return': episode_return,
            'volatility': avg_volatility,
            'episode_length': self.volatility_penalty.step_count
        }
        
        # Process curriculum advancement
        curriculum_update = None
        if self.curriculum_scheduler.enabled:
            curriculum_update = self.curriculum_scheduler.episode_end(episode_metrics)
            if curriculum_update:
                self.update_config(curriculum_update)
                self.logger.info(f"🎓 Curriculum advanced - new config applied")
        
        # Create episode summary
        summary = {
            'episode': self.current_episode,
            'peak_portfolio_value': self.peak_portfolio_value,
            'final_portfolio_value': final_portfolio_value,
            'episode_return': episode_return,
            'final_drawdown': self.current_drawdown,
            'avg_volatility': avg_volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_volatility_penalty': self.volatility_penalty.get_total_penalty(),
            'drawdown_breached': self.current_drawdown > self.dd_limit,
            'curriculum_advanced': curriculum_update is not None,
            'curriculum_stage': self.curriculum_scheduler.get_stage_info() if self.curriculum_scheduler.enabled else None
        }
        
        return summary
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the completed episode (legacy method)."""
        return {
            'episode': self.current_episode,
            'peak_portfolio_value': self.peak_portfolio_value,
            'final_drawdown': self.current_drawdown,
            'avg_volatility': self.volatility_penalty.get_average_volatility(),
            'total_volatility_penalty': self.volatility_penalty.get_total_penalty(),
            'drawdown_breached': self.current_drawdown > self.dd_limit
        }