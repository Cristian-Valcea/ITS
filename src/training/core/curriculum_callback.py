"""
Curriculum Learning Callback for Stable-Baselines3

Implements episode-based curriculum learning with performance gates:
- Progressive risk constraint tightening
- Performance-based stage advancement  
- AND logic for gate criteria (drawdown AND Sharpe)
- Integration with existing training infrastructure

Integrates with CurriculumManager for sophisticated curriculum control.
"""

import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import logging

try:
    from ...risk.curriculum_manager import CurriculumManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from risk.curriculum_manager import CurriculumManager


class CurriculumLearningCallback(BaseCallback):
    """
    Callback for curriculum learning with episode-based progression.
    
    Manages progressive risk constraint tightening based on agent performance,
    with performance gates requiring both drawdown control AND positive Sharpe.
    """
    
    def __init__(self, 
                 curriculum_config: Dict[str, Any],
                 risk_config: Dict[str, Any],
                 verbose: int = 1):
        """
        Initialize curriculum learning callback.
        
        Args:
            curriculum_config: Configuration for curriculum learning
            risk_config: Risk management configuration to update
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.curriculum_manager = CurriculumManager(curriculum_config)
        self.risk_config = risk_config
        self.logger = logging.getLogger(__name__)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_start_step = 0
        self.episode_returns = []
        self.episode_portfolio_values = []
        self.episode_trades = 0
        
        # Performance tracking for current episode
        self.episode_initial_value = None
        self.episode_peak_value = None
        self.episode_min_value = None
        
        # Stage tracking
        self.last_stage_name = None
        self.stage_changes = []
        
        self.logger.info("🎓 Curriculum Learning Callback initialized")
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.logger.info("🎓 Curriculum Learning: Training started")
        
        # Initialize with first stage constraints
        self._update_environment_constraints()
        
        # Log initial stage
        constraints = self.curriculum_manager.get_current_constraints()
        self.logger.info(f"🎯 Starting curriculum stage: {constraints['stage_name']}")
        self.logger.info(f"   Drawdown cap: {constraints['drawdown_cap']:.1%}")
        self.logger.info(f"   Lambda penalty: {constraints['lambda_penalty']}")
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:  # Episode ended
            self._on_episode_end()
        
        # Track episode metrics
        if hasattr(self.training_env, 'get_wrapper_attr'):
            try:
                # Get current portfolio value from environment
                info = self.locals.get('infos', [{}])[0]
                if 'portfolio_value' in info:
                    portfolio_value = info['portfolio_value']
                    self.episode_portfolio_values.append(portfolio_value)
                    
                    # Track episode statistics
                    if self.episode_initial_value is None:
                        self.episode_initial_value = portfolio_value
                        self.episode_peak_value = portfolio_value
                        self.episode_min_value = portfolio_value
                    else:
                        self.episode_peak_value = max(self.episode_peak_value, portfolio_value)
                        self.episode_min_value = min(self.episode_min_value, portfolio_value)
                
                # Track trades
                if 'last_trade_details' in info:
                    self.episode_trades += 1
                    
            except Exception as e:
                self.logger.debug(f"Error tracking episode metrics: {e}")
        
        return True
    
    def _on_episode_end(self) -> None:
        """Called when an episode ends."""
        self.current_episode += 1
        
        # Calculate episode performance metrics
        if self.episode_initial_value is not None and len(self.episode_portfolio_values) > 0:
            final_value = self.episode_portfolio_values[-1]
            total_return = (final_value - self.episode_initial_value) / self.episode_initial_value
            
            # Calculate maximum drawdown
            if self.episode_peak_value > self.episode_initial_value:
                max_drawdown = (self.episode_peak_value - self.episode_min_value) / self.episode_peak_value
            else:
                max_drawdown = 0.0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.episode_portfolio_values) > 1:
                returns = np.diff(self.episode_portfolio_values) / self.episode_portfolio_values[:-1]
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)  # Annualized
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate volatility
            if len(self.episode_portfolio_values) > 1:
                returns = np.diff(self.episode_portfolio_values) / self.episode_portfolio_values[:-1]
                volatility = np.std(returns) * np.sqrt(252 * 390) if len(returns) > 1 else 0.0
            else:
                volatility = 0.0
            
            # Record performance with curriculum manager
            curriculum_info = self.curriculum_manager.record_episode_performance(
                episode=self.current_episode,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                num_trades=self.episode_trades,
                final_portfolio_value=final_value
            )
            
            # Check for stage advancement
            if curriculum_info['advancement_info']['advanced']:
                self._handle_stage_advancement(curriculum_info['advancement_info'])
            
            # Log episode summary
            if self.verbose >= 1:
                stage_name = curriculum_info['current_stage']
                episodes_in_stage = curriculum_info['episodes_in_stage']
                
                self.logger.info(f"📊 Episode {self.current_episode} completed ({stage_name}, #{episodes_in_stage}):")
                self.logger.info(f"   Return: {total_return:.2%}, Drawdown: {max_drawdown:.2%}")
                self.logger.info(f"   Sharpe: {sharpe_ratio:.3f}, Volatility: {volatility:.2%}")
                self.logger.info(f"   Trades: {self.episode_trades}, Final Value: ${final_value:.2f}")
                
                # Log advancement info if gate check was performed
                adv_info = curriculum_info['advancement_info']
                if adv_info['gate_check_performed']:
                    self.logger.info(f"🚪 Gate Check: {adv_info['reason']}")
                    if 'criteria_met' in adv_info:
                        criteria = adv_info['criteria_met']
                        dd_status = "✅" if criteria['drawdown']['met'] else "❌"
                        sharpe_status = "✅" if criteria['sharpe']['met'] else "❌"
                        self.logger.info(f"   Drawdown: {criteria['drawdown']['value']:.2%} {dd_status}")
                        self.logger.info(f"   Sharpe: {criteria['sharpe']['value']:.3f} {sharpe_status}")
            
            # Log to tensorboard if available
            if hasattr(self.logger, 'record'):
                self.logger.record('curriculum/episode', self.current_episode)
                self.logger.record('curriculum/stage', curriculum_info['current_stage'])
                self.logger.record('curriculum/episodes_in_stage', curriculum_info['episodes_in_stage'])
                self.logger.record('curriculum/total_return', total_return)
                self.logger.record('curriculum/max_drawdown', max_drawdown)
                self.logger.record('curriculum/sharpe_ratio', sharpe_ratio)
                self.logger.record('curriculum/volatility', volatility)
                self.logger.record('curriculum/num_trades', self.episode_trades)
        
        # Reset episode tracking
        self._reset_episode_tracking()
    
    def _handle_stage_advancement(self, advancement_info: Dict[str, Any]) -> None:
        """Handle advancement to next curriculum stage."""
        previous_stage = advancement_info['previous_stage']
        new_stage = advancement_info['new_stage']
        
        self.logger.info(f"🎓 CURRICULUM ADVANCEMENT: {previous_stage} → {new_stage}")
        self.logger.info(f"   Reason: {advancement_info['reason']}")
        
        # Update environment constraints
        self._update_environment_constraints()
        
        # Log new constraints
        constraints = self.curriculum_manager.get_current_constraints()
        self.logger.info(f"🎯 New stage constraints:")
        self.logger.info(f"   Drawdown cap: {constraints['drawdown_cap']:.1%}")
        self.logger.info(f"   Lambda penalty: {constraints['lambda_penalty']}")
        
        # Record stage change
        self.stage_changes.append({
            'episode': self.current_episode,
            'from_stage': previous_stage,
            'to_stage': new_stage,
            'reason': advancement_info['reason']
        })
        
        # Log to tensorboard
        if hasattr(self.logger, 'record'):
            self.logger.record('curriculum/stage_advancement', len(self.stage_changes))
    
    def _update_environment_constraints(self) -> None:
        """Update environment risk constraints based on current curriculum stage."""
        constraints = self.curriculum_manager.get_current_constraints()
        
        # Update risk configuration
        if 'curriculum' in self.risk_config:
            # Update curriculum stage parameters
            self.risk_config['dd_limit'] = constraints['drawdown_cap']
            self.risk_config['penalty_lambda'] = constraints['lambda_penalty']
            
            # Update advanced reward shaping if enabled
            if 'advanced_reward_shaping' in self.risk_config:
                lagrangian_config = self.risk_config['advanced_reward_shaping'].get('lagrangian_constraint', {})
                if lagrangian_config.get('enabled', False):
                    # Update lambda value in advanced reward shaping
                    lagrangian_config['initial_lambda'] = constraints['lambda_penalty'] * 0.1  # Scale appropriately
        
        # Try to update environment directly if possible
        try:
            if hasattr(self.training_env, 'env_method'):
                # For VecEnv
                self.training_env.env_method('update_risk_constraints', constraints)
            elif hasattr(self.training_env, 'update_risk_constraints'):
                # For single env
                self.training_env.update_risk_constraints(constraints)
            elif hasattr(self.training_env, 'env') and hasattr(self.training_env.env, 'update_risk_constraints'):
                # For wrapped env
                self.training_env.env.update_risk_constraints(constraints)
        except Exception as e:
            self.logger.debug(f"Could not update environment constraints directly: {e}")
    
    def _reset_episode_tracking(self) -> None:
        """Reset episode-level tracking variables."""
        self.episode_start_step = self.num_timesteps
        self.episode_returns.clear()
        self.episode_portfolio_values.clear()
        self.episode_trades = 0
        self.episode_initial_value = None
        self.episode_peak_value = None
        self.episode_min_value = None
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        stats = self.curriculum_manager.get_stage_statistics()
        stats['stage_changes'] = self.stage_changes
        stats['current_episode'] = self.current_episode
        return stats
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        self.logger.info("🎓 Curriculum Learning: Training ended")
        
        # Log final curriculum statistics
        stats = self.get_curriculum_statistics()
        self.logger.info(f"📊 Final Curriculum Statistics:")
        self.logger.info(f"   Total episodes: {stats['total_episodes']}")
        self.logger.info(f"   Final stage: {stats['current_stage']}")
        self.logger.info(f"   Stage changes: {len(self.stage_changes)}")
        
        # Export curriculum data for analysis
        try:
            curriculum_df = self.curriculum_manager.export_curriculum_data()
            if not curriculum_df.empty:
                # Save to CSV for analysis
                save_path = Path(self.logger.get_dir()) / "curriculum_performance.csv"
                curriculum_df.to_csv(save_path, index=False)
                self.logger.info(f"📁 Curriculum data saved to: {save_path}")
        except Exception as e:
            self.logger.warning(f"Could not save curriculum data: {e}")