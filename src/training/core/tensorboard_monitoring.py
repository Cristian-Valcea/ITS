"""
TensorBoard Monitoring & Debugging Module

Provides comprehensive TensorBoard logging with custom scalars for:
- Volatility penalty tracking
- Drawdown percentage monitoring  
- Q-value variance (Q_max - Q_min)
- Lagrangian multiplier evolution
- Advanced reward shaping metrics
- Risk management indicators

This module enables deep insights into training dynamics and system behavior.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from collections import deque
import torch

# Core dependencies
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv

# Internal imports
try:
    from ...gym_env.intraday_trading_env import IntradayTradingEnv as TradingEnvironment
    from .q_value_tracker import create_q_value_tracker, is_q_value_compatible
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gym_env.intraday_trading_env import IntradayTradingEnv as TradingEnvironment
    from training.core.q_value_tracker import create_q_value_tracker, is_q_value_compatible


class TensorBoardMonitoringCallback(BaseCallback):
    """
    Advanced TensorBoard monitoring callback with custom scalars.
    
    Tracks and visualizes:
    - vol_penalty: Volatility penalty applied per step
    - drawdown_pct: Current drawdown percentage
    - Q_variance: Q_max - Q_min spread for variance monitoring
    - lambda_lagrangian: Lagrangian multiplier evolution (if enabled)
    - reward_components: Breakdown of reward shaping components
    - risk_metrics: Advanced risk management indicators
    """

    def __init__(
        self,
        config: Dict[str, Any],
        log_frequency: int = 100,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.config = config
        self.log_frequency = log_frequency
        
        # Initialize tracking variables
        self.step_count = 0
        self.episode_count = 0
        
        # Metrics buffers for smoothing
        self.vol_penalty_buffer = deque(maxlen=1000)
        self.drawdown_buffer = deque(maxlen=1000)
        self.q_variance_buffer = deque(maxlen=1000)
        self.lambda_buffer = deque(maxlen=1000)
        self.reward_buffer = deque(maxlen=1000)
        
        # Turnover penalty tracking buffers
        self.turnover_penalty_buffer = deque(maxlen=1000)
        self.turnover_normalized_buffer = deque(maxlen=1000)
        self.turnover_absolute_buffer = deque(maxlen=1000)
        self.turnover_target_buffer = deque(maxlen=1000)
        self.turnover_excess_buffer = deque(maxlen=1000)
        
        # Performance metrics buffers
        self.total_reward_buffer = deque(maxlen=1000)
        self.win_rate_buffer = deque(maxlen=100)  # Track wins over episodes
        self.sharpe_ratio_buffer = deque(maxlen=100)
        self.max_drawdown_buffer = deque(maxlen=100)
        self.portfolio_value_buffer = deque(maxlen=1000)
        
        # TensorBoard writer
        self.tb_writer = None
        
        # Advanced reward config
        self.advanced_reward_config = config.get('environment', {}).get('advanced_reward_config', {})
        self.lagrangian_enabled = self.advanced_reward_config.get('lagrangian_constraint', {}).get('enabled', False)
        
        # Q-value tracking
        self.q_value_tracker = None
        self.algorithm = config.get('training', {}).get('algorithm', '')
        self.track_q_values = is_q_value_compatible(self.algorithm)
        
        self._logger = logging.getLogger(__name__)
        self._logger.info("TensorBoard monitoring callback initialized")
        if self.track_q_values:
            self._logger.info(f"Q-value tracking enabled for {self.algorithm}")
        else:
            self._logger.info(f"Q-value tracking not available for {self.algorithm}")

    def _init_callback(self) -> None:
        """Initialize callback with TensorBoard writer."""
        # Get TensorBoard writer from logger
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break
        
        if self.tb_writer is None:
            self._logger.warning("TensorBoard writer not found - custom scalars disabled")
        else:
            self._logger.info("TensorBoard custom scalars enabled")
        
        # Initialize Q-value tracker if supported
        if self.track_q_values and self.model is not None:
            self.q_value_tracker = create_q_value_tracker(self.model, self.config)
            if self.q_value_tracker:
                self._logger.info("Q-value variance tracking initialized")
            else:
                self._logger.warning("Q-value tracker creation failed")

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1
        
        # Log metrics at specified frequency
        if self.step_count % self.log_frequency == 0:
            self._log_custom_scalars()
        
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        self._extract_environment_metrics()

    def _extract_environment_metrics(self) -> None:
        """Extract metrics from the training environment."""
        try:
            # Get environment from training
            env = self.training_env
            if isinstance(env, VecEnv):
                # Get first environment from vectorized env
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    base_env = env.envs[0]
                elif hasattr(env, 'env'):
                    base_env = env.env
                else:
                    return
            else:
                base_env = env
            
            # Unwrap to get TradingEnvironment
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if not isinstance(base_env, TradingEnvironment):
                return
            
            # Extract volatility penalty
            vol_penalty = getattr(base_env, 'last_vol_penalty', 0.0)
            self.vol_penalty_buffer.append(vol_penalty)
            
            # Extract drawdown percentage
            if hasattr(base_env, 'portfolio_value') and hasattr(base_env, 'initial_capital'):
                current_value = base_env.portfolio_value
                peak_value = getattr(base_env, 'peak_portfolio_value', base_env.initial_capital)
                drawdown_pct = max(0, (peak_value - current_value) / peak_value * 100)
                self.drawdown_buffer.append(drawdown_pct)
            
            # Extract Q-value variance if available
            if self.q_value_tracker and hasattr(base_env, 'last_observation'):
                obs = getattr(base_env, 'last_observation', None)
                if obs is not None:
                    q_variance = self.q_value_tracker.on_step(obs)
                    if q_variance is not None:
                        self.q_variance_buffer.append(q_variance)
            
            # Extract Lagrangian multiplier if enabled
            if self.lagrangian_enabled and hasattr(base_env, 'lagrangian_lambda'):
                lambda_val = getattr(base_env, 'lagrangian_lambda', 0.0)
                self.lambda_buffer.append(lambda_val)
            
            # Extract reward components
            last_reward = getattr(base_env, 'last_reward', 0.0)
            self.reward_buffer.append(abs(last_reward))
            
            # Extract turnover penalty metrics
            if hasattr(base_env, 'turnover_penalty_calculator') and base_env.turnover_penalty_calculator:
                calc = base_env.turnover_penalty_calculator
                
                # Get current daily turnover data
                current_turnover = getattr(base_env, 'total_traded_value', 0.0)
                
                # Always log turnover metrics (including zero turnover periods)
                portfolio_value = calc._get_current_portfolio_value()
                normalized_turnover = current_turnover / (portfolio_value + 1e-6)
                target_ratio = calc.target_ratio
                excess_ratio = normalized_turnover - target_ratio
                
                # Calculate penalty (even for zero turnover)
                turnover_penalty = calc.compute_penalty(current_turnover)
                
                # Store metrics
                self.turnover_penalty_buffer.append(turnover_penalty)
                self.turnover_normalized_buffer.append(normalized_turnover)
                self.turnover_absolute_buffer.append(current_turnover)
                self.turnover_target_buffer.append(target_ratio)
                self.turnover_excess_buffer.append(excess_ratio)
                
                # Log per-step turnover metrics to TensorBoard immediately
                if self.tb_writer:
                    self.tb_writer.add_scalar('turnover/ratio', normalized_turnover, self.step_count)
                    self.tb_writer.add_scalar('turnover/penalty', turnover_penalty, self.step_count)
                    self.tb_writer.add_scalar('turnover/absolute_value', current_turnover, self.step_count)
                    self.tb_writer.add_scalar('turnover/excess_over_target', excess_ratio, self.step_count)
                    
                    # Log penalty as percentage of NAV for better interpretation
                    penalty_pct_nav = abs(turnover_penalty) / (portfolio_value + 1e-6)
                    self.tb_writer.add_scalar('turnover/penalty_pct_nav', penalty_pct_nav, self.step_count)
                    
                    # 🎯 DYNAMIC CURRICULUM MONITORING
                    if hasattr(base_env, 'turnover_penalty_calculator'):
                        calc = base_env.turnover_penalty_calculator
                        if hasattr(calc, 'turnover_history_1d') and len(calc.turnover_history_1d) > 0:
                            avg_recent_turnover = sum(calc.turnover_history_1d) / len(calc.turnover_history_1d)
                            curriculum_multiplier = 1.0
                            if avg_recent_turnover > 10 * calc.target_ratio:
                                curriculum_multiplier = 7.0
                            elif avg_recent_turnover > 2 * calc.target_ratio:
                                curriculum_multiplier = 3.0
                            
                            self.tb_writer.add_scalar('curriculum/avg_recent_turnover', avg_recent_turnover, self.step_count)
                            self.tb_writer.add_scalar('curriculum/penalty_multiplier', curriculum_multiplier, self.step_count)
                            self.tb_writer.add_scalar('curriculum/overtrading_factor', avg_recent_turnover / calc.target_ratio, self.step_count)
                    
                    # 🎁 HOLD BONUS MONITORING
                    if hasattr(base_env, '_last_action'):
                        is_hold_action = float(base_env._last_action == 1)  # Action 1 = HOLD
                        self.tb_writer.add_scalar('actions/hold_frequency', is_hold_action, self.step_count)
                    
                    # Log daily turnover reset events
                    daily_reset = getattr(base_env, 'daily_turnover_reset', False)
                    self.tb_writer.add_scalar('turnover/daily_reset_event', float(daily_reset), self.step_count)
                    
                    # Reset the flag after logging
                    if hasattr(base_env, 'daily_turnover_reset'):
                        base_env.daily_turnover_reset = False

            
            # Extract portfolio performance metrics
            if hasattr(base_env, 'portfolio_value'):
                portfolio_value = base_env.portfolio_value
                self.portfolio_value_buffer.append(portfolio_value)
                
                # Calculate total reward for episode tracking
                if hasattr(base_env, 'episode_reward'):
                    episode_reward = getattr(base_env, 'episode_reward', 0.0)
                    self.total_reward_buffer.append(episode_reward)
            
            # Extract performance metrics at episode end
            if hasattr(base_env, 'done') and getattr(base_env, 'done', False):
                self._extract_episode_metrics(base_env)
            
        except Exception as e:
            self._logger.debug(f"Error extracting environment metrics: {e}")
    
    def _extract_episode_metrics(self, base_env) -> None:
        """Extract episode-level performance metrics."""
        try:
            # Calculate win rate (positive episode return)
            if hasattr(base_env, 'episode_reward'):
                episode_reward = getattr(base_env, 'episode_reward', 0.0)
                is_win = 1.0 if episode_reward > 0 else 0.0
                self.win_rate_buffer.append(is_win)
            
            # Calculate Sharpe ratio (if we have enough data)
            if len(self.total_reward_buffer) >= 30:  # Need at least 30 episodes
                returns = np.array(list(self.total_reward_buffer)[-30:])
                if np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    self.sharpe_ratio_buffer.append(sharpe_ratio)
            
            # Calculate max drawdown
            if len(self.portfolio_value_buffer) >= 10:
                portfolio_values = np.array(list(self.portfolio_value_buffer)[-100:])  # Last 100 values
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = np.max(drawdown) * 100  # Convert to percentage
                self.max_drawdown_buffer.append(max_drawdown)
                
        except Exception as e:
            self._logger.debug(f"Error extracting episode metrics: {e}")

    def _log_custom_scalars(self) -> None:
        """Log custom scalars to TensorBoard."""
        if self.tb_writer is None:
            return
        
        try:
            # Log volatility penalty
            if self.vol_penalty_buffer:
                vol_penalty_mean = np.mean(list(self.vol_penalty_buffer))
                self.tb_writer.add_scalar('monitoring/vol_penalty', vol_penalty_mean, self.step_count)
            
            # Log drawdown percentage
            if self.drawdown_buffer:
                drawdown_mean = np.mean(list(self.drawdown_buffer))
                drawdown_max = np.max(list(self.drawdown_buffer))
                self.tb_writer.add_scalar('monitoring/drawdown_pct_mean', drawdown_mean, self.step_count)
                self.tb_writer.add_scalar('monitoring/drawdown_pct_max', drawdown_max, self.step_count)
            
            # Log Q-value variance
            if self.q_variance_buffer:
                q_variance_mean = np.mean(list(self.q_variance_buffer))
                q_variance_max = np.max(list(self.q_variance_buffer))
                self.tb_writer.add_scalar('monitoring/q_variance_mean', q_variance_mean, self.step_count)
                self.tb_writer.add_scalar('monitoring/q_variance_max', q_variance_max, self.step_count)
            
            # Log Lagrangian multiplier
            if self.lagrangian_enabled and self.lambda_buffer:
                lambda_mean = np.mean(list(self.lambda_buffer))
                lambda_current = list(self.lambda_buffer)[-1] if self.lambda_buffer else 0.0
                self.tb_writer.add_scalar('monitoring/lambda_lagrangian_mean', lambda_mean, self.step_count)
                self.tb_writer.add_scalar('monitoring/lambda_lagrangian_current', lambda_current, self.step_count)
            
            # Log reward magnitude
            if self.reward_buffer:
                reward_mean = np.mean(list(self.reward_buffer))
                reward_std = np.std(list(self.reward_buffer))
                self.tb_writer.add_scalar('monitoring/reward_magnitude_mean', reward_mean, self.step_count)
                self.tb_writer.add_scalar('monitoring/reward_magnitude_std', reward_std, self.step_count)
            
            # Log turnover penalty metrics
            if self.turnover_penalty_buffer:
                turnover_penalty_mean = np.mean(list(self.turnover_penalty_buffer))
                turnover_penalty_current = list(self.turnover_penalty_buffer)[-1]
                self.tb_writer.add_scalar('turnover/penalty_mean', turnover_penalty_mean, self.step_count)
                self.tb_writer.add_scalar('turnover/penalty_current', turnover_penalty_current, self.step_count)
            
            if self.turnover_normalized_buffer:
                normalized_mean = np.mean(list(self.turnover_normalized_buffer))
                normalized_current = list(self.turnover_normalized_buffer)[-1]
                self.tb_writer.add_scalar('turnover/normalized_mean', normalized_mean, self.step_count)
                self.tb_writer.add_scalar('turnover/normalized_current', normalized_current, self.step_count)
            
            if self.turnover_absolute_buffer:
                absolute_mean = np.mean(list(self.turnover_absolute_buffer))
                absolute_current = list(self.turnover_absolute_buffer)[-1]
                self.tb_writer.add_scalar('turnover/absolute_mean', absolute_mean, self.step_count)
                self.tb_writer.add_scalar('turnover/absolute_current', absolute_current, self.step_count)
            
            if self.turnover_target_buffer:
                target_current = list(self.turnover_target_buffer)[-1]
                self.tb_writer.add_scalar('turnover/target', target_current, self.step_count)
            
            if self.turnover_excess_buffer:
                excess_mean = np.mean(list(self.turnover_excess_buffer))
                excess_current = list(self.turnover_excess_buffer)[-1]
                self.tb_writer.add_scalar('turnover/excess_mean', excess_mean, self.step_count)
                self.tb_writer.add_scalar('turnover/excess_current', excess_current, self.step_count)
            
            # Log performance metrics
            if self.total_reward_buffer:
                total_reward_mean = np.mean(list(self.total_reward_buffer))
                total_reward_current = list(self.total_reward_buffer)[-1] if self.total_reward_buffer else 0.0
                self.tb_writer.add_scalar('performance/total_reward_mean', total_reward_mean, self.step_count)
                self.tb_writer.add_scalar('performance/total_reward_current', total_reward_current, self.step_count)
            
            if self.win_rate_buffer:
                win_rate = np.mean(list(self.win_rate_buffer)) * 100  # Convert to percentage
                self.tb_writer.add_scalar('performance/win_rate', win_rate, self.step_count)
            
            if self.sharpe_ratio_buffer:
                sharpe_current = list(self.sharpe_ratio_buffer)[-1]
                sharpe_mean = np.mean(list(self.sharpe_ratio_buffer))
                self.tb_writer.add_scalar('performance/sharpe_ratio_current', sharpe_current, self.step_count)
                self.tb_writer.add_scalar('performance/sharpe_ratio_mean', sharpe_mean, self.step_count)
            
            if self.max_drawdown_buffer:
                max_dd_current = list(self.max_drawdown_buffer)[-1]
                max_dd_worst = np.max(list(self.max_drawdown_buffer))
                self.tb_writer.add_scalar('performance/max_drawdown_current', max_dd_current, self.step_count)
                self.tb_writer.add_scalar('performance/max_drawdown_worst', max_dd_worst, self.step_count)
            
            if self.portfolio_value_buffer:
                portfolio_current = list(self.portfolio_value_buffer)[-1]
                portfolio_mean = np.mean(list(self.portfolio_value_buffer))
                self.tb_writer.add_scalar('performance/portfolio_value_current', portfolio_current, self.step_count)
                self.tb_writer.add_scalar('performance/portfolio_value_mean', portfolio_mean, self.step_count)
            
            # Log buffer statistics
            self.tb_writer.add_scalar('monitoring/vol_penalty_buffer_size', len(self.vol_penalty_buffer), self.step_count)
            self.tb_writer.add_scalar('monitoring/drawdown_buffer_size', len(self.drawdown_buffer), self.step_count)
            self.tb_writer.add_scalar('monitoring/turnover_penalty_buffer_size', len(self.turnover_penalty_buffer), self.step_count)
            
            # Flush writer
            self.tb_writer.flush()
            
        except Exception as e:
            self._logger.debug(f"Error logging custom scalars: {e}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }
        
        if self.vol_penalty_buffer:
            stats['vol_penalty'] = {
                'mean': np.mean(list(self.vol_penalty_buffer)),
                'std': np.std(list(self.vol_penalty_buffer)),
                'max': np.max(list(self.vol_penalty_buffer))
            }
        
        if self.drawdown_buffer:
            stats['drawdown_pct'] = {
                'mean': np.mean(list(self.drawdown_buffer)),
                'std': np.std(list(self.drawdown_buffer)),
                'max': np.max(list(self.drawdown_buffer))
            }
        
        if self.q_variance_buffer:
            stats['q_variance'] = {
                'mean': np.mean(list(self.q_variance_buffer)),
                'std': np.std(list(self.q_variance_buffer)),
                'max': np.max(list(self.q_variance_buffer))
            }
        
        if self.lagrangian_enabled and self.lambda_buffer:
            stats['lambda_lagrangian'] = {
                'mean': np.mean(list(self.lambda_buffer)),
                'std': np.std(list(self.lambda_buffer)),
                'current': list(self.lambda_buffer)[-1] if self.lambda_buffer else 0.0
            }
        
        if self.reward_buffer:
            stats['reward_magnitude'] = {
                'mean': np.mean(list(self.reward_buffer)),
                'std': np.std(list(self.reward_buffer)),
                'max': np.max(list(self.reward_buffer))
            }
        
        # Turnover penalty statistics
        if self.turnover_penalty_buffer:
            stats['turnover_penalty'] = {
                'mean': np.mean(list(self.turnover_penalty_buffer)),
                'std': np.std(list(self.turnover_penalty_buffer)),
                'current': list(self.turnover_penalty_buffer)[-1]
            }
        
        if self.turnover_normalized_buffer:
            stats['turnover_normalized'] = {
                'mean': np.mean(list(self.turnover_normalized_buffer)),
                'std': np.std(list(self.turnover_normalized_buffer)),
                'current': list(self.turnover_normalized_buffer)[-1]
            }
        
        if self.turnover_excess_buffer:
            stats['turnover_excess'] = {
                'mean': np.mean(list(self.turnover_excess_buffer)),
                'std': np.std(list(self.turnover_excess_buffer)),
                'current': list(self.turnover_excess_buffer)[-1]
            }
        
        # Performance statistics
        if self.win_rate_buffer:
            stats['win_rate'] = np.mean(list(self.win_rate_buffer)) * 100
        
        if self.sharpe_ratio_buffer:
            stats['sharpe_ratio'] = {
                'mean': np.mean(list(self.sharpe_ratio_buffer)),
                'current': list(self.sharpe_ratio_buffer)[-1]
            }
        
        if self.max_drawdown_buffer:
            stats['max_drawdown'] = {
                'current': list(self.max_drawdown_buffer)[-1],
                'worst': np.max(list(self.max_drawdown_buffer))
            }
        
        if self.portfolio_value_buffer:
            stats['portfolio_value'] = {
                'current': list(self.portfolio_value_buffer)[-1],
                'mean': np.mean(list(self.portfolio_value_buffer))
            }
        
        return stats


class ReplayBufferAuditCallback(BaseCallback):
    """
    Replay buffer audit callback for monitoring reward magnitude.
    
    Every 50k steps, samples 1k transitions and reports mean |reward|.
    Helps detect if reward shaping is too weak (rewards near 0).
    """

    def __init__(
        self,
        audit_frequency: int = 50000,
        sample_size: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.audit_frequency = audit_frequency
        self.sample_size = sample_size
        self.step_count = 0
        self.last_audit_step = 0
        
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Replay buffer audit callback initialized (every {audit_frequency} steps)")

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1
        
        # Perform audit at specified frequency
        if self.step_count - self.last_audit_step >= self.audit_frequency:
            self._audit_replay_buffer()
            self.last_audit_step = self.step_count
        
        return True

    def _audit_replay_buffer(self) -> None:
        """Audit replay buffer for reward magnitude."""
        try:
            # Access replay buffer from model
            if not hasattr(self.model, 'replay_buffer'):
                self._logger.warning("Model has no replay_buffer attribute - audit skipped")
                return
            
            replay_buffer = self.model.replay_buffer
            
            # Check if buffer has enough samples
            if replay_buffer.size() < self.sample_size:
                self._logger.info(f"Replay buffer too small ({replay_buffer.size()}) - audit skipped")
                return
            
            # Sample transitions
            sample_indices = np.random.choice(replay_buffer.size(), self.sample_size, replace=False)
            
            # Extract rewards
            rewards = []
            for idx in sample_indices:
                # Different replay buffer implementations have different access patterns
                if hasattr(replay_buffer, 'rewards'):
                    reward = replay_buffer.rewards[idx]
                elif hasattr(replay_buffer, 'buffer') and hasattr(replay_buffer.buffer, 'rewards'):
                    reward = replay_buffer.buffer.rewards[idx]
                else:
                    # Try to sample a batch and extract rewards
                    try:
                        batch = replay_buffer.sample(1)
                        if hasattr(batch, 'rewards'):
                            reward = batch.rewards[0]
                        else:
                            continue
                    except:
                        continue
                
                rewards.append(float(reward))
            
            if not rewards:
                self._logger.warning("Could not extract rewards from replay buffer")
                return
            
            # Calculate statistics
            rewards = np.array(rewards)
            mean_abs_reward = np.mean(np.abs(rewards))
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # Log audit results
            self._logger.info(f"🔍 REPLAY BUFFER AUDIT (Step {self.step_count})")
            self._logger.info(f"   Sample size: {len(rewards)}")
            self._logger.info(f"   Mean |reward|: {mean_abs_reward:.6f}")
            self._logger.info(f"   Mean reward: {mean_reward:.6f}")
            self._logger.info(f"   Std reward: {std_reward:.6f}")
            
            # Warning for weak reward shaping
            if mean_abs_reward < 0.001:
                self._logger.warning("⚠️  WEAK REWARD SHAPING DETECTED!")
                self._logger.warning(f"   Mean |reward| = {mean_abs_reward:.6f} is very low")
                self._logger.warning("   Consider increasing reward scaling factors")
            elif mean_abs_reward < 0.01:
                self._logger.info("ℹ️  Reward magnitude is low - monitor for training effectiveness")
            else:
                self._logger.info("✅ Reward magnitude appears healthy")
            
            # Log to TensorBoard if available
            if hasattr(self, '_logger') and hasattr(self._logger, 'record'):
                self._logger.record('replay_audit/mean_abs_reward', mean_abs_reward)
                self._logger.record('replay_audit/mean_reward', mean_reward)
                self._logger.record('replay_audit/std_reward', std_reward)
                self._logger.record('replay_audit/sample_size', len(rewards))
            
        except Exception as e:
            self._logger.error(f"Error during replay buffer audit: {e}")
            import traceback
            traceback.print_exc()


def create_monitoring_callbacks(config: Dict[str, Any]) -> List[BaseCallback]:
    """
    Create monitoring and debugging callbacks.
    
    Args:
        config: Training configuration
        
    Returns:
        List of monitoring callbacks
    """
    callbacks = []
    
    # TensorBoard monitoring callback
    tb_callback = TensorBoardMonitoringCallback(
        config=config,
        log_frequency=config.get('monitoring', {}).get('tensorboard_frequency', 100),
        verbose=1
    )
    callbacks.append(tb_callback)
    
    # Replay buffer audit callback
    audit_callback = ReplayBufferAuditCallback(
        audit_frequency=config.get('monitoring', {}).get('audit_frequency', 50000),
        sample_size=config.get('monitoring', {}).get('audit_sample_size', 1000),
        verbose=1
    )
    callbacks.append(audit_callback)
    
    return callbacks


# Export public interface
__all__ = [
    'TensorBoardMonitoringCallback',
    'ReplayBufferAuditCallback', 
    'create_monitoring_callbacks'
]