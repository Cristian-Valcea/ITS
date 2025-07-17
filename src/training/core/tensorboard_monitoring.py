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
            
        except Exception as e:
            self._logger.debug(f"Error extracting environment metrics: {e}")

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
            
            # Log buffer statistics
            self.tb_writer.add_scalar('monitoring/vol_penalty_buffer_size', len(self.vol_penalty_buffer), self.step_count)
            self.tb_writer.add_scalar('monitoring/drawdown_buffer_size', len(self.drawdown_buffer), self.step_count)
            
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