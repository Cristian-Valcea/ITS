"""
Risk Callbacks Core Module

Contains risk-aware training callbacks extracted from TrainerAgent.
This module handles:
- Risk penalty callbacks for training
- Risk-aware training monitoring
- Reward shaping with risk constraints
- Early stopping based on risk metrics

This is an internal module - use src.training.TrainerAgent for public API.
"""

import logging
from typing import Dict, Any, Optional, Callable
import numpy as np

# Core dependencies
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

# Internal imports
try:
    from ..interfaces.risk_advisor import RiskAdvisor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from interfaces.risk_advisor import RiskAdvisor


class RiskPenaltyCallback(BaseCallback):
    """
    Callback for applying risk penalties during training.
    
    Monitors risk metrics and applies penalties to the reward signal
    to encourage risk-aware behavior.
    """

    def __init__(
        self,
        advisor: RiskAdvisor,
        lam: float = 0.1,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.advisor = advisor
        self.lam = lam  # Risk penalty weight
        self.total_penalties = 0.0
        self.penalty_count = 0
        self._logger = logging.getLogger("RiskPenaltyCallback")

    def _on_step(self) -> bool:
        """Called at each training step to apply risk penalties."""
        try:
            # Get current observation from environment
            obs = None
            
            # Try different ways to get the observation
            if hasattr(self.training_env, "get_attr"):
                # VecEnv case
                for attr_name in ['last_raw_obs', '_last_obs', 'last_obs']:
                    try:
                        obs_list = self.training_env.get_attr(attr_name)
                        if obs_list and obs_list[0] is not None:
                            obs = obs_list[0]
                            break
                    except:
                        continue
            else:
                # Single env case
                for attr_name in ['last_raw_obs', '_last_obs', 'last_obs']:
                    if hasattr(self.training_env, attr_name):
                        obs = getattr(self.training_env, attr_name)
                        if obs is not None:
                            break
            
            # If we still don't have an observation, skip this step
            if obs is None:
                return True
            
            # Convert observation to dict format for risk advisor
            obs_dict = self._convert_obs_to_dict(obs)
            
            # Evaluate risk
            risk = self.advisor.evaluate(obs_dict)
            
            # Calculate penalty based on drawdown velocity
            penalty = self.lam * risk.get('drawdown_vel', 0)
            
            if penalty > 0:
                # Track penalties for logging
                self.total_penalties += penalty
                self.penalty_count += 1
                
                if self.verbose > 1:
                    self._logger.debug(f"Risk penalty calculated: {penalty:.4f}")
                
                if self.verbose > 0 and self.penalty_count % 100 == 0:
                    avg_penalty = self.total_penalties / self.penalty_count
                    self._logger.info(f"Risk penalties applied: {self.penalty_count}, avg: {avg_penalty:.4f}")
        
        except Exception as e:
            self._logger.error(f"Risk penalty evaluation failed: {e}")
            # Continue training on errors
        
        return True
    
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        return {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }


class RiskAwareCallback(BaseCallback):
    """
    Callback for risk-aware training with early stopping and penalty injection.

    Monitors risk metrics during training and can:
    1. Apply early stopping when risk thresholds are breached
    2. Inject risk penalties into the reward signal
    3. Log risk metrics to TensorBoard
    """

    def __init__(
        self,
        risk_advisor: RiskAdvisor,
        penalty_weight: float = 0.1,
        early_stop_threshold: float = 0.8,
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.risk_advisor = risk_advisor
        self.penalty_weight = penalty_weight
        self.early_stop_threshold = early_stop_threshold
        self.log_freq = log_freq

        self.risk_violations = 0
        self.total_risk_penalty = 0.0
        self.episode_count = 0

        self._logger = logging.getLogger("RiskAwareCallback")

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Get current observation from environment
        obs = None
        
        if hasattr(self.training_env, "get_attr"):
            # VecEnv case
            for attr_name in ['last_raw_obs', '_last_obs', 'last_obs']:
                try:
                    obs_list = self.training_env.get_attr(attr_name)
                    if obs_list and obs_list[0] is not None:
                        obs = obs_list[0]
                        break
                except:
                    continue
        else:
            # Single env case
            for attr_name in ['last_raw_obs', '_last_obs', 'last_obs']:
                if hasattr(self.training_env, attr_name):
                    obs = getattr(self.training_env, attr_name)
                    if obs is not None:
                        break

        if obs is None:
            return True

        try:
            # Convert observation to dict format
            obs_dict = self._convert_obs_to_dict(obs)
            
            # Evaluate risk
            risk_metrics = self.risk_advisor.evaluate(obs_dict)
            
            # Check for risk violations
            risk_score = risk_metrics.get('overall_risk', 0.0)
            if risk_score > self.early_stop_threshold:
                self.risk_violations += 1
                self._logger.warning(f"Risk threshold exceeded: {risk_score:.3f}")
                
                # Early stopping if too many violations
                if self.risk_violations > 10:
                    self._logger.error("Too many risk violations, stopping training")
                    return False
            
            # Apply risk penalty
            penalty = self.penalty_weight * risk_score
            self.total_risk_penalty += penalty
            
            # Log metrics periodically
            if self.num_timesteps % self.log_freq == 0:
                self._log_risk_metrics(risk_metrics)
                
        except Exception as e:
            self._logger.error(f"Risk evaluation failed: {e}")
        
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        self.episode_count += 1
        
        if self.episode_count % 10 == 0:
            avg_penalty = self.total_risk_penalty / max(1, self.num_timesteps)
            self._logger.info(f"Episode {self.episode_count}: Avg risk penalty: {avg_penalty:.4f}")

    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        return {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }

    def _log_risk_metrics(self, risk_metrics: Dict[str, float]) -> None:
        """Log risk metrics to TensorBoard."""
        if self.logger is not None:
            for metric_name, metric_value in risk_metrics.items():
                self.logger.record(f"risk/{metric_name}", metric_value)


def reward_shaping_callback(
    locals_dict: Dict[str, Any],
    risk_advisor: Optional[RiskAdvisor],
    penalty_weight: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Apply reward shaping based on risk metrics.
    
    Args:
        locals_dict: Local variables from training step
        risk_advisor: Risk advisor instance
        penalty_weight: Weight for risk penalty
        logger: Optional logger instance
        
    Returns:
        Risk penalty to subtract from reward
    """
    logger = logger or logging.getLogger(__name__)
    
    if risk_advisor is None:
        return 0.0
    
    try:
        # Get observation from locals
        obs = locals_dict.get('obs', None)
        if obs is None:
            return 0.0
        
        # Convert to dict format
        obs_dict = {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }
        
        # Evaluate risk
        risk_metrics = risk_advisor.evaluate(obs_dict)
        
        # Calculate penalty
        risk_score = risk_metrics.get('overall_risk', 0.0)
        penalty = penalty_weight * risk_score
        
        return penalty
        
    except Exception as e:
        logger.error(f"Reward shaping failed: {e}")
        return 0.0


def early_stop_callback(
    eval_env: Any,
    risk_advisor: Optional[RiskAdvisor],
    threshold: float = 0.8,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Check if training should be stopped early due to risk concerns.
    
    Args:
        eval_env: Evaluation environment
        risk_advisor: Risk advisor instance
        threshold: Risk threshold for early stopping
        logger: Optional logger instance
        
    Returns:
        True if training should continue, False if it should stop
    """
    logger = logger or logging.getLogger(__name__)
    
    if risk_advisor is None:
        return True
    
    try:
        # Get current state from eval environment
        if hasattr(eval_env, 'get_attr'):
            obs_list = eval_env.get_attr('last_obs')
            if obs_list and obs_list[0] is not None:
                obs = obs_list[0]
            else:
                return True
        else:
            obs = getattr(eval_env, 'last_obs', None)
            if obs is None:
                return True
        
        # Convert to dict format
        obs_dict = {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }
        
        # Evaluate risk
        risk_metrics = risk_advisor.evaluate(obs_dict)
        risk_score = risk_metrics.get('overall_risk', 0.0)
        
        if risk_score > threshold:
            logger.warning(f"Early stopping triggered: risk score {risk_score:.3f} > {threshold}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Early stop check failed: {e}")
        return True  # Continue on error
        
    return base_reward - risk_penalty


def early_stop_callback(
    eval_env: Any,
    risk_advisor: Optional[Any],
    threshold: float = -1000.0,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Determine if training should stop early based on risk metrics.
    
    Args:
        eval_env: Evaluation environment
        risk_advisor: Risk advisor instance
        threshold: Risk threshold for early stopping
        logger: Optional logger instance
        
    Returns:
        True if training should stop, False otherwise
    """
    logger = logger or logging.getLogger(__name__)
    
    # TODO: Extract early stopping logic from trainer_agent.py
    
    # Placeholder implementation
    should_stop = False
    
    if risk_advisor is not None:
        # TODO: Check risk metrics and decide on early stopping
        pass
        
    return should_stop


class RiskPenaltyCallback:
    """
    Callback for applying risk penalties during training.
    
    This class will be populated with the actual implementation
    during the extraction phase.
    """
    
    def __init__(
        self,
        risk_advisor: Optional[Any],
        penalty_weight: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the risk penalty callback.
        
        Args:
            risk_advisor: Risk advisor instance
            penalty_weight: Weight for risk penalty
            logger: Optional logger instance
        """
        self.risk_advisor = risk_advisor
        self.penalty_weight = penalty_weight
        self.logger = logger or logging.getLogger(__name__)
        
        # TODO: Extract initialization logic from trainer_agent.py
        
    def on_step(self) -> bool:
        """
        Called at each training step.
        
        This method will be populated with the actual step logic
        during the extraction phase.
        
        Returns:
            True to continue training, False to stop
        """
        # TODO: Extract from _on_step in trainer_agent.py
        return True
        
    def calculate_risk_penalty(self, observation: np.ndarray) -> float:
        """
        Calculate risk penalty for the current observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Risk penalty value
        """
        if self.risk_advisor is None:
            return 0.0
            
        # TODO: Extract risk penalty calculation
        return 0.0


class RiskAwareCallback:
    """
    Callback for risk-aware training monitoring.
    
    This class will be populated with the actual implementation
    during the extraction phase.
    """
    
    def __init__(
        self,
        risk_advisor: Optional[Any],
        monitoring_frequency: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the risk-aware callback.
        
        Args:
            risk_advisor: Risk advisor instance
            monitoring_frequency: How often to monitor risk metrics
            logger: Optional logger instance
        """
        self.risk_advisor = risk_advisor
        self.monitoring_frequency = monitoring_frequency
        self.logger = logger or logging.getLogger(__name__)
        self.step_count = 0
        self.risk_history = []
        
        # TODO: Extract initialization logic from trainer_agent.py
        
    def on_step(self) -> bool:
        """
        Called at each training step.
        
        This method will be populated with the actual step logic
        during the extraction phase.
        
        Returns:
            True to continue training, False to stop
        """
        self.step_count += 1
        
        # TODO: Extract from _on_step in trainer_agent.py
        
        if self.step_count % self.monitoring_frequency == 0:
            self._monitor_risk_metrics()
            
        return True
        
    def on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        
        This method will be populated with the actual rollout end logic
        during the extraction phase.
        """
        # TODO: Extract from _on_rollout_end in trainer_agent.py
        pass
        
    def _monitor_risk_metrics(self) -> None:
        """Monitor and log risk metrics."""
        if self.risk_advisor is None:
            return
            
        # TODO: Extract risk monitoring logic
        risk_metrics = {}
        
        self.risk_history.append({
            'step': self.step_count,
            'metrics': risk_metrics
        })
        
        self._log_risk_metrics(risk_metrics)
        
    def _log_risk_metrics(self, risk_metrics: Dict[str, float]) -> None:
        """
        Log risk metrics.
        
        This method will be populated with the actual logging logic
        during the extraction phase.
        """
        # TODO: Extract from _log_risk_metrics in trainer_agent.py
        pass
        
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Convert observation array to dictionary format.
        
        This method will be populated with the actual conversion logic
        during the extraction phase.
        """
        # TODO: Extract from _convert_obs_to_dict in trainer_agent.py
        return {}
        
    def get_risk_history(self) -> list:
        """Get the history of risk metrics."""
        return self.risk_history.copy()
        
    def reset_risk_history(self) -> None:
        """Reset the risk metrics history."""
        self.risk_history.clear()
        self.step_count = 0


def create_risk_callbacks(
    risk_advisor: Optional[Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> list:
    """
    Create risk-related callbacks for training.
    
    Args:
        risk_advisor: Risk advisor instance
        config: Training configuration
        logger: Optional logger instance
        
    Returns:
        List of risk-related callbacks
    """
    logger = logger or logging.getLogger(__name__)
    
    callbacks = []
    
    # Risk penalty callback
    if config.get('use_risk_penalty', False):
        penalty_weight = config.get('risk_penalty_weight', 0.1)
        risk_penalty_cb = RiskPenaltyCallback(
            risk_advisor=risk_advisor,
            penalty_weight=penalty_weight,
            logger=logger
        )
        callbacks.append(risk_penalty_cb)
        
    # Risk monitoring callback
    if config.get('monitor_risk_metrics', True):
        monitoring_freq = config.get('risk_monitoring_frequency', 1000)
        risk_aware_cb = RiskAwareCallback(
            risk_advisor=risk_advisor,
            monitoring_frequency=monitoring_freq,
            logger=logger
        )
        callbacks.append(risk_aware_cb)
        
    logger.info(f"Created {len(callbacks)} risk-related callbacks")
    return callbacks