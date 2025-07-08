# src/training/callbacks/enhanced_risk_callback.py
"""
Enhanced Risk-Aware Callback with λ-weighted Multi-Risk Early Stopping.

This callback extends the basic early-stop functionality to use a λ-weighted sum
of all risk metrics (drawdown, ulcer index, market impact, feed staleness) to
prevent the DQN from learning to trade illiquid names or take excessive risks.

Key Features:
- Multi-risk metric evaluation (not just drawdown)
- λ-weighted risk scoring for comprehensive risk assessment
- Liquidity-aware penalties to discourage illiquid trading
- Adaptive thresholds based on market conditions
- Risk metric correlation analysis
- Early stopping with risk decomposition logging

Usage:
    callback = EnhancedRiskCallback(
        risk_advisor=risk_advisor,
        risk_weights={
            'drawdown_pct': 0.3,
            'ulcer_index': 0.25,
            'kyle_lambda': 0.25,  # Market impact
            'feed_staleness': 0.2
        },
        early_stop_threshold=0.75,
        liquidity_penalty_multiplier=2.0
    )
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

from stable_baselines3.common.callbacks import BaseCallback

# Import risk advisor interface
try:
    from ..interfaces.risk_advisor import RiskAdvisor
except ImportError:
    from src.training.interfaces.risk_advisor import RiskAdvisor


@dataclass
class RiskMetricHistory:
    """Track history of individual risk metrics."""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    violations: int = 0
    last_violation_time: Optional[datetime] = None
    
    def add_value(self, value: float, timestamp: datetime = None):
        """Add new value to history."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_recent_stats(self, lookback_minutes: int = 10) -> Dict[str, float]:
        """Get statistics for recent values."""
        if not self.values:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'trend': 0.0}
        
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_values = [
            val for val, ts in zip(self.values, self.timestamps)
            if ts >= cutoff_time
        ]
        
        if not recent_values:
            recent_values = list(self.values)[-10:]  # Last 10 values as fallback
        
        recent_array = np.array(recent_values)
        
        # Calculate trend (slope of linear regression)
        if len(recent_array) > 1:
            x = np.arange(len(recent_array))
            trend = np.polyfit(x, recent_array, 1)[0]
        else:
            trend = 0.0
        
        return {
            'mean': float(np.mean(recent_array)),
            'std': float(np.std(recent_array)),
            'max': float(np.max(recent_array)),
            'trend': trend
        }


@dataclass
class RiskWeightConfig:
    """Configuration for risk metric weights and thresholds."""
    # Primary risk weights (must sum to 1.0)
    drawdown_weight: float = 0.30
    ulcer_weight: float = 0.25
    market_impact_weight: float = 0.25  # Kyle lambda
    feed_staleness_weight: float = 0.20
    
    # Individual thresholds for early warning
    drawdown_threshold: float = 0.15  # 15% drawdown
    ulcer_threshold: float = 0.10     # 10% ulcer index
    market_impact_threshold: float = 0.05  # 5% market impact
    feed_staleness_threshold: float = 1000.0  # 1000ms staleness
    
    # Liquidity penalties
    liquidity_penalty_multiplier: float = 2.0
    illiquid_threshold: float = 0.02  # 2% market impact = illiquid
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    adaptation_lookback_hours: int = 2
    adaptation_sensitivity: float = 0.1
    
    def validate(self):
        """Validate weight configuration."""
        total_weight = (
            self.drawdown_weight + self.ulcer_weight + 
            self.market_impact_weight + self.feed_staleness_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Risk weights must sum to 1.0, got {total_weight}")
        
        if any(w < 0 for w in [self.drawdown_weight, self.ulcer_weight, 
                              self.market_impact_weight, self.feed_staleness_weight]):
            raise ValueError("All risk weights must be non-negative")


class EnhancedRiskCallback(BaseCallback):
    """
    Enhanced risk-aware callback with λ-weighted multi-risk early stopping.
    
    This callback evaluates multiple risk metrics and combines them using
    configurable weights to make early stopping decisions. It specifically
    targets preventing the DQN from learning to trade illiquid names by
    heavily penalizing high market impact scenarios.
    """
    
    def __init__(
        self,
        risk_advisor: RiskAdvisor,
        risk_weights: Optional[Dict[str, float]] = None,
        early_stop_threshold: float = 0.75,
        liquidity_penalty_multiplier: float = 2.0,
        consecutive_violations_limit: int = 5,
        evaluation_frequency: int = 100,  # Steps between evaluations
        log_frequency: int = 1000,
        enable_risk_decomposition: bool = True,
        verbose: int = 1
    ):
        """
        Initialize enhanced risk callback.
        
        Args:
            risk_advisor: RiskAdvisor instance for risk evaluation
            risk_weights: Dictionary of risk metric weights
            early_stop_threshold: Composite risk score threshold for early stopping
            liquidity_penalty_multiplier: Multiplier for illiquid trading penalties
            consecutive_violations_limit: Number of consecutive violations before stopping
            evaluation_frequency: Steps between risk evaluations
            log_frequency: Steps between detailed logging
            enable_risk_decomposition: Whether to log detailed risk breakdowns
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.risk_advisor = risk_advisor
        self.early_stop_threshold = early_stop_threshold
        self.liquidity_penalty_multiplier = liquidity_penalty_multiplier
        self.consecutive_violations_limit = consecutive_violations_limit
        self.evaluation_frequency = evaluation_frequency
        self.log_frequency = log_frequency
        self.enable_risk_decomposition = enable_risk_decomposition
        
        # Setup risk weights
        if risk_weights is None:
            risk_weights = {
                'drawdown_pct': 0.30,
                'ulcer_index': 0.25,
                'kyle_lambda': 0.25,
                'feed_staleness': 0.20
            }
        
        self.weight_config = RiskWeightConfig(
            drawdown_weight=risk_weights.get('drawdown_pct', 0.30),
            ulcer_weight=risk_weights.get('ulcer_index', 0.25),
            market_impact_weight=risk_weights.get('kyle_lambda', 0.25),
            feed_staleness_weight=risk_weights.get('feed_staleness', 0.20),
            liquidity_penalty_multiplier=liquidity_penalty_multiplier
        )
        self.weight_config.validate()
        
        # Risk tracking
        self.risk_history: Dict[str, RiskMetricHistory] = {
            'drawdown_pct': RiskMetricHistory(),
            'ulcer_index': RiskMetricHistory(),
            'kyle_lambda': RiskMetricHistory(),
            'feed_staleness': RiskMetricHistory(),
            'composite_risk': RiskMetricHistory()
        }
        
        # Violation tracking
        self.consecutive_violations = 0
        self.total_violations = 0
        self.violation_episodes = []
        
        # Performance tracking
        self.total_evaluations = 0
        self.evaluation_times = deque(maxlen=100)
        self.last_evaluation_step = 0
        
        # Liquidity tracking
        self.illiquid_trade_count = 0
        self.total_trade_count = 0
        self.illiquid_episodes = []
        
        self._logger = logging.getLogger("EnhancedRiskCallback")
        self._logger.info(f"Enhanced risk callback initialized with weights: {risk_weights}")
        self._logger.info(f"Early stop threshold: {early_stop_threshold}")
        self._logger.info(f"Liquidity penalty multiplier: {liquidity_penalty_multiplier}")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Only evaluate at specified frequency to maintain performance
        if (self.num_timesteps - self.last_evaluation_step) < self.evaluation_frequency:
            return True
        
        self.last_evaluation_step = self.num_timesteps
        
        # Get current observation
        obs = self._get_current_observation()
        if obs is None:
            return True
        
        try:
            # Evaluate risk metrics
            start_time = datetime.now()
            risk_metrics = self.risk_advisor.evaluate(obs)
            evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.evaluation_times.append(evaluation_time)
            self.total_evaluations += 1
            
            # Calculate composite risk score
            composite_risk = self._calculate_composite_risk_score(risk_metrics)
            
            # Update risk history
            self._update_risk_history(risk_metrics, composite_risk)
            
            # Check for liquidity violations
            self._check_liquidity_violations(risk_metrics)
            
            # Evaluate early stopping condition
            should_stop = self._evaluate_early_stopping(composite_risk, risk_metrics)
            
            # Log detailed information periodically
            if self.num_timesteps % self.log_frequency == 0:
                self._log_detailed_risk_status(risk_metrics, composite_risk)
            
            return not should_stop
            
        except Exception as e:
            self._logger.error(f"Risk evaluation failed at step {self.num_timesteps}: {e}")
            # Continue training on evaluation errors, but log the issue
            return True
    
    def _get_current_observation(self) -> Optional[Dict[str, Any]]:
        """Extract current observation from training environment."""
        obs = None
        
        try:
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
                env = self.training_env
                
                # If it's a Monitor wrapper, get the underlying environment
                if hasattr(env, 'env'):
                    env = env.env
                
                # Try different attribute names
                for attr_name in ['last_raw_obs', '_last_obs', 'last_obs']:
                    if hasattr(env, attr_name):
                        obs = getattr(env, attr_name)
                        if obs is not None:
                            break
            
            # Convert to expected format
            if obs is not None:
                return self._convert_obs_to_dict(obs)
            
        except Exception as e:
            self._logger.debug(f"Failed to get observation: {e}")
        
        return None
    
    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        # Enhanced conversion with more market data context
        current_time = datetime.now()
        
        # Extract components from observation (adjust based on your environment)
        if len(obs) > 10:
            # Assume structured observation with market features
            market_features = obs[:-2]  # All but last 2 elements
            position = obs[-2] if len(obs) > 1 else 0.0
            portfolio_value = obs[-1] if len(obs) > 0 else 100000.0
        else:
            # Simple observation
            market_features = obs[:-1] if len(obs) > 1 else obs
            position = obs[-1] if len(obs) > 1 else 0.0
            portfolio_value = 100000.0  # Default starting value
        
        # Create comprehensive observation dictionary
        obs_dict = {
            "market_features": market_features,
            "position": position,
            "portfolio_value": portfolio_value,
            "timestamp": current_time,
            
            # Additional context for risk evaluation
            "portfolio_values": [portfolio_value],  # For drawdown calculation
            "price_changes": market_features[:5] if len(market_features) >= 5 else [0.0] * 5,
            "order_flows": [position] * 5,  # Simplified order flow
            "feed_timestamps": {
                "primary": current_time,
                "secondary": current_time - timedelta(milliseconds=10)
            },
            "current_time": current_time,
            
            # Trading context
            "trade_size": abs(position) if position != 0 else 0.0,
            "is_trading": abs(position) > 0.01,
            "market_session": self._get_market_session(current_time)
        }
        
        return obs_dict
    
    def _get_market_session(self, timestamp: datetime) -> str:
        """Determine market session for risk context."""
        hour = timestamp.hour
        
        if 9 <= hour < 16:
            return "regular"
        elif 4 <= hour < 9:
            return "pre_market"
        elif 16 <= hour < 20:
            return "after_hours"
        else:
            return "overnight"
    
    def _calculate_composite_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """
        Calculate λ-weighted composite risk score from individual metrics.
        
        Args:
            risk_metrics: Dictionary of individual risk metrics
            
        Returns:
            Composite risk score (0-1, higher = more risky)
        """
        # Extract and normalize individual metrics
        drawdown = min(risk_metrics.get('drawdown_pct', 0.0), 1.0)
        ulcer = min(risk_metrics.get('ulcer_index', 0.0), 1.0)
        market_impact = min(risk_metrics.get('kyle_lambda', 0.0), 1.0)
        feed_staleness = min(risk_metrics.get('feed_staleness_ms', 0.0) / 5000.0, 1.0)  # Normalize to 5s max
        
        # Apply liquidity penalty multiplier for high market impact
        if market_impact > self.weight_config.illiquid_threshold:
            market_impact *= self.weight_config.liquidity_penalty_multiplier
            market_impact = min(market_impact, 1.0)  # Cap at 1.0
        
        # Calculate weighted composite score
        composite_score = (
            self.weight_config.drawdown_weight * drawdown +
            self.weight_config.ulcer_weight * ulcer +
            self.weight_config.market_impact_weight * market_impact +
            self.weight_config.feed_staleness_weight * feed_staleness
        )
        
        # Apply adaptive adjustments if enabled
        if self.weight_config.enable_adaptive_thresholds:
            composite_score = self._apply_adaptive_adjustments(composite_score, risk_metrics)
        
        return min(composite_score, 1.0)  # Ensure score stays within [0, 1]
    
    def _apply_adaptive_adjustments(self, base_score: float, risk_metrics: Dict[str, float]) -> float:
        """Apply adaptive threshold adjustments based on recent risk history."""
        try:
            # Get recent statistics for each metric
            recent_stats = {}
            for metric_name in ['drawdown_pct', 'ulcer_index', 'kyle_lambda', 'feed_staleness']:
                if metric_name in self.risk_history:
                    recent_stats[metric_name] = self.risk_history[metric_name].get_recent_stats(
                        lookback_minutes=self.weight_config.adaptation_lookback_hours * 60
                    )
            
            # Calculate adaptation factor based on recent trends
            adaptation_factor = 1.0
            
            for metric_name, stats in recent_stats.items():
                if stats['trend'] > 0:  # Increasing risk trend
                    adaptation_factor += self.weight_config.adaptation_sensitivity * stats['trend']
                elif stats['trend'] < 0:  # Decreasing risk trend
                    adaptation_factor -= self.weight_config.adaptation_sensitivity * abs(stats['trend'])
            
            # Apply adaptation with bounds
            adaptation_factor = max(0.5, min(2.0, adaptation_factor))
            adapted_score = base_score * adaptation_factor
            
            return adapted_score
            
        except Exception as e:
            self._logger.debug(f"Adaptive adjustment failed: {e}")
            return base_score
    
    def _update_risk_history(self, risk_metrics: Dict[str, float], composite_risk: float):
        """Update risk metric history."""
        current_time = datetime.now()
        
        # Update individual metric histories
        for metric_name in ['drawdown_pct', 'ulcer_index', 'kyle_lambda', 'feed_staleness']:
            if metric_name in risk_metrics and metric_name in self.risk_history:
                value = risk_metrics[metric_name]
                self.risk_history[metric_name].add_value(value, current_time)
                
                # Check for individual metric violations
                threshold_map = {
                    'drawdown_pct': self.weight_config.drawdown_threshold,
                    'ulcer_index': self.weight_config.ulcer_threshold,
                    'kyle_lambda': self.weight_config.market_impact_threshold,
                    'feed_staleness': self.weight_config.feed_staleness_threshold
                }
                threshold = threshold_map.get(metric_name, 1.0)
                if value > threshold:
                    self.risk_history[metric_name].violations += 1
                    self.risk_history[metric_name].last_violation_time = current_time
        
        # Update composite risk history
        self.risk_history['composite_risk'].add_value(composite_risk, current_time)
    
    def _check_liquidity_violations(self, risk_metrics: Dict[str, float]):
        """Check for liquidity violations and track illiquid trading."""
        market_impact = risk_metrics.get('kyle_lambda', 0.0)
        
        # Count total trades
        obs_dict = self._get_current_observation()
        if obs_dict and obs_dict.get('is_trading', False):
            self.total_trade_count += 1
            
            # Check if this is an illiquid trade
            if market_impact > self.weight_config.illiquid_threshold:
                self.illiquid_trade_count += 1
                
                # Log illiquid episode
                illiquid_episode = {
                    'step': self.num_timesteps,
                    'market_impact': market_impact,
                    'position_size': obs_dict.get('position', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
                self.illiquid_episodes.append(illiquid_episode)
                
                # Log warning
                if self.verbose >= 1:
                    self._logger.warning(
                        f"Illiquid trade detected at step {self.num_timesteps}: "
                        f"market_impact={market_impact:.4f} > threshold={self.weight_config.illiquid_threshold:.4f}"
                    )
    
    def _evaluate_early_stopping(self, composite_risk: float, risk_metrics: Dict[str, float]) -> bool:
        """
        Evaluate whether to trigger early stopping based on composite risk score.
        
        Args:
            composite_risk: Composite risk score
            risk_metrics: Individual risk metrics
            
        Returns:
            True if training should be stopped
        """
        # Check composite risk threshold
        if composite_risk > self.early_stop_threshold:
            self.consecutive_violations += 1
            self.total_violations += 1
            
            # Log violation details
            violation_info = {
                'step': self.num_timesteps,
                'composite_risk': composite_risk,
                'threshold': self.early_stop_threshold,
                'consecutive_violations': self.consecutive_violations,
                'risk_breakdown': {
                    'drawdown': risk_metrics.get('drawdown_pct', 0.0),
                    'ulcer': risk_metrics.get('ulcer_index', 0.0),
                    'market_impact': risk_metrics.get('kyle_lambda', 0.0),
                    'feed_staleness': risk_metrics.get('feed_staleness_ms', 0.0)
                },
                'timestamp': datetime.now().isoformat()
            }
            self.violation_episodes.append(violation_info)
            
            if self.verbose >= 1:
                self._logger.warning(
                    f"Risk violation {self.consecutive_violations}/{self.consecutive_violations_limit} "
                    f"at step {self.num_timesteps}: composite_risk={composite_risk:.4f} > "
                    f"threshold={self.early_stop_threshold:.4f}"
                )
                
                if self.enable_risk_decomposition:
                    self._log_risk_decomposition(risk_metrics, composite_risk)
            
            # Check if we should stop
            if self.consecutive_violations >= self.consecutive_violations_limit:
                self._logger.critical(
                    f"Early stopping triggered after {self.consecutive_violations} consecutive "
                    f"risk violations. Final composite risk: {composite_risk:.4f}"
                )
                return True
                
        else:
            # Reset consecutive violations on good behavior
            if self.consecutive_violations > 0:
                self._logger.info(
                    f"Risk violation streak ended at step {self.num_timesteps}. "
                    f"Composite risk: {composite_risk:.4f} <= threshold: {self.early_stop_threshold:.4f}"
                )
            self.consecutive_violations = 0
        
        return False
    
    def _log_risk_decomposition(self, risk_metrics: Dict[str, float], composite_risk: float):
        """Log detailed risk decomposition for analysis."""
        drawdown = risk_metrics.get('drawdown_pct', 0.0)
        ulcer = risk_metrics.get('ulcer_index', 0.0)
        market_impact = risk_metrics.get('kyle_lambda', 0.0)
        feed_staleness = risk_metrics.get('feed_staleness_ms', 0.0)
        
        # Calculate individual contributions
        drawdown_contrib = self.weight_config.drawdown_weight * drawdown
        ulcer_contrib = self.weight_config.ulcer_weight * ulcer
        impact_contrib = self.weight_config.market_impact_weight * market_impact
        staleness_contrib = self.weight_config.feed_staleness_weight * (feed_staleness / 5000.0)
        
        self._logger.info("Risk Decomposition:")
        self._logger.info(f"  Drawdown: {drawdown:.4f} × {self.weight_config.drawdown_weight:.2f} = {drawdown_contrib:.4f}")
        self._logger.info(f"  Ulcer Index: {ulcer:.4f} × {self.weight_config.ulcer_weight:.2f} = {ulcer_contrib:.4f}")
        self._logger.info(f"  Market Impact: {market_impact:.4f} × {self.weight_config.market_impact_weight:.2f} = {impact_contrib:.4f}")
        self._logger.info(f"  Feed Staleness: {feed_staleness:.1f}ms × {self.weight_config.feed_staleness_weight:.2f} = {staleness_contrib:.4f}")
        self._logger.info(f"  Composite Risk: {composite_risk:.4f}")
        
        # Highlight the dominant risk factor
        contributions = {
            'Drawdown': drawdown_contrib,
            'Ulcer Index': ulcer_contrib,
            'Market Impact': impact_contrib,
            'Feed Staleness': staleness_contrib
        }
        
        dominant_risk = max(contributions.items(), key=lambda x: x[1])
        self._logger.info(f"  Dominant Risk Factor: {dominant_risk[0]} ({dominant_risk[1]:.4f})")
    
    def _log_detailed_risk_status(self, risk_metrics: Dict[str, float], composite_risk: float):
        """Log detailed risk status periodically."""
        # Calculate illiquid trading rate
        illiquid_rate = (
            self.illiquid_trade_count / max(1, self.total_trade_count)
            if self.total_trade_count > 0 else 0.0
        )
        
        # Calculate average evaluation time
        avg_eval_time = np.mean(self.evaluation_times) if self.evaluation_times else 0.0
        
        self._logger.info(f"Risk Status at step {self.num_timesteps}:")
        self._logger.info(f"  Composite Risk: {composite_risk:.4f} / {self.early_stop_threshold:.4f}")
        self._logger.info(f"  Total Violations: {self.total_violations}")
        self._logger.info(f"  Consecutive Violations: {self.consecutive_violations}")
        self._logger.info(f"  Illiquid Trading Rate: {illiquid_rate:.2%} ({self.illiquid_trade_count}/{self.total_trade_count})")
        self._logger.info(f"  Avg Evaluation Time: {avg_eval_time:.2f}ms")
        
        # Log recent trends
        for metric_name in ['drawdown_pct', 'ulcer_index', 'kyle_lambda']:
            if metric_name in self.risk_history:
                stats = self.risk_history[metric_name].get_recent_stats()
                trend_direction = "↗" if stats['trend'] > 0.001 else "↘" if stats['trend'] < -0.001 else "→"
                self._logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} {trend_direction}")
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Log rollout summary
        if self.verbose >= 1:
            composite_history = self.risk_history['composite_risk']
            if composite_history.values:
                recent_risk = list(composite_history.values)[-1]
                avg_risk = np.mean(list(composite_history.values)[-100:])  # Last 100 evaluations
                
                self._logger.info(
                    f"Rollout ended - Recent risk: {recent_risk:.4f}, "
                    f"Avg risk (last 100): {avg_risk:.4f}, "
                    f"Total violations: {self.total_violations}"
                )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary for analysis."""
        # Calculate statistics
        illiquid_rate = (
            self.illiquid_trade_count / max(1, self.total_trade_count)
            if self.total_trade_count > 0 else 0.0
        )
        
        avg_eval_time = np.mean(self.evaluation_times) if self.evaluation_times else 0.0
        
        # Get recent risk statistics
        risk_stats = {}
        for metric_name, history in self.risk_history.items():
            if history.values:
                values = list(history.values)
                risk_stats[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'violations': history.violations
                }
        
        return {
            'total_evaluations': self.total_evaluations,
            'total_violations': self.total_violations,
            'consecutive_violations': self.consecutive_violations,
            'illiquid_trade_count': self.illiquid_trade_count,
            'total_trade_count': self.total_trade_count,
            'illiquid_trading_rate': illiquid_rate,
            'avg_evaluation_time_ms': avg_eval_time,
            'risk_statistics': risk_stats,
            'weight_config': {
                'drawdown_weight': self.weight_config.drawdown_weight,
                'ulcer_weight': self.weight_config.ulcer_weight,
                'market_impact_weight': self.weight_config.market_impact_weight,
                'feed_staleness_weight': self.weight_config.feed_staleness_weight,
                'liquidity_penalty_multiplier': self.weight_config.liquidity_penalty_multiplier
            },
            'violation_episodes': self.violation_episodes[-10:],  # Last 10 violations
            'illiquid_episodes': self.illiquid_episodes[-10:]     # Last 10 illiquid trades
        }
    
    def save_risk_analysis(self, filepath: str):
        """Save detailed risk analysis to file."""
        analysis = {
            'callback_config': {
                'early_stop_threshold': self.early_stop_threshold,
                'consecutive_violations_limit': self.consecutive_violations_limit,
                'evaluation_frequency': self.evaluation_frequency,
                'liquidity_penalty_multiplier': self.liquidity_penalty_multiplier
            },
            'risk_summary': self.get_risk_summary(),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self._logger.info(f"Risk analysis saved to {filepath}")


def create_enhanced_risk_callback(
    risk_advisor: RiskAdvisor,
    config: Optional[Dict[str, Any]] = None
) -> EnhancedRiskCallback:
    """
    Create an EnhancedRiskCallback with recommended settings.
    
    Args:
        risk_advisor: RiskAdvisor instance
        config: Optional configuration dictionary
        
    Returns:
        Configured EnhancedRiskCallback
    """
    default_config = {
        'risk_weights': {
            'drawdown_pct': 0.30,
            'ulcer_index': 0.25,
            'kyle_lambda': 0.25,
            'feed_staleness': 0.20
        },
        'early_stop_threshold': 0.75,
        'liquidity_penalty_multiplier': 2.0,
        'consecutive_violations_limit': 5,
        'evaluation_frequency': 100,
        'log_frequency': 1000,
        'enable_risk_decomposition': True,
        'verbose': 1
    }
    
    if config:
        default_config.update(config)
    
    return EnhancedRiskCallback(
        risk_advisor=risk_advisor,
        **default_config
    )


__all__ = [
    "EnhancedRiskCallback", 
    "RiskMetricHistory", 
    "RiskWeightConfig", 
    "create_enhanced_risk_callback"
]