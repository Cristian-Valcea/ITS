# src/training/interfaces/risk_advisor.py
"""
Risk advisor interface for training-time risk evaluation.
Provides risk metrics without enforcement - pure advisory mode.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import logging

from ...shared.constants import MAX_RISK_EVALUATION_LATENCY_US


class RiskAdvisor(ABC):
    """
    Abstract interface for risk evaluation during training.
    
    Provides risk metrics and penalties without triggering enforcement actions.
    Used for risk-aware reward shaping and early stopping during training.
    """
    
    def __init__(self, advisor_id: str):
        self.advisor_id = advisor_id
        self.logger = logging.getLogger(f"RiskAdvisor.{advisor_id}")
    
    @abstractmethod
    def evaluate(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate risk metrics for given observation.
        
        Args:
            obs: Observation dictionary containing market data and portfolio state
            
        Returns:
            Dictionary with risk metrics:
            {
                'drawdown_pct': float,           # Current drawdown percentage
                'drawdown_velocity': float,      # Rate of drawdown change
                'var_breach_severity': float,    # VaR breach severity (0-1)
                'position_concentration': float, # Position concentration risk
                'liquidity_risk': float,         # Market liquidity risk
                'overall_risk_score': float,     # Composite risk score (0-1)
                'breach_severity': float,        # Overall breach severity (0-1)
                'penalty': float,                # Risk penalty for reward shaping
            }
            
        Performance requirement: Must complete in <50µs for training efficiency.
        """
        pass
    
    @abstractmethod
    def get_risk_config(self) -> Dict[str, Any]:
        """
        Get current risk configuration.
        
        Returns:
            Dictionary with risk thresholds and parameters
        """
        pass
    
    def validate_evaluation_latency(self, sample_obs: Dict[str, Any], 
                                  num_trials: int = 100) -> Dict[str, float]:
        """
        Validate that risk evaluation meets latency requirements.
        
        Args:
            sample_obs: Sample observation for testing
            num_trials: Number of evaluation trials to run
            
        Returns:
            Dictionary with latency statistics
        """
        import time
        
        latencies = []
        for _ in range(num_trials):
            start_time = time.perf_counter()
            self.evaluate(sample_obs)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
        
        stats = {
            'mean_latency_us': np.mean(latencies),
            'p95_latency_us': np.percentile(latencies, 95),
            'p99_latency_us': np.percentile(latencies, 99),
            'max_latency_us': np.max(latencies),
            'slo_violations': sum(1 for lat in latencies if lat > MAX_RISK_EVALUATION_LATENCY_US),
            'slo_violation_rate': sum(1 for lat in latencies if lat > MAX_RISK_EVALUATION_LATENCY_US) / len(latencies)
        }
        
        if stats['slo_violation_rate'] > 0.01:  # More than 1% violations
            self.logger.warning(
                f"RiskAdvisor {self.advisor_id} violates latency SLO: "
                f"{stats['slo_violation_rate']:.2%} evaluations > {MAX_RISK_EVALUATION_LATENCY_US}µs"
            )
        
        return stats
    
    def compute_risk_penalty(self, risk_metrics: Dict[str, float], 
                           penalty_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute risk penalty for reward shaping.
        
        Args:
            risk_metrics: Risk metrics from evaluate()
            penalty_weights: Optional custom weights for different risk components
            
        Returns:
            Risk penalty value (higher = more risky)
        """
        if penalty_weights is None:
            penalty_weights = {
                'drawdown_pct': 2.0,
                'drawdown_velocity': 1.5,
                'var_breach_severity': 3.0,
                'position_concentration': 1.0,
                'liquidity_risk': 1.0,
            }
        
        penalty = 0.0
        for metric, weight in penalty_weights.items():
            if metric in risk_metrics:
                penalty += weight * risk_metrics[metric]
        
        return penalty


class ProductionRiskAdvisor(RiskAdvisor):
    """
    Production implementation of RiskAdvisor using RiskAgentV2.
    
    Wraps RiskAgentV2 with a thin facade to provide advisory-only risk evaluation
    without triggering enforcement actions.
    """
    
    def __init__(self, policy_yaml: Path, advisor_id: str = "production"):
        super().__init__(advisor_id)
        
        # Import here to avoid circular dependencies
        from ...risk.risk_agent_v2 import create_risk_agent_v2
        from ...risk.event_types import RiskEvent, EventType, EventPriority
        
        # Load risk configuration
        import yaml
        with open(policy_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create RiskAgentV2 in advisory mode
        self._risk_agent = create_risk_agent_v2(config)
        self._event_type = EventType.RISK_CALCULATION
        self._event_priority = EventPriority.HIGH
        
        self.logger.info(f"ProductionRiskAdvisor initialized with config: {policy_yaml}")
    
    def evaluate(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate risk using RiskAgentV2 calculators without enforcement.
        
        Args:
            obs: Observation containing market data and portfolio state
            
        Returns:
            Risk metrics dictionary
        """
        from ...risk.event_types import RiskEvent
        
        # Create risk event for calculation
        event = RiskEvent(
            event_type=self._event_type,
            priority=self._event_priority,
            source="trainer_advisor",
            data=obs
        )
        
        # Use RiskAgentV2 for calculation only (no enforcement)
        try:
            # Calculate risk metrics using all registered calculators
            risk_results = {}
            
            for calculator in self._risk_agent.calculators:
                calc_result = calculator.calculate(obs)
                if calc_result:
                    risk_results.update(calc_result)
            
            # Normalize and structure the results
            risk_metrics = self._normalize_risk_metrics(risk_results)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating risk: {e}")
            # Return safe defaults on error
            return self._get_default_risk_metrics()
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get current risk configuration from RiskAgentV2."""
        return self._risk_agent.limits_config
    
    def _normalize_risk_metrics(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize raw calculator results into standardized risk metrics.
        
        Args:
            raw_results: Raw results from risk calculators
            
        Returns:
            Normalized risk metrics
        """
        metrics = {}
        
        # Extract and normalize specific metrics
        metrics['drawdown_pct'] = abs(raw_results.get('current_drawdown', 0.0))
        metrics['drawdown_velocity'] = abs(raw_results.get('drawdown_velocity', 0.0))
        metrics['var_breach_severity'] = max(0.0, raw_results.get('var_breach', 0.0))
        metrics['position_concentration'] = raw_results.get('position_concentration', 0.0)
        metrics['liquidity_risk'] = raw_results.get('liquidity_risk', 0.0)
        
        # Compute overall risk score (0-1 scale)
        risk_components = [
            metrics['drawdown_pct'] / 0.1,  # Normalize by 10% drawdown
            metrics['drawdown_velocity'] / 0.05,  # Normalize by 5% velocity
            metrics['var_breach_severity'],  # Already 0-1
            metrics['position_concentration'],  # Already 0-1
            metrics['liquidity_risk'],  # Already 0-1
        ]
        
        metrics['overall_risk_score'] = min(1.0, np.mean([max(0.0, comp) for comp in risk_components]))
        metrics['breach_severity'] = metrics['overall_risk_score']
        
        # Compute penalty for reward shaping
        metrics['penalty'] = self.compute_risk_penalty(metrics)
        
        return metrics
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Return default risk metrics when evaluation fails."""
        return {
            'drawdown_pct': 0.0,
            'drawdown_velocity': 0.0,
            'var_breach_severity': 0.0,
            'position_concentration': 0.0,
            'liquidity_risk': 0.0,
            'overall_risk_score': 0.0,
            'breach_severity': 0.0,
            'penalty': 0.0,
        }


__all__ = ["RiskAdvisor", "ProductionRiskAdvisor"]