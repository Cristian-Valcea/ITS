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

try:
    from ...shared.constants import MAX_RISK_EVALUATION_LATENCY_US
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.constants import MAX_RISK_EVALUATION_LATENCY_US


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

    def validate_evaluation_latency(
        self, sample_obs: Dict[str, Any], num_trials: int = 100
    ) -> Dict[str, float]:
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
            "mean_latency_us": np.mean(latencies),
            "p95_latency_us": np.percentile(latencies, 95),
            "p99_latency_us": np.percentile(latencies, 99),
            "max_latency_us": np.max(latencies),
            "slo_violations": sum(1 for lat in latencies if lat > MAX_RISK_EVALUATION_LATENCY_US),
            "slo_violation_rate": sum(
                1 for lat in latencies if lat > MAX_RISK_EVALUATION_LATENCY_US
            )
            / len(latencies),
        }

        if stats["slo_violation_rate"] > 0.01:  # More than 1% violations
            self.logger.warning(
                f"RiskAdvisor {self.advisor_id} violates latency SLO: "
                f"{stats['slo_violation_rate']:.2%} evaluations > {MAX_RISK_EVALUATION_LATENCY_US}µs"
            )

        return stats

    def compute_risk_penalty(
        self, risk_metrics: Dict[str, float], penalty_weights: Optional[Dict[str, float]] = None
    ) -> float:
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
                "drawdown_pct": 2.0,
                "drawdown_velocity": 1.5,
                "var_breach_severity": 3.0,
                "position_concentration": 1.0,
                "liquidity_risk": 1.0,
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
        try:
            from ...risk.risk_agent_v2 import create_risk_agent_v2
            from ...risk.event_types import EventType, EventPriority
        except ImportError:
            from risk.risk_agent_v2 import create_risk_agent_v2
            from risk.event_types import EventType, EventPriority

        # Load risk configuration
        import yaml

        with open(policy_yaml, "r") as f:
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
        try:
            # Use RiskAgentV2's calculate_only method (no enforcement)
            import asyncio

            # Run the async calculate_only method
            if asyncio.iscoroutinefunction(self._risk_agent.calculate_only):
                # Handle async case
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're already in an async context, create a new task
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, self._risk_agent.calculate_only(obs)
                            )
                            risk_metrics = future.result(timeout=1.0)  # 1 second timeout
                    else:
                        risk_metrics = loop.run_until_complete(self._risk_agent.calculate_only(obs))
                except RuntimeError:
                    # No event loop, create one
                    risk_metrics = asyncio.run(self._risk_agent.calculate_only(obs))
            else:
                # Sync case
                risk_metrics = self._risk_agent.calculate_only(obs)

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error evaluating risk: {e}")
            # Return safe defaults on error
            return self._get_default_risk_metrics()

    def get_risk_config(self) -> Dict[str, Any]:
        """Get current risk configuration from RiskAgentV2."""
        return self._risk_agent.limits_config

    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Return default risk metrics when evaluation fails."""
        return self._risk_agent._get_default_risk_metrics()


__all__ = ["RiskAdvisor", "ProductionRiskAdvisor"]
