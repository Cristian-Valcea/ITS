# src/execution/execution_agent_stub.py
"""
ExecutionAgentStub: Minimal stub for contract testing policy bundles.

This stub validates that policy bundles can be loaded and executed
with the required <100µs latency SLO in a production-like environment.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    from ..shared.constants import MAX_PREDICTION_LATENCY_US
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.constants import MAX_PREDICTION_LATENCY_US


class HoldCashFallbackPolicy:
    """
    Ultra-fast fallback policy that always returns HOLD (action=1).
    
    Activated when primary policy.pt fails to load.
    Guarantees <10µs P95 latency with zero dependencies.
    """
    
    def __init__(self):
        self.policy_id = "fallback_hold_cash"
        self.action = 1  # HOLD action
        self.prediction_count = 0
        
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Ultra-fast prediction that always returns HOLD.
        
        Args:
            obs: Observation array (ignored)
            deterministic: Whether to use deterministic policy (ignored)
        
        Returns:
            Tuple of (HOLD_action=1, info_dict)
        """
        # High-precision timing for latency tracking
        start = time.perf_counter_ns()
        
        # Trivial computation - just return HOLD
        action = self.action
        
        # Calculate latency
        lat_us = (time.perf_counter_ns() - start) / 1_000
        
        # Update performance tracking
        self.prediction_count += 1
        
        info = {
            "policy_id": self.policy_id,
            "policy_type": "fallback_hold_cash",
            "action_reason": "fallback_safe_hold",
            "deterministic": True,  # Always deterministic
            "latency_us": lat_us,
            "is_fallback": True,
            "prediction_count": self.prediction_count,
        }
        
        return action, info


class ExecutionAgentStub:
    """
    Minimal execution agent stub for contract testing.

    Simulates the production execution environment with:
    - Policy bundle loading
    - Latency SLO validation
    - Minimal dependencies (no SB3, no gym)
    - CPU-only inference
    """

    def __init__(self, policy_bundle_path: Path, enable_soft_deadline: bool = True):
        """
        Initialize ExecutionAgentStub with policy bundle.

        Args:
            policy_bundle_path: Path to policy bundle directory
            enable_soft_deadline: Whether to enable soft-deadline assertions (default: True)
        """
        self.policy_bundle_path = Path(policy_bundle_path)
        self.enable_soft_deadline = enable_soft_deadline
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load policy
        self.policy = self._load_policy_bundle()

        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time_us = 0.0
        self.slo_violations = 0

        self.logger.info(f"ExecutionAgentStub initialized with bundle: {policy_bundle_path}")
        if not enable_soft_deadline:
            self.logger.warning("⚠️ Soft-deadline assertions disabled for testing")

    def _load_policy_bundle(self):
        """Load policy from bundle directory with fallback to hold-cash policy."""
        try:
            # Import here to avoid circular dependencies
            try:
                from ..training.policies.sb3_policy import TorchScriptPolicy
            except ImportError:
                from training.policies.sb3_policy import TorchScriptPolicy

            policy = TorchScriptPolicy.load_bundle(self.policy_bundle_path)
            self.logger.info(f"Policy loaded: {policy.policy_id}")
            return policy

        except Exception as e:
            self.logger.error(f"Failed to load policy bundle: {e}")
            self.logger.warning("🔄 Activating fallback hold-cash policy...")
            
            # Return fallback hold-cash policy
            return HoldCashFallbackPolicy()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Predict action with latency monitoring and soft-deadline assertion.

        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, info_dict)
        """
        # High-precision timing for soft-deadline check
        start = time.perf_counter_ns()

        try:
            # Make prediction
            action, info = self.policy.predict(obs, deterministic=deterministic)

            # Calculate latency with nanosecond precision
            lat_us = (time.perf_counter_ns() - start) / 1_000

            # Soft-deadline assertion - fail fast on SLA violation
            if self.enable_soft_deadline:
                assert lat_us < 100, f"Inference {lat_us:.1f}µs exceeds SLA"

            # Update performance tracking
            self.prediction_count += 1
            self.total_prediction_time_us += lat_us

            # Check SLO violation (for logging/monitoring)
            if lat_us > MAX_PREDICTION_LATENCY_US:
                self.slo_violations += 1
                self.logger.warning(
                    f"SLO violation: prediction took {lat_us:.1f}µs "
                    f"(limit: {MAX_PREDICTION_LATENCY_US}µs)"
                )

            # Add performance info
            info.update(
                {
                    "latency_us": lat_us,
                    "slo_violation": lat_us > MAX_PREDICTION_LATENCY_US,
                    "prediction_count": self.prediction_count,
                }
            )

            return action, info

        except AssertionError:
            # Re-raise assertion errors (soft-deadline violations)
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return safe default
            return 1, {"error": str(e), "latency_us": 0.0}

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.prediction_count == 0:
            return {
                "prediction_count": 0,
                "mean_latency_us": 0.0,
                "slo_violation_rate": 0.0,
                "slo_violations": 0,
            }

        return {
            "prediction_count": self.prediction_count,
            "mean_latency_us": self.total_prediction_time_us / self.prediction_count,
            "slo_violation_rate": self.slo_violations / self.prediction_count,
            "slo_violations": self.slo_violations,
            "total_prediction_time_us": self.total_prediction_time_us,
        }

    def validate_slo_compliance(
        self, num_trials: int = 100, sample_obs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Validate SLO compliance with multiple predictions.

        Args:
            num_trials: Number of prediction trials
            sample_obs: Sample observation (random if None)

        Returns:
            Validation results dictionary
        """
        if sample_obs is None:
            # Create sample observation based on policy metadata
            try:
                obs_shape = self.policy.metadata.get("obs_space", {}).get("shape", [10])
                sample_obs = np.random.randn(*obs_shape).astype(np.float32)
            except (KeyError, TypeError, ValueError):
                sample_obs = np.random.randn(10).astype(np.float32)

        self.logger.info(f"Running SLO validation with {num_trials} trials...")

        # Reset performance tracking
        initial_violations = self.slo_violations

        latencies = []

        # Temporarily disable soft-deadline assertions for validation
        original_soft_deadline = self.enable_soft_deadline
        self.enable_soft_deadline = False
        
        try:
            # Run trials
            for i in range(num_trials):
                start_time = time.perf_counter()
                action, info = self.predict(sample_obs, deterministic=True)
                end_time = time.perf_counter()

                latency_us = (end_time - start_time) * 1_000_000
                latencies.append(latency_us)
        finally:
            # Restore original setting
            self.enable_soft_deadline = original_soft_deadline

        # Calculate statistics
        latencies = np.array(latencies)

        results = {
            "num_trials": num_trials,
            "mean_latency_us": np.mean(latencies),
            "median_latency_us": np.median(latencies),
            "p95_latency_us": np.percentile(latencies, 95),
            "p99_latency_us": np.percentile(latencies, 99),
            "max_latency_us": np.max(latencies),
            "min_latency_us": np.min(latencies),
            "std_latency_us": np.std(latencies),
            "slo_violations": self.slo_violations - initial_violations,
            "slo_violation_rate": (self.slo_violations - initial_violations) / num_trials,
            "slo_compliant": (self.slo_violations - initial_violations) == 0,
            "slo_threshold_us": MAX_PREDICTION_LATENCY_US,
        }

        # Log results
        self.logger.info("SLO Validation Results:")
        self.logger.info(f"  Mean latency: {results['mean_latency_us']:.1f}µs")
        self.logger.info(f"  P95 latency: {results['p95_latency_us']:.1f}µs")
        self.logger.info(f"  P99 latency: {results['p99_latency_us']:.1f}µs")
        self.logger.info(
            f"  SLO violations: {results['slo_violations']}/{num_trials} ({results['slo_violation_rate']:.2%})"
        )

        if results["slo_compliant"]:
            self.logger.info("✅ Policy meets SLO requirements")
        else:
            self.logger.warning("⚠️ Policy violates SLO requirements")

        return results

    def benchmark_against_baseline(
        self, baseline_latency_us: float = 10.0, num_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark policy against a baseline latency.

        Args:
            baseline_latency_us: Expected baseline latency for comparison
            num_trials: Number of trials to run

        Returns:
            Benchmark results
        """
        validation_results = self.validate_slo_compliance(num_trials)

        overhead_us = validation_results["mean_latency_us"] - baseline_latency_us
        overhead_ratio = validation_results["mean_latency_us"] / baseline_latency_us

        benchmark_results = {
            **validation_results,
            "baseline_latency_us": baseline_latency_us,
            "overhead_us": overhead_us,
            "overhead_ratio": overhead_ratio,
            "efficiency_score": min(
                1.0, baseline_latency_us / validation_results["mean_latency_us"]
            ),
        }

        self.logger.info("Benchmark Results:")
        self.logger.info(f"  Baseline: {baseline_latency_us:.1f}µs")
        self.logger.info(f"  Actual: {validation_results['mean_latency_us']:.1f}µs")
        self.logger.info(f"  Overhead: {overhead_us:.1f}µs ({overhead_ratio:.2f}x)")
        self.logger.info(f"  Efficiency: {benchmark_results['efficiency_score']:.2%}")

        return benchmark_results


def create_execution_agent_stub(policy_bundle_path: Path, enable_soft_deadline: bool = True) -> ExecutionAgentStub:
    """
    Factory function to create ExecutionAgentStub with validation.

    Args:
        policy_bundle_path: Path to policy bundle
        enable_soft_deadline: Whether to enable soft-deadline assertions

    Returns:
        Configured ExecutionAgentStub
    """
    bundle_path = Path(policy_bundle_path)

    # Validate bundle exists
    if not bundle_path.exists():
        raise FileNotFoundError(f"Policy bundle not found: {bundle_path}")

    # Validate bundle structure
    required_files = ["policy.pt", "metadata.json"]
    for file_name in required_files:
        file_path = bundle_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required bundle file missing: {file_path}")

    return ExecutionAgentStub(bundle_path, enable_soft_deadline)


__all__ = ["ExecutionAgentStub", "create_execution_agent_stub"]
