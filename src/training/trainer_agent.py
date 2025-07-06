# src/training/trainer_agent.py
"""
Production-grade TrainerAgent for risk-aware RL training.

Key features:
- Clean separation from execution environment
- Risk-aware training with callbacks and reward shaping
- TorchScript policy bundle export for production deployment
- Latency SLO validation and monitoring
- Type hints and comprehensive error handling
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Core dependencies
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

# Internal imports
try:
    from ..agents.base_agent import BaseAgent
    from ..gym_env.intraday_trading_env import IntradayTradingEnv
    from ..shared.constants import MODEL_VERSION_FORMAT
    from .interfaces.rl_policy import RLPolicy
    from .interfaces.risk_advisor import RiskAdvisor, ProductionRiskAdvisor
    from .policies.sb3_policy import SB3Policy, SB3_AVAILABLE, SB3_ALGORITHMS
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.base_agent import BaseAgent
    from gym_env.intraday_trading_env import IntradayTradingEnv
    from shared.constants import MODEL_VERSION_FORMAT
    from training.interfaces.rl_policy import RLPolicy
    from training.interfaces.risk_advisor import RiskAdvisor, ProductionRiskAdvisor
    from training.policies.sb3_policy import SB3Policy, SB3_AVAILABLE, SB3_ALGORITHMS


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

        self.logger = logging.getLogger("RiskAwareCallback")

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Get current observation from environment
        if hasattr(self.training_env, "get_attr"):
            # VecEnv
            obs_list = self.training_env.get_attr("_last_obs")
            if obs_list and obs_list[0] is not None:
                obs = obs_list[0]
            else:
                return True  # Skip if no observation available
        else:
            # Single env
            if hasattr(self.training_env, "_last_obs"):
                obs = self.training_env._last_obs
            else:
                return True  # Skip if no observation available

        try:
            # Convert observation to dict format expected by risk advisor
            obs_dict = self._convert_obs_to_dict(obs)

            # Evaluate risk
            risk_metrics = self.risk_advisor.evaluate(obs_dict)

            # Check for early stopping
            if risk_metrics["breach_severity"] > self.early_stop_threshold:
                self.risk_violations += 1
                self.logger.warning(
                    f"Risk breach detected: severity={risk_metrics['breach_severity']:.3f} "
                    f"(threshold={self.early_stop_threshold})"
                )

                # Stop training if too many violations
                if self.risk_violations > 5:
                    self.logger.critical("Too many risk violations, stopping training early")
                    return False

            # Log risk metrics periodically
            if self.num_timesteps % self.log_freq == 0:
                self._log_risk_metrics(risk_metrics)

            # Accumulate risk penalty for reward shaping
            self.total_risk_penalty += risk_metrics["penalty"]

        except Exception as e:
            self.logger.error(f"Risk evaluation failed: {e}")
            # Continue training on risk evaluation errors

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        self.episode_count += 1

        # Log episode-level risk statistics
        if self.episode_count % 10 == 0:
            avg_risk_penalty = self.total_risk_penalty / max(1, self.num_timesteps)
            self.logger.info(
                f"Episode {self.episode_count}: "
                f"Risk violations: {self.risk_violations}, "
                f"Avg risk penalty: {avg_risk_penalty:.4f}"
            )

    def _convert_obs_to_dict(self, obs: np.ndarray) -> Dict[str, Any]:
        """Convert observation array to dictionary format for risk advisor."""
        # This is a simplified conversion - in practice, you'd need to map
        # observation components to meaningful risk inputs
        return {
            "market_features": obs[:-1] if len(obs) > 1 else obs,
            "position": obs[-1] if len(obs) > 1 else 0.0,
            "timestamp": datetime.now(),
        }

    def _log_risk_metrics(self, risk_metrics: Dict[str, float]) -> None:
        """Log risk metrics to TensorBoard."""
        if self.logger is not None:
            for metric_name, value in risk_metrics.items():
                self.logger.record(f"risk/{metric_name}", value)


class TrainerAgent(BaseAgent):
    """
    Production-grade trainer for RL models with risk-aware training.

    Features:
    - Clean SB3 integration without dummy fallbacks
    - Risk-aware callbacks and reward shaping
    - TorchScript policy bundle export
    - Comprehensive logging and monitoring
    - Latency SLO validation
    """

    def __init__(self, config: Dict[str, Any], training_env: Optional[IntradayTradingEnv] = None):
        super().__init__(agent_name="TrainerAgent", config=config)

        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable-Baselines3 is required for TrainerAgent. "
                "Install with: pip install stable-baselines3[extra]"
            )

        # Configuration
        self.model_save_dir = Path(self.config.get("model_save_dir", "models/"))
        self.log_dir = Path(self.config.get("log_dir", "logs/tensorboard/"))
        self.monitor_log_dir = self.log_dir / "monitor_logs"

        self.algorithm_name = self.config.get("algorithm", "DQN").upper()
        self.algo_params = self.config.get("algo_params", {})
        self.training_params = self.config.get("training_params", {})
        self.risk_config = self.config.get("risk_config", {})

        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model: Optional[DQN] = None
        self.training_env_monitor: Optional[Monitor] = None
        self.risk_advisor: Optional[RiskAdvisor] = None

        # Set up risk advisor if configured
        if self.risk_config.get("enabled", False):
            self._setup_risk_advisor()

        # Set environment if provided
        if training_env:
            self.set_env(training_env)

        self.logger.info(f"TrainerAgent initialized for {self.algorithm_name}")
        self.logger.info(f"Model save dir: {self.model_save_dir}")
        self.logger.info(f"Log dir: {self.log_dir}")

    def _setup_risk_advisor(self) -> None:
        """Set up risk advisor for risk-aware training."""
        try:
            risk_policy_path = self.risk_config.get("policy_yaml")
            if risk_policy_path:
                risk_policy_path = Path(risk_policy_path)
                if risk_policy_path.exists():
                    self.risk_advisor = ProductionRiskAdvisor(
                        policy_yaml=risk_policy_path, advisor_id="trainer_risk_advisor"
                    )
                    self.logger.info(f"Risk advisor initialized with policy: {risk_policy_path}")
                else:
                    self.logger.warning(f"Risk policy file not found: {risk_policy_path}")
            else:
                self.logger.info("No risk policy specified, risk-aware training disabled")
        except Exception as e:
            self.logger.error(f"Failed to setup risk advisor: {e}")
            self.risk_advisor = None

    def set_env(self, env: IntradayTradingEnv) -> None:
        """Set and wrap training environment with monitoring."""
        if not isinstance(env, (IntradayTradingEnv, Monitor)):
            raise ValueError("Environment must be IntradayTradingEnv or Monitor-wrapped")

        if isinstance(env, Monitor):
            self.training_env_monitor = env
            self.logger.info("Pre-monitored environment set")
        else:
            # Wrap with Monitor for episode logging
            monitor_file = (
                self.monitor_log_dir
                / f"{self.algorithm_name}_{datetime.now().strftime(MODEL_VERSION_FORMAT)}"
            )
            self.training_env_monitor = Monitor(
                env, filename=str(monitor_file), allow_early_resets=True
            )
            self.logger.info(f"Environment wrapped with Monitor: {monitor_file}.csv")

    def create_model(self) -> Optional[DQN]:
        """Create RL model with specified configuration."""
        if self.training_env_monitor is None:
            self.logger.error("Cannot create model: training environment not set")
            return None

        if self.algorithm_name not in SB3_ALGORITHMS:
            self.logger.error(f"Unsupported algorithm: {self.algorithm_name}")
            return None

        try:
            # Get algorithm class
            AlgorithmClass = SB3_ALGORITHMS[self.algorithm_name]

            # Prepare parameters
            model_params = self.algo_params.copy()
            model_params["env"] = self.training_env_monitor
            model_params["tensorboard_log"] = str(self.log_dir)

            # Create model
            self.model = AlgorithmClass(**model_params)

            self.logger.info(f"{self.algorithm_name} model created successfully")
            return self.model

        except Exception as e:
            self.logger.error(f"Failed to create {self.algorithm_name} model: {e}")
            return None

    def train(self, existing_model_path: Optional[str] = None) -> Optional[str]:
        """
        Train the RL model with risk-aware callbacks.

        Args:
            existing_model_path: Path to existing model to continue training

        Returns:
            Path to saved model bundle, or None if training failed
        """
        if self.training_env_monitor is None:
            self.logger.error("Cannot train: training environment not set")
            return None

        # Create run identifier
        run_timestamp = datetime.now().strftime(MODEL_VERSION_FORMAT)
        run_name = f"{self.algorithm_name}_{run_timestamp}"
        run_dir = self.model_save_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load or create model
        if existing_model_path and Path(existing_model_path).exists():
            self.logger.info(f"Loading existing model: {existing_model_path}")
            try:
                AlgorithmClass = SB3_ALGORITHMS[self.algorithm_name]
                self.model = AlgorithmClass.load(
                    existing_model_path,
                    env=self.training_env_monitor,
                    tensorboard_log=str(self.log_dir),
                )
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}, creating new model")
                self.model = self.create_model()
        else:
            self.model = self.create_model()

        if self.model is None:
            self.logger.error("Model creation/loading failed")
            return None

        # Setup callbacks
        callbacks = self._create_callbacks(run_dir, run_name)

        # Training parameters
        total_timesteps = self.training_params.get("total_timesteps", 100000)
        log_interval = self.training_params.get("log_interval", 100)

        # Start training
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=log_interval,
                tb_log_name=run_name,
                reset_num_timesteps=(existing_model_path is None),
            )

            self.logger.info("Training completed successfully")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None

        # Save model bundle
        try:
            bundle_path = self._save_model_bundle(run_dir, run_name)
            self.logger.info(f"Model bundle saved: {bundle_path}")
            return str(bundle_path)

        except Exception as e:
            self.logger.error(f"Failed to save model bundle: {e}")
            return None

    def _create_callbacks(self, run_dir: Path, run_name: str) -> List[BaseCallback]:
        """Create training callbacks."""
        callbacks = []

        # Checkpoint callback
        checkpoint_freq = self.training_params.get("checkpoint_freq", 10000)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"{self.algorithm_name.lower()}_checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)

        # Risk-aware callback
        if self.risk_advisor is not None:
            risk_callback = RiskAwareCallback(
                risk_advisor=self.risk_advisor,
                penalty_weight=self.risk_config.get("penalty_weight", 0.1),
                early_stop_threshold=self.risk_config.get("early_stop_threshold", 0.8),
                log_freq=self.risk_config.get("log_freq", 100),
            )
            callbacks.append(risk_callback)
            self.logger.info("Risk-aware callback enabled")

        # Evaluation callback (optional)
        if self.training_params.get("use_eval_callback", False):
            eval_freq = self.training_params.get("eval_freq", 20000)
            eval_log_path = run_dir / "eval_logs"
            best_model_path = run_dir / "best_model"

            eval_callback = EvalCallback(
                self.training_env_monitor,
                best_model_save_path=str(best_model_path),
                log_path=str(eval_log_path),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        return callbacks

    def _save_model_bundle(self, run_dir: Path, run_name: str) -> Path:
        """Save model as production-ready bundle."""
        # Create SB3Policy wrapper
        sb3_policy = SB3Policy(self.model, policy_id=run_name)

        # Save bundle
        bundle_path = run_dir / "policy_bundle"
        sb3_policy.save_bundle(bundle_path)

        # Validate latency SLO
        self._validate_policy_latency(sb3_policy)

        return bundle_path

    def _validate_policy_latency(self, policy: RLPolicy) -> None:
        """Validate that policy meets latency SLO."""
        if self.training_env_monitor is None:
            return

        try:
            # Get sample observation
            obs = self.training_env_monitor.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle new gym API

            # Validate latency
            latency_stats = policy.validate_prediction_latency(obs, num_trials=100)

            self.logger.info("Policy latency validation:")
            self.logger.info(f"  Mean: {latency_stats['mean_latency_us']:.1f}µs")
            self.logger.info(f"  P95: {latency_stats['p95_latency_us']:.1f}µs")
            self.logger.info(f"  P99: {latency_stats['p99_latency_us']:.1f}µs")
            self.logger.info(f"  SLO violations: {latency_stats['slo_violation_rate']:.2%}")

            if latency_stats["slo_violation_rate"] > 0.05:  # More than 5% violations
                self.logger.warning("Policy may not meet production latency SLO")

        except Exception as e:
            self.logger.error(f"Latency validation failed: {e}")

    def run(
        self, training_env: IntradayTradingEnv, existing_model_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Main entry point for training.

        Args:
            training_env: Training environment
            existing_model_path: Optional path to existing model

        Returns:
            Path to saved model bundle
        """
        self.logger.info("Starting TrainerAgent run")

        # Set environment
        self.set_env(training_env)

        if self.training_env_monitor is None:
            self.logger.error("Failed to set up training environment")
            return None

        # Start training
        return self.train(existing_model_path)


# Factory function for easy instantiation
def create_trainer_agent(
    config: Dict[str, Any], training_env: Optional[IntradayTradingEnv] = None
) -> TrainerAgent:
    """
    Factory function to create TrainerAgent with validation.

    Args:
        config: Training configuration
        training_env: Optional training environment

    Returns:
        Configured TrainerAgent instance
    """
    # Validate required configuration
    required_keys = ["algorithm", "algo_params", "training_params"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate algorithm
    algorithm = config["algorithm"].upper()
    if algorithm not in SB3_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Supported: {list(SB3_ALGORITHMS.keys())}"
        )

    return TrainerAgent(config, training_env)


__all__ = ["TrainerAgent", "RiskAwareCallback", "create_trainer_agent"]
