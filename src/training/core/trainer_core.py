"""
Trainer Core Module

Contains the main training logic extracted from TrainerAgent.
This module handles:
- Core training coordination
- Model management and lifecycle
- Training state management
- Risk advisor integration

This is an internal module - use src.training.TrainerAgent for public API.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import platform

# Optional dependency for detailed system info
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Core dependencies
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

# Internal imports
try:
    from ...gym_env.intraday_trading_env import IntradayTradingEnv
    from ...shared.constants import MODEL_VERSION_FORMAT
    from ..interfaces.rl_policy import RLPolicy
    from ..interfaces.risk_advisor import RiskAdvisor, ProductionRiskAdvisor
    from ..policies.sb3_policy import SB3Policy, SB3_AVAILABLE, SB3_ALGORITHMS
    from ...risk.risk_agent_v2 import RiskAgentV2
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gym_env.intraday_trading_env import IntradayTradingEnv
    from shared.constants import MODEL_VERSION_FORMAT
    from training.interfaces.rl_policy import RLPolicy
    from training.interfaces.risk_advisor import RiskAdvisor, ProductionRiskAdvisor
    from training.policies.sb3_policy import SB3Policy, SB3_AVAILABLE, SB3_ALGORITHMS
    from risk.risk_agent_v2 import RiskAgentV2


class TrainerCore:
    """
    Core training system for RL models.
    
    Handles the main training logic, model management, and coordination
    between different training components.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the trainer core.
        
        Args:
            config: Training configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
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
        self._risk_agent: Optional[RiskAgentV2] = None
        self.training_state: Dict[str, Any] = {}

        # Set up risk advisor if configured
        if self.risk_config.get("enabled", False):
            self.setup_risk_advisor()

        self.logger.info(f"TrainerCore initialized for {self.algorithm_name}")
        self.logger.info(f"Model save dir: {self.model_save_dir}")
        self.logger.info(f"Log dir: {self.log_dir}")
        
    def setup_risk_advisor(self) -> None:
        """Set up risk advisor for risk-aware training using RiskAgentV2.from_yaml()."""
        try:
            risk_policy_path = self.risk_config.get("policy_yaml")
            if risk_policy_path:
                risk_policy_path = Path(risk_policy_path)
                if risk_policy_path.exists():
                    # Use RiskAgentV2.from_yaml() instead of hand-rolled calculator list
                    self._risk_agent = RiskAgentV2.from_yaml(str(risk_policy_path))
                    self.risk_advisor = ProductionRiskAdvisor(self._risk_agent)
                    self.logger.info(f"Risk advisor loaded from {risk_policy_path}")
                else:
                    self.logger.warning(f"Risk policy file not found: {risk_policy_path}")
            else:
                self.logger.warning("Risk advisor enabled but no policy_yaml specified")
        except Exception as e:
            self.logger.error(f"Failed to setup risk advisor: {e}")
            self.risk_advisor = None
    
    @property
    def risk_agent(self) -> Optional[RiskAgentV2]:
        """Access to the underlying risk agent for compatibility."""
        return self._risk_agent
        
    def create_model(self) -> Optional[DQN]:
        """
        Create and configure the RL model.
        
        Returns:
            Configured RL model or None if creation fails
        """
        if self.training_env_monitor is None:
            self.logger.error("Cannot create model: training environment not set")
            return None

        try:
            # Get algorithm class
            if self.algorithm_name not in SB3_ALGORITHMS:
                self.logger.error(f"Unsupported algorithm: {self.algorithm_name}")
                return None

            AlgorithmClass = SB3_ALGORITHMS[self.algorithm_name]

            # Prepare algorithm parameters
            model_params = {
                "env": self.training_env_monitor,
                "tensorboard_log": str(self.log_dir),
                **self.algo_params
            }

            # Create model
            model = AlgorithmClass(**model_params)
            self.logger.info(f"Created {self.algorithm_name} model with params: {self.algo_params}")
            
            return model

        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return None
        
    def train_model(self, existing_model_path: Optional[str] = None) -> Optional[str]:
        """
        Execute the main training loop.
        
        Args:
            existing_model_path: Path to existing model to continue training
            
        Returns:
            Path to saved model or None if training fails
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

        # Log comprehensive hardware information
        self.log_hardware_info()

        # Setup callbacks (will be implemented in separate method)
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
            
            # Save model bundle
            model_bundle_path = self._save_model_bundle(run_dir, run_name)
            return str(model_bundle_path)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None
    
    def _create_callbacks(self, run_dir: Path, run_name: str) -> List[BaseCallback]:
        """Create training callbacks including risk-aware callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_freq = self.training_params.get("checkpoint_freq", 10000)
        if checkpoint_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"{self.algorithm_name}_checkpoint"
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_freq = self.training_params.get("eval_freq", 5000)
        if eval_freq > 0 and hasattr(self, 'eval_env'):
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval_logs"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Risk-aware callbacks
        if self.risk_advisor:
            from .risk_callbacks import RiskAwareCallback
            risk_callback = RiskAwareCallback(
                risk_advisor=self.risk_advisor,
                penalty_weight=self.risk_config.get("penalty_weight", 0.1),
                early_stop_threshold=self.risk_config.get("early_stop_threshold", 0.8),
                log_freq=self.training_params.get("log_interval", 100)
            )
            callbacks.append(risk_callback)
        
        return callbacks
    
    def _save_model_bundle(self, run_dir: Path, run_name: str) -> Path:
        """Save the trained model as a bundle with metadata."""
        try:
            # Save the SB3 model
            model_path = run_dir / f"{run_name}.zip"
            self.model.save(str(model_path))
            
            # Export TorchScript bundle (will be implemented in policy_export.py)
            from .policy_export import export_torchscript_bundle
            export_torchscript_bundle(self.model, run_dir, run_name)
            
            # Save metadata
            metadata = {
                "algorithm": self.algorithm_name,
                "training_timesteps": self.training_params.get("total_timesteps", 100000),
                "created_at": datetime.now().isoformat(),
                "config": self.config,
                "model_path": str(model_path),
                "torchscript_path": str(run_dir / f"{run_name}_torchscript.pt")
            }
            
            metadata_path = run_dir / f"{run_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model bundle saved: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model bundle: {e}")
            raise
        
    def set_environment(self, env: IntradayTradingEnv) -> None:
        """Set the training environment with monitoring wrapper."""
        try:
            # Wrap environment with Monitor for logging
            monitor_path = self.monitor_log_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.training_env_monitor = Monitor(
                env,
                filename=str(monitor_path),
                allow_early_resets=True
            )
            self.logger.info(f"Training environment set with monitor: {monitor_path}")
        except Exception as e:
            self.logger.error(f"Failed to set training environment: {e}")
            raise
    
    def log_hardware_info(self) -> None:
        """Log comprehensive hardware information for training reproducibility."""
        try:
            self.logger.info("=== Hardware Information ===")
            self.logger.info(f"Platform: {platform.platform()}")
            self.logger.info(f"Python: {platform.python_version()}")
            self.logger.info(f"PyTorch: {torch.__version__}")
            
            if PSUTIL_AVAILABLE:
                # CPU info
                cpu_count = psutil.cpu_count(logical=False)
                cpu_count_logical = psutil.cpu_count(logical=True)
                cpu_freq = psutil.cpu_freq()
                self.logger.info(f"CPU: {cpu_count} cores ({cpu_count_logical} logical)")
                if cpu_freq:
                    self.logger.info(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
                
                # Memory info
                memory = psutil.virtual_memory()
                self.logger.info(f"Memory: {memory.total / (1024**3):.1f} GB total, "
                               f"{memory.available / (1024**3):.1f} GB available")
                
                # GPU info (if available)
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.logger.info(f"CUDA GPUs: {gpu_count}")
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    self.logger.info("CUDA: Not available")
            else:
                self.logger.info("Detailed hardware info unavailable (psutil not installed)")
                
        except Exception as e:
            self.logger.warning(f"Failed to log hardware info: {e}")
        
    def get_training_state(self) -> Dict[str, Any]:
        """Get the current training state."""
        return {
            'model_created': self.model is not None,
            'environment_set': self.training_env is not None,
            'risk_advisor_setup': self.risk_advisor is not None,
            'training_active': self.training_state.get('active', False),
            **self.training_state
        }
        
    def update_training_state(self, state_update: Dict[str, Any]) -> None:
        """Update the training state."""
        self.training_state.update(state_update)
        
    def log_hardware_info(self) -> None:
        """
        Log hardware information for training optimization.
        
        This method will be populated with the actual logging logic
        during the extraction phase.
        """
        # TODO: Extract from _log_hardware_info in trainer_agent.py
        pass
        
    def validate_training_config(self) -> bool:
        """
        Validate the training configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = ['algorithm', 'total_timesteps', 'learning_rate']
        
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required config key: {key}")
                return False
                
        # Validate algorithm
        algorithm = self.config.get('algorithm')
        supported_algorithms = ['DQN', 'PPO', 'A2C', 'SAC']
        if algorithm not in supported_algorithms:
            self.logger.error(f"Unsupported algorithm: {algorithm}")
            return False
            
        # Validate timesteps
        total_timesteps = self.config.get('total_timesteps', 0)
        if total_timesteps <= 0:
            self.logger.error(f"Invalid total_timesteps: {total_timesteps}")
            return False
            
        return True
        
    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.model:
            del self.model
            self.model = None
            
        if self.training_env:
            try:
                self.training_env.close()
            except Exception as e:
                self.logger.warning(f"Error closing training environment: {e}")
            self.training_env = None
            
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Training resources cleaned up")