"""
Trainer Core Module

Contains the main training logic extracted from TrainerAgent.
This module handles:
- Core training coordination
- Model management and lifecycle
- Training state management
- Risk advisor integration

ENVIRONMENT MANAGEMENT:
- training_env_monitor: Main training environment (Monitor-wrapped)
- eval_env: Optional evaluation environment for EvalCallback
  * Set by TrainerAgent.set_evaluation_environment() if evaluation needed
  * If None, evaluation callbacks are automatically skipped
  * Used for periodic model evaluation during training

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
from stable_baselines3.common.vec_env import VecEnv

# Advanced algorithms from sb3-contrib
try:
    from sb3_contrib import QRDQN, RecurrentPPO
    SB3_CONTRIB_AVAILABLE = True
except ImportError:
    QRDQN = None
    RecurrentPPO = None
    SB3_CONTRIB_AVAILABLE = False

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
        
        # Extract algorithm-specific parameters from training_params for backward compatibility
        self._extract_algorithm_params()
        self.risk_config = self.config.get("risk_config", {})
        
        # GPU Configuration
        self._setup_device_config()

        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model: Optional[DQN] = None
        self.training_env_monitor: Optional[Monitor] = None
        self.eval_env: Optional[Any] = None  # Evaluation environment (set by TrainerAgent if needed)
        self.risk_advisor: Optional[RiskAdvisor] = None
        self._risk_agent: Optional[RiskAgentV2] = None
        self.training_state: Dict[str, Any] = {}

        # Set up risk advisor if configured
        if self.risk_config.get("enabled", False):
            self.setup_risk_advisor()

        self.logger.info(f"TrainerCore initialized for {self.algorithm_name}")
        self.logger.info(f"Model save dir: {self.model_save_dir}")
        self.logger.info(f"Log dir: {self.log_dir}")
    
    def _extract_algorithm_params(self) -> None:
        """
        Extract algorithm-specific parameters from training_params.
        
        This handles the case where algorithm parameters are mixed with training
        parameters in the YAML configuration for convenience.
        """
        # Algorithm-specific parameters that should be passed to the model constructor
        algorithm_param_keys = {
            'policy', 'policy_kwargs', 'buffer_size', 'batch_size', 'learning_rate',
            'gamma', 'exploration_fraction', 'exploration_initial_eps', 
            'exploration_final_eps', 'target_update_interval', 'train_freq',
            'gradient_steps', 'learning_starts', 'tau', 'prioritized_replay',
            'prioritized_replay_alpha', 'prioritized_replay_beta0', 
            'prioritized_replay_eps', 'optimize_memory_usage'
        }
        
        # Extract algorithm parameters from training_params
        for key in algorithm_param_keys:
            if key in self.training_params:
                self.algo_params[key] = self.training_params[key]
                self.logger.debug(f"Extracted algorithm param: {key} = {self.training_params[key]}")
        
        # Handle dueling DQN configuration
        self._setup_dueling_dqn()
        
        # Log the final algorithm parameters for Advanced DQN
        policy_kwargs = self.algo_params.get('policy_kwargs', {})
        if self.algorithm_name == 'QR-DQN':
            self.logger.info("🎯 Advanced QR-DQN Configuration:")
            self.logger.info(f"  - Algorithm: Quantile Regression DQN (Distributional RL)")
            self.logger.info(f"  - Policy: {self.algo_params.get('policy', 'MultiInputPolicy')}")
            self.logger.info(f"  - Network Architecture: {policy_kwargs.get('net_arch', 'default')}")
            self.logger.info(f"  - Quantiles: {policy_kwargs.get('n_quantiles', 200)}")
            self.logger.info(f"  - Buffer size: {self.algo_params.get('buffer_size', 'default')}")
            self.logger.info(f"  - Batch size: {self.algo_params.get('batch_size', 'default')}")
            self.logger.info("  - Benefits: Full return distribution + reduced overestimation bias")
        elif policy_kwargs:
            self.logger.info("🎯 Enhanced Double DQN Configuration:")
            self.logger.info(f"  - Policy: {self.algo_params.get('policy', 'MultiInputPolicy')}")
            self.logger.info(f"  - Network Architecture: {policy_kwargs.get('net_arch', 'default')}")
            self.logger.info(f"  - Activation Function: {policy_kwargs.get('activation_fn', 'default')}")
            self.logger.info(f"  - Buffer size: {self.algo_params.get('buffer_size', 'default')}")
            self.logger.info(f"  - Batch size: {self.algo_params.get('batch_size', 'default')}")
            self.logger.info("  - Benefits: Target network reduces overestimation bias + larger network capacity")
    
    def _setup_dueling_dqn(self) -> None:
        """Setup enhanced DQN configuration (Double DQN is built into SB3)."""
        policy_kwargs = self.training_params.get('policy_kwargs', {})
        
        if policy_kwargs:
            # Handle activation function string to class conversion
            if 'activation_fn' in policy_kwargs:
                activation_name = policy_kwargs['activation_fn']
                if isinstance(activation_name, str):
                    import torch.nn as nn
                    activation_map = {
                        'ReLU': nn.ReLU,
                        'Tanh': nn.Tanh,
                        'ELU': nn.ELU,
                        'LeakyReLU': nn.LeakyReLU,
                    }
                    if activation_name in activation_map:
                        policy_kwargs['activation_fn'] = activation_map[activation_name]
                        self.logger.info(f"✅ Activation function set to: {activation_name}")
            
            # Set the policy kwargs
            self.algo_params['policy_kwargs'] = policy_kwargs
            self.logger.info("✅ Enhanced DQN Policy configured")
            self.logger.info(f"   Network Architecture: {policy_kwargs.get('net_arch', 'default')}")
            self.logger.info(f"   Activation Function: {policy_kwargs.get('activation_fn', 'default')}")
            
        # Log Double DQN information (built into SB3)
        self.logger.info("🎯 Double DQN Features (Built into SB3):")
        self.logger.info("   - Target Network: ✅ (reduces overestimation bias)")
        self.logger.info("   - Experience Replay: ✅ (improves sample efficiency)")
        self.logger.info("   - Epsilon-Greedy Exploration: ✅ (balanced exploration/exploitation)")
    
    def _setup_device_config(self) -> None:
        """Setup GPU/CPU device configuration for training."""
        import torch
        
        # Get device preference from config
        device_config = self.training_params.get("device", "auto")
        
        if device_config == "auto":
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"🚀 GPU Auto-detected: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.device = "cpu"
                self.logger.info("💻 Using CPU (no GPU available)")
        else:
            # Use specified device
            self.device = device_config
            self.logger.info(f"🔧 Device set to: {self.device}")
        
        # Add device to algorithm parameters
        if "device" not in self.algo_params:
            self.algo_params["device"] = self.device
            
        # GPU memory management
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory_fraction = self.training_params.get("gpu_memory_fraction", 0.8)
            if gpu_memory_fraction < 1.0:
                # Set memory fraction (for TensorFlow compatibility)
                self.logger.info(f"🔧 GPU memory fraction: {gpu_memory_fraction}")
            
            # Enable mixed precision if requested
            mixed_precision = self.training_params.get("mixed_precision", False)
            if mixed_precision:
                self.logger.info("🔧 Mixed precision training enabled")
                # Note: SB3 handles mixed precision internally when device="cuda"
        
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
        
    def create_model(self) -> Optional[Any]:
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
            
            # Enhanced logging for Advanced DQN algorithms
            policy_kwargs = self.algo_params.get('policy_kwargs', {})
            if self.algorithm_name == 'QR-DQN':
                policy_name = getattr(self.algo_params.get('policy'), '__name__', str(self.algo_params.get('policy', 'MultiInputPolicy')))
                self.logger.info(f"✅ Created Advanced {self.algorithm_name} model")
                self.logger.info(f"   Algorithm: Quantile Regression DQN (Distributional RL)")
                self.logger.info(f"   Policy: {policy_name}")
                self.logger.info(f"   Network Architecture: {policy_kwargs.get('net_arch', 'default')}")
                self.logger.info(f"   Quantiles: {policy_kwargs.get('n_quantiles', 200)}")
                self.logger.info(f"   Buffer Size: {self.algo_params.get('buffer_size', 'default'):,}")
                self.logger.info(f"   Batch Size: {self.algo_params.get('batch_size', 'default')}")
                self.logger.info("   🎯 Benefits: Full return distribution learning + reduced overestimation bias")
            elif policy_kwargs:
                policy_name = getattr(self.algo_params.get('policy'), '__name__', str(self.algo_params.get('policy', 'MultiInputPolicy')))
                self.logger.info(f"✅ Created Enhanced Double {self.algorithm_name} model")
                self.logger.info(f"   Policy: {policy_name}")
                self.logger.info(f"   Network Architecture: {policy_kwargs.get('net_arch', 'default')}")
                self.logger.info(f"   Activation Function: {policy_kwargs.get('activation_fn', 'ReLU')}")
                self.logger.info(f"   Buffer Size: {self.algo_params.get('buffer_size', 'default'):,}")
                self.logger.info(f"   Batch Size: {self.algo_params.get('batch_size', 'default')}")
                self.logger.info("   🎯 Benefits: Target network reduces overestimation bias + enhanced capacity")
            else:
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
        total_timesteps = self.training_params.get("total_timesteps", 20000)
        log_interval = self.training_params.get("log_interval", 100)

        # Log training configuration details
        self.logger.info(f"=== Training Configuration ===")
        self.logger.info(f"Total timesteps: {total_timesteps:,}")
        self.logger.info(f"Log interval: {log_interval}")
        self.logger.info(f"Algorithm: {self.algorithm_name}")
        self.logger.info(f"Training params: {self.training_params}")
        self.logger.info(f"Environment: {type(self.training_env_monitor).__name__}")
        
        # Start training
        start_time = datetime.now()
        self.logger.info(f"Starting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Calling model.learn() with {total_timesteps:,} timesteps...")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=log_interval,
                tb_log_name=run_name,
                reset_num_timesteps=(existing_model_path is None),
            )
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            self.logger.info(f"Training completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Training duration: {training_duration}")
            self.logger.info(f"Timesteps per second: {total_timesteps / training_duration.total_seconds():.2f}")
            
            # Save model bundle
            model_bundle_path = self._save_model_bundle(run_dir, run_name)
            return str(model_bundle_path)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None
    
    def _create_callbacks(self, run_dir: Path, run_name: str) -> List[BaseCallback]:
        """Create training callbacks including risk-aware callbacks."""
        callbacks = []
        
        # Early stopping callback to prevent infinite loops
        from .early_stopping_callback import EarlyStoppingCallback
        from .curriculum_callback import CurriculumLearningCallback
        max_episodes = self.training_params.get("max_episodes", 200)  # Allow more episodes for meaningful training
        max_training_time = self.training_params.get("max_training_time_minutes", 15)  # At least 15 minutes
        
        # Get early stopping configuration
        early_stopping_config = self.training_params.get("early_stopping", {})
        plateau_patience = early_stopping_config.get("patience", 100)  # More patience
        min_improvement = early_stopping_config.get("min_improvement", 0.01)
        check_freq = early_stopping_config.get("check_freq", 10000)  # Check every 10k steps
        min_episodes_before_stopping = early_stopping_config.get("min_episodes_before_stopping", 50)
        verbose = early_stopping_config.get("verbose", 1)
        
        early_stopping = EarlyStoppingCallback(
            max_episodes=max_episodes,
            max_training_time_minutes=max_training_time,
            plateau_patience=plateau_patience,
            min_improvement=min_improvement,
            check_freq=check_freq,
            min_episodes_before_stopping=min_episodes_before_stopping,
            verbose=verbose
        )
        callbacks.append(early_stopping)
        
        # Curriculum learning callback
        curriculum_config = self.risk_config.get("curriculum", {})
        if curriculum_config.get("enabled", False):
            curriculum_callback = CurriculumLearningCallback(
                curriculum_config=curriculum_config,
                risk_config=self.risk_config,
                verbose=self.training_params.get("verbose", 1)
            )
            callbacks.append(curriculum_callback)
            self.logger.info("🎓 Curriculum Learning callback added")
        
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
        if eval_freq > 0 and hasattr(self, 'eval_env') and self.eval_env:
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
            # Save the SB3 model as zip (for SB3 compatibility)
            model_zip_path = run_dir / f"{run_name}.zip"
            self.model.save(str(model_zip_path))
            
            # Also save as policy.pt for evaluation compatibility (using torch.save for state dict)
            policy_path = run_dir / "policy.pt"
            import torch
            torch.save(self.model.policy.state_dict(), str(policy_path))
            
            # Export TorchScript bundle (will be implemented in policy_export.py)
            from .policy_export import export_torchscript_bundle
            torchscript_result = export_torchscript_bundle(self.model, run_dir, run_name)
            
            # Log TorchScript export result
            if torchscript_result:
                self.logger.info(f"TorchScript export successful: {torchscript_result}")
            else:
                self.logger.warning("TorchScript export failed, no deployment model available")
            
            # Save metadata
            metadata = {
                "algorithm": self.algorithm_name,
                "training_timesteps": self.training_params.get("total_timesteps", 20000),
                "created_at": datetime.now().isoformat(),
                "config": self.config,
                "model_path": str(model_zip_path),
                "policy_path": str(policy_path),
                "torchscript_path": str(torchscript_result) if torchscript_result else None
            }
            
            metadata_path = run_dir / f"{run_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                # NOTE: Consider gzip compression for large metadata files in future optimization
            
            self.logger.info(f"Model bundle saved: {model_zip_path}")
            self.logger.info(f"Policy model saved: {policy_path}")
            return model_zip_path
            
        except Exception as e:
            self.logger.error(f"Failed to save model bundle: {e}")
            raise
        
    def set_environment(self, env) -> None:
        """
        Set the training environment with monitoring wrapper.
        
        Args:
            env: Either a single IntradayTradingEnv or a VecEnv
        """
        try:
            if isinstance(env, VecEnv):
                # Vectorized environment - already has monitoring via VecMonitor
                self.training_env_monitor = env
                self.logger.info(f"Vectorized training environment set with {env.num_envs} workers")
            else:
                # Single environment - wrap with Monitor for logging
                monitor_path = self.monitor_log_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.training_env_monitor = Monitor(
                    env,
                    filename=str(monitor_path),
                    allow_early_resets=True
                )
                self.logger.info(f"Single training environment set with monitor: {monitor_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to set training environment: {e}")
            raise
    
    def create_vectorized_environment(
        self,
        symbols: List[str],
        data_dir: Path,
        env_config: Dict[str, Any],
        n_envs: Optional[int] = None,
        use_shared_memory: bool = True
    ) -> Any:
        """
        Create a vectorized training environment for improved throughput.
        
        This method creates multiple parallel environments using ShmemVecEnv
        for 3-4x faster experience collection compared to single-threaded rollouts.
        
        Args:
            symbols: List of trading symbols
            data_dir: Directory containing data files
            env_config: Environment configuration
            n_envs: Number of environments (auto-detected if None)
            use_shared_memory: Whether to use shared memory (requires SB3 1.8+)
            
        Returns:
            VecMonitor-wrapped vectorized environment
        """
        try:
            from .env_builder import build_vec_env, get_optimal_n_envs
            
            # Determine optimal number of environments
            if n_envs is None:
                n_envs = get_optimal_n_envs(symbols, max_envs=16)  # Reasonable upper limit
                self.logger.info(f"Auto-detected {n_envs} environments for training")
            
            # Create monitor path
            monitor_path = self.monitor_log_dir / f"vec_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create vectorized environment
            vec_env = build_vec_env(
                symbols=symbols,
                data_dir=data_dir,
                config=env_config,
                n_envs=n_envs,
                monitor_path=str(monitor_path),
                use_shared_memory=use_shared_memory,
                logger=self.logger
            )
            
            self.logger.info(f"✅ Vectorized environment created with {n_envs} workers")
            self.logger.info(f"Expected throughput improvement: {n_envs}x faster rollouts")
            
            # Store vectorized environment internally for consistency
            self.training_env_monitor = vec_env
            
            return vec_env
            
        except Exception as e:
            self.logger.error(f"Failed to create vectorized environment: {e}")
            self.logger.info("Falling back to single environment creation")
            raise
    
    def get_training_performance_info(self) -> Dict[str, Any]:
        """
        Get information about training performance and environment setup.
        
        Returns:
            Dictionary with performance information
        """
        info = {
            'timestamp': datetime.now().isoformat(),
            'environment_type': 'unknown',
            'num_workers': 1,
            'expected_speedup': '1x',
            'shared_memory': False
        }
        
        try:
            if hasattr(self, 'training_env_monitor'):
                env = self.training_env_monitor
                
                if isinstance(env, VecEnv):
                    info.update({
                        'environment_type': 'vectorized',
                        'num_workers': env.num_envs,
                        'expected_speedup': f"{env.num_envs}x",
                        'shared_memory': 'ShmemVecEnv' in str(type(env))
                    })
                else:
                    info.update({
                        'environment_type': 'single',
                        'num_workers': 1,
                        'expected_speedup': '1x',
                        'shared_memory': False
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error getting performance info: {e}")
            
        return info
    
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
            'environment_set': self.training_env_monitor is not None,
            'risk_advisor_setup': self.risk_advisor is not None,
            'training_active': self.training_state.get('active', False),
            **self.training_state
        }
        
    def update_training_state(self, state_update: Dict[str, Any]) -> None:
        """Update the training state."""
        self.training_state.update(state_update)
        
    def validate_training_config(self) -> bool:
        """
        Validate the training configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check top-level required keys
        required_top_level = ['algorithm']
        for key in required_top_level:
            if key not in self.config:
                self.logger.error(f"Missing required config key: {key}")
                return False
                
        # Validate algorithm
        algorithm = self.config.get('algorithm')
        supported_algorithms = ['DQN', 'QR-DQN', 'PPO', 'A2C', 'SAC']
        if algorithm not in supported_algorithms:
            self.logger.error(f"Unsupported algorithm: {algorithm}")
            return False
            
        # Check nested training parameters
        training_params = self.config.get('training_params', {})
        if 'total_timesteps' not in training_params:
            self.logger.error("Missing required training_params.total_timesteps")
            return False
            
        total_timesteps = training_params.get('total_timesteps', 0)
        if total_timesteps <= 0:
            self.logger.error(f"Invalid total_timesteps: {total_timesteps}")
            return False
            
        # Check algorithm parameters (learning_rate typically in algo_params)
        algo_params = self.config.get('algo_params', {})
        if 'learning_rate' not in algo_params:
            self.logger.warning("learning_rate not found in algo_params - using algorithm default")
            
        return True
        
    def cleanup(self) -> None:
        """Clean up training resources."""
        if self.model:
            del self.model
            self.model = None
            
        if self.training_env_monitor:
            try:
                self.training_env_monitor.close()
            except Exception as e:
                self.logger.warning(f"Error closing training environment: {e}")
            self.training_env_monitor = None
            
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Training resources cleaned up")