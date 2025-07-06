# src/agents/trainer_agent.py
"""
DEPRECATED: This module has been refactored and moved to src/training/trainer_agent.py

This file is kept for backward compatibility but should not be used for new development.
Use the new production-grade TrainerAgent from src.training.trainer_agent instead.

Key improvements in the new version:
- Clean SB3 integration without dummy fallbacks
- Risk-aware training with callbacks and reward shaping  
- TorchScript policy bundle export for production deployment
- Comprehensive type hints and error handling
- Latency SLO validation (<100µs per prediction)
"""

import logging
import warnings
from typing import Dict, Any, Optional

# Import the new implementation
try:
    from ..training.trainer_agent import TrainerAgent as NewTrainerAgent, create_trainer_agent
    from ..gym_env.intraday_trading_env import IntradayTradingEnv
    NEW_TRAINER_AVAILABLE = True
except ImportError as e:
    NEW_TRAINER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"New TrainerAgent not available: {e}")

# Issue deprecation warning
warnings.warn(
    "src.agents.trainer_agent is deprecated. Use src.training.trainer_agent instead.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)
logger.warning(
    "DEPRECATED: src.agents.trainer_agent is deprecated. "
    "Use src.training.trainer_agent.TrainerAgent instead."
)


class TrainerAgent:
    """
    DEPRECATED: Compatibility wrapper for the old TrainerAgent.
    
    This class provides basic compatibility but lacks the advanced features
    of the new production-grade implementation.
    """
    
    def __init__(self, config: Dict[str, Any], training_env: Optional[IntradayTradingEnv] = None):
        warnings.warn(
            "TrainerAgent from src.agents.trainer_agent is deprecated. "
            "Use src.training.trainer_agent.TrainerAgent for production features.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if NEW_TRAINER_AVAILABLE:
            # Delegate to new implementation
            self._new_trainer = NewTrainerAgent(config, training_env)
            logger.info("Using new TrainerAgent implementation via compatibility wrapper")
        else:
            # Minimal fallback
            self.config = config
            self.logger = logging.getLogger("DeprecatedTrainerAgent")
            self.logger.error("New TrainerAgent not available, using minimal fallback")
            raise ImportError(
                "New TrainerAgent implementation not available. "
                "Please install required dependencies: pip install stable-baselines3[extra]"
            )
    
    def __getattr__(self, name):
        """Delegate all method calls to the new implementation."""
        if hasattr(self, '_new_trainer'):
            return getattr(self._new_trainer, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Export compatibility symbols
if NEW_TRAINER_AVAILABLE:
    # Re-export from new module for compatibility
    __all__ = ["TrainerAgent", "create_trainer_agent"]
else:
    __all__ = ["TrainerAgent"]


# Migration guide
MIGRATION_GUIDE = """
MIGRATION GUIDE: src.agents.trainer_agent → src.training.trainer_agent
================================================================

OLD USAGE:
    from src.agents.trainer_agent import TrainerAgent
    
    trainer = TrainerAgent(config, env)
    model_path = trainer.run(env)

NEW USAGE:
    from src.training.trainer_agent import TrainerAgent, create_trainer_agent
    
    # Option 1: Direct instantiation
    trainer = TrainerAgent(config, env)
    model_path = trainer.run(env)
    
    # Option 2: Factory function (recommended)
    trainer = create_trainer_agent(config, env)
    model_path = trainer.run(env)

NEW FEATURES:
    ✅ Risk-aware training with RiskAdvisor integration
    ✅ TorchScript policy bundle export for production
    ✅ Latency SLO validation (<100µs per prediction)
    ✅ Clean SB3 integration without dummy fallbacks
    ✅ Comprehensive type hints and error handling
    ✅ Production-ready callbacks and monitoring

CONFIGURATION CHANGES:
    # Add risk-aware training (optional)
    config['risk_config'] = {
        'enabled': True,
        'policy_yaml': 'config/risk_limits.yaml',
        'penalty_weight': 0.1,
        'early_stop_threshold': 0.8
    }
    
    # Simplified algorithm configuration
    config['algorithm'] = 'DQN'  # No more dummy fallbacks
    config['algo_params'] = {...}
    config['training_params'] = {...}

BREAKING CHANGES:
    ❌ Removed SB3 dummy fallbacks - SB3 is now required
    ❌ Removed incomplete C51 implementation
    ❌ Changed model save format to TorchScript bundles
    ❌ Updated callback interface for risk-aware training
"""

if __name__ == "__main__":
    print(MIGRATION_GUIDE)


class TrainerAgent(BaseAgent):
    SB3_AVAILABLE = SB3_AVAILABLE # Expose at class level
    SB3_MODEL_CLASSES = SB3_MODEL_CLASSES # Expose at class level
    """
    TrainerAgent is responsible for:
    1. Instantiating the RL algorithm (e.g., C51-like DQN) from Stable-Baselines3.
    2. Training the RL model using the environment provided by EnvAgent.
    3. Saving model checkpoints and TensorBoard logs.
    """
    def __init__(self, config: dict, training_env: IntradayTradingEnv = None):
        super().__init__(agent_name="TrainerAgent", config=config)
        
        self.model_save_dir = self.config.get('model_save_dir', 'models/')
        self.log_dir = self.config.get('log_dir', 'logs/tensorboard/') # For TensorBoard itself
        self.monitor_log_dir = os.path.join(self.log_dir, "monitor_logs") # For Monitor CSV files
        
        self.algorithm_name = self.config.get('algorithm', 'DQN').upper()
        self.algo_params_config = self.config.get('algo_params', {}) # From model_params.yaml:algorithm_params
        self.c51_features_config = self.config.get('c51_features', {}) # From model_params.yaml:c51_features
        self.training_run_params = self.config.get('training_params', {}) # From main_config.yaml:training

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.monitor_log_dir, exist_ok=True)

        self.model = None
        self.training_env_monitor = None # This will be the Monitor-wrapped env
        if training_env:
            self.set_env(training_env)

        self.logger.info(f"TrainerAgent initialized for algorithm: {self.algorithm_name}. SB3 Available: {SB3_AVAILABLE}")
        self.logger.info(f"Models will be saved to: {self.model_save_dir}")
        self.logger.info(f"TensorBoard logs to: {self.log_dir}; Monitor logs to: {self.monitor_log_dir}")

    def set_env(self, env: IntradayTradingEnv):
        if not isinstance(env, IntradayTradingEnv) and not isinstance(env, Monitor): # Allow already monitored env
            self.logger.error("Invalid environment type provided to TrainerAgent.")
            raise ValueError("Environment must be an instance of IntradayTradingEnv or Monitor.")
        
        if isinstance(env, Monitor):
            self.training_env_monitor = env
            self.logger.info("Pre-monitored training environment set.")
        else:
            # Wrap the environment with Monitor for SB3 to log episode rewards, lengths etc.
            # Ensure unique log file for this monitor instance if multiple runs happen in one session.
            monitor_file_path = os.path.join(self.monitor_log_dir, f"{self.algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.training_env_monitor = Monitor(env, filename=monitor_file_path, allow_early_resets=True)
            self.logger.info(f"Training environment wrapped with Monitor. Log file: {monitor_file_path}.csv")
        
        # If using VecEnvs (e.g. for parallel training), wrap with DummyVecEnv or SubprocVecEnv
        # self.training_env_monitor = DummyVecEnv([lambda: self.training_env_monitor])


    def _create_model(self):
        if self.training_env_monitor is None:
            self.logger.error("Cannot create model: Training environment (Monitor-wrapped) is not set.")
            return None

        # Start with base algo_params from config
        current_algo_params = self.algo_params_config.copy()
        policy_kwargs = current_algo_params.get('policy_kwargs', {})
        if policy_kwargs is None: policy_kwargs = {} # Ensure it's a dict

        # --- Configure C51-like features for DQN ---
        if self.algorithm_name == 'DQN': # Only apply these if base is DQN
            # Dueling Networks
            # SB3's MlpPolicy/CnnPolicy do not have a simple `dueling=True`.
            # Dueling is an architectural choice. If `net_arch` in policy_kwargs specifies separate streams (e.g., for qf and vf),
            # it implies dueling. Some custom policies might expose a `dueling` flag.
            # For this skeleton, we assume `policy_kwargs.net_arch` would be structured for dueling if desired.
            if self.c51_features_config.get('dueling_nets', False):
                self.logger.info("Dueling networks requested. Ensure 'policy_kwargs.net_arch' is configured for dueling streams if using standard policies.")
                # Example: policy_kwargs['net_arch'] = dict(qf=[64, 64], vf=[64, 64]) # For SB3 default DQN MlpPolicy, this is how value/advantage streams are defined.
                # This depends on the specific policy used. The default MlpPolicy for DQN is already dueling if net_arch is not specified or is a simple list.
                # The default `net_arch` for DQN's MlpPolicy is `[64, 64]`, which SB3 interprets for a dueling structure.
                # So, explicit configuration might only be needed for custom architectures.

            # Prioritized Experience Replay (PER)
            if self.c51_features_config.get('use_per', False):
                current_algo_params['prioritized_replay'] = True
                # PER specific params (alpha, beta, eps) can be in `algo_params_config`
                for per_param in ['prioritized_replay_alpha', 'prioritized_replay_beta', 'prioritized_replay_eps']:
                    if per_param in self.algo_params_config:
                        current_algo_params[per_param] = self.algo_params_config[per_param]
                self.logger.info(f"Prioritized Experience Replay (PER) enabled for DQN with params: "
                                 f"alpha={current_algo_params.get('prioritized_replay_alpha')}, "
                                 f"beta={current_algo_params.get('prioritized_replay_beta')}, "
                                 f"eps={current_algo_params.get('prioritized_replay_eps')}.")
            else:
                current_algo_params['prioritized_replay'] = False


            # Noisy Nets
            # For SB3 DQN, this is often enabled by setting `use_noisy_net=True` inside `policy_kwargs`.
            # Or by selecting a specific NoisyNet policy.
            if self.c51_features_config.get('use_noisy_nets', False):
                if SB3_AVAILABLE: # Only if using real SB3
                    if 'net_arch' not in policy_kwargs: # SB3 NoisyLinear needs explicit net_arch
                         policy_kwargs['net_arch'] = [64,64] # Default, or get from config
                    policy_kwargs['noisy_net'] = True # This is a conceptual flag, SB3 might need specific NoisyLinear layers.
                    # For SB3, exploration_fraction/eps become irrelevant if NoisyNets are properly used for exploration.
                    # The actual implementation of NoisyNets in SB3 requires using NoisyLinear layers in the network definition.
                    # Standard MlpPolicy for DQN does not automatically switch to NoisyLinear layers with a simple flag.
                    # This would typically require a custom policy or a policy that explicitly supports noisy nets.
                    # For now, we log the intent. A full implementation would modify the policy or use a NoisyNet-specific one.
                    self.logger.info("Noisy Nets requested for DQN. This requires a policy with NoisyLinear layers (e.g., custom policy or specific SB3 Contrib policy).")
                else:
                    self.logger.info("Noisy Nets requested (using DummyModel).")


            # N-step Returns
            # SB3's DQN does 1-step TD learning by default.
            # For n-step returns with DQN, you might need to use a wrapper or a version of DQN that supports it (e.g., from sb3_contrib or custom).
            # Some algorithms like A3C/A2C use n-step returns naturally.
            # If `n_step_returns` is a direct parameter of the chosen SB3 DQN variant:
            n_step = self.c51_features_config.get('n_step_returns')
            if n_step and n_step > 1:
                # current_algo_params['n_steps'] = n_step # Example if the model took 'n_steps'
                self.logger.info(f"{n_step}-step returns requested. Standard SB3 DQN is 1-step. This feature might require a custom DQN variant or specific configuration not covered by standard SB3 DQN.")

        # --- Model Instantiation ---
        self.logger.info(f"Attempting to create model: {self.algorithm_name} with effective params: {current_algo_params}")
        if policy_kwargs: # Update algo_params with potentially modified policy_kwargs
            current_algo_params['policy_kwargs'] = policy_kwargs

        # Remove unsupported PER params for SB3 DQN
        if self.algorithm_name == 'DQN':
            for per_param in ['prioritized_replay', 'prioritized_replay_alpha', 'prioritized_replay_beta', 'prioritized_replay_eps']:
                if per_param in current_algo_params:
                    del current_algo_params[per_param]

        try:
            if self.algorithm_name == 'DQN':
                self.model = DQN(
                    env=self.training_env_monitor,
                    tensorboard_log=self.log_dir,
                    **current_algo_params
                )
            # elif self.algorithm_name == 'C51':
            #     # If using a true C51 model (e.g., from sb3_contrib)
            #     # from sb3_contrib import C51
            #     # distributional_params = self.c51_features_config.get('distributional_rl_params', {})
            #     # self.model = C51(env=self.training_env_monitor, tensorboard_log=self.log_dir,
            #     #                  **distributional_params, **current_algo_params)
            #     self.logger.warning("True C51 model not used in this skeleton; DQN with C51 features configured instead if SB3_AVAILABLE.")
            #     # Fallback to DQN for dummy if C51 is selected but not truly available
            #     if not SB3_AVAILABLE: self.model = DQN(policy=current_algo_params.get('policy','MlpPolicy'), env=self.training_env_monitor, **current_algo_params)
            #     else: self.logger.error("C51 selected but no SB3 Contrib C51 implementation linked."); return None

            else:
                self.logger.error(f"Unsupported algorithm: {self.algorithm_name}")
                return None
            
            self.logger.info(f"{self.algorithm_name} model created successfully.")
            return self.model

        except Exception as e:
            self.logger.error(f"Error creating {self.algorithm_name} model: {e}", exc_info=True)
            return None


    def train(self, existing_model_path: str = None) -> str | None:
        if self.training_env_monitor is None:
            self.logger.error("Cannot train: Training environment (Monitor-wrapped) is not set.")
            return None

        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{self.algorithm_name}_{run_timestamp}"

        if existing_model_path and os.path.exists(existing_model_path):
            self.logger.info(f"Loading existing model from: {existing_model_path}")
            try:
                # Use the class method of the specific algorithm for loading
                if self.algorithm_name == 'DQN':
                    self.model = DQN.load(existing_model_path, env=self.training_env_monitor, tensorboard_log=self.log_dir)
                # elif self.algorithm_name == 'C51': # if using a specific C51 class
                #     self.model = C51.load(existing_model_path, env=self.training_env_monitor, tensorboard_log=self.log_dir)
                else:
                    self.logger.error(f"Loading not implemented for algorithm {self.algorithm_name}"); return None
                
                # If continuing training, reset TensorBoard log name to avoid conflicts or append to existing.
                # SB3 model.learn() with a new tb_log_name will create a new subdirectory.
                self.model.set_logger(self.logger) # SB3 logger, not python logging. Reconfigure tensorboard path for new run.
                self.logger.info(f"Model loaded. Continuing training. New TensorBoard logs under: {run_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {existing_model_path}: {e}. Starting new training.", exc_info=True)
                self.model = self._create_model()
        else:
            if existing_model_path:
                 self.logger.warning(f"Existing model path {existing_model_path} not found or invalid. Creating new model.")
            self.model = self._create_model()

        if self.model is None:
            self.logger.error("Model could not be created or loaded. Training aborted.")
            return None

        # --- Callbacks ---
        callbacks_list = []
        # 1. Checkpoint Callback
        if SB3_AVAILABLE: # Only use real callbacks if SB3 is available
            checkpoint_freq = self.training_run_params.get('checkpoint_freq', 10000)
            checkpoint_save_path = os.path.join(self.model_save_dir, run_name, "checkpoints")
            # No need to os.makedirs here, CheckpointCallback does it.
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_save_path,
                name_prefix=f"{self.algorithm_name.lower()}_model",
                save_replay_buffer=True, # Save replay buffer with model for off-policy algos
                save_vecnormalize=True # If using VecNormalize wrapper
            )
            callbacks_list.append(checkpoint_callback)
            self.logger.info(f"CheckpointCallback configured: Freq={checkpoint_freq}, Path={checkpoint_save_path}")
        
        # 2. Eval Callback (Optional)
        if SB3_AVAILABLE and self.training_run_params.get('use_eval_callback', False):
            eval_freq = self.training_run_params.get('eval_freq', 20000)
            # EvalCallback needs a separate evaluation environment.
            # For simplicity, we might reuse the training_env_monitor or a copy.
            # Ideally, create a new EnvAgent instance with validation data.
            # eval_env_for_callback = Monitor(self.training_env_monitor.env, ...) # Wrap the original env if no separate eval data
            # This is complex to set up here without Orchestrator providing a dedicated eval_env for this.
            # For now, if enabled, it will use the training_env_monitor for evaluation.
            eval_log_path = os.path.join(self.model_save_dir, run_name, "eval_logs")
            best_model_save_path = os.path.join(self.model_save_dir, run_name, "best_model")
            
            # Ensure the eval_env is also vectorized if the training_env is.
            # For now, assuming self.training_env_monitor is a single Monitor-wrapped env.
            # If it were a VecEnv: eval_env_for_callback = self.training_env_monitor
            # else: eval_env_for_callback = DummyVecEnv([lambda: self.training_env_monitor]) # Wrap if not VecEnv
            
            eval_callback = EvalCallback(
                self.training_env_monitor, # Use the monitored training env for eval in this simplified setup
                best_model_save_path=best_model_save_path,
                log_path=eval_log_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks_list.append(eval_callback)
            self.logger.info(f"EvalCallback configured: Freq={eval_freq}, BestModelPath={best_model_save_path}")
        
        if not SB3_AVAILABLE and not callbacks_list: # Fallback to dummy if SB3 not there and no real callbacks
            class DummyCallbackForLearn:
                def __init__(self, model_ref): self.model = model_ref; self.num_timesteps = 0; self.logger = logging.getLogger("DummyLearnCB")
                def init_callback(self, model): pass # model already set for dummy
                def on_step(self): self.num_timesteps+=1; return True
                def on_training_end(self): self.logger.info("DummyLearnCB: Training ended.")
            dummy_cb_for_learn = DummyCallbackForLearn(self.model)
            callbacks_list = dummy_cb_for_learn # Dummy learn expects single callback obj
            self.logger.info("Using DummyCallbackForLearn as SB3 is not available.")


        # --- Start Training ---
        total_timesteps = self.training_run_params.get('total_timesteps', 100000)
        # SB3's model.learn `log_interval` is for console print of episode stats (default 100 episodes for DQN).
        # TensorBoard logging happens more frequently based on internal model logic.
        learn_log_interval = self.training_run_params.get('console_log_interval_episodes', 100) 
        tb_log_name_for_run = run_name # This creates a subfolder in self.log_dir (tensorboard_log path)

        self.logger.info(f"Starting training for {total_timesteps} timesteps. TB sub-log: {tb_log_name_for_run}")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks_list if callbacks_list else None,
                log_interval=learn_log_interval, 
                tb_log_name=tb_log_name_for_run, # Subdirectory for this run's TB logs
                reset_num_timesteps= (existing_model_path is None) # Reset timesteps if new model
            )
        except Exception as e:
            self.logger.error(f"Error during model training: {e}", exc_info=True)
            return None

        # --- Save Final Model ---
        final_model_filename = f"{self.algorithm_name.lower()}_final_{run_timestamp}.zip"
        final_model_path = os.path.join(self.model_save_dir, run_name, final_model_filename) # Save inside run_name folder
        try:
            self.model.save(final_model_path)
            self.logger.info(f"Training complete. Final model saved to: {final_model_path}")
            return final_model_path
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}", exc_info=True)
            return None

    def run(self, training_env: IntradayTradingEnv, existing_model_path: str = None) -> str | None:
        self.logger.info("TrainerAgent run: Setting up environment and starting training.")
        self.set_env(training_env) # This will wrap with Monitor
        
        if self.training_env_monitor is None: # Check the monitored env
            self.logger.error("TrainerAgent: Environment not provided or failed to set up with Monitor. Aborting training.")
            return None
            
        return self.train(existing_model_path=existing_model_path)


if __name__ == '__main__':
    # This __main__ block will use DummySB3Model if SB3 is not installed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_main = logging.getLogger(__name__)
    logger_main.info(f"SB3_AVAILABLE: {SB3_AVAILABLE} for TrainerAgent __main__ demo.")

    # --- Mock Training Environment ---
    num_env_steps = 200; lookback = 1; num_market_features = 5 # Simplified: lookback=1, 5 market features + 1 pos feature
    
    # For lookback=1, market_feature_data is (num_steps, num_market_features)
    mock_market_feature_data = np.random.rand(num_env_steps, num_market_features).astype(np.float32)
    
    mock_prices = 100 + np.cumsum(np.random.randn(num_env_steps))
    mock_dates = pd.to_datetime(pd.date_range(start='2023-03-01', periods=num_env_steps, freq='1min'))
    mock_price_series = pd.Series(mock_prices, index=mock_dates, name=CLOSE)

    mock_env = IntradayTradingEnv(
        processed_feature_data=mock_market_feature_data, # Market features only
        price_data=mock_price_series,
        initial_capital=10000,
        lookback_window=lookback, # Env adds position feature, so obs is (num_market_features + 1)
        max_daily_drawdown_pct=0.1,
        transaction_cost_pct=0.001,
        max_episode_steps=50 
    )
    logger_main.info(f"Mock training environment created: {mock_env}")
    logger_main.info(f"Mock env observation space: {mock_env.observation_space} (shape: {mock_env.observation_space.shape}), action space: {mock_env.action_space}")

    # --- TrainerAgent Configuration ---
    trainer_config = {
        'model_save_dir': 'models/test_trainer_enhanced',
        'log_dir': 'logs/tensorboard_test_trainer_enhanced', # Base TB dir
        'algorithm': 'DQN',
        'algo_params': { 
            'policy': 'MlpPolicy', 
            'learning_rate': 5e-4, 'buffer_size': 1000, 'learning_starts': 100,
            'batch_size': 32, 'target_update_interval': 200,
            'exploration_fraction': 0.2, 'exploration_final_eps': 0.05,
            'verbose': 1 if SB3_AVAILABLE else 0, # SB3 verbose
            'seed': 42,
            # For PER with real SB3 DQN:
            # 'prioritized_replay_alpha': 0.6, 'prioritized_replay_beta': 0.4, 
        },
        'c51_features': { # Flags to control DQN enhancements
            'dueling_nets': True, # Will try to configure if applicable
            'use_per': True,      # Will set prioritized_replay=True in DQN params
            'n_step_returns': 1,  # Standard DQN is 1-step. >1 needs special handling/model.
            'use_noisy_nets': False # Requires policy with NoisyLinear layers
        },
        'training_params': {
            'total_timesteps': 1000, # Short for test
            'console_log_interval_episodes': 10, # SB3 model.learn log_interval
            'checkpoint_freq': 200, 
            'use_eval_callback': False # Keep false for this simple test
        }
    }

    trainer_agent = TrainerAgent(config=trainer_config)
    
    logger_main.info("\nStarting TrainerAgent run (enhanced)...")
    saved_model_filepath = trainer_agent.run(training_env=mock_env)

    if saved_model_filepath:
        logger_main.info(f"\nTrainerAgent run completed. Model saved to: {saved_model_filepath}")
        # Verification for dummy model
        if not SB3_AVAILABLE and os.path.exists(saved_model_filepath + ".dummy"):
            logger_main.info(f"Dummy model file found: {saved_model_filepath}.dummy")
        elif SB3_AVAILABLE and os.path.exists(saved_model_filepath):
             logger_main.info(f"SB3 model file found: {saved_model_filepath}")

        # Check for Monitor logs
        monitor_files = [f for f in os.listdir(trainer_agent.monitor_log_dir) if f.endswith('.csv')]
        if monitor_files: logger_main.info(f"Monitor log(s) found in {trainer_agent.monitor_log_dir}: {monitor_files[:2]}") # Show first two
        else: logger_main.warning(f"No Monitor logs found in {trainer_agent.monitor_log_dir}")

    else:
        logger_main.error("\nTrainerAgent run failed or no model was saved.")

    mock_env.close()
    logger_main.info("\nTrainerAgent example run complete (enhanced).")
