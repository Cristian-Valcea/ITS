# src/agents/trainer_agent.py
import os
import logging
from datetime import datetime

# Attempt to import Stable-Baselines3 components
try:
    from stable_baselines3 import DQN # Using DQN as the base for C51-like features
    # from stable_baselines3.dqn import CnnPolicy, MlpPolicy # Policies
    # from sb3_contrib import QRDQN, C51 # If using sb3_contrib for true C51 or QRDQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    # For PER, SB3's DQN has built-in support via a parameter.
    # If a custom ReplayBuffer is needed:
    # from stable_baselines3.common.buffers import ReplayBuffer, PrioritizedReplayBuffer # (PrioritizedReplayBuffer might be in contrib or custom)
    SB3_AVAILABLE = True
    SB3_MODEL_CLASSES = {
        'DQN': DQN,
        # Add other SB3 models here if supported, e.g.
        # 'A2C': A2C,
        # 'PPO': PPO,
        # 'C51': C51, # If using from sb3_contrib
        # 'QRDQN': QRDQN, # If using from sb3_contrib
    }
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("Stable-Baselines3 components not found. TrainerAgent will use dummy implementations.")
    # Define dummy classes if SB3 is not available, to allow skeleton to run
    class DummySB3Model:
        def __init__(self, policy, env, **kwargs): self.logger = logging.getLogger("DummySB3Model"); self.env = env; self.kwargs = kwargs; self.num_timesteps = 0 # Added num_timesteps for predict
        def predict(self, observation, state=None, episode_start=None, deterministic=False): # Added predict
            self.logger.info(f"Dummy predict called with observation shape: {observation.shape if hasattr(observation, 'shape') else 'N/A'}, deterministic: {deterministic}")
            # Return a dummy action (e.g., random action from env space if env is available)
            if self.env and hasattr(self.env, 'action_space'):
                action = self.env.action_space.sample()
                self.logger.info(f"Dummy predict returning action: {action}")
                return action, None # action, state (None for SB3)
            self.logger.info(f"Dummy predict returning default action: 0")
            return 0, None # Default dummy action if no env
        def learn(self, total_timesteps, callback=None, tb_log_name="DummyRun", **kwargs):
            self.logger.info(f"Dummy learn for {total_timesteps} timesteps. TB log: {tb_log_name}. Callbacks: {callback}")
            if callback: callback.init_callback(self) # Mimic SB3 callback init
            for i in range(0, int(total_timesteps), 1000): # Simulate steps
                if callback:
                    if hasattr(callback, 'on_step'):
                         if not callback.on_step(): break # Callback signals to stop
                if (i/1000) % self.kwargs.get('log_interval', 100) == 0:
                    self.logger.info(f"Dummy learn: Timestep {i+1000}/{total_timesteps}")
            if callback and hasattr(callback, 'on_training_end'): callback.on_training_end()

        def save(self, path): self.logger.info(f"Dummy save model to {path}"); os.makedirs(os.path.dirname(path), exist_ok=True); open(path + ".dummy", "w").write("dummy")
        @classmethod
        def load(cls, path, env=None, **kwargs): logger = logging.getLogger("DummySB3Model"); logger.info(f"Dummy load model from {path}"); return cls(None,env) # Simplified
    
    class Monitor: # Dummy Monitor
        def __init__(self, env, filename=None, **kwargs): self.env = env; self.logger = logging.getLogger("DummyMonitor") ; self.logger.info(f"Dummy Monitor initialized for env. Log: {filename}")
        def __getattr__(self, name): return getattr(self.env, name) # Pass through attributes
        def reset(self, **kwargs): self.logger.debug("Dummy Monitor reset called"); return self.env.reset(**kwargs)
        def step(self, action): self.logger.debug(f"Dummy Monitor step with action {action}"); return self.env.step(action)
        def close(self): self.env.close()

    class CheckpointCallback: # Dummy CheckpointCallback
         def __init__(self, save_freq, save_path, name_prefix="rl_model", **kwargs): self.sf=save_freq; self.sp=save_path; self.np=name_prefix; self.logger=logging.getLogger("DummyCheckpointCb"); self.num_timesteps=0; os.makedirs(save_path, exist_ok=True)
         def init_callback(self, model): self.model = model # SB3 style
         def _on_step(self) -> bool: self.num_timesteps+=1; self.logger.debug(f"Dummy CB step {self.num_timesteps}"); return True
         def on_step(self) -> bool: # For the dummy model's loop
             self.num_timesteps+=1
             if self.num_timesteps % self.sf == 0: model_path=os.path.join(self.sp,f"{self.np}_{self.num_timesteps}_steps.zip"); self.logger.info(f"Dummy Checkpoint: Saving model to {model_path}"); self.model.save(model_path)
             return True
    class EvalCallback: # Dummy EvalCallback
        def __init__(self, eval_env, **kwargs): self.logger = logging.getLogger("DummyEvalCallback"); self.logger.info("Dummy EvalCallback initialized.")
        def init_callback(self, model): self.model = model
        def _on_step(self) -> bool: return True
        def on_step(self) -> bool: return True # Simplified for dummy loop

    # Assign dummy if real ones not available
    if not SB3_AVAILABLE:
        DQN_dummy_assign = DummySB3Model # Use a different name to avoid confusion with potential later import
        # MlpPolicy = "MlpPolicy" # Keep as string for DummySB3Model
        # CnnPolicy = "CnnPolicy"
        SB3_MODEL_CLASSES = { # Ensure SB3_MODEL_CLASSES is defined in the else block too
            'DQN': DQN_dummy_assign,
        }


from .base_agent import BaseAgent
# Ensure IntradayTradingEnv is available for type hinting even if SB3 is not
if SB3_AVAILABLE:
    from src.gym_env.intraday_trading_env import IntradayTradingEnv
else: # Define a placeholder if gym_env might also be missing or for type hinting consistency
    class IntradayTradingEnv: pass


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
