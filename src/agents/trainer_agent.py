# src/agents/trainer_agent.py
import os
import logging
from datetime import datetime
# import stable_baselines3 as sb3 # pip install stable-baselines3[extra]
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise # For continuous actions

# For C51 specifically (Categorical DQN)
# from stable_baselines3 import C51 # C51 is part of DQN in SB3 v1.x, separate in SB3 Contrib or built on DQN
# SB3's core DQN can be configured for C51-like properties (categorical output)
# However, a true C51 with distributional RL might require sb3_contrib or custom implementation.
# For this skeleton, let's assume we are using a standard DQN and will note where C51 specifics apply.
# If a dedicated C51 is available (e.g. from a contrib package), that would be used.
# For SB3 v2.x, C51 is not directly in core. Let's use DQN as a base.
# If we must use C51, we'd need to confirm its availability in the chosen SB3 version or use sb3_contrib.
# For now, let's use DQN and note where C51 features like Dueling nets, PER, n-step, noisy nets would be configured.
# from stable_baselines3 import DQN # Standard DQN

from .base_agent import BaseAgent
from src.gym_env.intraday_trading_env import IntradayTradingEnv
# from ..utils.config_loader import load_model_params # Example

class TrainerAgent(BaseAgent):
    """
    TrainerAgent is responsible for:
    1. Instantiating the RL algorithm (e.g., C51, DQN) from Stable-Baselines3.
    2. Training the RL model using the environment provided by EnvAgent.
    3. Saving model checkpoints and TensorBoard logs.
    4. Optionally, handling hyperparameter tuning (e.g., with Optuna).
    """
    def __init__(self, config: dict, training_env: IntradayTradingEnv = None):
        """
        Initializes the TrainerAgent.

        Args:
            config (dict): Configuration dictionary. Expected keys:
                           'model_save_dir': Path to save trained models.
                           'log_dir': Path for TensorBoard logs.
                           'algorithm': Name of the algorithm (e.g., 'C51', 'DQN').
                           'algo_params': Dictionary of parameters for the SB3 algorithm.
                                          (e.g., learning_rate, buffer_size, policy_kwargs for dueling).
                           'training_params': {'total_timesteps', 'log_interval', 'checkpoint_freq', 'eval_freq'}.
                           'use_per': bool (Prioritized Experience Replay)
                           'n_step_returns': int or None
                           'use_noisy_nets': bool
            training_env (IntradayTradingEnv, optional): Pre-initialized training environment.
                                                        If None, it's expected to be set later via `set_env`.
        """
        super().__init__(agent_name="TrainerAgent", config=config)
        
        self.model_save_dir = self.config.get('model_save_dir', 'models/')
        self.log_dir = self.config.get('log_dir', 'logs/tensorboard/')
        self.algorithm_name = self.config.get('algorithm', 'DQN') # Default to DQN
        self.algo_params = self.config.get('algo_params', {})
        self.training_params = self.config.get('training_params', {})

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = None
        self.training_env = None
        if training_env:
            self.set_env(training_env)

        self.logger.info(f"TrainerAgent initialized for algorithm: {self.algorithm_name}")
        self.logger.info(f"Models will be saved to: {self.model_save_dir}")
        self.logger.info(f"TensorBoard logs will be saved to: {self.log_dir}")

    def set_env(self, env: IntradayTradingEnv):
        """
        Sets the training environment. The environment should be wrapped with Monitor for SB3 logging.
        """
        if not isinstance(env, IntradayTradingEnv):
            self.logger.error("Invalid environment type provided to TrainerAgent.")
            raise ValueError("Environment must be an instance of IntradayTradingEnv.")
        
        # It's good practice to wrap the environment with Monitor for SB3 to log episode rewards, lengths etc.
        # log_subdir = os.path.join(self.log_dir, f"{self.algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_monitor")
        # os.makedirs(log_subdir, exist_ok=True)
        # self.training_env = Monitor(env, filename=log_subdir)
        # For simplicity in the skeleton, we'll use the raw env first. Monitor is crucial for full SB3 integration.
        self.training_env = env
        self.logger.info("Training environment set for TrainerAgent.")


    def _create_model(self):
        """
        Creates an RL model instance based on the configuration.
        This is where C51 specific configurations (Dueling, PER, NoisyNets, n-step) would be applied.
        """
        if self.training_env is None:
            self.logger.error("Cannot create model: Training environment is not set.")
            return None

        policy_kwargs = self.algo_params.get('policy_kwargs', {})
        
        # --- C51 / Advanced DQN Features ---
        # Dueling Networks: Often enabled via policy_kwargs in SB3's DQN.
        # e.g., policy_kwargs=dict(net_arch=[...], features_extractor=..., dueling=True) -> This depends on SB3 version.
        # SB3's DQN has `dueling_network` in `policy_kwargs` for some policies.
        # For C51, 'dueling' might be a direct parameter or within policy_kwargs.
        if self.config.get('dueling_nets', False): # Assuming a config flag 'dueling_nets'
            # This is highly dependent on how the chosen SB3 model (DQN or a specific C51) implements it.
            # For SB3's DQN, you might specify a policy that supports dueling or pass it in policy_kwargs.
            # policy_kwargs['dueling'] = True # Hypothetical, check SB3 docs for chosen model.
            # For a MlpPolicy in DQN, dueling is not a direct toggle. It's part of specific architectures.
            # If using CnnPolicy, it might have a dueling option.
            # Let's assume policy_kwargs = dict(net_arch=dict(qf=[64,64], vf=[64,64])) for dueling if supported.
            # Or, if the policy itself is 'DuelingMlpPolicy', etc.
            self.logger.info("Dueling networks requested (configuration depends on SB3 model).")
            # For SB3 DQN, there isn't a simple `dueling=True`. It's structural.
            # If C51 implies Dueling by default or has a specific policy, that's different.

        # Prioritized Experience Replay (PER):
        # SB3's DQN has `prioritized_replay` boolean parameter.
        if self.config.get('use_per', False):
            if 'prioritized_replay' in self.algo_params: # Check if already set
                 self.algo_params['prioritized_replay'] = True
                 self.logger.info("Prioritized Experience Replay (PER) enabled via algo_params.")
            # else: # Add it if not, assuming DQN model
            #    if self.algorithm_name.upper() == 'DQN': # Only for DQN like models
            #        self.algo_params['prioritized_replay'] = True
            #        self.logger.info("Prioritized Experience Replay (PER) enabled for DQN.")


        # Noisy Nets:
        # For SB3's DQN, this is typically enabled by setting `policy_kwargs=dict(noisy_net=True)` or similar.
        # Or by choosing a policy that inherently uses noisy nets.
        if self.config.get('use_noisy_nets', False):
            # policy_kwargs['noisy_net'] = True # Hypothetical for SB3 DQN
            self.logger.info("Noisy Nets requested (configuration depends on SB3 model).")


        # N-step Returns:
        # For SB3's DQN, this is `n_steps_target_update` or similar, or part of how target Q is calculated.
        # More directly, `train_freq` and `gradient_steps` interact with how often updates happen.
        # True n-step learning is often a direct parameter if supported.
        # SB3 DQN's `target_update_interval` relates to target network sync, not directly n-step returns.
        # `n_episodes_rollout` or similar in OffPolicyAlgorithm controls data collection.
        # For n-step returns, one might need to look at `gamma` and how returns are bootstrapped.
        # Some algorithms (like A2C/PPO) inherently use n-step returns. For DQN/C51, it's an extension.
        # If `n_step_returns` (e.g., 3 or 5) is a parameter for the chosen model:
        n_step = self.config.get('n_step_returns')
        if n_step and 'n_step_returns' not in self.algo_params: # Assuming a param name
             # self.algo_params['n_step_returns'] = n_step # Hypothetical
             self.logger.info(f"{n_step}-step returns requested (configuration depends on SB3 model).")


        # --- Model Instantiation ---
        # This part needs actual SB3 imports and calls.
        self.logger.info(f"Attempting to create model: {self.algorithm_name} with params: {self.algo_params}")
        self.logger.info(f"Policy kwargs: {policy_kwargs}")

        # Placeholder for actual model creation
        # Example for DQN:
        # if self.algorithm_name.upper() == 'DQN':
        #     try:
        #         from stable_baselines3 import DQN
        #         self.model = DQN(
        #             policy=self.algo_params.get('policy', 'MlpPolicy'), # e.g., 'MlpPolicy', 'CnnPolicy'
        #             env=self.training_env,
        #             learning_rate=self.algo_params.get('learning_rate', 1e-4),
        #             buffer_size=self.algo_params.get('buffer_size', 50000),
        #             learning_starts=self.algo_params.get('learning_starts', 1000),
        #             batch_size=self.algo_params.get('batch_size', 32),
        #             tau=self.algo_params.get('tau', 1.0),
        #             gamma=self.algo_params.get('gamma', 0.99),
        #             train_freq=self.algo_params.get('train_freq', 4), # (4, "step")
        #             gradient_steps=self.algo_params.get('gradient_steps', 1),
        #             # replay_buffer_class= # For PER, might need custom or from sb3_contrib
        #             # replay_buffer_kwargs= # For PER settings
        #             optimize_memory_usage=self.algo_params.get('optimize_memory_usage', False),
        #             target_update_interval=self.algo_params.get('target_update_interval', 10000),
        #             exploration_fraction=self.algo_params.get('exploration_fraction', 0.1),
        #             exploration_initial_eps=self.algo_params.get('exploration_initial_eps', 1.0),
        #             exploration_final_eps=self.algo_params.get('exploration_final_eps', 0.05),
        #             # prioritized_replay=self.algo_params.get('prioritized_replay', False), # If using PER
        #             policy_kwargs=policy_kwargs if policy_kwargs else None,
        #             tensorboard_log=self.log_dir,
        #             verbose=self.algo_params.get('verbose', 1),
        #             seed=self.algo_params.get('seed', None)
        #         )
        #         self.logger.info(f"DQN model created successfully.")
        #     except ImportError:
        #         self.logger.error("Stable Baselines3 DQN not found. Please install it.")
        #         return None
        #     except Exception as e:
        #         self.logger.error(f"Error creating DQN model: {e}", exc_info=True)
        #         return None

        # elif self.algorithm_name.upper() == 'C51':
        #     # This assumes C51 is available, possibly from sb3_contrib or a specific SB3 version.
        #     # The parameters for C51 would be different from DQN.
        #     # E.g., number of atoms, v_min, v_max for the distribution.
        #     # policy_kwargs for C51 might include `dueling=True`, `noisy=True` more directly.
        #     self.logger.warning("C51 model creation is a placeholder. Requires specific C51 implementation.")
        #     # Example structure if C51 was like DQN:
        #     # self.model = C51('MlpPolicy', self.training_env, policy_kwargs=policy_kwargs, verbose=1, ...)
        #     pass

        # else:
        #     self.logger.error(f"Unsupported algorithm: {self.algorithm_name}")
        #     return None
        
        # SKELETON: For now, we won't instantiate a real SB3 model to avoid mandatory dependency.
        # We'll simulate its creation.
        if self.algorithm_name.upper() in ['DQN', 'C51']: # Assuming C51 is handled similarly
            self.logger.info(f"Skeleton: Successfully created a placeholder for {self.algorithm_name} model.")
            # Simulate a model object with a learn() method and save/load
            class DummySB3Model:
                def __init__(self, name, env, tb_log_dir):
                    self.name = name
                    self.env = env
                    self.tb_log_dir = tb_log_dir
                    self.logger = logging.getLogger(f"RLTradingPlatform.DummyModel.{name}")

                def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="run"):
                    self.logger.info(f"DummyModel ({self.name}) starting 'learn' for {total_timesteps} timesteps.")
                    self.logger.info(f"TensorBoard log name: {tb_log_name}, Log interval: {log_interval}")
                    if callback:
                        self.logger.info(f"Callbacks: {callback}")
                        # Simulate callback interaction
                        if hasattr(callback, 'on_training_start'): callback.on_training_start(locals(), globals())
                        for i in range(0, int(total_timesteps), 1000): # Simulate steps
                            if hasattr(callback, 'on_step'): 
                                if not callback.on_step(): break # Callback signals to stop
                            if (i/1000) % log_interval == 0:
                                self.logger.info(f"Dummy learn: Timestep {i+1000}/{total_timesteps}")
                        if hasattr(callback, 'on_training_end'): callback.on_training_end()
                    self.logger.info(f"DummyModel ({self.name}) 'learn' complete.")

                def save(self, path):
                    self.logger.info(f"DummyModel ({self.name}) 'save' called for path: {path}. (Simulated save)")
                    # Create a dummy file to signify saving
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path + ".dummy", "w") as f: f.write("dummy model data")
                
                @classmethod
                def load(cls, path, env=None):
                    logger = logging.getLogger(f"RLTradingPlatform.DummyModel.Load")
                    logger.info(f"DummyModel 'load' called for path: {path}. (Simulated load)")
                    # Check if dummy file exists
                    if os.path.exists(path + ".dummy"):
                        # Return a new instance, mimicking SB3 load
                        # The name of the model isn't stored in the dummy file, so we use a generic one
                        model_name = os.path.basename(path).replace('.zip', '').replace('.dummy', '')
                        loaded_model = cls(name=f"Loaded_{model_name}", env=env, tb_log_dir=None)
                        logger.info(f"DummyModel loaded successfully from {path}")
                        return loaded_model
                    else:
                        logger.error(f"Dummy model file {path}.dummy not found.")
                        raise FileNotFoundError(f"No dummy model found at {path}.dummy")


            self.model = DummySB3Model(self.algorithm_name, self.training_env, self.log_dir)
            return self.model
        else:
            self.logger.error(f"Unsupported algorithm in skeleton: {self.algorithm_name}")
            return None


    def train(self, existing_model_path: str = None) -> str | None:
        """
        Trains the RL model.

        Args:
            existing_model_path (str, optional): Path to an existing model to continue training.
                                                 If None, a new model is created.

        Returns:
            str or None: Path to the saved trained model, or None if training failed.
        """
        if self.training_env is None:
            self.logger.error("Cannot train: Training environment is not set.")
            return None

        if existing_model_path and os.path.exists(existing_model_path):
            self.logger.info(f"Loading existing model from: {existing_model_path}")
            # self.model = self.algorithm_class.load(existing_model_path, env=self.training_env)
            # Placeholder for load:
            try:
                self.model = DummySB3Model.load(existing_model_path, env=self.training_env) # Use the dummy class method
                self.logger.info(f"Model loaded. Continuing training.")
            except Exception as e:
                self.logger.error(f"Failed to load model from {existing_model_path}: {e}. Starting new training.")
                self.model = self._create_model()
        else:
            if existing_model_path:
                 self.logger.warning(f"Existing model path {existing_model_path} not found. Creating new model.")
            self.model = self._create_model()

        if self.model is None:
            self.logger.error("Model could not be created or loaded. Training aborted.")
            return None

        # --- Callbacks ---
        # 1. Checkpoint Callback: Saves the model periodically.
        checkpoint_freq = self.training_params.get('checkpoint_freq', 10000) # Timesteps
        run_name = f"{self.algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_save_path = os.path.join(self.model_save_dir, run_name, "checkpoints")
        # checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_save_path,
        #                                         name_prefix='rl_model')
        # Placeholder callback:
        class DummyCheckpointCallback: # Mimics CheckpointCallback
            def __init__(self, save_freq, save_path, name_prefix):
                self.save_freq = save_freq
                self.save_path = save_path
                self.name_prefix = name_prefix
                self.logger = logging.getLogger("RLTradingPlatform.DummyCheckpointCallback")
                self.num_timesteps = 0
                os.makedirs(self.save_path, exist_ok=True)
            def on_step(self) -> bool:
                self.num_timesteps +=1
                if self.num_timesteps % self.save_freq == 0:
                    model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
                    self.logger.info(f"Dummy Checkpoint: Saving model to {model_path} (simulated)")
                    # In real SB3, self.model.save(model_path) would be here or implicitly handled
                    # For dummy model, we can call its save:
                    if hasattr(self.model, 'save'): self.model.save(model_path) 
                return True # Continue training
            def __str__(self): return f"DummyCheckpointCallback(save_freq={self.save_freq})"

        checkpoint_callback = DummyCheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_save_path, name_prefix='rl_model')
        checkpoint_callback.model = self.model # SB3 callbacks get self.model from training loop

        # TODO: 2. Eval Callback: Evaluates the model on a separate environment periodically.
        # eval_env = Monitor(self.training_env) # Ideally a separate validation env
        # eval_freq = self.training_params.get('eval_freq', 20000)
        # eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(self.model_save_dir, run_name, 'best_model'),
        #                              log_path=os.path.join(self.model_save_dir, run_name, 'eval_logs'),
        #                              eval_freq=eval_freq, deterministic=True, render=False)
        # For skeleton, we'll just use checkpoint callback.
        # callbacks_list = [checkpoint_callback, eval_callback]
        callbacks_list = [checkpoint_callback] # Simplified for skeleton
        self.logger.info(f"Using callbacks: {callbacks_list}")


        # --- Start Training ---
        total_timesteps = self.training_params.get('total_timesteps', 100000)
        log_interval = self.training_params.get('log_interval', 100) # For TensorBoard logging (episodes)
        tb_log_name = f"{self.algorithm_name}_{run_name}"

        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks_list, # Pass the list of callbacks
                log_interval=log_interval, # This is for console logging of episode stats by SB3
                tb_log_name=tb_log_name # This is the sub-directory within self.log_dir for this run
            )
        except Exception as e:
            self.logger.error(f"Error during model training: {e}", exc_info=True)
            return None

        # --- Save Final Model ---
        final_model_filename = f"{self.algorithm_name}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        final_model_path = os.path.join(self.model_save_dir, final_model_filename)
        try:
            self.model.save(final_model_path)
            self.logger.info(f"Training complete. Final model saved to: {final_model_path}")
            return final_model_path
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}", exc_info=True)
            return None

    def run(self, training_env: IntradayTradingEnv, existing_model_path: str = None) -> str | None:
        """
        Main method for TrainerAgent: sets environment and starts training.

        Args:
            training_env (IntradayTradingEnv): The environment to train on.
            existing_model_path (str, optional): Path to continue training from.

        Returns:
            str or None: Path to the saved trained model, or None if failed.
        """
        self.logger.info("TrainerAgent run: Setting up environment and starting training.")
        self.set_env(training_env)
        
        if self.training_env is None:
            self.logger.error("TrainerAgent: Environment not provided or failed to set. Aborting training.")
            return None
            
        return self.train(existing_model_path=existing_model_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Training Environment (from EnvAgent or IntradayTradingEnv directly) ---
    # This would typically be provided by EnvAgent.
    num_env_steps = 200
    lookback = 5 
    num_market_features = 4
    if lookback > 1:
        mock_feature_data = np.random.rand(num_env_steps, lookback, num_market_features).astype(np.float32)
    else:
        mock_feature_data = np.random.rand(num_env_steps, num_market_features).astype(np.float32)
    mock_prices = 100 + np.cumsum(np.random.randn(num_env_steps))
    mock_dates = pd.to_datetime(pd.date_range(start='2023-03-01', periods=num_env_steps, freq='1min'))
    mock_price_series = pd.Series(mock_prices, index=mock_dates, name='close')

    mock_env = IntradayTradingEnv(
        processed_feature_data=mock_feature_data,
        price_data=mock_price_series,
        initial_capital=10000,
        lookback_window=lookback,
        max_daily_drawdown_pct=0.1, # Higher for faster testing if it hits
        transaction_cost_pct=0.001,
        max_episode_steps=50 # Short episodes for testing
    )
    print(f"Mock training environment created: {mock_env}")
    print(f"Mock env observation space: {mock_env.observation_space}, action space: {mock_env.action_space}")


    # --- TrainerAgent Configuration ---
    trainer_config = {
        'model_save_dir': 'models/test_trainer',
        'log_dir': 'logs/tensorboard_test_trainer',
        'algorithm': 'DQN', # Could be 'C51' if available
        'algo_params': { # Parameters for the SB3 model (DQN example)
            'policy': 'MlpPolicy', # Assuming features are flat or handled by MlpPolicy
            'learning_rate': 5e-4,
            'buffer_size': 10000, # Smaller for quick test
            'learning_starts': 200, # Start learning quickly
            'batch_size': 64,
            'target_update_interval': 500,
            'exploration_fraction': 0.3,
            'exploration_final_eps': 0.05,
            'verbose': 0, # Less verbose for dummy model
            'seed': 42,
            # 'prioritized_replay': True, # Example for PER
        },
        'training_params': {
            'total_timesteps': 5000, # Short training for test
            'log_interval': 10, # Log every 10 episodes (dummy model logs per N steps)
            'checkpoint_freq': 1000, # Save checkpoint every 1000 steps
            # 'eval_freq': 2000, # Evaluate every 2000 steps
        },
        # 'use_per': True, # Example flag for PER
        # 'dueling_nets': True, # Example flag for Dueling
        # 'n_step_returns': 3, # Example for n-step
    }

    # --- Initialize and Run TrainerAgent ---
    trainer_agent = TrainerAgent(config=trainer_config)
    
    # The "run" method sets the env and starts training
    print("\nStarting TrainerAgent run...")
    saved_model_filepath = trainer_agent.run(training_env=mock_env)

    if saved_model_filepath:
        print(f"\nTrainerAgent run completed. Model saved to: {saved_model_filepath}")
        # Verify dummy model file exists
        if os.path.exists(saved_model_filepath + ".dummy"):
            print(f"Dummy model file found at {saved_model_filepath}.dummy")
        
        # Verify checkpoint files (simulated)
        run_name_part = [d for d in os.listdir(os.path.join(trainer_config['model_save_dir'])) if d.startswith(trainer_config['algorithm'])][0]
        checkpoint_dir = os.path.join(trainer_config['model_save_dir'], run_name_part, "checkpoints")
        if os.path.exists(checkpoint_dir):
            print(f"Checkpoints directory found: {checkpoint_dir}")
            ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip.dummy')]
            if ckpts:
                print(f"Found dummy checkpoint files: {ckpts}")
            else:
                print("No dummy checkpoint files found.")
        else:
            print(f"Checkpoints directory NOT found: {checkpoint_dir}")
            
        # Verify TensorBoard log directory (simulated structure)
        tb_main_log_dir = trainer_config['log_dir']
        tb_run_dirs = [d for d in os.listdir(tb_main_log_dir) if d.startswith(trainer_config['algorithm'])]
        if tb_run_dirs:
            print(f"TensorBoard run directory found: {os.path.join(tb_main_log_dir, tb_run_dirs[0])}")
        else:
            print(f"TensorBoard run directory NOT found under {tb_main_log_dir}")

    else:
        print("\nTrainerAgent run failed or no model was saved.")

    mock_env.close()
    print("\nTrainerAgent example run complete.")
