# src/agents/orchestrator_agent.py
import os
import logging
import yaml # pip install PyYAML
from datetime import datetime

# Import other specific agent classes
from .data_agent import DataAgent
from .feature_agent import FeatureAgent
from .env_agent import EnvAgent
from .trainer_agent import TrainerAgent
from .evaluator_agent import EvaluatorAgent
from .risk_agent import RiskAgent # For live trading mode

# BaseAgent might not be strictly needed by Orchestrator itself, unless it also inherits.
# from .base_agent import BaseAgent

class OrchestratorAgent: # Not inheriting BaseAgent for this skeleton, could if needed
    """
    OrchestratorAgent is responsible for:
    1. Loading overall configuration for the platform.
    2. Initializing and wiring together all other specialized agents.
    3. Managing the end-to-end pipeline for:
        - Data fetching and processing.
        - RL model training.
        - Model evaluation (including walk-forward validation).
        - Live trading execution (future capability).
    4. Scheduling periodic tasks (e.g., weekly retraining and re-evaluation).
    """
    def __init__(self, main_config_path: str, model_params_path: str, risk_limits_path: str):
        """
        Initializes the OrchestratorAgent.

        Args:
            main_config_path (str): Path to the main YAML configuration file.
            model_params_path (str): Path to the model hyperparameters YAML configuration file.
            risk_limits_path (str): Path to the risk limits YAML configuration file.
        """
        self.logger = logging.getLogger("RLTradingPlatform.OrchestratorAgent")
        self.logger.info("Initializing OrchestratorAgent...")

        self.main_config = self._load_yaml_config(main_config_path, "Main Config")
        self.model_params_config = self._load_yaml_config(model_params_path, "Model Params Config")
        self.risk_limits_config = self._load_yaml_config(risk_limits_path, "Risk Limits Config")

        if not all([self.main_config, self.model_params_config, self.risk_limits_config]):
            self.logger.error("One or more configuration files failed to load. Orchestrator cannot proceed.")
            raise ValueError("Configuration loading failed.")

        # --- Initialize other agents based on loaded configurations ---
        # Paths from main_config are used by agents (data_dir, model_dir, etc.)
        # Specific algo params from model_params_config go to TrainerAgent.
        # Risk limits from risk_limits_config go to RiskAgent and potentially EnvAgent.

        # DataAgent config: data_dir_raw, ibkr_conn (if used)
        data_agent_cfg = {
            'data_dir_raw': self.main_config.get('paths', {}).get('data_dir_raw', 'data/raw/'),
            'ibkr_conn': self.main_config.get('ibkr_connection', None) # Optional
        }
        self.data_agent = DataAgent(config=data_agent_cfg)

        # FeatureAgent config: data_dir_processed, scalers_dir, feature computation params
        feature_agent_cfg = {
            'data_dir_processed': self.main_config.get('paths', {}).get('data_dir_processed', 'data/processed/'),
            'scalers_dir': self.main_config.get('paths', {}).get('scalers_dir', 'data/scalers/'),
            **self.main_config.get('feature_engineering', {}) # RSI, EMA, VWAP, time_features settings
        }
        self.feature_agent = FeatureAgent(config=feature_agent_cfg)

        # EnvAgent config: build the complete 'env_config' sub-dictionary for IntradayTradingEnv
        env_specific_params = self.main_config.get('environment', {})
        env_agent_env_cfg = {
            'initial_capital': env_specific_params.get('initial_capital', 100000.0),
            'transaction_cost_pct': env_specific_params.get('transaction_cost_pct', 0.001),
            'reward_scaling': env_specific_params.get('reward_scaling', 1.0),
            'max_episode_steps': env_specific_params.get('max_episode_steps', None),
            'log_trades': env_specific_params.get('log_trades_in_env', True),
            'lookback_window': self.main_config.get('feature_engineering', {}).get('lookback_window', 1),
            'max_daily_drawdown_pct': self.risk_limits_config.get('max_daily_drawdown_pct', 0.02),
            'hourly_turnover_cap': self.risk_limits_config.get('max_hourly_turnover_ratio', 5.0),
            'turnover_penalty_factor': self.risk_limits_config.get('env_turnover_penalty_factor', 0.01) 
        }
        # EnvAgent config dictionary itself (passed to EnvAgent constructor)
        env_agent_cfg = {'env_config': env_agent_env_cfg}
        self.env_agent = EnvAgent(config=env_agent_cfg)

        # TrainerAgent config
        trainer_agent_cfg = {
            'model_save_dir': self.main_config.get('paths', {}).get('model_save_dir', 'models/'),
            'log_dir': self.main_config.get('paths', {}).get('tensorboard_log_dir', 'logs/tensorboard/'),
            'algorithm': self.model_params_config.get('algorithm_name', 'DQN'),
            'algo_params': self.model_params_config.get('algorithm_params', {}), # For SB3 model
            'c51_features': self.model_params_config.get('c51_features', {}),   # For specific feature flags like PER, Dueling
            'training_params': self.main_config.get('training', {}),            # For learn loop (total_timesteps, etc.)
        }
        self.trainer_agent = TrainerAgent(config=trainer_agent_cfg)

        # EvaluatorAgent config
        eval_agent_cfg = {
            'reports_dir': self.main_config.get('paths', {}).get('reports_dir', 'reports/'),
            'eval_metrics': self.main_config.get('evaluation', {}).get('metrics', ['sharpe', 'max_drawdown'])
        }
        self.evaluator_agent = EvaluatorAgent(config=eval_agent_cfg)

        # RiskAgent config
        self.risk_agent = RiskAgent(config=self.risk_limits_config)
        
        self.logger.info("All specialized agents initialized by Orchestrator with refined configurations.")


    def _load_yaml_config(self, config_path: str, config_name: str) -> dict:
        """Loads a YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.logger.info(f"{config_name} loaded successfully from {config_path}")
            if cfg is None: # Handle empty YAML file case
                self.logger.warning(f"{config_name} file at {config_path} is empty. Returning empty dict.")
                return {}
            return cfg
        except FileNotFoundError:
            self.logger.error(f"{config_name} file not found at {config_path}. Critical error.")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing {config_name} YAML file {config_path}: {e}. Critical error.")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading {config_name} from {config_path}: {e}. Critical error.")
            raise

    def run_training_pipeline(self, symbol: str, start_date: str, end_date: str, interval: str,
                              use_cached_data: bool = False, # No use_cached_features yet
                              continue_from_model: str = None,
                              run_evaluation_after_train: bool = True, # New flag
                              eval_start_date: str = None, # For post-train eval
                              eval_end_date: str = None,   # For post-train eval
                              eval_interval: str = None    # For post-train eval
                              ) -> str | None:
        """
        Executes the full data processing and model training pipeline.

        Args:
            symbol (str): Stock symbol to process.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD HH:MM:SS for IBKR).
            interval (str): Data interval (e.g., "1 min", "5 mins").
            use_cached_data (bool): If True, tries to load raw data from DataAgent's cache.
            use_cached_features (bool): If True, tries to load processed features. (Not fully implemented in FA skeleton)
            continue_from_model (str, optional): Path to a pre-trained model to continue training.

        Returns:
            str or None: Path to the final trained model, or None if pipeline failed.
        """
        self.logger.info(f"--- Starting Training Pipeline for {symbol} ({start_date} to {end_date}, {interval}) ---")

        # --- 1. Data Fetching (DataAgent) ---
        # DataAgent's run needs symbol, end_date (for IBKR), duration_str, interval.
        # We need to derive duration_str from start_date and end_date for DataAgent.
        # This is simplified in DataAgent's skeleton; assuming it can handle these.
        # For now, DataAgent.run uses start_date for cache naming and end_date+duration_str for fetch.
        # Let's use a placeholder duration_str or assume DataAgent figures it out.
        # A more robust Orchestrator would calculate this.
        # Example: duration_str = f"{(datetime.strptime(end_date.split(' ')[0], '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days +1} D"
        # This needs careful handling of IBKR limits.
        # For skeleton, let's assume a fixed duration or pass it via main_config.training.data_duration_for_fetch
        data_duration_str = self.main_config.get('training',{}).get('data_duration_for_fetch', "90 D") # e.g. 90 days of data ending at end_date

        raw_bars_df = self.data_agent.run(
            symbol=symbol,
            start_date=start_date, # Primarily for consistent cache naming, if DataAgent uses it
            end_date=end_date,     # Crucial for IBKR fetch end point
            interval=interval,
            duration_str=data_duration_str, # How far back from end_date to fetch
            force_fetch=not use_cached_data # If use_cached_data is True, force_fetch is False
        )
        if raw_bars_df is None or raw_bars_df.empty:
            self.logger.error("Data fetching failed. Training pipeline aborted.")
            return None
        self.logger.info(f"DataAgent returned raw data of shape: {raw_bars_df.shape}")

        # --- 2. Feature Engineering (FeatureAgent) ---
        # FeatureAgent needs raw_bars_df, symbol, and whether to fit_scaler (True for training).
        # Date/interval strings are for processed data cache naming.
        _df_features, feature_sequences, prices_for_env = self.feature_agent.run(
            raw_data_df=raw_bars_df,
            symbol=symbol,
            cache_processed_data=True, # Cache the features
            fit_scaler=True,           # Fit a new scaler during training
            start_date_str=start_date, # For cache naming
            end_date_str=end_date.split(' ')[0], # For cache naming
            interval_str=interval
        )
        if feature_sequences is None or prices_for_env is None:
            self.logger.error("Feature engineering failed. Training pipeline aborted.")
            return None
        self.logger.info(f"FeatureAgent returned sequences of shape: {feature_sequences.shape} and prices of shape {prices_for_env.shape}")

        # --- 3. Environment Creation (EnvAgent) ---
        # EnvAgent needs feature_sequences and prices_for_env.
        training_environment = self.env_agent.run(
            processed_feature_data=feature_sequences,
            price_data_for_env=prices_for_env
        )
        if training_environment is None:
            self.logger.error("Environment creation failed. Training pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created training environment: {training_environment}")

        # --- 4. Model Training (TrainerAgent) ---
        # TrainerAgent needs the training_environment and optionally a model to continue from.
        trained_model_path = self.trainer_agent.run(
            training_env=training_environment,
            existing_model_path=continue_from_model
        )
        if trained_model_path is None:
            self.logger.error("Model training failed. Training pipeline aborted.")
            return None
        self.logger.info(f"TrainerAgent completed training. Model saved at: {trained_model_path}")
        
        self.logger.info(f"--- Training Pipeline for {symbol} COMPLETED ---")
        return trained_model_path


    def run_evaluation_pipeline(self, symbol: str, start_date: str, end_date: str, interval: str,
                                model_path: str, use_cached_data: bool = False) -> dict | None:
        """
        Executes data processing (for eval data) and model evaluation.

        Args:
            symbol (str): Stock symbol for evaluation data.
            start_date (str): Start date for evaluation historical data.
            end_date (str): End date for evaluation historical data.
            interval (str): Data interval.
            model_path (str): Path to the trained model to be evaluated.
            use_cached_data (bool): If True, tries to load raw data from cache.

        Returns:
            dict or None: Dictionary of evaluation metrics, or None if pipeline failed.
        """
        self.logger.info(f"--- Starting Evaluation Pipeline for {symbol} ({start_date} to {end_date}, {interval}) on model: {model_path} ---")

        # --- 1. Data Fetching for Evaluation (DataAgent) ---
        eval_data_duration_str = self.main_config.get('evaluation',{}).get('data_duration_for_fetch', "30 D")
        raw_eval_bars_df = self.data_agent.run(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            duration_str=eval_data_duration_str,
            force_fetch=not use_cached_data
        )
        if raw_eval_bars_df is None or raw_eval_bars_df.empty:
            self.logger.error("Evaluation data fetching failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"DataAgent returned raw evaluation data of shape: {raw_eval_bars_df.shape}")

        # --- 2. Feature Engineering for Evaluation (FeatureAgent) ---
        # IMPORTANT: For evaluation, `fit_scaler` must be False to use the scaler fitted during training.
        _df_eval_features, eval_feature_sequences, eval_prices_for_env = self.feature_agent.run(
            raw_data_df=raw_eval_bars_df,
            symbol=symbol, # Symbol is used to load the correct scaler
            cache_processed_data=True, 
            fit_scaler=False, # Use existing scaler
            start_date_str=start_date,
            end_date_str=end_date.split(' ')[0],
            interval_str=interval
        )
        if eval_feature_sequences is None or eval_prices_for_env is None:
            self.logger.error("Evaluation feature engineering failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"FeatureAgent returned eval sequences shape: {eval_feature_sequences.shape}, prices shape: {eval_prices_for_env.shape}")

        # --- 3. Environment Creation for Evaluation (EnvAgent) ---
        evaluation_environment = self.env_agent.run(
            processed_feature_data=eval_feature_sequences,
            price_data_for_env=eval_prices_for_env
        )
        if evaluation_environment is None:
            self.logger.error("Evaluation environment creation failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created evaluation environment: {evaluation_environment}")

        # --- 4. Model Evaluation (EvaluatorAgent) ---
        # EvaluatorAgent needs the environment, path to the model, and algorithm name.
        model_name_tag = os.path.basename(model_path).replace(".zip", "").replace(".dummy", "") # For report naming
        algo_name_from_config = self.model_params_config.get('algorithm_name', 'DQN')
        
        eval_metrics = self.evaluator_agent.run(
            eval_env=evaluation_environment,
            model_path=model_path,
            algorithm_name=algo_name_from_config,
            model_name_tag=f"{symbol}_{model_name_tag}" # Add symbol to tag
        )
        if eval_metrics is None:
            self.logger.error("Model evaluation failed.")
            # Continue to log pipeline completion, but result is None
        else:
            self.logger.info(f"EvaluatorAgent completed. Metrics: {eval_metrics}")
            
        self.logger.info(f"--- Evaluation Pipeline for {symbol} on model {model_path} COMPLETED ---")
        return eval_metrics

    def run_walk_forward_evaluation(self, symbol: str, walk_forward_config: dict):
        """
        Runs a walk-forward validation process.
        This involves multiple train-evaluate cycles on rolling/expanding windows.
        
        Args:
            symbol (str): Stock symbol.
            walk_forward_config (dict): Configuration for walk-forward splits. E.g.,
                {
                    "start_date": "2020-01-01",
                    "end_date": "2023-12-31",
                    "train_window_days": 365,
                    "eval_window_days": 90,
                    "step_days": 90, // How much to slide the windows
                    "interval": "1 day" // Or "1 min" etc.
                }
        """
        # TODO: Implement walk-forward logic.
        # This will be a loop:
        # 1. Determine current training window (start_train, end_train).
        # 2. Determine current evaluation window (start_eval, end_eval).
        # 3. Call run_training_pipeline for training window.
        # 4. Call run_evaluation_pipeline for evaluation window using the model from step 3.
        # 5. Store/aggregate results.
        # 6. Slide windows forward and repeat.
        self.logger.warning("Walk-forward evaluation is not yet implemented in this skeleton.")
        pass

    def run_live_trading(self, symbol: str):
        """
        Initiates and manages the live trading loop. (Conceptual for skeleton)
        """
        # TODO: Implement live trading logic.
        # This would involve:
        # 1. Loading the latest production-ready model.
        # 2. DataAgent fetching live bars.
        # 3. FeatureAgent processing live bars (using scaler from training).
        # 4. Model predicting action.
        # 5. RiskAgent assessing proposed trade.
        # 6. Orchestrator (or a dedicated ExecutionAgent) placing orders via IBKR.
        # 7. Continuous monitoring and updates.
        self.logger.warning("Live trading is not yet implemented in this skeleton.")
        pass
        
    def schedule_weekly_retrain(self):
        """
        Placeholder for scheduling logic (e.g., using cron, APScheduler).
        This method would define what happens during a scheduled weekly run.
        """
        # TODO: Implement scheduling integration.
        # Example tasks for a weekly run:
        # - Define the new data window for retraining (e.g., last 2 years).
        # - Call run_training_pipeline().
        # - Define the new data window for evaluation (e.g., next 3 months, held-out).
        # - Call run_evaluation_pipeline() on the newly trained model.
        # - If results are good, potentially promote the new model to "production".
        self.logger.warning("Scheduling logic for weekly retraining is not yet implemented.")
        self.logger.info("Conceptual weekly retrain: Call run_training_pipeline and run_evaluation_pipeline with updated date windows.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- Create dummy config files for the test ---
    # These paths should align with the example directory structure.
    # Assume Orchestrator is run from the root of `rl_trading_platform/`
    CONFIG_DIR = "config" # Relative to where this script is run from.
    os.makedirs(CONFIG_DIR, exist_ok=True)

    main_cfg_path = os.path.join(CONFIG_DIR, "main_config_orchestrator_test.yaml")
    model_params_cfg_path = os.path.join(CONFIG_DIR, "model_params_orchestrator_test.yaml")
    risk_limits_cfg_path = os.path.join(CONFIG_DIR, "risk_limits_orchestrator_test.yaml")

    # Dummy main_config.yaml
    dummy_main_config = {
        'paths': {
            'data_dir_raw': 'data/raw_orch_test/',
            'data_dir_processed': 'data/processed_orch_test/',
            'scalers_dir': 'data/scalers_orch_test/',
            'model_save_dir': 'models/orch_test/',
            'tensorboard_log_dir': 'logs/tensorboard_orch_test/',
            'reports_dir': 'reports/orch_test/'
        },
        'ibkr_connection': None, # No live IBKR for this test
        'feature_engineering': {
            'features': ['RSI', 'EMA'], 'rsi': {'window': 14}, 'ema': {'windows': [10, 20]},
            'time_features': ['hour'], 'lookback_window': 3,
            'feature_cols_to_scale': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day'], # Example
            'observation_feature_cols': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day', 'close'] # Example
        },
        'environment': {
            'initial_capital': 50000.0, 'transaction_cost_pct': 0.001, 'reward_scaling': 1.0,
            'log_trades_in_env': True
        },
        'training': {
            'total_timesteps': 2000, 'checkpoint_freq': 500, 'log_interval': 50,
            'data_duration_for_fetch': "10 D" # For DataAgent.run
        },
        'evaluation': {
            'metrics': ['total_return', 'num_trades'],
            'data_duration_for_fetch': "5 D" # For DataAgent.run during eval
        }
    }
    with open(main_cfg_path, 'w') as f: yaml.dump(dummy_main_config, f)

    # Dummy model_params.yaml
    dummy_model_params = {
        'algorithm_name': 'DQN',
        'algorithm_params': {'policy': 'MlpPolicy', 'learning_rate': 1e-3, 'buffer_size': 5000, 'verbose':0},
        'c51_features': {'dueling_nets': False, 'use_per': False, 'n_step_returns': 1, 'use_noisy_nets': False}
    }
    with open(model_params_cfg_path, 'w') as f: yaml.dump(dummy_model_params, f)

    # Dummy risk_limits.yaml
    dummy_risk_limits = {
        'max_daily_drawdown_pct': 0.025, 'max_hourly_turnover_ratio': 1.5, 'max_daily_turnover_ratio': 4.0,
        'halt_on_breach': True
    }
    with open(risk_limits_cfg_path, 'w') as f: yaml.dump(dummy_risk_limits, f)
    
    print(f"Dummy config files created in {CONFIG_DIR}/")

    # --- Initialize OrchestratorAgent ---
    try:
        orchestrator = OrchestratorAgent(
            main_config_path=main_cfg_path,
            model_params_path=model_params_cfg_path,
            risk_limits_path=risk_limits_cfg_path
        )
        print("\nOrchestratorAgent initialized successfully.")

        # --- Test Training Pipeline ---
        print("\n--- Testing Training Pipeline via Orchestrator ---")
        # Using dummy symbol and short date range for quick test.
        # DataAgent will use dummy data due to no IBKR connection.
        # FeatureAgent, EnvAgent, TrainerAgent use dummy logic/data.
        trained_model_file = orchestrator.run_training_pipeline(
            symbol="DUMMYTRAIN",
            start_date="2023-01-01", # Used by DataAgent for cache name part
            end_date="2023-01-10 23:59:59", # Used by DataAgent for fetch end point
            interval="1min", # DataAgent uses for cache name part and fetch
            use_cached_data=False, # Force (dummy) fetch
            continue_from_model=None
        )
        if trained_model_file:
            print(f"Training pipeline test complete. Trained model: {trained_model_file}")

            # --- Test Evaluation Pipeline ---
            print("\n--- Testing Evaluation Pipeline via Orchestrator ---")
            # Use the model just "trained" (dummy model file path)
            eval_results = orchestrator.run_evaluation_pipeline(
                symbol="DUMMYEVAL",
                start_date="2023-02-01",
                end_date="2023-02-05 23:59:59",
                interval="1min",
                model_path=trained_model_file, # From previous step
                use_cached_data=False
            )
            if eval_results:
                print(f"Evaluation pipeline test complete. Results: {eval_results}")
            else:
                print("Evaluation pipeline test failed or returned no results.")
        else:
            print("Training pipeline test failed. Skipping evaluation test.")

    except ValueError as e:
        print(f"Orchestrator initialization failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during OrchestratorAgent test: {e}", exc_info=True)
    finally:
        # Clean up dummy config files
        # if os.path.exists(main_cfg_path): os.remove(main_cfg_path)
        # if os.path.exists(model_params_cfg_path): os.remove(model_params_cfg_path)
        # if os.path.exists(risk_limits_cfg_path): os.remove(risk_limits_cfg_path)
        # Potentially remove dummy data/model directories if desired.
        # For manual inspection, leave them.
        print(f"\nOrchestratorAgent example run complete. Check directories specified in {main_cfg_path} for outputs.")
        print(f"Remember to install dependencies from requirements.txt (e.g. PyYAML, stable-baselines3, gymnasium, pandas, numpy, ta, ib_insync).")
