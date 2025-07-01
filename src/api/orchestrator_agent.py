import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import yaml

from .data_agent import DataAgent
from .feature_agent import FeatureAgent
from .env_agent import EnvAgent
from .trainer_agent import TrainerAgent
from .evaluator_agent import EvaluatorAgent
from .risk_agent import RiskAgent

class OrchestratorAgent:
    """
    OrchestratorAgent coordinates the RL trading platform:
    - Loads and validates configuration.
    - Initializes and wires all specialized agents.
    - Manages end-to-end pipelines for data, training, evaluation, and (future) live trading.
    """

    def __init__(self, main_config_path: str, model_params_path: str, risk_limits_path: str):
        self.logger = logging.getLogger("RLTradingPlatform.OrchestratorAgent")
        self.logger.info("Initializing OrchestratorAgent...")

        # Load configs
        self.main_config = self._load_yaml_config(main_config_path, "Main Config")
        self.model_params_config = self._load_yaml_config(model_params_path, "Model Params Config")
        self.risk_limits_config = self._load_yaml_config(risk_limits_path, "Risk Limits Config")

        if not all([self.main_config, self.model_params_config, self.risk_limits_config]):
            self.logger.error("One or more configuration files failed to load. Orchestrator cannot proceed.")
            raise ValueError("Configuration loading failed.")

        # Initialize agents
        self.data_agent = DataAgent(config={
            'data_dir_raw': self.main_config.get('paths', {}).get('data_dir_raw', 'data/raw/'),
            'ibkr_conn': self.main_config.get('ibkr_connection', None)
        })

        self.feature_agent = FeatureAgent(config={
            'data_dir_processed': self.main_config.get('paths', {}).get('data_dir_processed', 'data/processed/'),
            'scalers_dir': self.main_config.get('paths', {}).get('scalers_dir', 'data/scalers/'),
            **self.main_config.get('feature_engineering', {})
        })

        env_cfg = self.main_config.get('environment', {})
        self.env_agent = EnvAgent(config={
            'env_config': {
                'initial_capital': env_cfg.get('initial_capital', 100000.0),
                'transaction_cost_pct': env_cfg.get('transaction_cost_pct', 0.001),
                'reward_scaling': env_cfg.get('reward_scaling', 1.0),
                'max_episode_steps': env_cfg.get('max_episode_steps', None),
                'log_trades': env_cfg.get('log_trades_in_env', True),
                'lookback_window': self.main_config.get('feature_engineering', {}).get('lookback_window', 1),
                'max_daily_drawdown_pct': self.risk_limits_config.get('max_daily_drawdown_pct', 0.02),
                'hourly_turnover_cap': self.risk_limits_config.get('max_hourly_turnover_ratio', 5.0),
                'turnover_penalty_factor': self.risk_limits_config.get('env_turnover_penalty_factor', 0.01)
            }
        })

        self.trainer_agent = TrainerAgent(config={
            'model_save_dir': self.main_config.get('paths', {}).get('model_save_dir', 'models/'),
            'log_dir': self.main_config.get('paths', {}).get('tensorboard_log_dir', 'logs/tensorboard/'),
            'algorithm': self.model_params_config.get('algorithm_name', 'DQN'),
            'algo_params': self.model_params_config.get('algorithm_params', {}),
            'c51_features': self.model_params_config.get('c51_features', {}),
            'training_params': self.main_config.get('training', {}),
        })

        self.evaluator_agent = EvaluatorAgent(config={
            'reports_dir': self.main_config.get('paths', {}).get('reports_dir', 'reports/'),
            'eval_metrics': self.main_config.get('evaluation', {}).get('metrics', ['sharpe', 'max_drawdown'])
        })

        self.risk_agent = RiskAgent(config=self.risk_limits_config)

        self.logger.info("All specialized agents initialized by Orchestrator.")

    def _load_yaml_config(self, config_path: str, config_name: str) -> Dict[str, Any]:
        """Load a YAML configuration file and return its contents as a dict."""
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            self.logger.info(f"{config_name} loaded successfully from {config_path}")
            return cfg or {}
        except FileNotFoundError:
            self.logger.error(f"{config_name} file not found at {config_path}.")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing {config_name} YAML file {config_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading {config_name} from {config_path}: {e}")
            raise

    def run_training_pipeline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        use_cached_data: bool = False,
        continue_from_model: Optional[str] = None,
        run_evaluation_after_train: bool = True,
        eval_start_date: Optional[str] = None,
        eval_end_date: Optional[str] = None,
        eval_interval: Optional[str] = None
    ) -> Optional[str]:
        """
        Executes the full data processing and model training pipeline.
        Returns the path to the trained model, or None if failed.
        """
        self.logger.info(f"--- Starting Training Pipeline for {symbol} ({start_date} to {end_date}, {interval}) ---")

        # 1. Data Fetching
        data_duration_str = self.main_config.get('training', {}).get('data_duration_for_fetch', "90 D")
        raw_bars_df = self.data_agent.run(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            duration_str=data_duration_str,
            force_fetch=not use_cached_data
        )
        if raw_bars_df is None or raw_bars_df.empty:
            self.logger.error("Data fetching failed. Training pipeline aborted.")
            return None
        self.logger.info(f"DataAgent returned raw data of shape: {raw_bars_df.shape}")

        # 2. Feature Engineering
        _df_features, feature_sequences, prices_for_env = self.feature_agent.run(
            raw_data_df=raw_bars_df,
            symbol=symbol,
            cache_processed_data=True,
            fit_scaler=True,
            start_date_str=start_date,
            end_date_str=end_date.split(' ')[0],
            interval_str=interval
        )
        if feature_sequences is None or prices_for_env is None:
            self.logger.error("Feature engineering failed. Training pipeline aborted.")
            return None
        self.logger.info(f"FeatureAgent returned sequences of shape: {feature_sequences.shape} and prices of shape {prices_for_env.shape}")

        # 3. Environment Creation
        training_environment = self.env_agent.run(
            processed_feature_data=feature_sequences,
            price_data_for_env=prices_for_env
        )
        if training_environment is None:
            self.logger.error("Environment creation failed. Training pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created training environment: {training_environment}")

        # 4. Model Training
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

    def run_evaluation_pipeline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        model_path: str,
        use_cached_data: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Executes data processing and model evaluation.
        Returns a dictionary of evaluation metrics, or None if failed.
        """
        self.logger.info(f"--- Starting Evaluation Pipeline for {symbol} ({start_date} to {end_date}, {interval}) on model: {model_path} ---")

        # 1. Data Fetching for Evaluation
        eval_data_duration_str = self.main_config.get('evaluation', {}).get('data_duration_for_fetch', "30 D")
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

        # 2. Feature Engineering for Evaluation
        _df_eval_features, eval_feature_sequences, eval_prices_for_env = self.feature_agent.run(
            raw_data_df=raw_eval_bars_df,
            symbol=symbol,
            cache_processed_data=True,
            fit_scaler=False,
            start_date_str=start_date,
            end_date_str=end_date.split(' ')[0],
            interval_str=interval
        )
        if eval_feature_sequences is None or eval_prices_for_env is None:
            self.logger.error("Evaluation feature engineering failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"FeatureAgent returned eval sequences shape: {eval_feature_sequences.shape}, prices shape: {eval_prices_for_env.shape}")

        # 3. Environment Creation for Evaluation
        evaluation_environment = self.env_agent.run(
            processed_feature_data=eval_feature_sequences,
            price_data_for_env=eval_prices_for_env
        )
        if evaluation_environment is None:
            self.logger.error("Evaluation environment creation failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created evaluation environment: {evaluation_environment}")

        # 4. Model Evaluation
        model_name_tag = os.path.basename(model_path).replace(".zip", "").replace(".dummy", "")
        algo_name_from_config = self.model_params_config.get('algorithm_name', 'DQN')
        eval_metrics = self.evaluator_agent.run(
            eval_env=evaluation_environment,
            model_path=model_path,
            algorithm_name=algo_name_from_config,
            model_name_tag=f"{symbol}_{model_name_tag}"
        )
        if eval_metrics is None:
            self.logger.error("Model evaluation failed.")
        else:
            self.logger.info(f"EvaluatorAgent completed. Metrics: {eval_metrics}")

        self.logger.info(f"--- Evaluation Pipeline for {symbol} on model {model_path} COMPLETED ---")
        return eval_metrics

    def run_walk_forward_evaluation(self, symbol: str, walk_forward_config: Dict[str, Any]) -> None:
        """
        Runs a walk-forward validation process (not yet implemented).
        """
        self.logger.warning("Walk-forward evaluation is not yet implemented in this skeleton.")

    def run_live_trading(self, symbol: str) -> None:
        """
        Initiates and manages the live trading loop (not yet implemented).
        """
        self.logger.warning("Live trading is not yet implemented in this skeleton.")

    def schedule_weekly_retrain(self) -> None:
        """
        Placeholder for scheduling logic (not yet implemented).
        """
        self.logger.warning("Scheduling logic for weekly retraining is not yet implemented.")
        self.logger.info("Conceptual weekly retrain: Call run_training_pipeline and run_evaluation_pipeline with updated date windows.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Create dummy config files for the test ---
    CONFIG_DIR = "config"
    os.makedirs(CONFIG_DIR, exist_ok=True)

    main_cfg_path = os.path.join(CONFIG_DIR, "main_config_orchestrator_test.yaml")
    model_params_cfg_path = os.path.join(CONFIG_DIR, "model_params_orchestrator_test.yaml")
    risk_limits_cfg_path = os.path.join(CONFIG_DIR, "risk_limits_orchestrator_test.yaml")

    dummy_main_config = {
        'paths': {
            'data_dir_raw': 'data/raw_orch_test/',
            'data_dir_processed': 'data/processed_orch_test/',
            'scalers_dir': 'data/scalers_orch_test/',
            'model_save_dir': 'models/orch_test/',
            'tensorboard_log_dir': 'logs/tensorboard_orch_test/',
            'reports_dir': 'reports/orch_test/'
        },
        'ibkr_connection': None,
        'feature_engineering': {
            'features': ['RSI', 'EMA'], 'rsi': {'window': 14}, 'ema': {'windows': [10, 20]},
            'time_features': ['hour'], 'lookback_window': 3,
            'feature_cols_to_scale': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day'],
            'observation_feature_cols': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day', 'close']
        },
        'environment': {
            'initial_capital': 50000.0, 'transaction_cost_pct': 0.001, 'reward_scaling': 1.0,
            'log_trades_in_env': True
        },
        'training': {
            'total_timesteps': 2000, 'checkpoint_freq': 500, 'log_interval': 50,
            'data_duration_for_fetch': "10 D"
        },
        'evaluation': {
            'metrics': ['total_return', 'num_trades'],
            'data_duration_for_fetch': "5 D"
        }
    }
    with open(main_cfg_path, 'w') as f: yaml.dump(dummy_main_config, f)

    dummy_model_params = {
        'algorithm_name': 'DQN',
        'algorithm_params': {'policy': 'MlpPolicy', 'learning_rate': 1e-3, 'buffer_size': 5000, 'verbose': 0},
        'c51_features': {'dueling_nets': False, 'use_per': False, 'n_step_returns': 1, 'use_noisy_nets': False}
    }
    with open(model_params_cfg_path, 'w') as f: yaml.dump(dummy_model_params, f)

    dummy_risk_limits = {
        'max_daily_drawdown_pct': 0.025, 'max_hourly_turnover_ratio': 1.5, 'max_daily_turnover_ratio': 4.0,
        'halt_on_breach': True
    }
    with open(risk_limits_cfg_path, 'w') as f: yaml.dump(dummy_risk_limits, f)

    print(f"Dummy config files created in {CONFIG_DIR}/")

    try:
        orchestrator = OrchestratorAgent(
            main_config_path=main_cfg_path,
            model_params_path=model_params_cfg_path,
            risk_limits_path=risk_limits_cfg_path
        )
        print("\nOrchestratorAgent initialized successfully.")

        print("\n--- Testing Training Pipeline via Orchestrator ---")
        trained_model_file = orchestrator.run_training_pipeline(
            symbol="DUMMYTRAIN",
            start_date="2023-01-01",
            end_date="2023-01-10 23:59:59",
            interval="1min",
            use_cached_data=False,
            continue_from_model=None
        )
        if trained_model_file:
            print(f"Training pipeline test complete. Trained model: {trained_model_file}")

            print("\n--- Testing Evaluation Pipeline via Orchestrator ---")
            eval_results = orchestrator.run_evaluation_pipeline(
                symbol="DUMMYEVAL",
                start_date="2023-02-01",
                end_date="2023-02-05 23:59:59",
                interval="1min",
                model_path=trained_model_file,
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
        print(f"An unexpected error occurred during OrchestratorAgent test: {e}")
    finally:
        print(f"\nOrchestratorAgent example run complete. Check directories specified in {main_cfg_path} for outputs.")
        print("Remember to install dependencies from requirements.txt (e.g. PyYAML, stable-baselines3, gymnasium, pandas, numpy, ta, ib_insync).")
