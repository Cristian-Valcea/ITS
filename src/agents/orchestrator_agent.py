import os
import sys
import logging
import importlib
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List

import yaml

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---

from src.column_names import CLOSE
from src.agents.data_agent import DataAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.env_agent import EnvAgent
from src.agents.trainer_agent import TrainerAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.risk_agent import RiskAgent

def optional_import(module_path: str, attr: str):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr, None)
    except ImportError:
        return None

run_data_provisioning = optional_import("src.graph_ai_agents.orchestrator_data_provisioning", "run_data_provisioning")
DataProvisioningOrchestrator = optional_import("src.ai_agents.dqn_data_agent_system", "DataProvisioningOrchestrator")

def run_async(coro):
    """Run async code safely, even if an event loop is already running."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(coro)
        raise

class OrchestratorAgent:
    """
    Coordinates the RL trading platform:
    - Loads and validates configuration.
    - Initializes and wires all specialized agents.
    - Manages end-to-end pipelines for data, training, evaluation, and live trading.
    - Supports hooks/callbacks for pipeline events.
    """

    def __init__(
        self,
        main_config_path: str,
        model_params_path: str,
        risk_limits_path: str,
        hooks: Optional[Dict[str, Callable]] = None
    ):
        self.logger = logging.getLogger("RLTradingPlatform.OrchestratorAgent")
        self.logger.info("Initializing OrchestratorAgent...")

        # Load configs
        self.main_config = self._load_yaml_config(main_config_path, "Main Config")
        self.model_params_config = self._load_yaml_config(model_params_path, "Model Params Config")
        self.risk_limits_config = self._load_yaml_config(risk_limits_path, "Risk Limits Config")
        self._validate_configs()

        # Initialize agents
        self._init_agents()
        self.hooks = hooks or {}
        self.logger.info("All specialized agents initialized by Orchestrator.")

    def _init_agents(self):
        """Initialize all specialized agents with configs."""
        paths = self.main_config.get('paths', {})
        feature_engineering = self.main_config.get('feature_engineering', {})
        env_cfg = self.main_config.get('environment', {})
        risk_cfg = self.risk_limits_config

        self.data_agent = DataAgent(config={
            'data_dir_raw': paths.get('data_dir_raw', 'data/raw/'),
            'ibkr_conn': self.main_config.get('ibkr_connection', None)
        })
        self.feature_agent = FeatureAgent(config={
            'data_dir_processed': paths.get('data_dir_processed', 'data/processed/'),
            'scalers_dir': paths.get('scalers_dir', 'data/scalers/'),
            'feature_engineering': feature_engineering
        })
        self.env_agent = EnvAgent(config={
            'env_config': {
                'initial_capital': env_cfg.get('initial_capital', 100000.0),
                'transaction_cost_pct': env_cfg.get('transaction_cost_pct', 0.001),
                'reward_scaling': env_cfg.get('reward_scaling', 1.0),
                'max_episode_steps': env_cfg.get('max_episode_steps', None),
                'log_trades': env_cfg.get('log_trades_in_env', True),
                'lookback_window': feature_engineering.get('lookback_window', 1),
                'max_daily_drawdown_pct': risk_cfg.get('max_daily_drawdown_pct', 0.02),
                'hourly_turnover_cap': risk_cfg.get('max_hourly_turnover_ratio', 5.0),
                'turnover_penalty_factor': risk_cfg.get('env_turnover_penalty_factor', 0.01),
                # New parameters from collaborator's improvements
                'position_sizing_pct_capital': env_cfg.get('position_sizing_pct_capital', 0.25),
                'trade_cooldown_steps': env_cfg.get('trade_cooldown_steps', 0),
                'terminate_on_turnover_breach': risk_cfg.get('env_terminate_on_turnover_breach', False),
                'turnover_termination_threshold_multiplier': risk_cfg.get('env_turnover_termination_threshold_multiplier', 2.0)
            }
        })
        self.trainer_agent = TrainerAgent(config={
            'model_save_dir': paths.get('model_save_dir', 'models/'),
            'log_dir': paths.get('tensorboard_log_dir', 'logs/tensorboard/'),
            'algorithm': self.model_params_config.get('algorithm_name', 'DQN'),
            'algo_params': self.model_params_config.get('algorithm_params', {}),
            'c51_features': self.model_params_config.get('c51_features', {}),
            'training_params': self.main_config.get('training', {}),
        })
        self.evaluator_agent = EvaluatorAgent(config={
            'reports_dir': paths.get('reports_dir', 'reports/'),
            'eval_metrics': self.main_config.get('evaluation', {}).get('metrics', ['sharpe', 'max_drawdown'])
        })
        self.risk_agent = RiskAgent(config=risk_cfg)

    def _validate_configs(self):
        """Basic config validation. Extend as needed."""
        for cfg, name in [
            (self.main_config, "main_config"),
            (self.model_params_config, "model_params_config"),
            (self.risk_limits_config, "risk_limits_config")
        ]:
            if not cfg:
                self.logger.error(f"{name} is missing or empty.")
                raise ValueError(f"{name} is missing or empty.")

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

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for a pipeline event."""
        self.hooks[event] = callback

    def _trigger_hook(self, event: str, *args, **kwargs):
        if event in self.hooks:
            try:
                self.hooks[event](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in hook '{event}': {e}")

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
        eval_interval: Optional[str] = None,
        use_ai_agents: bool = False,
        use_group_ai_agents: bool = False,
    ) -> Optional[str]:
        """
        Run the full training pipeline: data, features, environment, training, (optional) evaluation.
        """
        self.logger.info(f"--- Starting Training Pipeline for {symbol} ({start_date} to {end_date}, {interval}) ---")
        feature_sequences = prices_for_env = None

        if use_ai_agents or use_group_ai_agents:
            ai_parameters = {
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "use_cached_data": use_cached_data,
                "max_missing_pct": 0.05,
                "min_volume_per_bar": 100
            }
            try:
                if use_ai_agents and DataProvisioningOrchestrator:
                    orchestrator = DataProvisioningOrchestrator(ai_parameters)
                    results = run_async(orchestrator.run(
                        objectives=["momentum", "mean-reversion", "reversal"],
                        user_universe=["AAPL", "MSFT", "NVDA", "TSLA"]
                    ))
                    validated_files = results.get("validated_files", [])
                    self.logger.info(f"Validated data files: {validated_files}")
                elif use_group_ai_agents and run_data_provisioning:
                    initial_request = {
                        "objectives": ["momentum", "mean_reversion", "reversal"],
                        "candidates": ["AAPL", "MSFT", "NVDA"]
                    }
                    ai_result_str = run_async(run_data_provisioning(initial_request, ai_parameters))
                    ai_result = json.loads(ai_result_str)
                    feature_sequences = ai_result.get("feature_sequences")
                    prices_for_env = ai_result.get("prices_for_env")
                    if feature_sequences is None or prices_for_env is None:
                        self.logger.error("AI agent pipeline did not return required data.")
                        return None
                else:
                    self.logger.error("AI agent pipeline not available.")
                    return None
            except Exception as e:
                self.logger.error(f"AI agent pipeline failed: {e}", exc_info=True)
                return None
        else:
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

        training_environment = self.env_agent.run(
            processed_feature_data=feature_sequences,
            price_data_for_env=prices_for_env
        )
        if training_environment is None:
            self.logger.error("Environment creation failed. Training pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created training environment: {training_environment}")

        trained_model_path = self.trainer_agent.run(
            training_env=training_environment,
            existing_model_path=continue_from_model
        )
        if trained_model_path is None:
            self.logger.error("Model training failed. Training pipeline aborted.")
            return None
        self.logger.info(f"TrainerAgent completed training. Model saved at: {trained_model_path}")

        self._trigger_hook('after_training', trained_model_path=trained_model_path)

        if run_evaluation_after_train and trained_model_path:
            eval_results = self.run_evaluation_pipeline(
                symbol=symbol,
                start_date=eval_start_date or start_date,
                end_date=eval_end_date or end_date,
                interval=eval_interval or interval,
                model_path=trained_model_path,
                use_cached_data=use_cached_data
            )
            self._trigger_hook('after_evaluation', eval_results=eval_results)

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
        Run the evaluation pipeline: data, features, environment, evaluation.
        """
        self.logger.info(f"--- Starting Evaluation Pipeline for {symbol} ({start_date} to {end_date}, {interval}) on model: {model_path} ---")

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

        evaluation_environment = self.env_agent.run(
            processed_feature_data=eval_feature_sequences,
            price_data_for_env=eval_prices_for_env
        )
        if evaluation_environment is None:
            self.logger.error("Evaluation environment creation failed. Evaluation pipeline aborted.")
            return None
        self.logger.info(f"EnvAgent created evaluation environment: {evaluation_environment}")

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

    # The rest of the methods (walk-forward, live trading, etc.) can be similarly refactored as above.
    # For brevity, they are omitted here, but the same principles apply.

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
            'features': ['RSI', 'EMA', 'Time'], 'rsi': {'window': 14}, 'ema': {'windows': [10, 20]},
            'time_features': ['hour_of_day'], 'lookback_window': 3,
            'feature_cols_to_scale': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day'],
            'observation_feature_cols': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day', CLOSE]
        },
        'environment': {
            'initial_capital': 50000.0, 'transaction_cost_pct': 0.001, 'reward_scaling': 1.0,
            'log_trades_in_env': True,
            'position_sizing_pct_capital': 0.25,
            'trade_cooldown_steps': 0
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
        'max_daily_drawdown_pct': 0.025, 'max_hourly_turnover_ratio': 5.0, 'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True,
        'env_turnover_penalty_factor': 0.01,
        'env_terminate_on_turnover_breach': False,
        'env_turnover_termination_threshold_multiplier': 2.0
    }
    with open(risk_limits_cfg_path, 'w') as f: yaml.dump(dummy_risk_limits, f)

    logging.info(f"Dummy config files created in {CONFIG_DIR}/")

    def after_training_hook(trained_model_path=None, **kwargs):
        logging.info(f"[HOOK] Training completed. Model path: {trained_model_path}")

    def after_evaluation_hook(eval_results=None, **kwargs):
        logging.info(f"[HOOK] Evaluation completed. Results: {eval_results}")

    try:
        orchestrator = OrchestratorAgent(
            main_config_path=main_cfg_path,
            model_params_path=model_params_cfg_path,
            risk_limits_path=risk_limits_cfg_path,
            hooks={
                'after_training': after_training_hook,
                'after_evaluation': after_evaluation_hook
            }
        )
        logging.info("OrchestratorAgent initialized successfully.")

        logging.info("--- Testing Training Pipeline via Orchestrator ---")
        trained_model_file = orchestrator.run_training_pipeline(
            symbol="NVDA",
            start_date="2025-01-01",
            end_date="2025-01-10 23:59:59",
            interval="1min",
            use_cached_data=False,
            continue_from_model=None
        )
        if trained_model_file:
            logging.info(f"Training pipeline test complete. Trained model: {trained_model_file}")

            logging.info("--- Testing Evaluation Pipeline via Orchestrator ---")
            eval_results = orchestrator.run_evaluation_pipeline(
                symbol="NVDA",
                start_date="2025-01-01",
                end_date="2025-01-10 23:59:59",
                interval="1min",
                model_path=trained_model_file,
                use_cached_data=False
            )
            if eval_results:
                logging.info(f"Evaluation pipeline test complete. Results: {eval_results}")
            else:
                logging.warning("Evaluation pipeline test failed or returned no results.")
        else:
            logging.warning("Training pipeline test failed. Skipping evaluation test.")

    except ValueError as e:
        logging.error(f"Orchestrator initialization failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during OrchestratorAgent test: {e}")
    finally:
        logging.info(f"OrchestratorAgent example run complete. Check directories specified in {main_cfg_path} for outputs.")
        #logging.info("Remember to install dependencies from requirements.txt (e.g. PyYAML, stable-baselines3, gymnasium, pandas, numpy, ta, ib_insync).")

