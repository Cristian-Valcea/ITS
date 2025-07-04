import os
import sys
import logging
import importlib
import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List

import yaml

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---

from src.column_names import COL_CLOSE  # Commented out to avoid numpy conflict
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

    def _preprocess_features(self, features: np.ndarray, config: dict) -> np.ndarray:
        """Apply preprocessing steps to features."""
        # Impute missing values
        if config.get('impute_missing', True):
            if np.isnan(features).any():
                # Simple mean imputation
                col_means = np.nanmean(features, axis=0)
                inds = np.where(np.isnan(features))
                features[inds] = np.take(col_means, inds[1])
        # Normalize (if not already done)
        if config.get('normalize', False):
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1
            features = (features - mean) / std
        return features

    def _augment_features(self, features: np.ndarray, config: dict) -> np.ndarray:
        """Apply data augmentation techniques to features."""
        # Noise Injection
        if config.get('noise_injection', False):
            noise_level = config.get('noise_level', 0.01)
            features = features + np.random.normal(0, noise_level, features.shape)
        # Scaling
        if config.get('random_scaling', False):
            scale = np.random.uniform(config.get('scale_min', 0.98), config.get('scale_max', 1.02))
            features = features * scale
        # Window Slicing
        if config.get('window_slicing', False):
            window_size = config.get('window_size', features.shape[0])
            if features.shape[0] > window_size:
                start = np.random.randint(0, features.shape[0] - window_size)
                features = features[start:start+window_size]
        # Add more as needed (time warping, permutation, etc.)
        return features

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

        # --- Advanced Data Handling: Preprocessing & Augmentation ---
        preprocessing_cfg = self.main_config.get('data_preprocessing', {})
        augmentation_cfg = self.main_config.get('data_augmentation', {})

        if feature_sequences is not None:
            feature_sequences = self._preprocess_features(feature_sequences, preprocessing_cfg)
            feature_sequences = self._augment_features(feature_sequences, augmentation_cfg)

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

    def run_walk_forward_evaluation(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Runs a walk-forward validation process based on settings in main_config.yaml.

        Args:
            symbol (str): The stock symbol to perform walk-forward evaluation on.
            
        Returns:
            Optional[List[Dict[str, Any]]]: A list of evaluation metrics for each fold, 
                                             or None if walk-forward is not enabled or configured.
        """
        self.logger.info(f"--- Starting Walk-Forward Evaluation for {symbol} ---")

        wf_config_main = self.main_config.get('walk_forward')
        wf_config_eval = self.main_config.get('evaluation', {}).get('walk_forward') 
        
        wf_config_dict = wf_config_main if wf_config_main is not None else wf_config_eval

        if not wf_config_dict or not wf_config_dict.get('enabled', False):
            self.logger.warning("Walk-forward evaluation is not enabled or not configured in main_config.yaml. Skipping.")
            return None

        try:
            overall_start_date_str = wf_config_dict['overall_start_date']
            overall_end_date_str = wf_config_dict['overall_end_date']
            train_window_days = int(wf_config_dict['train_window_days'])
            eval_window_days = int(wf_config_dict['eval_window_days'])
            step_days = int(wf_config_dict['step_days'])
            data_interval = wf_config_dict['data_interval']
            use_cached_data = wf_config_dict.get('use_cached_data', False) 
        except KeyError as e:
            self.logger.error(f"Missing key in walk_forward_config: {e}. Aborting walk-forward evaluation.")
            return None
        except ValueError as e: 
            self.logger.error(f"Invalid value in walk_forward_config: {e}. Aborting walk-forward evaluation.")
            return None
        except TypeError as e: # Handles case where wf_config_dict is None
            self.logger.error(f"Walk-forward configuration is not correctly structured or is missing: {e}. Aborting.")
            return None


        self.logger.info(f"Walk-Forward Parameters: Symbol={symbol}, Interval={data_interval}")
        self.logger.info(f"  Overall Period: {overall_start_date_str} to {overall_end_date_str}")
        self.logger.info(f"  Train Window: {train_window_days} days, Eval Window: {eval_window_days} days, Step: {step_days} days")

        all_fold_results = []
        
        try:
            current_train_start_dt = pd.to_datetime(overall_start_date_str)
            overall_end_dt = pd.to_datetime(overall_end_date_str)
        except Exception as e:
            self.logger.error(f"Error parsing overall_start_date ('{overall_start_date_str}') or overall_end_date ('{overall_end_date_str}'): {e}. Aborting walk-forward.")
            return None

        fold_number = 1
        while True:
            train_start_dt = current_train_start_dt
            train_end_dt = train_start_dt + pd.DateOffset(days=train_window_days - 1) 
            
            eval_start_dt = train_end_dt + pd.DateOffset(days=1)
            eval_end_dt_candidate = eval_start_dt + pd.DateOffset(days=eval_window_days - 1)

            if train_end_dt > overall_end_dt : 
                 self.logger.info(f"Fold {fold_number}: Training window (ends {train_end_dt.strftime('%Y-%m-%d')}) would extend beyond overall end date {overall_end_dt.strftime('%Y-%m-%d')}. Ending walk-forward.")
                 break
            
            if eval_start_dt > overall_end_dt:
                self.logger.info(f"Fold {fold_number}: Evaluation period (starts {eval_start_dt.strftime('%Y-%m-%d')}) starts after overall end date {overall_end_dt.strftime('%Y-%m-%d')}. Ending walk-forward.")
                break
            
            eval_end_dt = min(eval_end_dt_candidate, overall_end_dt)

            if eval_start_dt > eval_end_dt : 
                self.logger.info(f"Fold {fold_number}: Not enough data for a valid evaluation period (start: {eval_start_dt.strftime('%Y-%m-%d')}, end: {eval_end_dt.strftime('%Y-%m-%d')}). Ending walk-forward.")
                break
            
            time_suffix = " 23:59:59" if "min" in data_interval.lower() or "hour" in data_interval.lower() else ""
            
            train_start_str = train_start_dt.strftime('%Y-%m-%d')
            # Ensure end_date for training also includes time if interval is intraday
            train_end_str = train_end_dt.strftime('%Y-%m-%d') + time_suffix 
            
            eval_start_str = eval_start_dt.strftime('%Y-%m-%d')
            eval_end_str = eval_end_dt.strftime('%Y-%m-%d') + time_suffix

            self.logger.info(f"\n--- Fold {fold_number} ---")
            self.logger.info(f"Training Window: {train_start_str} to {train_end_str}")
            self.logger.info(f"Evaluation Window: {eval_start_str} to {eval_end_str}")

            # For run_training_pipeline, the eval_..._override parameters are not relevant here
            # as we are controlling the evaluation segment explicitly in this walk-forward loop.
            trained_model_path = self.run_training_pipeline(
                symbol=symbol,
                start_date=train_start_str,
                end_date=train_end_str, # Pass the calculated training end date
                interval=data_interval,
                use_cached_data=use_cached_data,
                run_evaluation_after_train=False # We handle evaluation within this loop
            )

            fold_result_summary = {
                "fold": fold_number,
                "train_start": train_start_str, "train_end": train_end_str,
                "eval_start": eval_start_str, "eval_end": eval_end_str,
                "model_path": trained_model_path,
                "metrics": None,
                "status": "TrainingFailed" if not trained_model_path else "PendingEvaluation"
            }

            if trained_model_path:
                self.logger.info(f"Fold {fold_number}: Training successful. Model: {trained_model_path}")
                
                fold_metrics = self.run_evaluation_pipeline(
                    symbol=symbol,
                    start_date=eval_start_str,
                    end_date=eval_end_str, # Pass the calculated evaluation end date
                    interval=data_interval,
                    model_path=trained_model_path,
                    use_cached_data=use_cached_data 
                )
                if fold_metrics:
                    self.logger.info(f"Fold {fold_number}: Evaluation successful. Metrics: {fold_metrics}")
                    fold_result_summary["metrics"] = fold_metrics
                    fold_result_summary["status"] = "EvaluationSuccessful"
                else:
                    self.logger.warning(f"Fold {fold_number}: Evaluation failed or returned no metrics.")
                    fold_result_summary["metrics"] = {"error": "Evaluation failed or no metrics"}
                    fold_result_summary["status"] = "EvaluationFailed"
            else:
                self.logger.error(f"Fold {fold_number}: Training failed. Skipping evaluation for this fold.")
            
            all_fold_results.append(fold_result_summary)

            current_train_start_dt += pd.DateOffset(days=step_days)
            fold_number += 1
            
            if current_train_start_dt > overall_end_dt:
                self.logger.info("Next training window starts after overall end date. Ending walk-forward.")
                break
        
        self.logger.info(f"--- Walk-Forward Evaluation for {symbol} COMPLETED ---")
        self.logger.info(f"Total folds processed: {len(all_fold_results)}")
        
        reports_dir_config = self.main_config.get('paths', {})
        reports_dir = reports_dir_config.get('reports_dir', 'reports/') 
        os.makedirs(reports_dir, exist_ok=True) 
        
        wf_summary_filename = f"walk_forward_summary_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        wf_summary_path = os.path.join(reports_dir, wf_summary_filename)
        
        try:
            with open(wf_summary_path, 'w') as f:
                yaml.dump(all_fold_results, f, sort_keys=False, default_flow_style=False, indent=2)
            self.logger.info(f"Walk-forward summary report saved to: {wf_summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save walk-forward summary report: {e}", exc_info=True)
            
        return all_fold_results

    def _run_live_trading_loop_conceptual(self, symbol: str) -> None:
        """
        Initiates and manages the live trading loop.
        This is a high-level skeleton and requires significant implementation
        for actual live trading, especially around broker interaction and real-time data handling.
        """
        self.logger.info(f"--- Attempting to Start LIVE TRADING for {symbol} ---")
        self.logger.warning("Live trading is highly conceptual in this skeleton and NOT functional for real trades.")

        # --- 1. Configuration and Setup ---
        live_trading_config = self.main_config.get('live_trading', {})
        if not live_trading_config.get('enabled', False):
            self.logger.warning(f"Live trading is not enabled for {symbol} in the configuration. Aborting.")
            return

        model_path = live_trading_config.get('production_model_path')
        if not model_path or not os.path.exists(model_path): # Add .dummy check if using dummy models
             if not model_path or (not os.path.exists(model_path) and not os.path.exists(model_path + ".dummy")):
                self.logger.error(f"Production model path not found or not specified ('{model_path}'). Aborting live trading.")
                return
        
        live_interval = live_trading_config.get('data_interval', '1min') # e.g., "1min", "5mins"
        # TODO: Define trade_quantity_type: "fixed_shares", "fixed_notional", "percent_of_capital"
        trade_quantity_type = live_trading_config.get('trade_quantity_type', 'fixed_shares')
        trade_quantity_value = live_trading_config.get('trade_quantity_value', 1) # Shares or notional amount or percentage

        self.logger.info(f"Live trading setup: Symbol={symbol}, Interval={live_interval}, Model={model_path}")
        self.logger.info(f"Trade Quantity Type: {trade_quantity_type}, Value: {trade_quantity_value}")

        # --- 2. Load Production Model ---
        # Assuming TrainerAgent has a suitable load_model method or using SB3 directly
        # For SB3, the environment used for loading might not be strictly necessary if only predicting,
        # but it's good practice if the policy needs observation/action space info.
        # We might not have a full historical dataset for a live "eval_env".
        # For predict, often `env=None` is acceptable if model structure doesn't require full env.
        algo_name = self.model_params_config.get('algorithm_name', 'DQN')
        try:
            # This is conceptual. A real model loading might be more direct via SB3.
            # self.evaluator_agent.load_model(model_path, algorithm_name=algo_name) # Evaluator has load_model
            # live_model = self.evaluator_agent.model_to_evaluate 
            # A more direct approach:
            if self.trainer_agent.SB3_AVAILABLE and algo_name in self.trainer_agent.SB3_MODEL_CLASSES:
                ModelClass = self.trainer_agent.SB3_MODEL_CLASSES[algo_name]
                live_model = ModelClass.load(model_path, env=None) # Env might be needed depending on policy
                self.logger.info(f"Successfully loaded live model from {model_path}")
            elif os.path.exists(model_path + ".dummy"): # Check for dummy model
                from .trainer_agent import DummySB3Model # Temporary import for dummy
                live_model = DummySB3Model.load(model_path, env=None)
                self.logger.info(f"Successfully loaded DUMMY live model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file {model_path} not found or SB3 not available for {algo_name}")

            if live_model is None:
                self.logger.error("Failed to load the production model. Aborting live trading.")
                return
        except Exception as e:
            self.logger.error(f"Error loading production model {model_path}: {e}", exc_info=True)
            return

        # --- 3. Initialize DataAgent for Live Data & RiskAgent for session ---
        # DataAgent needs to be connected to IBKR
        if not self.data_agent.ib_connected:
            self.logger.error("DataAgent not connected to IBKR. Cannot start live trading.")
            # Attempt connection if not already done by DataAgent's init
            self.data_agent._connect_ibkr() # Call internal connect method
            if not self.data_agent.ib_connected:
                self.logger.error("Failed to connect DataAgent to IBKR. Aborting live trading.")
                return
        
        # RiskAgent: reset daily limits at the start of a trading session
        # TODO: Get current portfolio value from broker to initialize RiskAgent correctly.
        # current_portfolio_value_from_broker = ... # Placeholder
        # self.risk_agent.reset_daily_limits(current_portfolio_value_from_broker, datetime.now())
        self.logger.warning("Live trading: Initial portfolio value for RiskAgent needs to be fetched from broker.")


        # --- 4. Main Trading Loop ---
        self.logger.info(f"Entering main live trading loop for {symbol}...")
        # This loop structure is highly conceptual and needs real-time event handling (new bar, order fills, etc.)
        try:
            # TODO: Implement a proper loop with sleep/event-driven logic.
            # For now, this is just a placeholder for the structure.
            while True: # Condition to run during market hours / strategy activity
                # self.logger.debug("Live trading loop iteration...")

                # TODO: 4a. Get latest bar data from DataAgent
                # This would involve DataAgent having a method like `get_live_bar(symbol, live_interval)`
                # which might use `ib.reqRealTimeBars` or poll `reqHistoricalData` for the latest.
                # latest_bar_df = self.data_agent.get_latest_bar(symbol, live_interval) # Conceptual
                # if latest_bar_df is None or latest_bar_df.empty:
                #     # self.ib.sleep(10) # Wait for next bar (ib_insync sleep)
                #     continue
                self.logger.warning("Live data fetching (get_latest_bar) is not implemented.")


                # TODO: 4b. Process bar with FeatureAgent
                # Needs to handle single bar updates or a small rolling window for features.
                # The scaler MUST be the one fitted during training.
                # _, live_feature_sequence, current_price_for_trade = self.feature_agent.run(
                #     raw_data_df=latest_bar_df, # This needs careful handling for sequence features.
                #     symbol=symbol,
                #     fit_scaler=False # CRITICAL
                # )
                # if live_feature_sequence is None:
                #     continue
                self.logger.warning("Live feature processing is not implemented.")
                # current_price_for_trade = latest_bar_df['close'].iloc[-1] # Example


                # TODO: 4c. Get action from the model
                # observation = ... # Construct observation including current position from broker
                # action, _ = live_model.predict(observation, deterministic=True)
                # desired_position_signal = self.env_agent.env._action_map[action] # Accessing env's action map
                self.logger.warning("Live model prediction is not implemented.")


                # TODO: 4d. Update RiskAgent with current portfolio value from broker
                # portfolio_value_broker = ... # Fetch from broker
                # self.risk_agent.update_portfolio_value(portfolio_value_broker, datetime.now())


                # TODO: 4e. Assess proposed trade with RiskAgent
                # proposed_trade_monetary_value = ... # Calculate based on action, quantity, current_price_for_trade
                # is_safe, reason, should_liquidate = self.risk_agent.assess_trade_risk(proposed_trade_monetary_value, datetime.now())


                # TODO: 4f. If trade is safe, execute trade
                # if is_safe:
                #     # EXECUTION LOGIC HERE (e.g., using self.data_agent.ib to place orders)
                #     # contract = Stock(symbol, "SMART", "USD") # Or get from DataAgent
                #     # order_action = "BUY" if desired_position_signal == 1 else "SELL"
                #     # order = MarketOrder(order_action, trade_quantity_value) # Example
                #     # trade = self.data_agent.ib.placeOrder(contract, order)
                #     # self.logger.info(f"Placed order: {trade}")
                #     # Need to monitor trade status, fills.
                #     # On fill: self.risk_agent.record_trade(fill_price * fill_quantity, datetime.now())
                #     pass
                # else:
                #     self.logger.warning(f"Trade for {symbol} blocked by RiskAgent: {reason}")
                #     if should_liquidate:
                #         self.logger.critical(f"LIQUIDATION SIGNAL for {symbol} due to: {reason}. Implement liquidation logic.")
                #         # ... logic to liquidate positions ...
                #         break # Exit trading loop
                self.logger.warning("Live trade execution and risk assessment loop not implemented.")


                # TODO: 4g. Update portfolio state (cash, positions) based on actual fills from broker.


                # TODO: 4h. Sleep until next decision point (e.g., start of next bar).
                # self.ib.sleep(60) # Sleep for 60 seconds (for 1-min bars)
                # break # For skeleton, run once conceptually
            pass # End of conceptual while loop

        except KeyboardInterrupt:
            self.logger.info("Live trading interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            self.logger.error(f"Exception in live trading loop for {symbol}: {e}", exc_info=True)
        finally:
            self.logger.info(f"--- Live Trading for {symbol} STOPPED ---")
            # --- 5. Graceful Shutdown ---
            # Disconnect DataAgent from IBKR if it was managing the connection
            if self.data_agent and hasattr(self.data_agent, 'disconnect_ibkr'):
                 self.data_agent.disconnect_ibkr()


    def schedule_weekly_retrain(self) -> None:
        """
        Executes the logic for a scheduled weekly retraining and evaluation run.
        This method would typically be called by an external scheduler (cron, Task Scheduler).
        """
        self.logger.info("--- Starting Scheduled Weekly Retrain ---")
        
        schedule_cfg = self.main_config.get('scheduling')
        if not schedule_cfg: # Check if the 'scheduling' key itself exists
            self.logger.error("Scheduling configuration ('scheduling') key not found in main_config.yaml. Aborting.")
            return
        if not isinstance(schedule_cfg, dict): # Check if it's a dictionary
            self.logger.error("'scheduling' configuration is not a valid dictionary. Aborting.")
            return

        try:
            symbol = schedule_cfg['retrain_symbol']
            interval = schedule_cfg['retrain_interval']
            train_start_offset_days = int(schedule_cfg['retrain_data_start_offset_days'])
            train_end_offset_days = int(schedule_cfg['retrain_data_end_offset_days'])
            eval_duration_days = int(schedule_cfg['evaluate_after_retrain_duration_days'])
            use_cached_data = schedule_cfg.get('use_cached_data_for_scheduled_run', False)
        except KeyError as e:
            self.logger.error(f"Missing key in scheduling_config: {e}. Aborting scheduled retrain.")
            return
        except (ValueError, TypeError) as e: # Catch errors from int() conversion or if a value is None unexpectedly
            self.logger.error(f"Invalid value type in scheduling_config: {e}. Aborting scheduled retrain.")
            return

        self.logger.info(f"Scheduled Retrain Parameters: Symbol={symbol}, Interval={interval}")
        self.logger.info(f"  Train Start Offset: {train_start_offset_days} days, Train End Offset: {train_end_offset_days} days")
        self.logger.info(f"  Evaluation Duration: {eval_duration_days} days after training.")

        today = datetime.now().date()
        
        train_end_dt = today - timedelta(days=train_end_offset_days)
        train_start_dt = today - timedelta(days=train_start_offset_days) 

        if train_start_dt >= train_end_dt:
            self.logger.error(f"Calculated training start date ({train_start_dt.strftime('%Y-%m-%d')}) is not before training end date ({train_end_dt.strftime('%Y-%m-%d')}). Check offset configurations. Aborting.")
            return

        eval_start_dt = train_end_dt + timedelta(days=1)
        eval_end_dt = eval_start_dt + timedelta(days=eval_duration_days - 1)

        time_suffix = " 23:59:59" if "min" in interval.lower() or "hour" in interval.lower() else ""
        
        train_start_str = train_start_dt.strftime('%Y-%m-%d')
        train_end_str = train_end_dt.strftime('%Y-%m-%d') + time_suffix
        
        eval_start_str = eval_start_dt.strftime('%Y-%m-%d')
        eval_end_str = eval_end_dt.strftime('%Y-%m-%d') + time_suffix

        self.logger.info(f"Calculated Training Window for scheduled run: {train_start_str} to {train_end_str}")
        self.logger.info(f"Calculated Evaluation Window for scheduled run: {eval_start_str} to {eval_end_str}")

        trained_model_path = self.run_training_pipeline(
            symbol=symbol,
            start_date=train_start_str,
            end_date=train_end_str,
            interval=interval,
            use_cached_data=use_cached_data,
            run_evaluation_after_train=False 
        )

        if trained_model_path:
            self.logger.info(f"Scheduled retraining successful for {symbol}. New model: {trained_model_path}")
            
            self.logger.info(f"Proceeding to evaluate newly trained model from scheduled run: {trained_model_path}")
            eval_metrics = self.run_evaluation_pipeline(
                symbol=symbol,
                start_date=eval_start_str,
                end_date=eval_end_str,
                interval=interval, 
                model_path=trained_model_path,
                use_cached_data=use_cached_data 
            )
            
            if eval_metrics:
                self.logger.info(f"Scheduled evaluation successful for {symbol}. Metrics: {eval_metrics}")
                self.logger.info("Model promotion logic based on metrics is a TODO.")
            else:
                self.logger.warning(f"Scheduled evaluation for {symbol} on new model {trained_model_path} failed or returned no metrics.")
        else:
            self.logger.error(f"Scheduled retraining failed for {symbol}. No new model was produced.")

        self.logger.info(f"--- Scheduled Weekly Retrain for {symbol} COMPLETED ---")





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
            'observation_feature_cols': ['rsi_14', 'ema_10', 'ema_20', 'hour_of_day', 'Close']
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
        },
        'data_preprocessing': {
            'impute_missing': True,
            'normalize': False
        },
        'data_augmentation': {
            'noise_injection': True,
            'noise_level': 0.02,
            'random_scaling': True,
            'scale_min': 0.98,
            'scale_max': 1.02,
            'window_slicing': False,
            'window_size': 60
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
        'max_daily_drawdown_pct': 0.025, 
        'max_hourly_turnover_ratio': 5.0, 
        'max_daily_turnover_ratio': 20.0,
        'halt_on_breach': True,
        'env_turnover_penalty_factor': 0.015,
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

        # Test walk-forward evaluation
        print("\n--- Testing Walk-Forward Evaluation ---")
        wf_config_from_main = orchestrator.main_config.get('walk_forward')
        wf_config_from_eval = orchestrator.main_config.get('evaluation', {}).get('walk_forward')
        
        if wf_config_from_main and wf_config_from_main.get('enabled'):
             wf_results = orchestrator.run_walk_forward_evaluation(symbol="DUMMYWF_MAIN")
        elif wf_config_from_eval and wf_config_from_eval.get('enabled'):
             wf_results = orchestrator.run_walk_forward_evaluation(symbol="DUMMYWF_EVAL")
        else:
            wf_results = None
            print("Walk-forward not enabled in any configuration location for the test.")

        if wf_results is not None: 
            print(f"Walk-forward evaluation completed. Number of folds: {len(wf_results)}")
            for i, fold_res in enumerate(wf_results):
                print(f"  Fold {fold_res.get('fold', i+1)}:") 
                print(f"    Train: {fold_res.get('train_start')} to {fold_res.get('train_end')}")
                print(f"    Eval: {fold_res.get('eval_start')} to {fold_res.get('eval_end')}")
                print(f"    Model: {fold_res.get('model_path')}")
                print(f"    Metrics: {fold_res.get('metrics')}")
                print(f"    Status: {fold_res.get('status')}")
        else:
            print("Walk-forward evaluation did not run or returned no results (check config and logs if enabled).")

        # Test scheduled weekly retraining logic
        print("\n--- Testing Scheduled Weekly Retrain Logic ---")
        orchestrator.schedule_weekly_retrain() # This will use DUMMYSCHED from config
        print("Scheduled weekly retrain process finished (check logs for details).")
            
    except ValueError as e:
        logging.error(f"Orchestrator initialization failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during OrchestratorAgent test: {e}")
    finally:
        logging.info(f"OrchestratorAgent example run complete. Check directories specified in {main_cfg_path} for outputs.")
        #logging.info("Remember to install dependencies from requirements.txt (e.g. PyYAML, stable-baselines3, gymnasium, pandas, numpy, ta, ib_insync).")

