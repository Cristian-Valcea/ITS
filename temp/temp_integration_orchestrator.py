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

from src.column_names import COL_CLOSE, COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME # Added OHLCV
from src.agents.data_agent import DataAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.env_agent import EnvAgent
from src.training.trainer_agent import create_trainer_agent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.risk_agent import RiskAgent

# Attempt to import ib_insync specific types for type hinting if available
try:
    from ib_insync import Trade as IBTrade, Fill as IBFill, CommissionReport as IBCommissionReport
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    IBTrade = Any # type: ignore
    IBFill = Any # type: ignore
    IBCommissionReport = Any # type: ignore


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

        self.main_config = self._load_yaml_config(main_config_path, "Main Config")
        self.model_params_config = self._load_yaml_config(model_params_path, "Model Params Config")
        self.risk_limits_config = self._load_yaml_config(risk_limits_path, "Risk Limits Config")
        self._validate_configs()

        self._init_agents()
        self.hooks = hooks or {}

        # Attributes for live trading state
        self.live_trading_active = False
        self.portfolio_state: Dict[str, Any] = {} # See _run_live_trading_loop_conceptual for structure
        self.live_model: Optional[Any] = None # Stores the loaded SB3 model for live trading
        self.open_trades: Dict[int, IBTrade] = {} # orderId: ib_insync.Trade object

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
        self.feature_agent = FeatureAgent(config={ # Pass full feature_engineering config
            **self.main_config.get('feature_engineering', {}), # Original content
            'data_dir_processed': paths.get('data_dir_processed', 'data/processed/'), # Add these paths
            'scalers_dir': paths.get('scalers_dir', 'data/scalers/')
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
        for cfg, name in [(self.main_config, "main_config"), (self.model_params_config, "model_params_config"), (self.risk_limits_config, "risk_limits_config")]:
            if not cfg: self.logger.error(f"{name} is missing or empty."); raise ValueError(f"{name} is missing or empty.")

    def _load_yaml_config(self, config_path: str, config_name: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
            self.logger.info(f"{config_name} loaded successfully from {config_path}"); return cfg or {}
        except FileNotFoundError: self.logger.error(f"{config_name} file not found at {config_path}."); raise
        except yaml.YAMLError as e: self.logger.error(f"Error parsing {config_name} YAML file {config_path}: {e}"); raise
        except Exception as e: self.logger.error(f"Unexpected error loading {config_name} from {config_path}: {e}"); raise

    def register_hook(self, event: str, callback: Callable): self.hooks[event] = callback
    def _trigger_hook(self, event: str, *args, **kwargs):
        if event in self.hooks:
            try: self.hooks[event](*args, **kwargs)
            except Exception as e: self.logger.error(f"Error in hook '{event}': {e}")

    def _preprocess_features(self, features: np.ndarray, config: dict) -> np.ndarray:
        if config.get('impute_missing', True) and np.isnan(features).any():
            col_means = np.nanmean(features, axis=0); inds = np.where(np.isnan(features)); features[inds] = np.take(col_means, inds[1])
        if config.get('normalize', False): # Note: FeatureAgent handles its own scaling usually
            mean = np.mean(features, axis=0); std = np.std(features, axis=0); std[std == 0] = 1; features = (features - mean) / std
        return features

    def _augment_features(self, features: np.ndarray, config: dict) -> np.ndarray:
        if config.get('noise_injection', False): features = features + np.random.normal(0, config.get('noise_level', 0.01), features.shape)
        if config.get('random_scaling', False): features = features * np.random.uniform(config.get('scale_min', 0.98), config.get('scale_max', 1.02))
        if config.get('window_slicing', False) and features.shape[0] > config.get('window_size', features.shape[0]):
            start = np.random.randint(0, features.shape[0] - config['window_size']); features = features[start:start+config['window_size']]
        return features

    def run_training_pipeline(self, symbol: str, start_date: str, end_date: str, interval: str, use_cached_data: bool = False, continue_from_model: Optional[str] = None, run_evaluation_after_train: bool = True, eval_start_date: Optional[str] = None, eval_end_date: Optional[str] = None, eval_interval: Optional[str] = None, use_ai_agents: bool = False, use_group_ai_agents: bool = False) -> Optional[str]:
        self.logger.info(f"--- Starting Training Pipeline for {symbol} ({start_date} to {end_date}, {interval}) ---")
        # ... (rest of training pipeline - unchanged for this step)
        # [Training pipeline logic as previously defined]
        feature_sequences = prices_for_env = None
        if use_ai_agents or use_group_ai_agents: # ... (AI agent data provisioning)
            pass # Placeholder for brevity
        else:
            data_duration_str = self.main_config.get('training', {}).get('data_duration_for_fetch', "90 D")
            raw_bars_df = self.data_agent.run(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval, duration_str=data_duration_str, force_fetch=not use_cached_data)
            if raw_bars_df is None or raw_bars_df.empty: self.logger.error("Data fetching failed."); return None
            _df_features, feature_sequences, prices_for_env = self.feature_agent.run(raw_bars_df, symbol, True, True, start_date, end_date.split(' ')[0], interval)
            if feature_sequences is None or prices_for_env is None: self.logger.error("Feature engineering failed."); return None
        
        preprocessing_cfg = self.main_config.get('data_preprocessing', {}); augmentation_cfg = self.main_config.get('data_augmentation', {})
        if feature_sequences is not None:
            feature_sequences = self._preprocess_features(feature_sequences, preprocessing_cfg)
            feature_sequences = self._augment_features(feature_sequences, augmentation_cfg)

        training_environment = self.env_agent.run(feature_sequences, prices_for_env)
        if training_environment is None: self.logger.error("Environment creation failed."); return None
        trained_model_path = self.trainer_agent.run(training_environment, continue_from_model)
        if trained_model_path is None: self.logger.error("Model training failed."); return None
        self._trigger_hook('after_training', trained_model_path=trained_model_path)
        if run_evaluation_after_train: self.run_evaluation_pipeline(symbol, eval_start_date or start_date, eval_end_date or end_date, eval_interval or interval, trained_model_path, use_cached_data)
        self.logger.info(f"--- Training Pipeline for {symbol} COMPLETED ---")
        return trained_model_path


    def run_evaluation_pipeline( self, symbol: str, start_date: str, end_date: str, interval: str, model_path: str, use_cached_data: bool = False) -> Optional[Dict[str, Any]]:
        self.logger.info(f"--- Starting Evaluation Pipeline for {symbol} ({start_date} to {end_date}, {interval}) on model: {model_path} ---")
        # ... (rest of evaluation pipeline - unchanged for this step)
        # [Evaluation pipeline logic as previously defined]
        eval_data_duration_str = self.main_config.get('evaluation', {}).get('data_duration_for_fetch', "30 D")
        raw_eval_bars_df = self.data_agent.run(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval, duration_str=eval_data_duration_str, force_fetch=not use_cached_data)
        if raw_eval_bars_df is None or raw_eval_bars_df.empty: self.logger.error("Eval data fetching failed."); return None
        _df_eval_features, eval_feature_sequences, eval_prices_for_env = self.feature_agent.run(raw_eval_bars_df, symbol, True, False, start_date, end_date.split(' ')[0], interval)
        if eval_feature_sequences is None or eval_prices_for_env is None: self.logger.error("Eval feature engineering failed."); return None
        evaluation_environment = self.env_agent.run(eval_feature_sequences, eval_prices_for_env)
        if evaluation_environment is None: self.logger.error("Eval environment creation failed."); return None
        model_name_tag = os.path.basename(model_path).replace(".zip", "").replace(".dummy", "")
        algo_name_from_config = self.model_params_config.get('algorithm_name', 'DQN')
        eval_metrics = self.evaluator_agent.run(evaluation_environment, model_path, algo_name_from_config, f"{symbol}_{model_name_tag}")
        if eval_metrics is None: self.logger.error("Model evaluation failed.")
        self.logger.info(f"--- Evaluation Pipeline for {symbol} on model {model_path} COMPLETED ---")
        return eval_metrics

    def run_walk_forward_evaluation(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        # ... (walk-forward logic - unchanged for this step)
        # [Walk-forward logic as previously defined]
        self.logger.info(f"--- Starting Walk-Forward Evaluation for {symbol} ---") # Simplified for brevity
        return [] # Placeholder

    def _run_live_trading_loop_conceptual(self, symbol: str) -> None:
        self.logger.info(f"--- Attempting to Start LIVE TRADING for {symbol} ---")
        live_trading_config = self.main_config.get('live_trading', {})
        if not live_trading_config.get('enabled', False):
            self.logger.warning(f"Live trading is not enabled for '{symbol}' in config. Aborting."); return

        model_path = live_trading_config.get('production_model_path')
        # ... (config loading and model loading as previously defined) ...
        initial_capital_for_risk = self.main_config.get('environment', {}).get('initial_capital', 100000) # Fallback
        algo_name = self.model_params_config.get('algorithm_name', 'DQN')
        loaded_model = None
        model_exists = os.path.exists(model_path) if model_path else False
        dummy_model_exists = os.path.exists(model_path + ".dummy") if model_path else False
        if not model_path or (not model_exists and not dummy_model_exists):
            self.logger.error(f"Production model not found: '{model_path}'. Aborting."); return
        try:
            if self.trainer_agent.SB3_AVAILABLE and algo_name in self.trainer_agent.SB3_MODEL_CLASSES and model_exists:
                ModelClass = self.trainer_agent.SB3_MODEL_CLASSES[algo_name]
                loaded_model = ModelClass.load(model_path, env=None)
            elif dummy_model_exists:
                from .trainer_agent import DummySB3Model
                loaded_model = DummySB3Model.load(model_path + ".dummy", env=None)
            if loaded_model is None: self.logger.error("Failed to load production model."); return
            self.live_model = loaded_model # Store for callbacks
        except Exception as e: self.logger.error(f"Error loading model {model_path}: {e}", exc_info=True); return
        
        self.live_trading_active = False
        self.portfolio_state = {}
        self.open_trades = {}

        try:
            if not self.data_agent.ib_connected: self.data_agent._connect_ibkr()
            if not self.data_agent.ib_connected: self.logger.error("Failed to connect DataAgent to IBKR. Aborting."); return

            self.logger.info("Fetching initial account summary...")
            account_summary = self.data_agent.get_account_summary(tags="NetLiquidation,TotalCashValue,AvailableFunds")
            if not account_summary: self.logger.error("Failed to fetch initial account summary. Aborting."); return
            self.portfolio_state['cash'] = account_summary.get('TotalCashValue', initial_capital_for_risk)
            self.portfolio_state['net_liquidation'] = account_summary.get('NetLiquidation', initial_capital_for_risk)
            self.portfolio_state['available_funds'] = account_summary.get('AvailableFunds', initial_capital_for_risk)
            self.portfolio_state['positions'] = {}
            initial_positions = self.data_agent.get_portfolio_positions()
            for pos_item in initial_positions:
                self.portfolio_state['positions'][pos_item['symbol']] = {"shares": pos_item['position'], "average_cost": pos_item['average_cost'], "market_value": pos_item['market_value']}
            self.logger.info(f"Initial portfolio: {self.portfolio_state}")

            # Fetch warmup data for FeatureAgent
            historical_warmup_bars = int(live_trading_config.get('historical_warmup_bars', 0))
            warmup_df = None
            if historical_warmup_bars > 0:
                self.logger.info(f"Attempting to fetch {historical_warmup_bars} historical bars for FeatureAgent warmup...")
                # Determine end_datetime for historical fetch (just before now)
                # Note: DataAgent.fetch_ibkr_bars expects YYYYMMDD HH:MM:SS format for end_datetime_str
                # For live, this should be as close to 'now' as possible.
                # However, fetching historical data right up to 'now' can be tricky with IBKR's caching/timing.
                # A slight delay might be needed, or fetch up to previous day's close for daily/longer intervals.
                # For intraday (e.g. '1 min', '5 secs'), fetching up to a few minutes ago is typical.
                
                # For simplicity, using a fixed end time for now if needed by underlying fetcher.
                # A robust solution would calculate this based on current time and ensure no overlap with live feed.
                # end_dt_warmup = datetime.now() - timedelta(minutes=1) # e.g., data up to 1 min ago
                # end_dt_warmup_str = end_dt_warmup.strftime("%Y%m%d %H:%M:%S")
                end_dt_warmup_str = "" # Empty string for "now" in reqHistoricalData

                warmup_interval_str = live_trading_config.get('data_interval', '5 secs') # Use live data interval for warmup
                
                # Calculate appropriate duration_str for DataAgent.fetch_ibkr_bars
                # This is a simplified calculation. A robust one would handle various intervals precisely.
                # Example: 200 bars of "1 min" data needs "200 M" or "4 H" or "1 D"
                # Example: 200 bars of "5 secs" data needs "1000 S"
                duration_str_warmup = self._calculate_duration_for_warmup(historical_warmup_bars, warmup_interval_str)
                if not duration_str_warmup:
                    self.logger.warning("Could not calculate valid duration for warmup data. Skipping warmup.")
                else:
                    self.logger.info(f"Fetching warmup data: {duration_str_warmup} of {warmup_interval_str} ending '{end_dt_warmup_str if end_dt_warmup_str else 'now'}'")
                    warmup_df = self.data_agent.fetch_ibkr_bars(
                        symbol=symbol,
                        end_datetime_str=end_dt_warmup_str, # "" means now
                        duration_str=duration_str_warmup,
                        bar_size_str=warmup_interval_str, # e.g., "1 min", "5 secs"
                        use_rth=live_trading_config.get('contract_details',{}).get('use_rth_for_warmup', True), # Configurable RTH for warmup
                        force_fetch=True # Always fetch fresh for warmup
                    )
                    if warmup_df is not None and not warmup_df.empty:
                        self.logger.info(f"Fetched {len(warmup_df)} bars for warmup.")
                        # Ensure warmup_df has DatetimeIndex for FeatureAgent
                        if not isinstance(warmup_df.index, pd.DatetimeIndex):
                            try:
                                warmup_df.index = pd.to_datetime(warmup_df.index)
                                self.logger.info("Warmup data index converted to DatetimeIndex.")
                            except Exception as e_idx:
                                self.logger.error(f"Failed to convert warmup_df index to DatetimeIndex: {e_idx}. Skipping warmup.")
                                warmup_df = None
                    else:
                        self.logger.warning(f"Failed to fetch sufficient warmup data for {symbol}. Proceeding without.")
                        warmup_df = None
            
            self.feature_agent.initialize_live_session(symbol=symbol, historical_data_for_warmup=warmup_df)
            if not self.feature_agent._is_live_session_initialized: self.logger.error("Failed to init FeatureAgent. Aborting."); return
            
            self.risk_agent.reset_daily_limits(current_portfolio_value=self.portfolio_state['net_liquidation'], timestamp=datetime.now())
            
            self.data_agent.set_bar_update_callback(self._process_incoming_bar)
            # Subscribe to IBKR events for order status and executions
            if self.data_agent.ib:
                self.data_agent.ib.orderStatusEvent += self._on_order_status_update
                self.data_agent.ib.execDetailsEvent += self._on_execution_details
                self.data_agent.ib.commissionReportEvent += self._on_commission_report
                self.logger.info("Subscribed to IBKR order, execution, and commission events.")
            else: self.logger.error("DataAgent.ib not initialized, cannot subscribe to order events."); return

            live_contract_details = live_trading_config.get('contract_details', {})
            if not self.data_agent.subscribe_live_bars(symbol=symbol, sec_type=live_contract_details.get('sec_type', 'STK'), exchange=live_contract_details.get('exchange', 'SMART'), currency=live_contract_details.get('currency', 'USD')):
                self.logger.error(f"Failed to subscribe to live bars for {symbol}. Aborting."); return
            
            self.live_trading_active = True
            self.logger.info(f"Live trading loop starting for {symbol}...")
            last_portfolio_sync_time = datetime.now()
            portfolio_sync_interval = timedelta(minutes=live_trading_config.get('portfolio_sync_interval_minutes', 15))

            while self.live_trading_active:
                if self.data_agent.ib and self.data_agent.ib.isConnected(): self.data_agent.ib.sleep(1)
                else: self.logger.error("IB connection lost. Halting."); self.live_trading_active = False; break
                
                current_loop_time = datetime.now()
                if current_loop_time - last_portfolio_sync_time > portfolio_sync_interval:
                    self._synchronize_portfolio_state_with_broker(symbol)
                    last_portfolio_sync_time = current_loop_time
                # TODO: Market hours check

        except KeyboardInterrupt: self.logger.info("Live trading interrupted by user.")
        except Exception as e: self.logger.error(f"Unhandled exception in live trading loop for {symbol}: {e}", exc_info=True)
        finally:
            self.live_trading_active = False
            self.logger.info(f"--- Live Trading for {symbol} STOPPING ---")
            if self.data_agent.ib and self.data_agent.ib.isConnected():
                try:
                    self.data_agent.ib.orderStatusEvent -= self._on_order_status_update
                    self.data_agent.ib.execDetailsEvent -= self._on_execution_details
                    self.data_agent.ib.commissionReportEvent -= self._on_commission_report
                    self.logger.info("Unsubscribed from IBKR order events.")
                except Exception as e_unsub: self.logger.error(f"Error unsubscribing order events: {e_unsub}")
                
                live_contract_details_final = live_trading_config.get('contract_details', {})
                self.data_agent.unsubscribe_live_bars(symbol=symbol, sec_type=live_contract_details_final.get('sec_type', 'STK'), exchange=live_contract_details_final.get('exchange', 'SMART'), currency=live_contract_details_final.get('currency', 'USD'))
                self.data_agent.disconnect_ibkr()
            self.logger.info(f"--- Live Trading for {symbol} FULLY STOPPED ---")

    def _calculate_shares_and_action(self, target_position_signal: int, current_holdings_shares: float, current_price: float, cash_for_sizing: float, trade_quantity_type: str, trade_quantity_value: float, symbol: str) -> tuple[float, Optional[str]]:
        self.logger.info(f"Calculating shares for {symbol}: TargetSignal={target_position_signal}, CurrentShares={current_holdings_shares}, Price={current_price:.2f}, CashForSizing={cash_for_sizing:.2f}, QtyType={trade_quantity_type}, QtyVal={trade_quantity_value}")
        target_desired_shares = 0.0
        if target_position_signal == 1: # Target Long
            if trade_quantity_type == 'fixed_shares': target_desired_shares = abs(trade_quantity_value)
            elif trade_quantity_type == 'percent_of_capital': capital_to_invest = cash_for_sizing * trade_quantity_value; target_desired_shares = np.floor(capital_to_invest / current_price) if current_price > 0 else 0
            elif trade_quantity_type == 'fixed_notional': target_desired_shares = np.floor(trade_quantity_value / current_price) if current_price > 0 else 0
            else: self.logger.warning(f"Unknown qty_type: {trade_quantity_type}"); return 0.0, None
        elif target_position_signal == -1: # Target Short
            if trade_quantity_type == 'fixed_shares': target_desired_shares = abs(trade_quantity_value)
            elif trade_quantity_type == 'percent_of_capital': capital_to_invest_notional = cash_for_sizing * trade_quantity_value; target_desired_shares = np.floor(capital_to_invest_notional / current_price) if current_price > 0 else 0
            elif trade_quantity_type == 'fixed_notional': target_desired_shares = np.floor(trade_quantity_value / current_price) if current_price > 0 else 0
            else: self.logger.warning(f"Unknown qty_type: {trade_quantity_type}"); return 0.0, None
        
        final_target_holding_shares = target_desired_shares if target_position_signal == 1 else -target_desired_shares if target_position_signal == -1 else 0.0
        delta_shares = final_target_holding_shares - current_holdings_shares
        abs_shares_to_trade = np.floor(abs(delta_shares)) # Assume whole shares
        order_action_str = None
        if delta_shares > 0: order_action_str = "BUY"
        elif delta_shares < 0: order_action_str = "SELL"
        
        if order_action_str == "BUY": # Basic affordability check
            estimated_cost = abs_shares_to_trade * current_price
            available_capital = self.portfolio_state.get('available_funds', self.portfolio_state.get('cash', 0.0))
            if estimated_cost > available_capital: self.logger.warning(f"BUY for {abs_shares_to_trade:.0f} {symbol} may be unaffordable. Cost: {estimated_cost:.2f}, Avail: {available_capital:.2f}.")
        
        if abs_shares_to_trade > 0 and order_action_str: self.logger.info(f"Calc trade for {symbol}: {order_action_str} {abs_shares_to_trade:.0f}"); return abs_shares_to_trade, order_action_str
        self.logger.debug(f"No change in position for {symbol}."); return 0.0, None

    def _process_incoming_bar(self, new_bar_df: pd.DataFrame, symbol: str):
        if not self.live_trading_active: self.logger.debug(f"Bar for {symbol}, live trading inactive."); return
        current_time_of_bar = new_bar_df.index[-1].to_pydatetime()
        self.logger.info(f"Processing bar for {symbol} at {current_time_of_bar}: Px={new_bar_df[COL_CLOSE].iloc[0]:.2f if COL_CLOSE in new_bar_df.columns else 'N/A'}")
        try:
            observation_sequence, latest_price_series = self.feature_agent.process_live_bar(new_bar_df, symbol)
            if observation_sequence is None or latest_price_series is None: self.logger.debug(f"No valid obs for {symbol} at {current_time_of_bar}."); return
            current_price = latest_price_series[COL_CLOSE]
            if pd.isna(current_price): self.logger.warning(f"NaN price for {symbol} at {current_time_of_bar}. Skip."); return

            if not hasattr(self, 'live_model') or self.live_model is None: self.logger.error("Live model not loaded! Halting."); self.live_trading_active = False; return
            action_raw, _ = self.live_model.predict(observation_sequence, deterministic=True)
            self.logger.info(f"Model prediction for {symbol}: RawAction={action_raw}, Px={current_price:.2f}")

            target_position_signal = {0: -1, 1: 0, 2: 1}.get(int(action_raw), 0) # Default to HOLD
            if target_position_signal != {0: -1, 1: 0, 2: 1}.get(int(action_raw)): self.logger.error(f"Invalid raw action {action_raw} from model. Defaulted to HOLD.")
            self.logger.info(f"Mapped Action for {symbol}: TargetSignal={target_position_signal}")

            current_holdings = self.portfolio_state.get('positions', {}).get(symbol, {}).get('shares', 0.0)
            cash_for_sizing = self.portfolio_state.get('available_funds', self.portfolio_state.get('cash', 0.0))
            cfg = self.main_config.get('live_trading', {})
            shares_to_trade, order_action = self._calculate_shares_and_action(target_position_signal, current_holdings, current_price, cash_for_sizing, cfg.get('trade_quantity_type','fixed_shares'), float(cfg.get('trade_quantity_value',1.0)), symbol)

            if shares_to_trade > 0 and order_action:
                self.logger.info(f"Trade Decision for {symbol}: {order_action} {shares_to_trade:.0f} @ Px={current_price:.2f}")
                is_safe, reason = self.risk_agent.assess_trade_risk(abs(shares_to_trade * current_price), current_time_of_bar)
                if is_safe:
                    self.logger.info(f"Trade for {symbol} safe by RiskAgent: {reason}")
                    live_cfg = self.main_config.get('live_trading', {})
                    contract_cfg = live_cfg.get('contract_details', {})
                    order_details = {'symbol': symbol, 'sec_type': contract_cfg.get('sec_type','STK'), 'exchange': contract_cfg.get('exchange','SMART'), 'currency': contract_cfg.get('currency','USD'), 'primary_exchange': contract_cfg.get('primary_exchange'), 'action': order_action, 'quantity': shares_to_trade, 'order_type': live_cfg.get('order_type',"MKT").upper(), 'limit_price': None, 'tif': live_cfg.get('time_in_force',"DAY").upper(), 'account': self.main_config.get('ibkr_connection',{}).get('account_id')}
                    if order_details['order_type'] == "LMT": self.logger.warning("LMT order needs limit_price logic.")
                    trade_obj = self.data_agent.place_order(order_details)
                    if trade_obj and trade_obj.order and trade_obj.orderStatus:
                        self.logger.info(f"Order for {symbol} ({order_action} {shares_to_trade:.0f}) submitted. ID: {trade_obj.order.orderId}, Status: {trade_obj.orderStatus.status}")
                        self.open_trades[trade_obj.order.orderId] = trade_obj
                    else: self.logger.error(f"Failed to place order for {symbol} or trade_obj invalid.")
                else: 
                    self.logger.warning(f"Trade for {symbol} blocked by RiskAgent: {reason}")
                    if "HALT" in reason and self.risk_agent.halt_on_breach: # Check RiskAgent's own halt_on_breach config first
                        self.logger.critical(f"HALT signal from RiskAgent is active for {symbol} due to: {reason}.")
                        if self.risk_limits_config.get('liquidate_on_halt', False): # Now check Orchestrator's config for liquidation
                            self.logger.critical(f"LIQUIDATION required for {symbol} as per configuration.")
                            current_holdings_for_liq = self.portfolio_state.get('positions', {}).get(symbol, {}).get('shares', 0.0)
                            if current_holdings_for_liq != 0:
                                liquidation_action = "SELL" if current_holdings_for_liq > 0 else "BUY"
                                liquidation_quantity = abs(current_holdings_for_liq)
                                self.logger.info(f"Attempting to liquidate {liquidation_quantity} shares of {symbol} via {liquidation_action} order.")
                                
                                live_cfg_liq = self.main_config.get('live_trading', {}) # Re-fetch for safety or pass
                                contract_cfg_liq = live_cfg_liq.get('contract_details', {})
                                liquidation_order_details = {
                                    'symbol': symbol,
                                    'sec_type': contract_cfg_liq.get('sec_type','STK'),
                                    'exchange': contract_cfg_liq.get('exchange','SMART'),
                                    'currency': contract_cfg_liq.get('currency','USD'),
                                    'primary_exchange': contract_cfg_liq.get('primary_exchange'),
                                    'action': liquidation_action,
                                    'quantity': liquidation_quantity,
                                    'order_type': "MKT", # Liquidation usually with MKT order
                                    'tif': "DAY", # Ensure it executes quickly
                                    'account': self.main_config.get('ibkr_connection',{}).get('account_id')
                                }
                                liq_trade_object = self.data_agent.place_order(liquidation_order_details)
                                if liq_trade_object and liq_trade_object.order and liq_trade_object.orderStatus:
                                    self.logger.info(f"Liquidation order for {symbol} ({liquidation_action} {liquidation_quantity:.0f}) submitted. ID: {liq_trade_object.order.orderId}, Status: {liq_trade_object.orderStatus.status}")
                                    self.open_trades[liq_trade_object.order.orderId] = liq_trade_object
                                else:
                                    self.logger.error(f"Failed to place LIQUIDATION order for {symbol}.")
                                # Potentially set a flag to prevent new trades for this symbol or stop all live trading
                                # self.live_trading_active = False # Example: Stop all trading
                                self.logger.warning(f"Further trading for {symbol} might be suspended post-liquidation signal.")
                            else:
                                self.logger.info(f"Liquidation for {symbol} not needed, already flat.")
                        else:
                             self.logger.info(f"HALT active for {symbol}, but liquidate_on_halt is false. No liquidation order placed.")
            else: self.logger.debug(f"No trade for {symbol} at {current_time_of_bar}. Target={target_position_signal}, CurrentShares={current_holdings}")
            self.risk_agent.update_portfolio_value(self.portfolio_state.get('net_liquidation',0.0), current_time_of_bar)
        except Exception as e: self.logger.error(f"Critical error in _process_incoming_bar for {symbol} at {current_time_of_bar}: {e}", exc_info=True)

    def _synchronize_portfolio_state_with_broker(self, symbol_traded: Optional[str] = None): # symbol_traded not strictly needed if syncing all
        self.logger.info("Attempting to synchronize portfolio state with broker...")
        try:
            new_summary = self.data_agent.get_account_summary(tags="NetLiquidation,TotalCashValue,AvailableFunds")
            if new_summary:
                self.portfolio_state['cash'] = new_summary.get('TotalCashValue', self.portfolio_state.get('cash',0.0))
                self.portfolio_state['net_liquidation'] = new_summary.get('NetLiquidation', self.portfolio_state.get('net_liquidation',0.0))
                self.portfolio_state['available_funds'] = new_summary.get('AvailableFunds', self.portfolio_state.get('available_funds',0.0))
                self.logger.info(f"Synced summary. NetLiq: {self.portfolio_state['net_liquidation']:.2f}, Cash: {self.portfolio_state['cash']:.2f}")
            new_positions = {p['symbol']: {"shares":p['position'],"average_cost":p['average_cost'],"market_value":p['market_value']} for p in self.data_agent.get_portfolio_positions()}
            self.portfolio_state['positions'] = new_positions
            self.logger.info(f"Synced positions. Local positions now: {self.portfolio_state['positions']}")
        except Exception as e: self.logger.error(f"Error during portfolio sync: {e}", exc_info=True)

    def _calculate_duration_for_warmup(self, num_bars: int, bar_size_str: str) -> Optional[str]:
        """
        Calculates an IBKR-compatible duration string to fetch approximately num_bars.
        Example: 200 bars of "1 min" -> "200 M" (but IB has limits, so might need "4 H" or "1 D")
                 200 bars of "5 secs" -> "1000 S"
        This is a simplified helper. More precise calculation might be needed based on IBKR limits for duration vs bar size.
        """
        bar_size_str = bar_size_str.lower().strip()
        total_seconds_needed = 0

        if "sec" in bar_size_str:
            try:
                secs = int(bar_size_str.split(" ")[0])
                total_seconds_needed = num_bars * secs
            except:
                self.logger.error(f"Could not parse seconds from bar_size_str: {bar_size_str}")
                return None
        elif "min" in bar_size_str:
            try:
                mins = int(bar_size_str.split(" ")[0])
                total_seconds_needed = num_bars * mins * 60
            except:
                self.logger.error(f"Could not parse minutes from bar_size_str: {bar_size_str}")
                return None
        elif "hour" in bar_size_str:
            try:
                hours = int(bar_size_str.split(" ")[0])
                total_seconds_needed = num_bars * hours * 3600
            except:
                self.logger.error(f"Could not parse hours from bar_size_str: {bar_size_str}")
                return None
        # TODO: Add "day", "week", "month" parsing if needed for very long warmups / large bar sizes
        else:
            self.logger.error(f"Unsupported bar_size_str for warmup duration calculation: {bar_size_str}")
            return None

        if total_seconds_needed <= 0: return None

        # IBKR Max Durations:
        # For seconds: up to "1800 S" (30 mins) for 1-sec bars, "7200 S" (2 hours) for 5-sec bars.
        # Generally, for intraday, "D" is safer for longer periods.
        # Max duration for "1 min" bars is often "1 D" or "2 D" for intraday data requests.
        # Max duration for "5 secs" bars is often "2 H" (7200 S).

        if total_seconds_needed <= 7200 : # Up to 2 hours, can use seconds
             # Ensure it doesn't exceed typical max for "S" duration for common small bar sizes
            if "sec" in bar_size_str and int(bar_size_str.split(" ")[0]) >= 5 and total_seconds_needed > 7200:
                 self.logger.warning(f"Calculated {total_seconds_needed}S exceeds typical 7200S limit for >=5sec bars. Capping to 7200 S.")
                 return "7200 S"
            if "sec" in bar_size_str and int(bar_size_str.split(" ")[0]) < 5 and total_seconds_needed > 1800: # e.g. 1-sec bars
                 self.logger.warning(f"Calculated {total_seconds_needed}S exceeds typical 1800S limit for <5sec bars. Capping to 1800 S.")
                 return "1800 S"
            return f"{total_seconds_needed} S"
        
        total_days_needed = total_seconds_needed / (24 * 3600)
        if total_days_needed <= 30 : # If it fits within a month, use days
             # IBKR often limits intraday history to 1-7 days for small bar sizes, or more for larger.
             # For "1 min" bars, "1 D" or "2 D" is usually safe.
             # Let's be conservative for intraday intervals.
             if "min" in bar_size_str or "sec" in bar_size_str:
                 # If many minutes/seconds are needed, it might span more than a few days of trading hours.
                 # e.g. 200 "1 min" bars = 3h 20m. If market is 6.5h, "1 D" is fine.
                 # 1000 "1 min" bars = ~16h. Needs "2 D" or "3 D".
                 # Let's cap to a few days if total duration is long for intraday.
                 days_to_request = int(np.ceil(total_days_needed))
                 if days_to_request == 0: days_to_request = 1 # At least 1 day
                 
                 # Heuristic: if requesting many intraday bars, cap duration to avoid IBKR error
                 if ("min" in bar_size_str or "sec" in bar_size_str) and days_to_request > 5:
                     self.logger.warning(f"Warmup requires {days_to_request} days for {bar_size_str}. Capping to 5 D to be safe with IBKR limits for intraday history. May fetch fewer than {num_bars} bars.")
                     return "5 D" # A common safe limit for intraday history of min/sec bars
                 return f"{days_to_request} D"

        total_months_needed = total_days_needed / 30 # Approximate
        if total_months_needed <= 12:
            return f"{int(np.ceil(total_months_needed))} M"
        
        total_years_needed = total_months_needed / 12
        return f"{int(np.ceil(total_years_needed))} Y"


    # --- IBKR Event Handlers for Order/Execution Status (Step 1 of new plan) ---
    def _on_order_status_update(self, trade: IBTrade): # Use IBTrade type hint
        if not self.live_trading_active or not trade or not trade.orderStatus: return
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        self.logger.info(f"Order Status: ID {order_id}, Sym {trade.contract.symbol}, Status: {status}, Filled: {trade.orderStatus.filled}, Rem: {trade.orderStatus.remaining}, AvgFillPx: {trade.orderStatus.avgFillPrice:.2f if trade.orderStatus.avgFillPrice else 'N/A'}")
        if status in ['Cancelled', 'ApiCancelled', 'Inactive', 'PendingCancel', 'ApiPendingCancel', 'Expired']: # Terminal states (non-fill)
            if order_id in self.open_trades: self.logger.info(f"Order {order_id} ({status}) removed from active monitoring."); self.open_trades.pop(order_id, None)
        elif status == 'Filled': # Fully filled
            if order_id in self.open_trades: self.logger.info(f"Order {order_id} fully filled reported by OrderStatus. Awaiting ExecDetails. Keeping in open_trades for now."); # Let ExecDetails handle removal
            # Actual fill processing and portfolio update should primarily happen in _on_execution_details

    def _on_execution_details(self, trade: IBTrade, fill: IBFill): # Use IBTrade, IBFill
        if not self.live_trading_active or not fill or not fill.execution: return
        exec_id = fill.execution.execId; order_id = fill.execution.orderId; symbol = trade.contract.symbol
        shares = fill.execution.shares; price = fill.execution.price; action_side = fill.execution.side # "BOT" or "SLD"
        exec_time = pd.to_datetime(fill.time) if fill.time else datetime.now() # fill.time is datetime
        self.logger.info(f"Execution Detail: ExecID {exec_id}, OrderID {order_id} for {symbol}: {action_side} {shares} @ {price:.2f} at {exec_time}")

        if symbol not in self.portfolio_state.get('positions', {}): self.portfolio_state['positions'][symbol] = {'shares': 0.0, 'average_cost': 0.0}
        current_pos_info = self.portfolio_state['positions'][symbol]
        current_shares = current_pos_info.get('shares', 0.0)
        current_avg_cost = current_pos_info.get('average_cost', 0.0)
        trade_value = shares * price
        
        # Commission is handled by _on_commission_report, for now assume 0 or estimate
        commission = getattr(fill, 'commissionReport', {}).get('commission', 0.0) # Check if report is part of fill
        if commission == 0.0: self.logger.debug(f"Commission not in fill object for ExecID {exec_id}, will await CommissionReport or use estimate.")

        # Update portfolio based on fill
        if action_side == "BOT":
            new_total_shares = current_shares + shares
            if current_shares < 0 and new_total_shares > 0: current_pos_info['average_cost'] = price # Flipped short to long
            elif new_total_shares != 0: current_pos_info['average_cost'] = ((current_avg_cost * current_shares) + trade_value) / new_total_shares if current_shares >=0 else price # new long or adding to long
            else: current_pos_info['average_cost'] = 0.0 # Went flat
            current_pos_info['shares'] = new_total_shares
            self.portfolio_state['cash'] -= (trade_value + commission)
        elif action_side == "SLD":
            new_total_shares = current_shares - shares
            if current_shares > 0 and new_total_shares < 0: current_pos_info['average_cost'] = price # Flipped long to short
            elif new_total_shares != 0: current_pos_info['average_cost'] = ((current_avg_cost * abs(current_shares)) + trade_value) / abs(new_total_shares) if current_shares < 0 else current_pos_info['average_cost'] # new short or adding to short
            else: current_pos_info['average_cost'] = 0.0 # Went flat
            current_pos_info['shares'] = new_total_shares
            self.portfolio_state['cash'] += (trade_value - commission)
        
        self.logger.info(f"Portfolio for {symbol}: Shares={current_pos_info['shares']:.0f}, AvgCost={current_pos_info['average_cost']:.2f}, Cash={self.portfolio_state['cash']:.2f}")
        
        # Update RiskAgent
        self.risk_agent.record_trade(trade_value, exec_time)
        self._update_net_liquidation_and_risk_agent(exec_time) # New helper for this

        # Manage open_trades
        if order_id in self.open_trades:
            # Check if order is fully filled based on this execution vs order's total quantity
            order_total_quantity = self.open_trades[order_id].order.totalQuantity
            # Sum of fills for this orderId would be more robust if partial fills are common
            # For now, if this fill makes it complete:
            if self.open_trades[order_id].orderStatus.status == 'Filled' or \
               self.open_trades[order_id].orderStatus.remaining == 0 or \
               (hasattr(trade, 'remaining') and trade.remaining() == 0) : # Check trade.remaining() if available
                self.logger.info(f"Order {order_id} fully filled. Removing from open_trades.")
                self.open_trades.pop(order_id, None)
            else:
                self.logger.info(f"Order {order_id} partially filled or status not yet 'Filled'. Kept in open_trades. Filled: {self.open_trades[order_id].orderStatus.filled}")
        else:
            self.logger.warning(f"Execution detail for unknown OrderID {order_id}. Trade log: {trade.log}")


    def _on_commission_report(self, trade: IBTrade, fill: IBFill, commission_report: IBCommissionReport):
        if not self.live_trading_active: return
        exec_id = commission_report.execId; commission = commission_report.commission; currency = commission_report.currency
        realized_pnl = commission_report.realizedPNL
        self.logger.info(f"Commission Rpt: ExecID {exec_id}, Comm {commission:.2f} {currency}, RealizedPNL (fill): {realized_pnl if realized_pnl is not None else 'N/A'}")
        
        # Adjust cash by the difference between any estimated commission and actual.
        # This assumes _on_execution_details might have used an estimate if commission wasn't on the fill.
        # This part is tricky because we need to associate this commission with the specific fill's impact on cash.
        # If _on_execution_details already accounted for commission if present on fill.commissionReport, this might double count or just refine.
        # Simplest: Assume _on_execution_details used 0 or estimate if commission_report was not on fill object.
        # This requires storing estimated commission or finding the fill to adjust.
        # For now, primarily for logging. Robust accounting would be:
        # 1. When fill received in _on_execution_details, if commission_report is there, use it.
        # 2. If not, store that fill's execId and that its commission is pending.
        # 3. When this _on_commission_report arrives, find the pending fill by execId, apply commission to cash, mark as processed.
        self.logger.warning("Commission report processing logic to precisely adjust cash against prior estimates is a TODO.")


    def _update_net_liquidation_and_risk_agent(self, timestamp: datetime):
        """Helper to recalculate net liquidation and update RiskAgent."""
        # Recalculate net liquidation based on current cash and market value of all positions
        # This requires fetching current market prices for all holdings if they are not the current symbol.
        # This is a simplification for now. A full implementation would iterate all positions,
        # get their current market prices (e.g. from DataAgent if it caches or can fetch quotes),
        # then sum up cash + total market value of positions.
        # For now, we only have the current price of the *traded* symbol.
        # This will lead to inaccurate NLV if holding other symbols.
        
        # A very rough estimate if only one symbol is traded:
        current_nlv = self.portfolio_state.get('cash', 0.0)
        for sym, pos_data in self.portfolio_state.get('positions', {}).items():
            # If sym is the one just traded, its market value might be shares * last_fill_price
            # If other symbols, their market_value in portfolio_state is stale.
            # This is a significant simplification.
            current_nlv += pos_data.get('shares', 0.0) * pos_data.get('average_cost', 0.0) # Using avg_cost as proxy for market_price is wrong
                                                                                      # Needs actual market price.
        self.logger.warning(f"Net Liquidation calculation is highly simplified and likely inaccurate for multi-asset portfolios or if market prices changed. Using {current_nlv:.2f} for RiskAgent update.")
        self.portfolio_state['net_liquidation'] = current_nlv # Update local state with this rough estimate.
        self.risk_agent.update_portfolio_value(self.portfolio_state['net_liquidation'], timestamp)


    def schedule_weekly_retrain(self) -> None:
        # ... (rest of schedule_weekly_retrain - unchanged for this step)
        # [schedule_weekly_retrain logic as previously defined]
        self.logger.info("--- Starting Scheduled Weekly Retrain ---") # Simplified for brevity
        pass # Placeholder

    # ... (rest of the class, like __main__)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ... (__main__ block as previously defined)
    pass # Placeholder for brevity
