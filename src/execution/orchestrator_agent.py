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

# Bounded context imports following new architecture
try:
    # Shared constants (centralized)
    from ..shared.constants import CLOSE, OPEN_PRICE, HIGH, LOW, VOLUME
    from ..shared.duckdb_manager import (
        close_all_duckdb_connections, 
        close_write_duckdb_connections,
        training_phase,
        evaluation_phase
    )
    
    # Legacy agents (still in agents/ - will be migrated)
    from ..agents.data_agent import DataAgent
    from ..agents.feature_agent import FeatureAgent
    from ..agents.env_agent import EnvAgent
    from ..agents.evaluator_agent import EvaluatorAgent
    
    # Training context (bounded)
    from ..training.trainer_agent import create_trainer_agent
    
    # Execution context (this module)
    from .execution_agent_stub import create_execution_agent_stub
    
    # Core execution modules
    from .core.execution_loop import ExecutionLoop
    from .core.order_router import OrderRouter
    from .core.pnl_tracker import PnLTracker
    from .core.live_data_loader import LiveDataLoader
    
    # Risk system
    from ..risk.risk_agent_adapter import RiskAgentAdapter
    from ..risk.risk_agent_v2 import RiskAgentV2
    
    # New modular risk controls
    from ..risk.controls.risk_manager import RiskManager
    from ..gym_env.wrappers.risk_wrapper import RiskObsWrapper, VolatilityPenaltyReward
    
    # Legacy column names (fallback)
    try:
        from src.column_names import COL_CLOSE, COL_OPEN, COL_HIGH, COL_LOW, COL_VOLUME
    except ImportError:
        # Use new constants as fallback
        COL_CLOSE = CLOSE
        COL_OPEN = OPEN_PRICE
        COL_HIGH = HIGH
        COL_LOW = LOW
        COL_VOLUME = VOLUME
        
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from shared.constants import CLOSE, OPEN_PRICE, HIGH, LOW, VOLUME
    from shared.duckdb_manager import (
        close_all_duckdb_connections, 
        close_write_duckdb_connections,
        training_phase,
        evaluation_phase
    )
    from agents.data_agent import DataAgent
    from agents.feature_agent import FeatureAgent
    from agents.env_agent import EnvAgent
    from agents.evaluator_agent import EvaluatorAgent
    from training.trainer_agent import create_trainer_agent
    from execution.execution_agent_stub import create_execution_agent_stub
    from execution.core.execution_loop import ExecutionLoop
    from execution.core.order_router import OrderRouter
    from execution.core.pnl_tracker import PnLTracker
    from execution.core.live_data_loader import LiveDataLoader
    from risk.risk_agent_adapter import RiskAgentAdapter
    from risk.risk_agent_v2 import RiskAgentV2
    
    # Legacy column names
    COL_CLOSE = CLOSE
    COL_OPEN = OPEN_PRICE
    COL_HIGH = HIGH
    COL_LOW = LOW
    COL_VOLUME = VOLUME

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
        hooks: Optional[Dict[str, Callable]] = None,
        read_only: bool = False
    ):
        self.logger = logging.getLogger("RLTradingPlatform.OrchestratorAgent")
        self.read_only = read_only
        self.logger.info("Initializing OrchestratorAgent...")

        # Load configs
        self.main_config = self._load_yaml_config(main_config_path, "Main Config")
        self.model_params_config = self._load_yaml_config(model_params_path, "Model Params Config")
        self.risk_limits_config = self._load_yaml_config(risk_limits_path, "Risk Limits Config")
        self._validate_configs()

        # Initialize agents
        self._init_agents()
        self.hooks = hooks or {}

        # Initialize core execution modules
        self._init_core_modules()

        # Attributes for live trading state (maintained for backward compatibility)
        self.live_trading_active = False
        self.portfolio_state: Dict[str, Any] = {} # Delegated to PnLTracker
        self.live_model: Optional[Any] = None # Stores the loaded SB3 model for live trading
        self.open_trades: Dict[int, IBTrade] = {} # orderId: ib_insync.Trade object

        self.logger.info("All specialized agents and core modules initialized by Orchestrator.")

    def _init_agents(self):
        """Initialize all specialized agents with configs."""
        paths = self.main_config.get('paths', {})
        feature_engineering = self.main_config.get('feature_engineering', {})
        env_cfg = self.main_config.get('environment', {})
        risk_cfg = self.risk_limits_config

        self.data_agent = DataAgent(config={
            'data_dir_raw': paths.get('data_dir_raw', 'data/raw/'),
            'ibkr_conn': self.main_config.get('ibkr_conn', None)
        })
        self.feature_agent = FeatureAgent(config={ # Pass full feature_engineering config
            **self.main_config.get('feature_engineering', {}), # Original content
            'data_dir_processed': paths.get('data_dir_processed', 'data/processed/'), # Add these paths
            'scalers_dir': paths.get('scalers_dir', 'data/scalers/')
        }, read_only=self.read_only)
        self.env_agent = EnvAgent(config={
            'env_config': {
                'initial_capital': env_cfg.get('initial_capital', 100000.0),
                'transaction_cost_pct': env_cfg.get('transaction_cost_pct', 0.001),
                'reward_scaling': env_cfg.get('reward_scaling', 1.0),
                'max_episode_steps': env_cfg.get('max_episode_steps', None),
                'log_trades': env_cfg.get('log_trades_in_env', True),
                'lookback_window': feature_engineering.get('lookback_window', 1),
                'max_daily_drawdown_pct': risk_cfg.get('max_daily_drawdown_pct', 0.02),
                'hourly_turnover_cap': risk_cfg.get('hourly_turnover_cap', 5.0),
                'turnover_penalty_factor': risk_cfg.get('turnover_penalty_factor', 0.01),
                'position_sizing_pct_capital': env_cfg.get('position_sizing_pct_capital', 0.25),
                'trade_cooldown_steps': env_cfg.get('trade_cooldown_steps', 0),
                'terminate_on_turnover_breach': risk_cfg.get('terminate_on_turnover_breach', False),
                'turnover_termination_threshold_multiplier': risk_cfg.get('turnover_termination_threshold_multiplier', 2.0),
                # Enhanced turnover enforcement parameters
                'turnover_exponential_penalty_factor': risk_cfg.get('turnover_exponential_penalty_factor', 0.1),
                'turnover_termination_penalty_pct': risk_cfg.get('turnover_termination_penalty_pct', 0.05),
                # Enhanced Kyle Lambda fill simulation
                'enable_kyle_lambda_fills': env_cfg.get('kyle_lambda_fills', {}).get('enable_kyle_lambda_fills', True),
                'fill_simulator_config': env_cfg.get('kyle_lambda_fills', {}).get('fill_simulator_config', {})
            }
        })
        # Use new TrainerAgent factory function (bounded context)
        risk_limits_path="config/risk_limits_orchestrator_test.yaml"

        trainer_config = {
            'model_save_dir': paths.get('model_save_dir', 'models/'),
            'log_dir': paths.get('tensorboard_log_dir', 'logs/tensorboard/'),
            'algorithm': self.model_params_config.get('algorithm_name', 'DQN'),
            'algo_params': self.model_params_config.get('algorithm_params', {}),
            'training_params': self.main_config.get('training', {}),
            'risk_config': {
                'enabled': self.risk_limits_config.get('risk_aware_training', {}).get('enabled', False),
                'policy_yaml': risk_limits_path,
                'penalty_weight': self.risk_limits_config.get('risk_aware_training', {}).get('penalty_weight', 0.1),
                'early_stop_threshold': self.risk_limits_config.get('risk_aware_training', {}).get('early_stop_threshold', 0.8),
            }
        }
        self.trainer_agent = create_trainer_agent(trainer_config)
        self.evaluator_agent = EvaluatorAgent(config={
            'reports_dir': paths.get('reports_dir', 'reports/'),
            'eval_metrics': self.main_config.get('evaluation', {}).get('metrics', ['sharpe', 'max_drawdown'])
        })
        self.risk_agent = RiskAgentAdapter(config=risk_cfg)

    def _init_core_modules(self):
        """Initialize core execution modules."""
        try:
            # Initialize High-Performance Audit System FIRST (critical for latency)
            self._init_high_perf_audit()
            
            # Initialize ExecutionLoop
            self.execution_loop = ExecutionLoop(
                config=self.main_config,
                logger=self.logger.getChild("ExecutionLoop")
            )
            
            # Initialize OrderRouter
            self.order_router = OrderRouter(
                config=self.main_config,
                logger=self.logger.getChild("OrderRouter")
            )
            
            # Initialize PnLTracker
            self.pnl_tracker = PnLTracker(
                config=self.main_config,
                logger=self.logger.getChild("PnLTracker")
            )
            
            # Initialize LiveDataLoader
            self.live_data_loader = LiveDataLoader(
                config=self.main_config,
                logger=self.logger.getChild("LiveDataLoader")
            )
            
            # Register hooks between modules
            self.execution_loop.register_hook("action_generated", self._handle_action_generated)
            
            self.logger.info("Core execution modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing core modules: {e}")
            raise
    
    def _init_high_perf_audit(self):
        """Initialize high-performance audit system for critical trading paths."""
        try:
            from .core.high_perf_audit import initialize_global_audit_logger
            from .core.latency_monitor import initialize_global_latency_monitor
            
            # Configuration for high-performance audit (INCREASED BUFFER SIZES)
            audit_config = {
                'buffer_size': 131072,  # 128K records = 8MB ring buffer (DOUBLED to prevent backpressure)
                'emergency_buffer_size': 16384,  # 16K records for kill switches (DOUBLED)
                'log_directory': self.main_config.get('paths', {}).get('audit_log_dir', 'logs/audit_hiperf'),
                'flush_interval_ms': 2,  # 2ms flush interval (REDUCED from 5ms for faster I/O)
            }
            
            # Configuration for latency monitoring
            latency_config = {
                'enabled': True,
                'kill_switch_threshold_us': 10.0,  # Alert if KILL_SWITCH > 10µs
                'trade_execution_threshold_us': 50.0,
                'risk_check_threshold_us': 20.0,
                'audit_logging_threshold_us': 5.0,
                'max_samples_per_category': 50000,  # Keep more samples for analysis
            }
            
            # Initialize systems
            self.high_perf_audit = initialize_global_audit_logger(audit_config)
            self.latency_monitor = initialize_global_latency_monitor(latency_config)
            
            # Add alert callback for latency issues
            self.latency_monitor.add_alert_callback(self._handle_latency_alert)
            
            self.logger.info("High-performance audit system initialized - KILL_SWITCH latency optimized")
            self.logger.info("Latency monitoring active - targeting <10µs for KILL_SWITCH operations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize high-performance audit system: {e}")
            # Don't fail initialization - audit is important but not critical for basic operation
            self.high_perf_audit = None
            self.latency_monitor = None
    
    def _handle_latency_alert(self, alert_data: Dict):
        """Handle latency alerts from the monitoring system."""
        category = alert_data['category']
        latency_us = alert_data['latency_us']
        threshold_us = alert_data['threshold_us']
        operation = alert_data['operation']
        
        # Log critical latency issues
        if category == 'kill_switch' and latency_us > 35.0:
            self.logger.critical(
                f"CRITICAL LATENCY SPIKE DETECTED: KILL_SWITCH operation '{operation}' "
                f"took {latency_us:.2f}µs (similar to original 38µs issue)"
            )
            
            # Could trigger additional actions here:
            # - Increase audit buffer sizes dynamically
            # - Switch to emergency audit mode
            # - Alert operations team
        
        elif latency_us > threshold_us * 2:  # Double threshold breach
            self.logger.error(
                f"SEVERE LATENCY BREACH: {category} operation '{operation}' "
                f"took {latency_us:.2f}µs (threshold: {threshold_us:.2f}µs)"
            )

    def _handle_action_generated(self, symbol: str, action: int, current_bar: pd.Series):
        """Handle action generated by ExecutionLoop."""
        try:
            # Delegate to OrderRouter
            success = self.order_router.calculate_and_execute_action(
                symbol=symbol,
                action=action,
                current_bar=current_bar,
                portfolio_state=self.pnl_tracker.get_portfolio_state(),
                risk_agent=self.risk_agent
            )
            
            if success:
                # Update PnL tracker
                self.pnl_tracker.synchronize_portfolio_state_with_broker()
                self.pnl_tracker.update_net_liquidation_and_risk_agent(self.risk_agent)
                
        except Exception as e:
            self.logger.error(f"Error handling action: {e}")

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

    def _create_risk_wrapped_environment(self, base_env, is_training: bool = True):
        """
        Wrap the base environment with risk controls.
        
        Args:
            base_env: Base trading environment from EnvAgent
            is_training: Whether this is for training (vs evaluation)
            
        Returns:
            Risk-wrapped environment with volatility penalty and risk observations
        """
        # Get risk configuration
        risk_config = self.main_config.get('risk', {})
        
        # Use different drawdown limits for training vs evaluation
        if is_training:
            dd_limit = risk_config.get('dd_limit', 0.03)  # 3% for training
        else:
            dd_limit = risk_config.get('eval_dd_limit', 0.02)  # 2% for evaluation
        
        # Create risk manager configuration
        risk_manager_config = {
            'vol_window': risk_config.get('vol_window', 60),
            'penalty_lambda': risk_config.get('penalty_lambda', 0.25),
            'dd_limit': dd_limit
        }
        
        # Create RiskManager instance
        risk_manager = RiskManager(risk_manager_config)
        
        # Apply wrappers in order
        wrapped_env = base_env
        
        # 1. Add risk features to observation space (if enabled)
        if risk_config.get('include_risk_features', True):
            wrapped_env = RiskObsWrapper(wrapped_env, risk_manager)
            self.logger.info(f"✅ Applied RiskObsWrapper - extended observation space")
        
        # 2. Add volatility penalty to rewards
        wrapped_env = VolatilityPenaltyReward(wrapped_env, risk_manager)
        self.logger.info(f"✅ Applied VolatilityPenaltyReward - λ={risk_manager_config['penalty_lambda']}")
        
        # Store risk manager reference for logging
        wrapped_env._risk_manager = risk_manager
        
        phase_name = "training" if is_training else "evaluation"
        self.logger.info(f"🛡️ Risk-wrapped environment created for {phase_name} - "
                        f"dd_limit: {dd_limit:.1%}, vol_window: {risk_manager_config['vol_window']}")
        
        return wrapped_env

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
            data_duration_str = self.main_config.get('training', {}).get('data_duration_for_fetch')
            if data_duration_str is None:
                # Calculate duration from start_date to end_date
                from datetime import datetime
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date.split(' ')[0], "%Y-%m-%d")
                    duration_days = (end_dt - start_dt).days + 1
                    data_duration_str = f"{duration_days} D"
                    self.logger.info(f"Calculated duration from date range: {data_duration_str}")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate duration from dates: {e}, using default 90 D")
                    data_duration_str = "90 D"
            else:
                self.logger.info(f"Using configured data duration: {data_duration_str}")
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

        # Apply risk control wrappers
        training_environment = self._create_risk_wrapped_environment(
            training_environment, 
            is_training=True
        )

        # Log hardware info before training starts
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                gpu_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                self.logger.info(f"🚀 Training will use GPU: {gpu_name} ({gpu_memory / (1024**3):.1f} GB)")
            else:
                self.logger.info("💻 Training will use CPU (no CUDA GPU detected)")
        except Exception as e:
            self.logger.debug(f"Could not check hardware: {e}")

        # New TrainerAgent returns policy bundle path, not model path
        policy_bundle_path = self.trainer_agent.run(
            training_env=training_environment
            # Note: continue_from_model not supported in new architecture yet
        )
        if policy_bundle_path is None:
            self.logger.error("Model training failed. Training pipeline aborted.")
            return None
        self.logger.info(f"TrainerAgent completed training. Policy bundle saved at: {policy_bundle_path}")
        
        # Log risk metrics from training
        if hasattr(training_environment, '_risk_manager'):
            risk_summary = training_environment._risk_manager.get_episode_summary()
            self.logger.info(f"🛡️ Training Risk Summary: {risk_summary}")
        
        # For backward compatibility, extract model path from bundle
        bundle_path = Path(policy_bundle_path)
        if bundle_path.suffix == '.zip':
            # If bundle is a zip file, use it directly (contains complete SB3 model)
            trained_model_path = str(bundle_path)
        else:
            # If bundle is a directory, policy.pt is inside it
            trained_model_path = str(bundle_path / "policy.pt")

        self._trigger_hook('after_training', trained_model_path=trained_model_path)

        # Close write connections before evaluation to prevent DuckDB conflicts
        self.logger.info("🔒 Closing DuckDB write connections before evaluation...")
        try:
            close_write_duckdb_connections()
        except NameError:
            self.logger.warning("⚠️ close_write_duckdb_connections not available, skipping cleanup")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to close DuckDB connections: {e}")

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

        # Final cleanup of all DuckDB connections after training pipeline
        self.logger.info("🧹 Cleaning up all DuckDB connections after training...")
        close_all_duckdb_connections()
        
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
        
        # Ensure clean DuckDB state for evaluation (read-only operations)
        self.logger.info("🔒 Ensuring clean DuckDB state for evaluation...")
        try:
            close_write_duckdb_connections()
        except NameError:
            self.logger.warning("⚠️ close_write_duckdb_connections not available, skipping cleanup")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to close DuckDB connections: {e}")
        
        # Enable evaluation cache to prevent DuckDB re-connections during episodes
        try:
            from shared.evaluation_cache import get_evaluation_cache
        except ImportError:
            # Fallback for different execution contexts
            from src.shared.evaluation_cache import get_evaluation_cache
        eval_cache = get_evaluation_cache()
        eval_cache.enable()
        self.logger.info("📋 Evaluation feature cache enabled - preventing DuckDB thundering herd")

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

        # Apply risk control wrappers (with stricter evaluation limits)
        evaluation_environment = self._create_risk_wrapped_environment(
            evaluation_environment, 
            is_training=False
        )

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

        # Disable evaluation cache after evaluation completes
        cache_stats = eval_cache.get_stats()
        self.logger.info(f"📋 Evaluation cache stats: {cache_stats['entries']} entries, {cache_stats['total_memory_mb']:.1f} MB")
        eval_cache.disable()
        self.logger.info("📋 Evaluation feature cache disabled and cleared")
        
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
                
                # Close write connections between training and evaluation in walk-forward
                self.logger.info(f"🔒 Fold {fold_number}: Closing write connections before evaluation...")
                try:
                    close_write_duckdb_connections()
                except NameError:
                    self.logger.warning("⚠️ close_write_duckdb_connections not available, skipping cleanup")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to close DuckDB connections: {e}")
                
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

        # --- 2. Load Production Policy Bundle ---
        # New architecture: Load policy bundle using ExecutionAgentStub
        # This provides <100µs latency SLO and minimal dependencies
        
        # Check if model_path is a policy bundle directory or legacy model file
        policy_bundle_path = None
        if model_path and os.path.isdir(model_path):
            # Already a policy bundle directory
            policy_bundle_path = model_path
        elif model_path and os.path.exists(model_path):
            # Legacy model file - check if there's a corresponding bundle
            bundle_dir = Path(model_path).parent / f"{Path(model_path).stem}_bundle"
            if bundle_dir.exists():
                policy_bundle_path = str(bundle_dir)
                self.logger.info(f"Found policy bundle for legacy model: {policy_bundle_path}")
            else:
                self.logger.warning(f"Legacy model found but no policy bundle: {model_path}")
                self.logger.warning("Consider retraining to generate policy bundle for optimal performance")
                # TODO: Could implement legacy model loading as fallback
                self.logger.error("Legacy model loading not implemented in new architecture")
                return
        
        if not policy_bundle_path:
            self.logger.error(f"No policy bundle found for model path: {model_path}")
            return
        
        try:
            # Load policy bundle with ExecutionAgentStub
            self.execution_agent = create_execution_agent_stub(Path(policy_bundle_path))
            
            # Validate SLO compliance
            slo_results = self.execution_agent.validate_slo_compliance(num_trials=10)
            self.logger.info(f"Policy SLO validation: mean={slo_results['mean_latency_us']:.1f}µs, "
                           f"violations={slo_results['slo_violation_rate']:.2%}")
            
            if slo_results['slo_violation_rate'] > 0.1:  # More than 10% violations
                self.logger.warning("Policy may not meet production latency SLO")
            
            self.live_model = self.execution_agent  # Store for callbacks
            self.logger.info(f"Production policy bundle loaded: {policy_bundle_path}")
            
        except Exception as e: 
            self.logger.error(f"Error loading policy bundle {policy_bundle_path}: {e}", exc_info=True)
            return
        
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
        price_str = f"{new_bar_df[COL_CLOSE].iloc[0]:.2f}" if COL_CLOSE in new_bar_df.columns else 'N/A'
        self.logger.info(f"Processing bar for {symbol} at {current_time_of_bar}: Px={price_str}")
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
                # Comprehensive pre-trade risk check using all sensors
                quantity_signed = shares_to_trade if order_action == "BUY" else -shares_to_trade
                is_safe, action, detailed_reason = self.risk_agent.pre_trade_check(
                    symbol=symbol,
                    quantity=quantity_signed,
                    price=current_price,
                    timestamp=current_time_of_bar,
                    market_data=self._gather_market_data_for_risk_check(symbol, current_time_of_bar)
                )
                
                if is_safe:
                    self.logger.info(f"Trade for {symbol} approved by comprehensive risk check: {detailed_reason}")
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
                    # Handle different risk actions
                    if action == "LIQUIDATE":
                        self.logger.critical(f"LIQUIDATE signal for {symbol}: {detailed_reason}")
                        # Use existing halt logic for liquidation
                        if self.risk_limits_config.get('liquidate_on_halt', False):
                            self.logger.critical(f"LIQUIDATION required for {symbol} as per configuration.")
                    elif action == "HALT":
                        self.logger.critical(f"Trading halted for {symbol}: {detailed_reason}")
                        if self.risk_agent.halt_on_breach:
                            self.logger.critical(f"HALT signal from comprehensive risk check is active for {symbol}.")
                    else:
                        self.logger.warning(f"Trade blocked for {symbol}: {detailed_reason}")
                        if "HALT" in detailed_reason and self.risk_agent.halt_on_breach:
                            self.logger.critical(f"HALT signal from comprehensive risk check is active for {symbol}.")
                    if "HALT" in detailed_reason and self.risk_agent.halt_on_breach: # Check RiskAgent's own halt_on_breach config first
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

    def run_live_trading(self, symbol: str) -> None:
        """
        Public method to start live trading for a given symbol.
        This is the main entry point for live trading operations.
        
        Args:
            symbol (str): The trading symbol to trade live (e.g., "AAPL")
        """
        print(f"🚀 STARTING LIVE TRADING FOR {symbol}")
        self.logger.info(f"=== STARTING LIVE TRADING FOR {symbol} ===")
        try:
            # Delegate to ExecutionLoop with setup
            self._setup_live_trading(symbol)
            self.execution_loop.start_live_trading_loop(
                symbol=symbol,
                data_agent=self.data_agent,
                feature_agent=self.feature_agent,
                risk_agent=self.risk_agent,
                live_model=self.live_model
            )
        except Exception as e:
            print(f"❌ CRITICAL ERROR in live trading for {symbol}: {e}")
            self.logger.error(f"Critical error in live trading for {symbol}: {e}", exc_info=True)
        finally:
            print(f"🏁 LIVE TRADING FOR {symbol} ENDED")
            self.logger.info(f"=== LIVE TRADING FOR {symbol} ENDED ===")


    def _setup_live_trading(self, symbol: str) -> None:
        """
        Setup live trading by loading model and initializing portfolio state.
        
        Args:
            symbol: Trading symbol
        """
        # Get live trading configuration
        live_trading_config = self.main_config.get('live_trading', {})
        
        if not live_trading_config.get('enabled', False):
            raise ValueError(f"Live trading is not enabled for '{symbol}' in config")

        # Load production model
        model_path = live_trading_config.get('production_model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Production model not found: '{model_path}'")
            
        try:
            # Load model using trainer agent
            algo_name = self.model_params_config.get('algorithm_name', 'DQN')
            self.live_model = self.trainer_agent.load_model(model_path, algo_name)
            self.logger.info(f"Production model loaded successfully: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading production model: {e}")

        # Initialize portfolio state
        initial_capital = self.main_config.get('environment', {}).get('initial_capital', 100000)
        if not self.pnl_tracker.initialize_portfolio_state(initial_capital):
            raise RuntimeError("Failed to initialize portfolio state")

        # Sync portfolio state reference for backward compatibility
        self.portfolio_state = self.pnl_tracker.get_portfolio_state()

        # Perform warmup
        warmup_data = self.live_data_loader.load_warmup_data(symbol, self.data_agent)
        if warmup_data is None or warmup_data.empty:
            raise RuntimeError("Failed to load warmup data")
            
        # Process warmup data through feature agent
        warmup_features = self.feature_agent.engineer_features(warmup_data)
        if warmup_features is None or warmup_features.empty:
            raise RuntimeError("Failed to engineer features for warmup data")
            
        self.logger.info(f"Live trading setup completed for {symbol}")

    # Public API delegation methods for backward compatibility
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state (delegates to PnLTracker)."""
        return self.pnl_tracker.get_portfolio_state()
    
    def get_open_orders(self) -> Dict[int, Any]:
        """Get open orders (delegates to OrderRouter)."""
        return self.order_router.get_open_orders()
    
    def stop_live_trading(self, reason: str = "Manual stop", emergency: bool = False):
        """
        Stop live trading with ultra-low latency audit logging.
        
        Args:
            reason: Reason for stopping trading
            emergency: Whether this is an emergency stop (affects audit priority)
        """
        # CRITICAL PATH: Ultra-fast audit logging for emergency stops
        if emergency and self.high_perf_audit:
            try:
                from .core.high_perf_audit import audit_kill_switch, KillSwitchReason
                
                # Get current portfolio state for audit
                portfolio_state = self.pnl_tracker.get_portfolio_state()
                total_pnl_cents = int(portfolio_state.get('total_pnl', 0) * 100)  # Convert to cents
                
                # Determine reason code
                reason_lower = reason.lower()
                if 'loss' in reason_lower or 'drawdown' in reason_lower:
                    reason_code = KillSwitchReason.DAILY_LOSS_LIMIT
                elif 'position' in reason_lower:
                    reason_code = KillSwitchReason.POSITION_LIMIT
                elif 'risk' in reason_lower:
                    reason_code = KillSwitchReason.RISK_BREACH
                elif 'volatility' in reason_lower:
                    reason_code = KillSwitchReason.MARKET_VOLATILITY
                elif 'connectivity' in reason_lower or 'connection' in reason_lower:
                    reason_code = KillSwitchReason.CONNECTIVITY_LOSS
                elif 'error' in reason_lower:
                    reason_code = KillSwitchReason.SYSTEM_ERROR
                else:
                    reason_code = KillSwitchReason.MANUAL_STOP
                
                # Ultra-fast audit logging (sub-microsecond target)
                audit_kill_switch(
                    reason_code=reason_code,
                    symbol_id=0,  # Could be enhanced to include current symbol
                    position_size=0,  # Could be enhanced to include total position
                    pnl_cents=total_pnl_cents
                )
                
            except Exception as e:
                # Don't let audit failure block emergency stop
                self.logger.warning(f"High-perf audit failed during emergency stop: {e}")
        
        # Standard logging
        if emergency:
            self.logger.critical(f"EMERGENCY STOP: {reason}")
        else:
            self.logger.info(f"Stopping live trading: {reason}")
        
        # Delegate to ExecutionLoop
        self.execution_loop.stop_live_trading_loop()
        self.live_trading_active = False
    
    def emergency_stop(self, reason: str, symbol_id: int = 0, position_size: int = 0):
        """
        CRITICAL PATH: Emergency stop with ultra-low latency audit and monitoring.
        
        This method is optimized for minimum latency in critical situations.
        Target: <10µs total latency (solving the original 38µs spike issue)
        
        Args:
            reason: Emergency reason
            symbol_id: Symbol identifier (numeric for performance)
            position_size: Current position size
        """
        # ULTRA-CRITICAL PATH: Measure latency of the entire kill switch operation
        from .core.latency_monitor import measure_kill_switch_latency
        
        with measure_kill_switch_latency(f"emergency_stop_{reason[:20]}"):
            # STEP 1: Ultra-fast audit logging (target: <1µs)
            if self.high_perf_audit:
                try:
                    from .core.high_perf_audit import audit_kill_switch, KillSwitchReason
                    
                    # Get PnL quickly (pre-computed when possible)
                    portfolio_state = self.pnl_tracker.get_portfolio_state()
                    pnl_cents = int(portfolio_state.get('total_pnl', 0) * 100)
                    
                    # Fast reason code mapping (optimized lookup)
                    reason_lower = reason.lower()
                    if 'loss' in reason_lower:
                        reason_code = KillSwitchReason.DAILY_LOSS_LIMIT
                    elif 'position' in reason_lower:
                        reason_code = KillSwitchReason.POSITION_LIMIT
                    elif 'volatility' in reason_lower:
                        reason_code = KillSwitchReason.MARKET_VOLATILITY
                    elif 'error' in reason_lower:
                        reason_code = KillSwitchReason.SYSTEM_ERROR
                    elif 'connectivity' in reason_lower:
                        reason_code = KillSwitchReason.CONNECTIVITY_LOSS
                    else:
                        reason_code = KillSwitchReason.RISK_BREACH
                    
                    # CRITICAL: Sub-microsecond audit logging to shared memory ring buffer
                    audit_kill_switch(reason_code, symbol_id, position_size, pnl_cents)
                    
                except:
                    # Absolutely no exceptions allowed to escape this path
                    pass
            
            # STEP 2: Stop trading immediately (target: <5µs)
            self.stop_live_trading(reason=reason, emergency=True)
    
    def is_live_trading_active(self) -> bool:
        """Check if live trading is active (delegates to ExecutionLoop)."""
        return self.execution_loop.is_running

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

    def _run_live_trading_loop_conceptual(self, symbol: str) -> None:
        print(f"🚀 LIVE TRADING STARTED FOR {symbol} - Check console for progress!")
        self.logger.info(f"--- Attempting to Start LIVE TRADING for {symbol} ---")
        print(f"📝 Logger name: {self.logger.name}")
        print(f"📝 Logger level: {self.logger.level}")
        print(f"📝 Logger handlers: {self.logger.handlers}")
        live_trading_config = self.main_config.get('live_trading', {})
        print(f"📋 Live trading config: {live_trading_config}")
        if not live_trading_config.get('enabled', False):
            print(f"❌ Live trading is not enabled for '{symbol}' in config. Aborting.")
            self.logger.warning(f"Live trading is not enabled for '{symbol}' in config. Aborting."); return
        else:
            print(f"✅ Live trading is enabled for '{symbol}'")

        # Check if we're in simulation mode early
        ibkr_config = self.main_config.get('ibkr_conn', {}) or self.main_config.get('ibkr_connection', {})
        self.simulation_mode = ibkr_config.get('simulation_mode', False)
        simulation_mode = self.simulation_mode  # Keep local variable for compatibility

        def check_connection_and_sleep():
            """Helper function to handle connection checking and sleeping"""
            if self.simulation_mode:
                import time
                time.sleep(1)
                return True
            elif self.data_agent.ib and self.data_agent.ib.isConnected():
                self.data_agent.ib.sleep(1)
                return True
            else:
                return False

        model_path = live_trading_config.get('production_model_path')
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
            # Check if we're in simulation mode
            ibkr_config = self.main_config.get('ibkr_conn', {})
            simulation_mode = ibkr_config.get('simulation_mode', False)
            
            if simulation_mode:
                print("🎮 SIMULATION MODE: Skipping IBKR connection, using cached data")
                self.logger.info("🎮 SIMULATION MODE: Skipping IBKR connection, using cached data")
            else:
                print("🔌 Checking IBKR connection...")
                self.logger.info("🔌 Checking IBKR connection...")
                if not self.data_agent.ib_connected: 
                    print("🔌 Connecting to IBKR...")
                    self.logger.info("🔌 Connecting to IBKR...")
                    
                    # Add timeout for IBKR connection
                    import threading
                    import time
                    
                    connection_result = {"success": False, "error": None}
                    
                    def connect_with_timeout():
                        try:
                            self.data_agent._connect_ibkr()
                            connection_result["success"] = self.data_agent.ib_connected
                        except Exception as e:
                            connection_result["error"] = str(e)
                    
                    # Start connection in separate thread
                    connect_thread = threading.Thread(target=connect_with_timeout)
                    connect_thread.daemon = True
                    connect_thread.start()
                    
                    # Wait for connection with timeout
                    connect_thread.join(timeout=15)  # 15 second timeout
                    
                    if connect_thread.is_alive():
                        print("⏰ IBKR connection timed out after 15 seconds!")
                        self.logger.error("❌ IBKR connection timed out!")
                        return
                    elif connection_result["error"]:
                        print(f"❌ IBKR connection failed: {connection_result['error']}")
                        self.logger.error(f"❌ IBKR connection failed: {connection_result['error']}")
                        return
                    elif connection_result["success"]:
                        print("✅ Successfully connected to IBKR!")
                        self.logger.info("✅ Successfully connected to IBKR!")
                    else:
                        print("❌ Failed to connect to IBKR!")
                        self.logger.error("❌ Failed to connect to IBKR!")
                        return
                else:
                    print("✅ Already connected to IBKR!")
                    self.logger.info("✅ Already connected to IBKR!")
            
            # simulation_mode already defined at the top of function
            
            if not self.data_agent.ib_connected and not simulation_mode:
                self.logger.error("Failed to connect DataAgent to IBKR. Aborting."); return
            elif simulation_mode:
                self.logger.info("Running in simulation mode - using mock portfolio data")
                # Initialize mock portfolio for simulation
                self.portfolio_state['cash'] = initial_capital_for_risk
                self.portfolio_state['net_liquidation'] = initial_capital_for_risk
                self.portfolio_state['available_funds'] = initial_capital_for_risk
                self.portfolio_state['positions'] = {}
                self.logger.info(f"Mock portfolio initialized: {self.portfolio_state}")
            else:
                self.logger.info("💰 Fetching initial account summary...")
                account_summary = self.data_agent.get_account_summary(tags="NetLiquidation,TotalCashValue,AvailableFunds")
                if not account_summary: 
                    self.logger.error("❌ Failed to fetch initial account summary. Aborting."); return
                else:
                    self.logger.info(f"✅ Account summary received: {account_summary}")
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
                self.logger.info(f"📊 Attempting to fetch {historical_warmup_bars} historical bars for FeatureAgent warmup...")
                # Use current date for end_datetime_str in YYYY-MM-DD format
                from datetime import datetime
                end_dt_warmup_str = datetime.now().strftime("%Y-%m-%d")

                warmup_interval_str = live_trading_config.get('data_interval', '5 secs') # Use live data interval for warmup
                
                # Calculate appropriate duration_str for DataAgent.fetch_ibkr_bars
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
            elif not simulation_mode:
                self.logger.error("DataAgent.ib not initialized, cannot subscribe to order events."); return
            else:
                self.logger.info("Simulation mode: Skipping IBKR event subscriptions")

            live_contract_details = live_trading_config.get('contract_details', {})
            if not simulation_mode and not self.data_agent.subscribe_live_bars(symbol=symbol, sec_type=live_contract_details.get('sec_type', 'STK'), exchange=live_contract_details.get('exchange', 'SMART'), currency=live_contract_details.get('currency', 'USD')):
                self.logger.error(f"Failed to subscribe to live bars for {symbol}. Aborting."); return
            elif simulation_mode:
                self.logger.info("Simulation mode: Skipping live bar subscription")
            
            self.live_trading_active = True
            self.logger.info(f"Live trading loop starting for {symbol}...")
            last_portfolio_sync_time = datetime.now()
            portfolio_sync_interval = timedelta(minutes=live_trading_config.get('portfolio_sync_interval_minutes', 15))

            # Initialize simulation data feeding if in simulation mode
            simulation_data_index = 0
            simulation_data = None
            if simulation_mode:
                # Get the cached data for simulation
                simulation_data = self.data_agent.get_cached_data(symbol)
                if simulation_data is not None and not simulation_data.empty:
                    self.logger.info(f"🎮 Simulation: Starting to feed {len(simulation_data)} cached bars for {symbol}")
                else:
                    self.logger.warning("🎮 Simulation: No cached data available for feeding")
                    self.live_trading_active = False  # End simulation if no data

            while self.live_trading_active:
                if simulation_mode:
                    import time
                    # Feed cached data bar by bar in simulation mode
                    if simulation_data is not None and simulation_data_index < len(simulation_data):
                        # Get current bar
                        current_bar = simulation_data.iloc[simulation_data_index:simulation_data_index+1]
                        self.logger.info(f"🎮 Simulation: Feeding bar {simulation_data_index+1}/{len(simulation_data)} for {symbol}")
                        
                        # Process the bar through the trading pipeline
                        self._process_incoming_bar(current_bar, symbol)
                        
                        simulation_data_index += 1
                        time.sleep(2)  # Simulate real-time delay between bars
                    else:
                        if simulation_data is not None and simulation_data_index >= len(simulation_data):
                            self.logger.info(f"🎮 Simulation: All {len(simulation_data)} bars processed for {symbol}. Ending simulation.")
                        else:
                            self.logger.info("🎮 Simulation: No data available. Ending simulation.")
                        self.live_trading_active = False
                        break
                elif self.data_agent.ib and self.data_agent.ib.isConnected():
                    self.data_agent.ib.sleep(1)
                else: self.logger.error("IB connection lost. Halting."); self.live_trading_active = False; break
                
                current_loop_time = datetime.now()
                if current_loop_time - last_portfolio_sync_time > portfolio_sync_interval:
                    self._synchronize_portfolio_state_with_broker(symbol)
                    last_portfolio_sync_time = current_loop_time

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
        price_str = f"{new_bar_df[COL_CLOSE].iloc[0]:.2f}" if COL_CLOSE in new_bar_df.columns else 'N/A'
        self.logger.info(f"Processing bar for {symbol} at {current_time_of_bar}: Px={price_str}")
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
                # Comprehensive pre-trade risk check using all sensors
                quantity_signed = shares_to_trade if order_action == "BUY" else -shares_to_trade
                is_safe, action, detailed_reason = self.risk_agent.pre_trade_check(
                    symbol=symbol,
                    quantity=quantity_signed,
                    price=current_price,
                    timestamp=current_time_of_bar,
                    market_data=self._gather_market_data_for_risk_check(symbol, current_time_of_bar)
                )
                
                if is_safe:
                    self.logger.info(f"Trade for {symbol} approved by comprehensive risk check: {detailed_reason}")
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
                    # Handle different risk actions
                    if action == "LIQUIDATE":
                        self.logger.critical(f"LIQUIDATE signal for {symbol}: {detailed_reason}")
                        # Use existing halt logic for liquidation
                        if self.risk_limits_config.get('liquidate_on_halt', False):
                            self.logger.critical(f"LIQUIDATION required for {symbol} as per configuration.")
                    elif action == "HALT":
                        self.logger.critical(f"Trading halted for {symbol}: {detailed_reason}")
                        if self.risk_agent.halt_on_breach:
                            self.logger.critical(f"HALT signal from comprehensive risk check is active for {symbol}.")
                    else:
                        self.logger.warning(f"Trade blocked for {symbol}: {detailed_reason}")
                        if "HALT" in detailed_reason and self.risk_agent.halt_on_breach:
                            self.logger.critical(f"HALT signal from comprehensive risk check is active for {symbol}.")
                    if "HALT" in detailed_reason and self.risk_agent.halt_on_breach:
                        self.logger.critical(f"HALT signal from RiskAgent is active for {symbol} due to: {reason}.")
                        if self.risk_limits_config.get('liquidate_on_halt', False):
                            self.logger.critical(f"LIQUIDATION required for {symbol} as per configuration.")
                            current_holdings_for_liq = self.portfolio_state.get('positions', {}).get(symbol, {}).get('shares', 0.0)
                            if current_holdings_for_liq != 0:
                                liquidation_action = "SELL" if current_holdings_for_liq > 0 else "BUY"
                                liquidation_quantity = abs(current_holdings_for_liq)
                                self.logger.info(f"Attempting to liquidate {liquidation_quantity} shares of {symbol} via {liquidation_action} order.")
                                
                                live_cfg_liq = self.main_config.get('live_trading', {})
                                contract_cfg_liq = live_cfg_liq.get('contract_details', {})
                                liquidation_order_details = {
                                    'symbol': symbol,
                                    'sec_type': contract_cfg_liq.get('sec_type','STK'),
                                    'exchange': contract_cfg_liq.get('exchange','SMART'),
                                    'currency': contract_cfg_liq.get('currency','USD'),
                                    'primary_exchange': contract_cfg_liq.get('primary_exchange'),
                                    'action': liquidation_action,
                                    'quantity': liquidation_quantity,
                                    'order_type': "MKT",
                                    'tif': "DAY",
                                    'account': self.main_config.get('ibkr_connection',{}).get('account_id')
                                }
                                liq_trade_object = self.data_agent.place_order(liquidation_order_details)
                                if liq_trade_object and liq_trade_object.order and liq_trade_object.orderStatus:
                                    self.logger.info(f"Liquidation order for {symbol} ({liquidation_action} {liquidation_quantity:.0f}) submitted. ID: {liq_trade_object.order.orderId}, Status: {liq_trade_object.orderStatus.status}")
                                    self.open_trades[liq_trade_object.order.orderId] = liq_trade_object
                                else:
                                    self.logger.error(f"Failed to place LIQUIDATION order for {symbol}.")
                                self.logger.warning(f"Further trading for {symbol} might be suspended post-liquidation signal.")
                            else:
                                self.logger.info(f"Liquidation for {symbol} not needed, already flat.")
                        else:
                             self.logger.info(f"HALT active for {symbol}, but liquidate_on_halt is false. No liquidation order placed.")
            else: self.logger.debug(f"No trade for {symbol} at {current_time_of_bar}. Target={target_position_signal}, CurrentShares={current_holdings}")
            self.risk_agent.update_portfolio_value(self.portfolio_state.get('net_liquidation',0.0), current_time_of_bar)
        except Exception as e: self.logger.error(f"Critical error in _process_incoming_bar for {symbol} at {current_time_of_bar}: {e}", exc_info=True)

    def _synchronize_portfolio_state_with_broker(self, symbol_traded: Optional[str] = None):
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
        """Calculates an IBKR-compatible duration string to fetch approximately num_bars."""
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
                # Handle both "1 min" and "1min" formats
                if " " in bar_size_str:
                    mins = int(bar_size_str.split(" ")[0])
                else:
                    mins = int(bar_size_str.replace("min", ""))
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
        else:
            self.logger.error(f"Unsupported bar_size_str for warmup duration calculation: {bar_size_str}")
            return None

        if total_seconds_needed <= 0: return None

        if total_seconds_needed <= 7200:
            if "sec" in bar_size_str and int(bar_size_str.split(" ")[0]) >= 5 and total_seconds_needed > 7200:
                 self.logger.warning(f"Calculated {total_seconds_needed}S exceeds typical 7200S limit for >=5sec bars. Capping to 7200 S.")
                 return "7200 S"
            if "sec" in bar_size_str and int(bar_size_str.split(" ")[0]) < 5 and total_seconds_needed > 1800:
                 self.logger.warning(f"Calculated {total_seconds_needed}S exceeds typical 1800S limit for <5sec bars. Capping to 1800 S.")
                 return "1800 S"
            return f"{total_seconds_needed} S"
        
        total_days_needed = total_seconds_needed / (24 * 3600)
        if total_days_needed <= 30:
             if "min" in bar_size_str or "sec" in bar_size_str:
                 days_to_request = int(np.ceil(total_days_needed))
                 if days_to_request == 0: days_to_request = 1
                 
                 if ("min" in bar_size_str or "sec" in bar_size_str) and days_to_request > 5:
                     self.logger.warning(f"Warmup requires {days_to_request} days for {bar_size_str}. Capping to 5 D to be safe with IBKR limits for intraday history. May fetch fewer than {num_bars} bars.")
                     return "5 D"
                 return f"{days_to_request} D"

        total_months_needed = total_days_needed / 30
        if total_months_needed <= 12:
            return f"{int(np.ceil(total_months_needed))} M"
        
        total_years_needed = total_months_needed / 12
        return f"{int(np.ceil(total_years_needed))} Y"

    # --- IBKR Event Handlers for Order/Execution Status ---
    def _on_order_status_update(self, trade: IBTrade):
        if not self.live_trading_active or not trade or not trade.orderStatus: return
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        self.logger.info(f"Order Status: ID {order_id}, Sym {trade.contract.symbol}, Status: {status}, Filled: {trade.orderStatus.filled}, Rem: {trade.orderStatus.remaining}, AvgFillPx: {trade.orderStatus.avgFillPrice:.2f if trade.orderStatus.avgFillPrice else 'N/A'}")
        if status in ['Cancelled', 'ApiCancelled', 'Inactive', 'PendingCancel', 'ApiPendingCancel', 'Expired']:
            if order_id in self.open_trades: self.logger.info(f"Order {order_id} ({status}) removed from active monitoring."); self.open_trades.pop(order_id, None)
        elif status == 'Filled':
            if order_id in self.open_trades: self.logger.info(f"Order {order_id} fully filled reported by OrderStatus. Awaiting ExecDetails. Keeping in open_trades for now.")

    def _on_execution_details(self, trade: IBTrade, fill: IBFill):
        if not self.live_trading_active or not fill or not fill.execution: return
        exec_id = fill.execution.execId; order_id = fill.execution.orderId; symbol = trade.contract.symbol
        shares = fill.execution.shares; price = fill.execution.price; action_side = fill.execution.side
        exec_time = pd.to_datetime(fill.time) if fill.time else datetime.now()
        self.logger.info(f"Execution Detail: ExecID {exec_id}, OrderID {order_id} for {symbol}: {action_side} {shares} @ {price:.2f} at {exec_time}")

        if symbol not in self.portfolio_state.get('positions', {}): self.portfolio_state['positions'][symbol] = {'shares': 0.0, 'average_cost': 0.0}
        current_pos_info = self.portfolio_state['positions'][symbol]
        current_shares = current_pos_info.get('shares', 0.0)
        current_avg_cost = current_pos_info.get('average_cost', 0.0)
        trade_value = shares * price
        
        commission = getattr(fill, 'commissionReport', {}).get('commission', 0.0)
        if commission == 0.0: self.logger.debug(f"Commission not in fill object for ExecID {exec_id}, will await CommissionReport or use estimate.")

        # Update portfolio based on fill
        if action_side == "BOT":
            new_total_shares = current_shares + shares
            if current_shares < 0 and new_total_shares > 0: current_pos_info['average_cost'] = price
            elif new_total_shares != 0: current_pos_info['average_cost'] = ((current_avg_cost * current_shares) + trade_value) / new_total_shares if current_shares >=0 else price
            else: current_pos_info['average_cost'] = 0.0
            current_pos_info['shares'] = new_total_shares
            self.portfolio_state['cash'] -= (trade_value + commission)
        elif action_side == "SLD":
            new_total_shares = current_shares - shares
            if current_shares > 0 and new_total_shares < 0: current_pos_info['average_cost'] = price
            elif new_total_shares != 0: current_pos_info['average_cost'] = ((current_avg_cost * abs(current_shares)) + trade_value) / abs(new_total_shares) if current_shares < 0 else current_pos_info['average_cost']
            else: current_pos_info['average_cost'] = 0.0
            current_pos_info['shares'] = new_total_shares
            self.portfolio_state['cash'] += (trade_value - commission)
        
        self.logger.info(f"Portfolio for {symbol}: Shares={current_pos_info['shares']:.0f}, AvgCost={current_pos_info['average_cost']:.2f}, Cash={self.portfolio_state['cash']:.2f}")
        
        # Update RiskAgent
        self.risk_agent.record_trade(trade_value, exec_time)
        self._update_net_liquidation_and_risk_agent(exec_time)

        # Manage open_trades
        if order_id in self.open_trades:
            if self.open_trades[order_id].orderStatus.status == 'Filled' or \
               self.open_trades[order_id].orderStatus.remaining == 0 or \
               (hasattr(trade, 'remaining') and trade.remaining() == 0):
                self.logger.info(f"Order {order_id} fully filled. Removing from open_trades.")
                self.open_trades.pop(order_id, None)
            else:
                self.logger.debug(f"Order {order_id} partially filled. Remaining in open_trades.")

    def _on_commission_report(self, trade: IBTrade, fill: IBFill, commission_report: IBCommissionReport):
        if not self.live_trading_active: return
        exec_id = commission_report.execId; commission = commission_report.commission; currency = commission_report.currency
        realized_pnl = commission_report.realizedPNL
        self.logger.info(f"Commission Rpt: ExecID {exec_id}, Comm {commission:.2f} {currency}, RealizedPNL (fill): {realized_pnl if realized_pnl is not None else 'N/A'}")

    def _update_net_liquidation_and_risk_agent(self, current_time: datetime):
        """Helper to update net liquidation value and notify RiskAgent."""
        estimated_net_liq = self.portfolio_state.get('cash', 0.0)
        for symbol, pos_info in self.portfolio_state.get('positions', {}).items():
            estimated_net_liq += pos_info.get('market_value', 0.0)
        
        self.portfolio_state['net_liquidation'] = estimated_net_liq
        self.risk_agent.update_portfolio_value(estimated_net_liq, current_time)
        # ... (rest of schedule_weekly_retrain - unchanged for this step)
        # [schedule_weekly_retrain logic as previously defined]
        self.logger.info("--- Starting Scheduled Weekly Retrain ---") # Simplified for brevity
        pass # Placeholder
    
    def _gather_market_data_for_risk_check(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Gather market data for comprehensive risk assessment.
        
        In production, this would collect:
        - Order book depth and spread
        - Recent trade history and volumes
        - Feed timestamps and latencies
        - Market volatility and liquidity metrics
        
        For now, returns basic data structure that sensors can work with.
        """
        try:
            # Get current market data from data agent if available
            market_data = {}
            
            # Try to get recent price data
            if hasattr(self.data_agent, 'get_recent_bars'):
                try:
                    recent_bars = self.data_agent.get_recent_bars(symbol, count=100)
                    if recent_bars is not None and len(recent_bars) > 0:
                        market_data['recent_prices'] = recent_bars[COL_CLOSE].values.tolist()
                        market_data['recent_volumes'] = recent_bars[COL_VOLUME].values.tolist()
                        market_data['recent_highs'] = recent_bars[COL_HIGH].values.tolist()
                        market_data['recent_lows'] = recent_bars[COL_LOW].values.tolist()
                except Exception as e:
                    self.logger.debug(f"Could not get recent bars for {symbol}: {e}")
            
            # Add portfolio context
            market_data.update({
                'symbol': symbol,
                'current_positions': self.portfolio_state.get('positions', {}),
                'available_funds': self.portfolio_state.get('available_funds', 0.0),
                'net_liquidation': self.portfolio_state.get('net_liquidation', 0.0),
                'timestamp': timestamp,
                
                # Mock feed timestamps (in production, get from actual feeds)
                'feed_timestamps': {
                    'market_data': timestamp.timestamp() - 0.1,  # 100ms old
                    'order_book': timestamp.timestamp() - 0.05,  # 50ms old
                    'trades': timestamp.timestamp() - 0.2,       # 200ms old
                    'news': timestamp.timestamp() - 1.0         # 1s old
                },
                
                # Mock order book (in production, get from actual order book)
                'order_book_depth': {
                    symbol: {
                        'bids': [(149.95, 1000), (149.90, 2000), (149.85, 1500)],
                        'asks': [(150.05, 1200), (150.10, 1800), (150.15, 2200)]
                    }
                },
                
                # Mock recent order latencies (in production, track actual latencies)
                'order_latencies': [45.0, 52.0, 48.0, 55.0, 47.0],  # milliseconds
                
                # Daily volume data (mock - in production get from market data)
                'daily_volumes': {
                    symbol: [50000000] * 20  # Mock 20-day volume history
                }
            })
            
            return market_data
            
        except Exception as e:
            self.logger.warning(f"Failed to gather market data for risk check: {e}")
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'feed_timestamps': {
                    'market_data': timestamp.timestamp() - 0.1,
                    'order_book': timestamp.timestamp() - 0.05,
                    'trades': timestamp.timestamp() - 0.2,
                }
            }
    
    def _gather_market_data_for_risk_check(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Gather market data for comprehensive risk assessment.
        
        In production, this would collect:
        - Order book depth and spread
        - Recent trade history and volumes
        - Feed timestamps and latencies
        - Market volatility and liquidity metrics
        
        For now, returns basic data structure that sensors can work with.
        """
        try:
            # Get current market data from data agent if available
            market_data = {}
            
            # Try to get recent price data
            if hasattr(self.data_agent, 'get_recent_bars'):
                try:
                    recent_bars = self.data_agent.get_recent_bars(symbol, count=100)
                    if recent_bars is not None and len(recent_bars) > 0:
                        market_data['recent_prices'] = recent_bars[COL_CLOSE].values.tolist()
                        market_data['recent_volumes'] = recent_bars[COL_VOLUME].values.tolist()
                        market_data['recent_highs'] = recent_bars[COL_HIGH].values.tolist()
                        market_data['recent_lows'] = recent_bars[COL_LOW].values.tolist()
                except Exception as e:
                    self.logger.debug(f"Could not get recent bars for {symbol}: {e}")
            
            # Add portfolio context
            market_data.update({
                'symbol': symbol,
                'current_positions': self.portfolio_state.get('positions', {}),
                'available_funds': self.portfolio_state.get('available_funds', 0.0),
                'net_liquidation': self.portfolio_state.get('net_liquidation', 0.0),
                'timestamp': timestamp,
                
                # Mock feed timestamps (in production, get from actual feeds)
                'feed_timestamps': {
                    'market_data': timestamp.timestamp() - 0.1,  # 100ms old
                    'order_book': timestamp.timestamp() - 0.05,  # 50ms old
                    'trades': timestamp.timestamp() - 0.2,       # 200ms old
                    'news': timestamp.timestamp() - 1.0         # 1s old
                },
                
                # Mock order book (in production, get from actual order book)
                'order_book_depth': {
                    symbol: {
                        'bids': [(149.95, 1000), (149.90, 2000), (149.85, 1500)],
                        'asks': [(150.05, 1200), (150.10, 1800), (150.15, 2200)]
                    }
                },
                
                # Mock recent order latencies (in production, track actual latencies)
                'order_latencies': [45.0, 52.0, 48.0, 55.0, 47.0],  # milliseconds
                
                # Additional data for calculators
                'portfolio_values': [self.portfolio_state.get('net_liquidation', 0.0)] * 10,
                'trade_values': [100000.0] * 5,  # Recent trade values
                'timestamps': [timestamp.timestamp() - i*60 for i in range(5)],
                'price_changes': [0.001, -0.002, 0.0015, -0.0005, 0.0008],
                'returns': [0.001, -0.002, 0.0015, -0.0005, 0.0008],
                'positions': self.portfolio_state.get('positions', {}),
                
                # Daily volume data (mock - in production get from market data)
                'daily_volumes': {
                    symbol: [50000000] * 20  # Mock 20-day volume history
                }
            })
            
            return market_data
            
        except Exception as e:
            self.logger.warning(f"Failed to gather market data for risk check: {e}")
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'feed_timestamps': {
                    'market_data': timestamp.timestamp() - 0.1,
                    'order_book': timestamp.timestamp() - 0.05,
                    'trades': timestamp.timestamp() - 0.2,
                }
            }


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
            'time': ['hour_of_day'], 'lookback_window': 3,
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
        },
        'scheduling': {
            'retrain_symbol': "AAPL",
            'retrain_interval': "1min",
            'retrain_data_start_offset_days': 30,
            'retrain_data_end_offset_days': 7,
            'evaluate_after_retrain_duration_days': 5,
            'use_cached_data_for_scheduled_run': False
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

