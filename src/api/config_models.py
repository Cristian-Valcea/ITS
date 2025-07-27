# src/api/config_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

# Pydantic models should ideally mirror the structure of your YAML files
# to enable validation and clear API request/response schemas.

# --- main_config.yaml Models ---

class PathsConfig(BaseModel):
    data_dir_raw: str = Field("data/raw/", description="Path for raw market data")
    data_dir_processed: str = Field("data/processed/", description="Path for engineered features")
    scalers_dir: str = Field("data/scalers/", description="Path for saved feature scalers")
    model_save_dir: str = Field("models/", description="Path for saved trained RL models")
    tensorboard_log_dir: str = Field("logs/tensorboard/", description="Path for TensorBoard logs")
    application_log_dir: str = Field("logs/", description="Path for general application logs")
    reports_dir: str = Field("reports/", description="Path for evaluation reports")

class IbkrConnectionConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7497
    clientId: int = 100
    timeout_seconds: int = 10
    readonly: bool = False
    account_number: Optional[str] = None

class DefaultDataParamsConfig(BaseModel):
    symbol: str = "SPY"
    interval: str = "1min"
    what_to_show: str = "TRADES"
    use_rth: bool = True

class RsiFeatureConfig(BaseModel):
    window: int = 14

class EmaFeatureConfig(BaseModel):
    windows: List[int] = [10, 20, 50]
    ema_diff: Optional[Dict[str, int]] = Field(None, example={'fast_ema_idx': 0, 'slow_ema_idx': 1})

class VwapFeatureConfig(BaseModel):
    window: Optional[int] = None # For rolling VWAP
    group_by_day: bool = True # For daily VWAP
    calculate_deviation: bool = True

class OrderBookImbalanceConfig(BaseModel): # Placeholder
    levels_to_consider: int = 5

class FeatureEngineeringConfig(BaseModel):
    features_to_calculate: List[str] = ['RSI', 'EMA', 'VWAP', 'Time']
    rsi: RsiFeatureConfig = Field(default_factory=RsiFeatureConfig)
    ema: EmaFeatureConfig = Field(default_factory=EmaFeatureConfig)
    vwap: VwapFeatureConfig = Field(default_factory=VwapFeatureConfig)
    order_book_imbalance: Optional[OrderBookImbalanceConfig] = None
    time_features_list: List[str] = Field(default_factory=lambda: ['hour_of_day', 'day_of_week'])
    sin_cos_encode: List[str] = Field(default_factory=lambda: ['hour_of_day'])
    lookback_window: int = Field(1, ge=1)
    feature_cols_to_scale: List[str] = Field(default_factory=list)
    observation_feature_cols: List[str] = Field(default_factory=list) # Market features for env

class EnvironmentConfig(BaseModel):
    initial_capital: float = 100000.0
    transaction_cost_pct: float = 0.0005
    reward_scaling: float = 1.0
    max_episode_steps: Optional[int] = None
    log_trades_in_env: bool = True
    # Risk params like max_daily_drawdown_pct, hourly_turnover_cap, turnover_penalty_factor
    # will be dynamically added by Orchestrator from risk_limits.yaml into the env_config for EnvAgent
    # but are part of IntradayTradingEnv's constructor, so they could be defined here too if preferred.
    # For now, keeping them separate as Orchestrator merges them.

class TrainingConfig(BaseModel):
    total_timesteps: int = 20000
    log_interval: int = 100 # For SB3 console log (episodes)
    checkpoint_freq: int = 50000
    eval_freq: int = 100000
    use_eval_callback: bool = False
    data_duration_for_fetch: str = "365 D"

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ['total_return_pct', 'sharpe_ratio'])
    data_duration_for_fetch: str = "90 D"
    default_post_train_eval_duration_days: int = 30
    risk_free_rate_annual: float = Field(0.0, description="Annual risk-free rate for Sharpe Ratio")
    periods_per_year_for_sharpe: int = Field(252, description="Number of periods in a year for Sharpe Ratio annualization (e.g., 252 for daily, 52 for weekly)")


class SchedulingConfig(BaseModel): # Conceptual
    retrain_cron_schedule: str = "0 0 * * 0"
    retrain_symbol: str = "SPY"
    # ... other scheduling params

class MainConfig(BaseModel):
    project_name: str = "IntradayRLTrader"
    version: str = "0.1.0"
    global_seed: Optional[int] = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ibkr_connection: Optional[IbkrConnectionConfig] = None
    default_data_params: DefaultDataParamsConfig = Field(default_factory=DefaultDataParamsConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig) # Basic env settings
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    scheduling: Optional[SchedulingConfig] = None


# --- model_params.yaml Models ---

class C51FeaturesConfig(BaseModel):
    dueling_nets: bool = True
    use_per: bool = True
    n_step_returns: int = Field(3, ge=1)
    use_noisy_nets: bool = False
    distributional_rl_params: Optional[Dict[str, Any]] = Field(None, example={'num_atoms': 51, 'v_min': -10, 'v_max': 10})

class ModelParamsConfig(BaseModel):
    algorithm_name: str = Field("DQN", description="RL Algorithm (e.g., DQN, C51)")
    # algorithm_params can be quite varied, using Dict[str, Any] for flexibility.
    # For specific algorithms, you might create more detailed Pydantic models.
    algorithm_params: Dict[str, Any] = Field(default_factory=lambda: {
        "policy": "MlpPolicy",
        "learning_rate": 0.0001,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "verbose": 1,
        "seed": None # Will use global_seed if None
    })
    c51_features: C51FeaturesConfig = Field(default_factory=C51FeaturesConfig)
    # optuna_hpo: Optional[Dict[str, Any]] = None # Placeholder for Optuna config


# --- risk_limits.yaml Models ---

class RiskLimitsConfig(BaseModel):
    max_daily_drawdown_pct: float = Field(0.02, ge=0, le=1)
    max_hourly_turnover_ratio: float = Field(5.0, ge=0)
    max_daily_turnover_ratio: float = Field(20.0, ge=0)
    halt_on_breach: bool = True
    liquidate_on_halt: bool = False
    # This is for the environment's penalty, distinct from RiskAgent's live trading halt
    env_turnover_penalty_factor: float = Field(0.01, ge=0) 


# Example usage (not part of API, just for testing model definitions)
if __name__ == "__main__":
    main_conf_data = {
        "paths": {"data_dir_raw": "data/custom_raw/"},
        "feature_engineering": {"lookback_window": 10, "rsi": {"window": 20}},
        "environment": {"initial_capital": 5000}
    }
    main_config_obj = MainConfig(**main_conf_data)
    print("Parsed MainConfig example:")
    print(main_config_obj.model_dump_json(indent=2))
    print(f"\nData dir raw: {main_config_obj.paths.data_dir_raw}")
    print(f"RSI window: {main_config_obj.feature_engineering.rsi.window}")
    print(f"Lookback window: {main_config_obj.feature_engineering.lookback_window}")
    print(f"Initial capital: {main_config_obj.environment.initial_capital}")

    model_p_data = {
        "algorithm_name": "DQN",
        "algorithm_params": {"learning_rate": 0.0005, "policy_kwargs": {"net_arch": [128, 128]}},
        "c51_features": {"use_per": True, "n_step_returns": 5}
    }
    model_params_obj = ModelParamsConfig(**model_p_data)
    print("\nParsed ModelParamsConfig example:")
    print(model_params_obj.model_dump_json(indent=2))
    print(f"Algorithm: {model_params_obj.algorithm_name}")
    print(f"Learning Rate: {model_params_obj.algorithm_params['learning_rate']}")
    print(f"Use PER: {model_params_obj.c51_features.use_per}")

    risk_l_data = {"max_daily_drawdown_pct": 0.03, "halt_on_breach": False}
    risk_limits_obj = RiskLimitsConfig(**risk_l_data)
    print("\nParsed RiskLimitsConfig example:")
    print(risk_limits_obj.model_dump_json(indent=2))
    print(f"Max Daily Drawdown: {risk_limits_obj.max_daily_drawdown_pct}")
    print(f"Halt on Breach: {risk_limits_obj.halt_on_breach}")

    # Test default instantiation
    default_main = MainConfig()
    print("\nDefault MainConfig:")
    print(default_main.model_dump_json(indent=2))
