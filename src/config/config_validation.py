#!/usr/bin/env python3
"""
Schema-Validated Configuration System
Addresses reviewer concern about YAML config sprawl and type safety

Uses Pydantic for schema validation to catch configuration errors at CI time
rather than runtime failures in production.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
import yaml
import logging
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Market regime classifications"""
    POST_COVID_BULL = "post_covid_bull_2022_h2"
    RATE_HIKE_CHOP = "rate_hike_chop_2023"
    AI_MELT_UP = "ai_melt_up_2024_2025"


class StorageBackend(str, Enum):
    """Storage backend options"""
    HDD = "HDD"
    SSD = "SSD"
    S3 = "s3"
    MINIO = "minio"


class DataConfig(BaseModel):
    """Data configuration with validation"""
    
    class DataHorizon(BaseModel):
        start_date: date = Field(..., description="Start date for data collection")
        end_date: Union[date, Literal["auto"]] = Field(default="auto")
        justification: str = Field(..., min_length=50)
        
        @field_validator('start_date')
        @classmethod
        def validate_start_date(cls, v):
            if v > date.today():
                raise ValueError("Start date cannot be in the future")
            if v < date(2020, 1, 1):
                raise ValueError("Start date too far in the past")
            return v
    
    class DataSplits(BaseModel):
        train_pct: int = Field(..., ge=50, le=80, description="Training split percentage")
        validation_pct: int = Field(..., ge=10, le=30)
        holdout_pct: int = Field(..., ge=5, le=20)
        lockbox_pct: int = Field(..., ge=5, le=20)
        shuffle_allowed: bool = Field(default=False)
        zero_peek_enforcement: bool = Field(default=True)
        lockbox_access_log: str = Field(default="lockbox_access_audit.json")
        
        @model_validator(mode='after')
        def validate_split_totals(self):
            total = (self.train_pct + 
                    self.validation_pct + 
                    self.holdout_pct + 
                    self.lockbox_pct)
            if total != 100:
                raise ValueError(f"Split percentages must sum to 100, got {total}")
            return self
    
    class BarResolution(BaseModel):
        core_training: Literal["1min", "5min", "15min"] = "1min"
        justification: str = Field(..., min_length=30)
        feature_windows: List[str] = Field(default=["5min", "15min"])
        
        @field_validator('feature_windows')
        @classmethod
        def validate_feature_windows(cls, v):
            valid_windows = ["1min", "5min", "15min", "30min", "1h"]
            for window in v:
                if window not in valid_windows:
                    raise ValueError(f"Invalid feature window: {window}")
            return v
    
    class FeatureEngineering(BaseModel):
        lag_enforcement: bool = Field(default=True)
        max_allowed_lag: int = Field(default=0, ge=0)
        stationarity_required: bool = Field(default=True)
        preprocessing_pipeline: List[str] = Field(
            default=["log_returns", "z_score_normalize", "outlier_winsorize"]
        )
    
    class ExclusionFilters(BaseModel):
        class EarningsExclusion(BaseModel):
            enabled: bool = True
            cost_benefit_analysis: str = Field(..., min_length=100)
            days_before: int = Field(default=1, ge=0, le=5)
            days_after: int = Field(default=1, ge=0, le=5)
        
        class FOMCExclusion(BaseModel):
            enabled: bool = True
            adaptive_window: bool = Field(default=True)
            static_window_start: str = Field(default="14:00")
            static_window_end: str = Field(default="15:00")
        
        class ExtremeVolatilityFilter(BaseModel):
            method: Literal["static", "adaptive_mad"] = "adaptive_mad"
            static_threshold: Optional[float] = Field(None, gt=0, le=0.20)
            mad_multiplier: float = Field(default=5.0, gt=0)
            
            @root_validator
            def validate_method_params(cls, values):
                method = values.get('method')
                if method == 'static' and values.get('static_threshold') is None:
                    raise ValueError("static_threshold required when method='static'")
                return values
        
        earnings_exclusion: EarningsExclusion
        fomc_exclusion: FOMCExclusion
        extreme_volatility_filter: ExtremeVolatilityFilter
    
    class QualityGuardrails(BaseModel):
        min_median_volume_1min: int = Field(default=20000, gt=0)
        min_price_threshold: float = Field(default=5.0, gt=0)
        survivorship_bias_handling: Dict[str, Union[bool, float]] = Field(
            default={
                "enabled": True,
                "delisting_recovery_rate": 0.30,
                "point_in_time_universe": True
            }
        )
    
    data_horizon: DataHorizon
    data_splits: DataSplits
    bar_resolution: BarResolution
    feature_engineering: FeatureEngineering
    exclusion_filters: ExclusionFilters
    volume_price_guardrails: QualityGuardrails


class ModelConfig(BaseModel):
    """Model configuration with validation"""
    
    class ModelArchitecture(BaseModel):
        type: Literal["RecurrentPPO", "PPO", "DQN"] = "RecurrentPPO"
        observation_space: int = Field(..., gt=0)
        action_space: int = Field(..., gt=0)
        
    class CrossValidation(BaseModel):
        class WalkForward(BaseModel):
            train_window_months: int = Field(..., ge=6, le=36)
            validation_window_months: int = Field(..., ge=1, le=12)
            num_folds: int = Field(..., ge=3, le=10)
        
        class EnsembleStrategy(BaseModel):
            method: Literal["fold_ensemble", "simple_average", "performance_weighted"] = "fold_ensemble"
            ensemble_weights: Literal["equal", "performance_weighted"] = "performance_weighted"
            cross_fold_variance_threshold: float = Field(default=0.15, gt=0, le=1.0)
        
        walk_forward: WalkForward
        ensemble_strategy: EnsembleStrategy
    
    class PerformanceMetrics(BaseModel):
        class Primary(BaseModel):
            metric: Literal["deflated_sharpe_ratio", "sharpe_ratio"] = "deflated_sharpe_ratio"
            threshold: float = Field(..., gt=0)
            p_value_threshold: float = Field(default=0.05, gt=0, lt=1.0)
        
        class StabilityRequirements(BaseModel):
            rolling_sharpe_drawdown_max: float = Field(default=0.30, gt=0, le=1.0)
            turnover_penalty_weight: float = Field(default=0.02, ge=0, le=0.10)
        
        primary: Primary
        stability_requirements: StabilityRequirements
    
    model_architecture: ModelArchitecture
    cross_validation: CrossValidation
    performance_metrics: PerformanceMetrics


class RiskConfig(BaseModel):
    """Risk management configuration with validation"""
    
    class PositionLimits(BaseModel):
        max_notional_per_symbol: float = Field(..., gt=0)
        max_total_notional: float = Field(..., gt=0)
        max_intraday_drawdown_pct: float = Field(..., gt=0, le=0.10)
        daily_loss_limit: float = Field(..., gt=0)
        
        class RiskFactorLimits(BaseModel):
            market_beta_limit: float = Field(default=1.5, gt=0, le=3.0)
            sector_concentration_limit: float = Field(default=0.6, gt=0, le=1.0)
            correlation_limit: float = Field(default=0.8, gt=0, le=1.0)
        
        risk_factor_limits: RiskFactorLimits
        
        @validator('max_total_notional')
        def validate_total_vs_per_symbol(cls, v, values):
            per_symbol = values.get('max_notional_per_symbol', 0)
            if v < per_symbol:
                raise ValueError("Total notional must be >= per symbol limit")
            return v
    
    class StressTesting(BaseModel):
        class VarCalculation(BaseModel):
            method: Literal["historical_simulation", "parametric", "monte_carlo"] = "historical_simulation"
            confidence_level: float = Field(default=0.99, gt=0.9, lt=1.0)
            lookback_days: int = Field(default=252, ge=30, le=1000)
        
        var_calculation: VarCalculation
        var_limit_usd: float = Field(..., gt=0)
        es_limit_usd: float = Field(..., gt=0)
        
        @validator('es_limit_usd')
        def validate_es_vs_var(cls, v, values):
            var_limit = values.get('var_limit_usd', 0)
            if v <= var_limit:
                raise ValueError("Expected Shortfall limit must be > VaR limit")
            return v
    
    position_limits: PositionLimits
    stress_testing: StressTesting


class OperationsConfig(BaseModel):
    """Operations configuration with validation"""
    
    class PolygonAPI(BaseModel):
        class RateLimiting(BaseModel):
            requests_per_minute: int = Field(default=5, ge=1, le=100)
            burst_capacity: int = Field(default=10, ge=5, le=50)
            token_persistence_backend: Literal["redis", "sqlite"] = "redis"
            redis_url: str = Field(default="redis://localhost:6379/0")
        
        rate_limiting: RateLimiting
        circuit_breaker_threshold: int = Field(default=5, ge=1)
    
    class Storage(BaseModel):
        class ParquetOptimization(BaseModel):
            row_group_size: int = Field(default=50000, ge=10000, le=1000000)
            compression: Literal["snappy", "gzip", "lz4"] = "snappy"
            use_dictionary: bool = Field(default=True)
        
        parquet_optimization: ParquetOptimization
        cold_storage: StorageBackend = StorageBackend.HDD
        hot_storage: StorageBackend = StorageBackend.SSD
    
    class LatencySLA(BaseModel):
        market_data_ingestion_ms: int = Field(default=10, ge=1, le=50)
        feature_calculation_ms: int = Field(default=15, ge=5, le=50)
        model_inference_ms: int = Field(default=20, ge=5, le=100)
        risk_checks_ms: int = Field(default=10, ge=1, le=30)
        broker_acknowledgment_ms: int = Field(default=50, ge=10, le=200)
        
        @property
        def total_budget_ms(self) -> int:
            return (self.market_data_ingestion_ms + 
                   self.feature_calculation_ms + 
                   self.model_inference_ms + 
                   self.risk_checks_ms + 
                   self.broker_acknowledgment_ms)
    
    polygon_api: PolygonAPI
    storage: Storage
    latency_sla: LatencySLA
    
    @validator('latency_sla')
    def validate_total_latency(cls, v):
        if v.total_budget_ms > 150:  # 150ms max end-to-end
            raise ValueError(f"Total latency budget {v.total_budget_ms}ms exceeds 150ms limit")
        return v


class ProfessionalConfig(BaseModel):
    """Master configuration combining all modules"""
    
    data: DataConfig
    model: ModelConfig
    risk: RiskConfig
    operations: OperationsConfig
    
    class Config:
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Serialize enums as values


class ConfigurationManager:
    """Manages loading and validation of configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_cache: Optional[ProfessionalConfig] = None
    
    def load_config(self, force_reload: bool = False) -> ProfessionalConfig:
        """Load and validate complete configuration"""
        
        if self.config_cache is not None and not force_reload:
            return self.config_cache
        
        try:
            # Load individual config files
            data_config = self._load_yaml_file("data.yaml")
            model_config = self._load_yaml_file("model.yaml")
            risk_config = self._load_yaml_file("risk.yaml")
            operations_config = self._load_yaml_file("operations.yaml")
            
            # Validate and combine
            config_dict = {
                "data": data_config,
                "model": model_config,
                "risk": risk_config,
                "operations": operations_config
            }
            
            # Pydantic validation
            validated_config = ProfessionalConfig(**config_dict)
            
            # Cache for performance
            self.config_cache = validated_config
            
            logger.info("✅ Configuration loaded and validated successfully")
            logger.info(f"   Data splits: {validated_config.data.data_splits.train_pct}/"
                       f"{validated_config.data.data_splits.validation_pct}/"
                       f"{validated_config.data.data_splits.holdout_pct}/"
                       f"{validated_config.data.data_splits.lockbox_pct}")
            logger.info(f"   Latency budget: {validated_config.operations.latency_sla.total_budget_ms}ms")
            
            return validated_config
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load and parse YAML file"""
        
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")
    
    def validate_config_files(self) -> Dict[str, bool]:
        """Validate all configuration files individually"""
        
        results = {}
        
        config_schemas = {
            "data.yaml": DataConfig,
            "model.yaml": ModelConfig, 
            "risk.yaml": RiskConfig,
            "operations.yaml": OperationsConfig
        }
        
        for filename, schema_class in config_schemas.items():
            try:
                config_data = self._load_yaml_file(filename)
                schema_class(**config_data)  # Validate
                results[filename] = True
                logger.info(f"✅ {filename} validation passed")
            except Exception as e:
                results[filename] = False
                logger.error(f"❌ {filename} validation failed: {e}")
        
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        
        config = self.load_config()
        
        return {
            "data_horizon_days": (config.data.data_horizon.end_date - 
                                config.data.data_horizon.start_date).days if config.data.data_horizon.end_date != "auto" else "auto",
            "total_latency_budget_ms": config.operations.latency_sla.total_budget_ms,
            "risk_var_limit": config.risk.stress_testing.var_limit_usd,
            "api_rate_limit": config.operations.polygon_api.rate_limiting.requests_per_minute,
            "model_type": config.model.model_architecture.type,
            "primary_metric": config.model.performance_metrics.primary.metric
        }


def main():
    """Test configuration validation"""
    
    logging.basicConfig(level=logging.INFO)
    
    config_manager = ConfigurationManager("config")
    
    # Test individual file validation
    print("Testing individual configuration files:")
    results = config_manager.validate_config_files()
    
    for filename, passed in results.items():
        print(f"  {filename}: {'✅' if passed else '❌'}")
    
    # Test complete configuration loading
    if all(results.values()):
        print("\nLoading complete configuration:")
        try:
            config = config_manager.load_config()
            summary = config_manager.get_config_summary()
            
            print("Configuration Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"❌ Complete configuration failed: {e}")
    else:
        print("❌ Individual validation failures prevent complete loading")


if __name__ == "__main__":
    main()