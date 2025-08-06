"""
Stress Test Configuration Management

Centralized configuration for all stress testing parameters with
production-ready defaults and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from pathlib import Path


@dataclass
class StressTestConfig:
    """Central configuration for all stress testing parameters."""
    
    # === FLASH CRASH SIMULATION ===
    slippage_levels: int = 1  # Take next-level price for realistic fills
    broker_rtt_ms: int = 30   # Realistic broker round-trip time
    crash_duration_s: int = 30  # Flash crash duration
    max_drawdown_pct: float = 0.15  # 15% maximum allowed drawdown
    spread_multiplier: float = 3.0  # Spreads widen 3x during crash
    depth_reduction: float = 0.8   # Depth reduces by 80%
    
    # === DECISION FLOOD LOAD TESTING ===
    decisions_per_second: int = 1000  # Target load rate
    test_duration_s: int = 600        # 10 minutes sustained load
    min_samples: int = 600_000        # Minimum samples for statistics
    latency_threshold_ms: int = 15    # P99 latency threshold
    observation_buffer_size: int = 1000  # Realistic obs buffer
    
    # === BROKER FAILURE INJECTION ===
    broker_outage_s: int = 10         # Outage duration
    recovery_timeout_s: int = 60      # Max time to wait for recovery
    max_recovery_time_s: int = 30     # Target recovery time
    data_freshness_ms: int = 500      # Market data freshness requirement
    num_failure_tests: int = 2        # Run failure scenario twice
    
    # === PORTFOLIO INTEGRITY VALIDATION ===
    position_tolerance_usd: float = 1.0   # $1 tolerance for position delta
    cash_tolerance_usd: float = 1.0       # $1 tolerance for cash delta
    transaction_log_required: bool = True  # Require complete transaction logs
    
    # === DATA SOURCES ===
    historical_data_path: Path = field(default_factory=lambda: Path("stress_testing/data/historical"))
    nvda_crash_date: str = "2023-10-17"  # Historical flash crash date
    backup_data_sources: List[str] = field(default_factory=lambda: [
        "https://data-host/nvda_l2_20231017.parquet",
        "s3://trading-data/historical/nvda_l2_20231017.parquet"
    ])
    
    # === MONITORING & METRICS ===
    prometheus_port: int = 8000
    metrics_export_interval_s: int = 5
    alert_latency_threshold_ms: int = 20  # Alert if P99 > 20ms
    dashboard_refresh_s: int = 30
    
    # === CI/CD INTEGRATION ===
    ci_timeout_s: int = 1800  # 30 minutes max CI runtime
    required_pass_rate: float = 1.0  # 100% pass rate required
    slack_webhook_url: Optional[str] = None
    html_report_path: Path = field(default_factory=lambda: Path("stress_testing/results/report.html"))
    
    # === SAFETY LIMITS ===
    max_test_position_usd: float = 1000.0  # Max position during testing
    emergency_stop_threshold_s: float = 2.0  # Emergency stop time limit
    hard_limit_breach_tolerance: int = 0   # Zero tolerance for breaches
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_config()
        self._setup_paths()
    
    def _validate_config(self):
        """Validate configuration parameters for safety and consistency."""
        # Latency thresholds
        if self.latency_threshold_ms <= 0:
            raise ValueError("Latency threshold must be positive")
        
        if self.alert_latency_threshold_ms <= self.latency_threshold_ms:
            raise ValueError("Alert threshold should be higher than test threshold")
        
        # Sample size validation
        min_required_samples = self.decisions_per_second * 60  # At least 1 minute
        if self.min_samples < min_required_samples:
            raise ValueError(f"Minimum samples too low: {self.min_samples} < {min_required_samples}")
        
        # Recovery time validation
        if self.max_recovery_time_s >= self.recovery_timeout_s:
            raise ValueError("Max recovery time should be less than timeout")
        
        # Drawdown validation
        if not 0 < self.max_drawdown_pct < 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        
        # Tolerance validation
        if self.position_tolerance_usd <= 0 or self.cash_tolerance_usd <= 0:
            raise ValueError("Tolerance values must be positive")
    
    def _setup_paths(self):
        """Ensure required directories exist."""
        self.historical_data_path.mkdir(parents=True, exist_ok=True)
        self.html_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'StressTestConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if slack_url := os.getenv('STRESS_TEST_SLACK_WEBHOOK'):
            config.slack_webhook_url = slack_url
        
        if prometheus_port := os.getenv('STRESS_TEST_PROMETHEUS_PORT'):
            config.prometheus_port = int(prometheus_port)
        
        if latency_threshold := os.getenv('STRESS_TEST_LATENCY_THRESHOLD_MS'):
            config.latency_threshold_ms = int(latency_threshold)
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'flash_crash': {
                'slippage_levels': self.slippage_levels,
                'broker_rtt_ms': self.broker_rtt_ms,
                'crash_duration_s': self.crash_duration_s,
                'max_drawdown_pct': self.max_drawdown_pct,
                'spread_multiplier': self.spread_multiplier,
                'depth_reduction': self.depth_reduction
            },
            'load_testing': {
                'decisions_per_second': self.decisions_per_second,
                'test_duration_s': self.test_duration_s,
                'min_samples': self.min_samples,
                'latency_threshold_ms': self.latency_threshold_ms
            },
            'failure_injection': {
                'broker_outage_s': self.broker_outage_s,
                'recovery_timeout_s': self.recovery_timeout_s,
                'max_recovery_time_s': self.max_recovery_time_s,
                'data_freshness_ms': self.data_freshness_ms
            },
            'validation': {
                'position_tolerance_usd': self.position_tolerance_usd,
                'cash_tolerance_usd': self.cash_tolerance_usd,
                'transaction_log_required': self.transaction_log_required
            }
        }
    
    def get_test_scenarios(self) -> List[str]:
        """Get list of enabled test scenarios."""
        return [
            'flash_crash',
            'decision_flood', 
            'broker_failure',
            'portfolio_integrity'
        ]
    
    def is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        return os.getenv('CI', '').lower() in ('true', '1', 'yes')
    
    def get_data_file_path(self, symbol: str, date: str) -> Path:
        """Get path to historical data file."""
        filename = f"{symbol.lower()}_l2_{date.replace('-', '')}.parquet"
        return self.historical_data_path / filename


# Global configuration instance
DEFAULT_CONFIG = StressTestConfig()


def get_config() -> StressTestConfig:
    """Get the global stress test configuration."""
    return DEFAULT_CONFIG


def set_config(config: StressTestConfig):
    """Set the global stress test configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config