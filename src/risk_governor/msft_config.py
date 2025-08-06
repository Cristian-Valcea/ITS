"""
MSFT-Specific Configuration for Production Risk Governor
Lower volatility trading with conservative risk parameters
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json
import redis
import logging
from datetime import datetime, timezone
import os

@dataclass
class MSFTRiskConfig:
    """MSFT-specific risk configuration"""
    
    # Symbol-specific parameters  
    symbol: str = "MSFT"
    base_volatility: float = 0.02  # Expected daily volatility ~2%
    
    # Conservative position sizing for MSFT
    max_position_notional: float = 500.0    # $500 max position
    max_single_trade: float = 50.0          # $50 max single trade  
    max_daily_turnover: float = 2000.0      # $2000 max daily turnover
    max_intraday_loss: float = 100.0        # $100 max daily loss
    
    # ATR-based position sizing
    atr_lookback_periods: int = 20          # 20-period ATR
    atr_position_multiplier: float = 0.3    # Conservative 0.3x ATR (vs 0.5x standard)
    
    # Drawdown zones (conservative for MSFT)
    dd_yellow_zone: float = 0.04           # 4% DD triggers caution (vs 5% standard)
    dd_red_zone: float = 0.06              # 6% DD triggers restrictions (vs 8% standard)  
    dd_hard_stop: float = 0.08             # 8% DD forces flat (vs 10% standard)
    
    # Performance requirements
    max_governor_latency_ms: float = 5.0   # <5ms decision latency
    min_sharpe_for_scaling: float = 1.0    # Require 1.0+ Sharpe before increasing limits
    
    # Market hours (ET)
    market_open: str = "09:30"
    market_close: str = "16:00"
    pre_market_start: str = "04:00"
    after_hours_end: str = "20:00"

class RiskStateManager:
    """
    Redis-based state persistence for risk governor
    Ensures risk limits survive system restarts
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.logger = logging.getLogger("RiskStateManager")
        self.key_prefix = "risk_governor:MSFT"
        
        try:
            # Test connection
            self.redis_client.ping()
            self.logger.info("Connected to Redis for risk state persistence")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def save_daily_state(self, portfolio_state: Dict) -> bool:
        """Save current daily risk state"""
        if not self.redis_client:
            return False
            
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = f"{self.key_prefix}:daily:{today}"
            
            state_data = {
                "daily_turnover": portfolio_state.get("daily_turnover", 0.0),
                "realized_pnl": portfolio_state.get("realized_pnl", 0.0),
                "max_daily_drawdown": portfolio_state.get("max_daily_drawdown", 0.0),
                "current_position": portfolio_state.get("current_position", 0.0),
                "last_update": datetime.now(timezone.utc).isoformat(),
                "breach_count": portfolio_state.get("breach_count", 0)
            }
            
            self.redis_client.hset(key, mapping=state_data)
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save daily state: {e}")
            return False
    
    def load_daily_state(self) -> Optional[Dict]:
        """Load current daily risk state"""
        if not self.redis_client:
            return None
            
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = f"{self.key_prefix}:daily:{today}"
            
            state_data = self.redis_client.hgetall(key)
            
            if not state_data:
                return None
                
            # Convert string values back to appropriate types
            return {
                "daily_turnover": float(state_data.get("daily_turnover", 0.0)),
                "realized_pnl": float(state_data.get("realized_pnl", 0.0)),
                "max_daily_drawdown": float(state_data.get("max_daily_drawdown", 0.0)),
                "current_position": float(state_data.get("current_position", 0.0)),
                "last_update": state_data.get("last_update"),
                "breach_count": int(state_data.get("breach_count", 0))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load daily state: {e}")
            return None
    
    def log_risk_decision(self, decision_data: Dict) -> bool:
        """Log risk governor decision for audit trail"""
        if not self.redis_client:
            return False
            
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            key = f"{self.key_prefix}:decisions:{timestamp}"
            
            audit_data = {
                "timestamp": timestamp,
                "raw_action": decision_data.get("raw_action"),
                "safe_increment": str(decision_data.get("safe_increment")),
                "position_reason": decision_data.get("position_reason"),
                "drawdown_reason": decision_data.get("drawdown_reason"),
                "latency_ms": str(decision_data.get("total_latency_ms")),
                "risk_limits_applied": str(decision_data.get("risk_limits_applied"))
            }
            
            self.redis_client.hset(key, mapping=audit_data)
            self.redis_client.expire(key, 86400 * 30)  # Keep audit for 30 days
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log decision: {e}")
            return False
    
    def get_recent_decisions(self, limit: int = 100) -> list:
        """Get recent risk governor decisions for analysis"""
        if not self.redis_client:
            return []
            
        try:
            pattern = f"{self.key_prefix}:decisions:*"
            keys = self.redis_client.keys(pattern)
            keys.sort(reverse=True)  # Most recent first
            
            decisions = []
            for key in keys[:limit]:
                decision_data = self.redis_client.hgetall(key)
                if decision_data:
                    decisions.append(decision_data)
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Failed to get recent decisions: {e}")
            return []
    
    def reset_daily_limits(self) -> bool:
        """Reset daily risk limits (called at market open)"""
        if not self.redis_client:
            return False
            
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = f"{self.key_prefix}:daily:{today}"
            
            reset_state = {
                "daily_turnover": "0.0",
                "realized_pnl": "0.0", 
                "max_daily_drawdown": "0.0",
                "current_position": "0.0",
                "last_update": datetime.now(timezone.utc).isoformat(),
                "breach_count": "0"
            }
            
            self.redis_client.hset(key, mapping=reset_state)
            self.logger.info(f"Reset daily risk limits for {today}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset daily limits: {e}")
            return False

class MSFTMarketRegimeDetector:
    """
    Detect market regime for MSFT to adjust risk parameters
    Conservative approach focused on volatility and time-of-day
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MSFTMarketRegime")
        
        # Historical MSFT volatility percentiles (approximate)
        self.volatility_percentiles = {
            "P10": 0.008,   # Very low volatility
            "P25": 0.012,   # Low volatility  
            "P50": 0.018,   # Normal volatility
            "P75": 0.025,   # High volatility
            "P90": 0.035,   # Very high volatility
            "P95": 0.045    # Extreme volatility
        }
    
    def detect_regime(self, current_volatility: float, time_of_day: str) -> Dict:
        """
        Detect current market regime and recommend risk adjustments
        Returns risk multiplier (1.0 = normal, <1.0 = reduce risk, >1.0 = increase risk)
        """
        
        # Base risk multiplier
        risk_multiplier = 1.0
        regime_description = "NORMAL"
        
        # Volatility-based adjustment
        if current_volatility > self.volatility_percentiles["P95"]:
            risk_multiplier = 0.25
            regime_description = "EXTREME_VOLATILITY"
        elif current_volatility > self.volatility_percentiles["P90"]:
            risk_multiplier = 0.5  
            regime_description = "HIGH_VOLATILITY"
        elif current_volatility > self.volatility_percentiles["P75"]:
            risk_multiplier = 0.75
            regime_description = "ELEVATED_VOLATILITY"
        elif current_volatility < self.volatility_percentiles["P25"]:
            risk_multiplier = 1.25
            regime_description = "LOW_VOLATILITY"
        
        # Time-of-day adjustment
        hour = int(time_of_day.split(":")[0])
        minute = int(time_of_day.split(":")[1])
        time_decimal = hour + minute / 60.0
        
        # Market open/close periods (higher risk)
        if (9.5 <= time_decimal <= 10.0) or (15.5 <= time_decimal <= 16.0):
            risk_multiplier *= 0.5
            regime_description += "_OPEN_CLOSE"
        
        # Lunch period (lower volume, different dynamics)
        elif 12.0 <= time_decimal <= 13.0:
            risk_multiplier *= 0.8
            regime_description += "_LUNCH"
        
        return {
            "risk_multiplier": risk_multiplier,
            "regime": regime_description,
            "volatility_percentile": self._get_volatility_percentile(current_volatility),
            "time_factor": "OPEN_CLOSE" if "_OPEN_CLOSE" in regime_description else "NORMAL"
        }
    
    def _get_volatility_percentile(self, volatility: float) -> str:
        """Determine which volatility percentile current value falls into"""
        for percentile in ["P95", "P90", "P75", "P50", "P25", "P10"]:
            if volatility >= self.volatility_percentiles[percentile]:
                return percentile
        return "P10"

# Create singleton instances for easy import
msft_config = MSFTRiskConfig()
risk_state_manager = RiskStateManager()  
market_regime_detector = MSFTMarketRegimeDetector()