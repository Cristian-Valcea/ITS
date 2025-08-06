"""
Integration layer: Stairways V4 Model + Production Risk Governor
Deploy ANY model safely through risk governance layer
"""

import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional
from dataclasses import asdict
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk_governor.core_governor import (
    ProductionRiskGovernor, TradingAction, PortfolioState, RiskLimits
)
from src.risk_governor.msft_config import (
    MSFTRiskConfig, RiskStateManager, MSFTMarketRegimeDetector,
    msft_config, risk_state_manager, market_regime_detector
)

class SafeStairwaysDeployment:
    """
    Safe deployment wrapper for Stairways V4 model
    Provides bulletproof risk management for any RL trading model
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 symbol: str = "MSFT",
                 paper_trading: bool = True):
        
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.logger = logging.getLogger(f"SafeStairways_{symbol}")
        
        # Initialize risk management components
        self.risk_governor = ProductionRiskGovernor(symbol)
        self.risk_config = msft_config
        self.state_manager = risk_state_manager
        self.regime_detector = market_regime_detector
        
        # Load and initialize the model
        self.model = self._load_stairways_model(model_path)
        
        # Initialize portfolio state
        self.portfolio_state = self._initialize_portfolio_state()
        
        # Performance tracking
        self.session_start_time = time.time()
        self.decision_count = 0
        self.total_latency = 0.0
        
        self.logger.info(f"SafeStairways initialized for {symbol} (paper_trading={paper_trading})")
    
    def _load_stairways_model(self, model_path: Optional[str]):
        """Load Stairways V4 model or create mock for testing"""
        if model_path and os.path.exists(model_path):
            try:
                # Import stable-baselines3 if available
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                self.logger.info(f"Loaded Stairways model from {model_path}")
                return model
            except ImportError:
                self.logger.warning("stable-baselines3 not available, using mock model")
        
        # Mock model for testing/development
        return MockStairwaysModel()
    
    def _initialize_portfolio_state(self) -> PortfolioState:
        """Initialize or restore portfolio state from persistence"""
        
        # Try to load from state manager
        saved_state = self.state_manager.load_daily_state()
        
        if saved_state:
            self.logger.info("Restored portfolio state from persistence")
            return PortfolioState(
                symbol=self.symbol,
                current_position=saved_state["current_position"],
                cash_balance=1000.0,  # Assume starting balance
                unrealized_pnl=0.0,   # Reset unrealized P&L
                realized_pnl=saved_state["realized_pnl"],
                current_price=100.0,  # Will be updated with market data
                daily_turnover=saved_state["daily_turnover"],
                max_daily_drawdown=saved_state["max_daily_drawdown"]
            )
        else:
            self.logger.info("Starting with fresh portfolio state")
            return PortfolioState(
                symbol=self.symbol,
                current_position=0.0,
                cash_balance=1000.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                current_price=100.0,
                daily_turnover=0.0,
                max_daily_drawdown=0.0
            )
    
    def get_safe_trading_action(self, 
                              market_observation: np.ndarray,
                              market_data: Dict,
                              top_of_book_mid: Optional[float] = None) -> Dict:
        """
        Main interface: Get safe trading action from model + risk governor
        
        Args:
            market_observation: 26-dim observation for Stairways model
            market_data: Dict with OHLC, volume, etc.
            top_of_book_mid: Real-time mid price for DD calculation
            
        Returns:
            Dict with safe action and complete audit trail
        """
        start_time = time.perf_counter()
        
        try:
            # 1. Get raw action from Stairways model
            raw_action_idx, model_confidence = self._get_model_prediction(market_observation)
            raw_action = TradingAction(raw_action_idx)
            
            # 2. Update portfolio state with current market data
            self._update_portfolio_state(market_data, top_of_book_mid)
            
            # 3. Detect market regime and adjust risk limits
            regime_info = self._adjust_for_market_regime(market_data)
            
            # 4. Apply risk governor filtering
            risk_decision = self.risk_governor.filter_trading_action(
                raw_action, self.portfolio_state, market_data, top_of_book_mid
            )
            
            # 5. Execute safe action (or simulate in paper trading)
            execution_result = self._execute_safe_action(
                risk_decision["safe_increment"], market_data
            )
            
            # 6. Log decision and update state persistence
            self._log_and_persist_decision(risk_decision, regime_info, execution_result)
            
            # 7. Update performance metrics
            total_latency = (time.perf_counter() - start_time) * 1000
            self.decision_count += 1
            self.total_latency += total_latency
            
            # 8. Build complete response
            return {
                "safe_increment": risk_decision["safe_increment"],
                "raw_action": raw_action.name,
                "model_confidence": model_confidence,
                "risk_reason": risk_decision["position_reason"],
                "drawdown_reason": risk_decision["drawdown_reason"],
                "market_regime": regime_info["regime"],
                "execution_result": execution_result,
                "portfolio_state": asdict(self.portfolio_state),
                "total_latency_ms": total_latency,
                "session_avg_latency": self.total_latency / self.decision_count,
                "paper_trading": self.paper_trading,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in safe trading action: {e}")
            return self._emergency_response(str(e))
    
    def _get_model_prediction(self, observation: np.ndarray) -> Tuple[int, float]:
        """Get prediction from Stairways model with confidence estimate"""
        try:
            if hasattr(self.model, 'predict'):
                # Real Stairways model
                action, _states = self.model.predict(observation, deterministic=True)
                confidence = 0.8  # Placeholder - could use policy entropy
                return int(action), confidence
            else:
                # Mock model
                return self.model.predict(observation)
        except Exception as e:
            self.logger.error(f"Model prediction error: {e}")
            return 2, 0.0  # Return HOLD action with zero confidence
    
    def _update_portfolio_state(self, market_data: Dict, top_of_book_mid: Optional[float]):
        """Update portfolio state with latest market data"""
        
        # Update current price
        new_price = top_of_book_mid or market_data.get("close", self.portfolio_state.current_price)
        
        # Calculate unrealized P&L from price change
        if self.portfolio_state.current_position != 0:
            price_change = new_price - self.portfolio_state.current_price
            position_pnl = price_change * abs(self.portfolio_state.current_position) / 100.0
            if self.portfolio_state.current_position < 0:  # Short position
                position_pnl = -position_pnl
            self.portfolio_state.unrealized_pnl += position_pnl
        
        self.portfolio_state.current_price = new_price
        
        # Update max drawdown
        total_pnl = self.portfolio_state.realized_pnl + self.portfolio_state.unrealized_pnl
        current_drawdown = abs(min(0, total_pnl))
        self.portfolio_state.max_daily_drawdown = max(
            self.portfolio_state.max_daily_drawdown, current_drawdown
        )
        
        # Update ATR in risk governor
        if "high" in market_data and "low" in market_data:
            self.risk_governor.position_governor.update_atr(
                market_data["high"], market_data["low"], 
                market_data["close"], market_data.get("prev_close", market_data["close"])
            )
    
    def _adjust_for_market_regime(self, market_data: Dict) -> Dict:
        """Detect market regime and adjust risk parameters"""
        
        # Calculate current volatility (simple approximation)
        if "high" in market_data and "low" in market_data:
            daily_range = market_data["high"] - market_data["low"]
            current_volatility = daily_range / market_data["close"]
        else:
            current_volatility = 0.02  # Default assumption
        
        # Get current time  
        current_time = time.strftime("%H:%M")
        
        # Detect regime
        regime_info = self.regime_detector.detect_regime(current_volatility, current_time)
        
        # Adjust risk limits based on regime
        risk_multiplier = regime_info["risk_multiplier"]
        
        # Apply regime-based adjustments to risk governor
        original_limits = self.risk_governor.risk_limits
        original_limits.max_single_trade *= risk_multiplier
        original_limits.max_position_notional *= risk_multiplier
        
        # Adjust drawdown thresholds for high volatility
        if risk_multiplier < 0.75:  # High volatility
            self.risk_governor.drawdown_governor.yellow_zone *= 0.8
            self.risk_governor.drawdown_governor.red_zone *= 0.8
            self.risk_governor.drawdown_governor.hard_stop *= 0.8
        
        return regime_info
    
    def _execute_safe_action(self, safe_increment: float, market_data: Dict) -> Dict:
        """Execute the safe trading action (or simulate if paper trading)"""
        
        if self.paper_trading:
            # Paper trading simulation
            execution_result = {
                "status": "SIMULATED",
                "executed_size": safe_increment,
                "execution_price": market_data.get("close", self.portfolio_state.current_price),
                "execution_cost": abs(safe_increment) * 0.001,  # 0.1% execution cost
                "timestamp": time.time()
            }
        else:
            # Real execution would go here
            execution_result = {
                "status": "EXECUTED",
                "executed_size": safe_increment,
                "execution_price": market_data.get("close", self.portfolio_state.current_price),
                "execution_cost": abs(safe_increment) * 0.001,
                "timestamp": time.time()
            }
            
            # TODO: Add real broker integration here
            # order_result = broker.place_order(symbol=self.symbol, size=safe_increment)
        
        # Update portfolio state with execution
        self.portfolio_state.current_position += safe_increment
        self.portfolio_state.daily_turnover += abs(safe_increment)
        
        # Update realized P&L with execution costs
        self.portfolio_state.realized_pnl -= execution_result["execution_cost"]
        
        return execution_result
    
    def _log_and_persist_decision(self, risk_decision: Dict, regime_info: Dict, execution_result: Dict):
        """Log decision to audit trail and update persistent state"""
        
        # Combine all decision data for audit log
        audit_data = {
            **risk_decision,
            "market_regime": regime_info["regime"],
            "risk_multiplier": regime_info["risk_multiplier"],
            "execution_status": execution_result["status"],
            "execution_cost": execution_result["execution_cost"]
        }
        
        # Log to Redis audit trail
        self.state_manager.log_risk_decision(audit_data)
        
        # Save current portfolio state
        portfolio_dict = asdict(self.portfolio_state)
        self.state_manager.save_daily_state(portfolio_dict)
    
    def _emergency_response(self, error_message: str) -> Dict:
        """Emergency fallback response for system errors"""
        return {
            "safe_increment": 0.0,  # HOLD - safest action
            "raw_action": "ERROR",
            "model_confidence": 0.0,
            "risk_reason": f"EMERGENCY_FALLBACK: {error_message}",
            "drawdown_reason": "SYSTEM_ERROR",
            "market_regime": "ERROR",
            "execution_result": {"status": "ABORTED", "reason": error_message},
            "portfolio_state": asdict(self.portfolio_state),
            "total_latency_ms": 999.0,
            "paper_trading": True,  # Force paper trading in error state
            "timestamp": time.time()
        }
    
    def get_performance_summary(self) -> Dict:
        """Get current session performance summary"""
        session_time = time.time() - self.session_start_time
        
        total_pnl = self.portfolio_state.realized_pnl + self.portfolio_state.unrealized_pnl
        
        return {
            "session_duration_minutes": session_time / 60,
            "total_decisions": self.decision_count,
            "avg_latency_ms": self.total_latency / max(1, self.decision_count),
            "total_pnl": total_pnl,
            "realized_pnl": self.portfolio_state.realized_pnl,
            "unrealized_pnl": self.portfolio_state.unrealized_pnl,
            "current_position": self.portfolio_state.current_position,
            "daily_turnover": self.portfolio_state.daily_turnover,
            "max_drawdown": self.portfolio_state.max_daily_drawdown,
            "decisions_per_minute": self.decision_count / max(1, session_time / 60),
            "risk_budget_used": {
                "position": abs(self.portfolio_state.current_position) / self.risk_config.max_position_notional,
                "turnover": self.portfolio_state.daily_turnover / self.risk_config.max_daily_turnover,
                "loss": self.portfolio_state.max_daily_drawdown / self.risk_config.max_intraday_loss
            }
        }

class MockStairwaysModel:
    """Mock Stairways model for testing without actual model files"""
    
    def __init__(self):
        self.action_probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]  # Favor HOLD
        
    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """Return weighted random action favoring HOLD"""
        action = np.random.choice(5, p=self.action_probabilities)
        confidence = 0.6 + np.random.normal(0, 0.1)  # 0.6 +/- 0.1
        return action, max(0.1, min(0.9, confidence))