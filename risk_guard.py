#!/usr/bin/env python3
"""
🛡️ RISK GUARD SERVICE
Monitors trading recommendations and applies risk controls before execution
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import redis
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class RiskLimits:
    max_daily_loss: float = 100.0          # $100 daily loss limit
    max_total_drawdown: float = 200.0      # $200 total drawdown limit
    max_position_size: float = 1000.0      # $1000 max per asset
    max_portfolio_value: float = 110000.0  # Max portfolio value
    min_portfolio_value: float = 95000.0   # Min portfolio value (5% drawdown)
    max_trades_per_hour: int = 20          # Trade frequency limit
    max_turnover_per_day: float = 0.5      # 50% max daily turnover

@dataclass 
class PortfolioState:
    total_value: float = 100000.0
    cash: float = 100000.0
    positions: Dict[str, float] = None
    daily_pnl: float = 0.0
    total_drawdown: float = 0.0
    trades_today: int = 0
    last_trade_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {"NVDA": 0.0, "MSFT": 0.0}

@dataclass
class TradingRecommendation:
    timestamp: float
    action: int
    action_name: str
    confidence: float
    portfolio_value: float
    positions: Dict[str, float]
    risk_metrics: Dict[str, float]

@dataclass
class RiskDecisionResult:
    decision: RiskDecision
    original_action: int
    approved_action: Optional[int]
    reason: str
    risk_metrics: Dict[str, Any]

class RiskGuardService:
    """Risk management service for trading recommendations"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.risk_limits = RiskLimits()
        self.portfolio_state = PortfolioState()
        self.trade_history: List[Dict] = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Risk statistics
        self.risk_stats = {
            "total_recommendations": 0,
            "approved": 0,
            "rejected": 0,
            "modified": 0,
            "risk_breaches": [],
            "last_decision": None
        }
        
    def reset_daily_limits(self):
        """Reset daily risk limits"""
        
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if current_date > self.daily_reset_time:
            logger.info("🔄 Resetting daily risk limits")
            self.portfolio_state.daily_pnl = 0.0
            self.portfolio_state.trades_today = 0
            self.trade_history = []
            self.daily_reset_time = current_date
            
    def calculate_position_risk(self, action: int, current_positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk metrics for a proposed action"""
        
        # Action mapping (simplified)
        action_effects = {
            0: {"NVDA": 0, "MSFT": 0},      # HOLD_BOTH
            1: {"NVDA": 1, "MSFT": 0},      # BUY_NVDA_HOLD_MSFT
            2: {"NVDA": -1, "MSFT": 0},     # SELL_NVDA_HOLD_MSFT
            3: {"NVDA": 0, "MSFT": 1},      # HOLD_NVDA_BUY_MSFT
            4: {"NVDA": 1, "MSFT": 1},      # BUY_BOTH
            5: {"NVDA": -1, "MSFT": 1},     # SELL_NVDA_BUY_MSFT
            6: {"NVDA": 0, "MSFT": -1},     # HOLD_NVDA_SELL_MSFT
            7: {"NVDA": 1, "MSFT": -1},     # BUY_NVDA_SELL_MSFT
            8: {"NVDA": -1, "MSFT": -1}     # SELL_BOTH
        }
        
        effect = action_effects.get(action, {"NVDA": 0, "MSFT": 0})
        
        # Calculate new positions (simplified - assumes unit position changes)
        new_positions = {}
        for symbol in ["NVDA", "MSFT"]:
            new_positions[symbol] = current_positions.get(symbol, 0) + effect[symbol] * 100  # $100 per unit
        
        # Calculate risk metrics
        total_exposure = abs(new_positions["NVDA"]) + abs(new_positions["MSFT"])
        max_single_position = max(abs(new_positions["NVDA"]), abs(new_positions["MSFT"]))
        
        return {
            "new_positions": new_positions,
            "total_exposure": total_exposure,
            "max_single_position": max_single_position,
            "position_change": effect,
            "concentration_risk": max_single_position / max(total_exposure, 1)
        }
        
    def check_drawdown_limits(self, proposed_portfolio_value: float) -> tuple[bool, str]:
        """Check if proposed trade violates drawdown limits"""
        
        # Daily loss check
        daily_loss = self.portfolio_state.total_value - proposed_portfolio_value
        if daily_loss > self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: ${daily_loss:.2f} > ${self.risk_limits.max_daily_loss}"
        
        # Total drawdown check
        initial_value = 100000.0  # Starting portfolio value
        total_drawdown = initial_value - proposed_portfolio_value
        if total_drawdown > self.risk_limits.max_total_drawdown:
            return False, f"Total drawdown limit exceeded: ${total_drawdown:.2f} > ${self.risk_limits.max_total_drawdown}"
        
        # Portfolio value bounds
        if proposed_portfolio_value < self.risk_limits.min_portfolio_value:
            return False, f"Portfolio value below minimum: ${proposed_portfolio_value:.2f} < ${self.risk_limits.min_portfolio_value}"
            
        if proposed_portfolio_value > self.risk_limits.max_portfolio_value:
            return False, f"Portfolio value above maximum: ${proposed_portfolio_value:.2f} > ${self.risk_limits.max_portfolio_value}"
        
        return True, "Drawdown limits OK"
    
    def check_position_limits(self, risk_metrics: Dict[str, Any]) -> tuple[bool, str]:
        """Check if proposed positions violate position limits"""
        
        # Single position size limit
        if risk_metrics["max_single_position"] > self.risk_limits.max_position_size:
            return False, f"Position size limit exceeded: ${risk_metrics['max_single_position']:.2f} > ${self.risk_limits.max_position_size}"
        
        # Concentration risk (optional additional check)
        if risk_metrics["concentration_risk"] > 0.8:  # 80% max in single asset
            return False, f"Concentration risk too high: {risk_metrics['concentration_risk']:.1%}"
        
        return True, "Position limits OK"
    
    def check_frequency_limits(self) -> tuple[bool, str]:
        """Check trading frequency limits"""
        
        # Reset daily counters if needed
        self.reset_daily_limits()
        
        # Check daily trade count
        if self.portfolio_state.trades_today >= self.risk_limits.max_trades_per_hour * 8:  # 8 trading hours
            return False, f"Daily trade limit exceeded: {self.portfolio_state.trades_today}"
        
        # Check hourly frequency
        if self.portfolio_state.last_trade_time:
            time_since_last = datetime.now() - self.portfolio_state.last_trade_time
            if time_since_last < timedelta(minutes=3):  # Min 3 minutes between trades
                return False, f"Trading too frequently: {time_since_last.total_seconds():.0f}s since last trade"
        
        return True, "Frequency limits OK"
    
    def evaluate_recommendation(self, recommendation: TradingRecommendation) -> RiskDecisionResult:
        """Evaluate a trading recommendation against risk limits"""
        
        logger.debug(f"🛡️ Evaluating recommendation: {recommendation.action_name}")
        
        # Update statistics
        self.risk_stats["total_recommendations"] += 1
        
        # Calculate risk metrics
        risk_metrics = self.calculate_position_risk(
            recommendation.action, 
            recommendation.positions
        )
        
        # Check all risk limits
        checks = []
        
        # 1. Drawdown limits
        drawdown_ok, drawdown_msg = self.check_drawdown_limits(recommendation.portfolio_value)
        checks.append(("drawdown", drawdown_ok, drawdown_msg))
        
        # 2. Position limits
        position_ok, position_msg = self.check_position_limits(risk_metrics)
        checks.append(("position", position_ok, position_msg))
        
        # 3. Frequency limits
        frequency_ok, frequency_msg = self.check_frequency_limits()
        checks.append(("frequency", frequency_ok, frequency_msg))
        
        # Determine decision
        failed_checks = [(name, msg) for name, ok, msg in checks if not ok]
        
        if not failed_checks:
            # All checks passed - APPROVE
            decision = RiskDecision.APPROVED
            approved_action = recommendation.action
            reason = "All risk checks passed"
            self.risk_stats["approved"] += 1
            
            # Update portfolio state
            self.portfolio_state.total_value = recommendation.portfolio_value
            self.portfolio_state.positions = risk_metrics["new_positions"]
            self.portfolio_state.trades_today += 1
            self.portfolio_state.last_trade_time = datetime.now()
            
        elif len(failed_checks) == 1 and failed_checks[0][0] == "frequency":
            # Only frequency check failed - DELAY (modify to HOLD)
            decision = RiskDecision.MODIFIED
            approved_action = 0  # HOLD_BOTH
            reason = f"Modified to HOLD due to: {failed_checks[0][1]}"
            self.risk_stats["modified"] += 1
            
        else:
            # Multiple failures or critical risk breach - REJECT
            decision = RiskDecision.REJECTED
            approved_action = None
            reason = f"Risk limits violated: {', '.join([msg for _, msg in failed_checks])}"
            self.risk_stats["rejected"] += 1
            
            # Log risk breach
            breach = {
                "timestamp": datetime.now().isoformat(),
                "action": recommendation.action_name,
                "reason": reason,
                "failed_checks": failed_checks
            }
            self.risk_stats["risk_breaches"].append(breach)
        
        # Create result
        result = RiskDecisionResult(
            decision=decision,
            original_action=recommendation.action,
            approved_action=approved_action,
            reason=reason,
            risk_metrics=risk_metrics
        )
        
        self.risk_stats["last_decision"] = datetime.now().isoformat()
        
        logger.info(f"🛡️ Risk decision: {decision.value.upper()} - {reason}")
        
        return result
    
    async def process_recommendations(self):
        """Main loop to process trading recommendations"""
        
        logger.info("🛡️ Risk Guard service started")
        logger.info(f"   Daily loss limit: ${self.risk_limits.max_daily_loss}")
        logger.info(f"   Total drawdown limit: ${self.risk_limits.max_total_drawdown}")
        logger.info(f"   Max position size: ${self.risk_limits.max_position_size}")
        
        while True:
            try:
                # Read recommendations from Redis stream
                messages = self.redis_client.xread({'trading:recommendations': '$'}, count=1, block=1000)
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self.process_single_recommendation(fields)
                        
            except Exception as e:
                logger.error(f"❌ Error in risk guard processing: {e}")
                await asyncio.sleep(5)
    
    async def process_single_recommendation(self, rec_data: Dict[str, str]):
        """Process a single trading recommendation"""
        
        try:
            # Parse recommendation
            recommendation = TradingRecommendation(
                timestamp=float(rec_data.get('timestamp', time.time())),
                action=int(rec_data.get('action', 0)),
                action_name=rec_data.get('action_name', 'UNKNOWN'),
                confidence=float(rec_data.get('confidence', 0.0)),
                portfolio_value=float(rec_data.get('portfolio_value', 100000.0)),
                positions=json.loads(rec_data.get('positions', '{"NVDA": 0, "MSFT": 0}')),
                risk_metrics=json.loads(rec_data.get('risk_metrics', '{}'))
            )
            
            # Evaluate against risk limits
            decision = self.evaluate_recommendation(recommendation)
            
            # Publish decision
            if decision.decision == RiskDecision.APPROVED:
                # Send to execution queue
                order_data = {
                    "timestamp": recommendation.timestamp,
                    "action": decision.approved_action,
                    "action_name": recommendation.action_name,
                    "confidence": recommendation.confidence,
                    "risk_approved": True,
                    "risk_reason": decision.reason
                }
                self.redis_client.xadd("trading:orders", order_data)
                
            elif decision.decision == RiskDecision.MODIFIED:
                # Send modified action
                order_data = {
                    "timestamp": recommendation.timestamp,
                    "action": decision.approved_action,
                    "action_name": "HOLD_BOTH",
                    "confidence": recommendation.confidence * 0.5,  # Reduce confidence
                    "risk_approved": True,
                    "risk_reason": decision.reason
                }
                self.redis_client.xadd("trading:orders", order_data)
                
            else:
                # Log rejection
                logger.warning(f"🚫 Trade REJECTED: {decision.reason}")
            
            # Push metrics to monitoring
            self.push_risk_metrics(decision)
            
        except Exception as e:
            logger.error(f"❌ Error processing recommendation: {e}")
    
    def push_risk_metrics(self, decision: RiskDecisionResult):
        """Push risk metrics to monitoring systems"""
        
        try:
            # Push to Pushgateway (simplified)
            metrics_data = {
                "risk_guard_decisions_total": self.risk_stats["total_recommendations"],
                "risk_guard_approved_total": self.risk_stats["approved"],
                "risk_guard_rejected_total": self.risk_stats["rejected"],
                "risk_guard_modified_total": self.risk_stats["modified"],
                "portfolio_value": self.portfolio_state.total_value,
                "daily_trades": self.portfolio_state.trades_today,
                "total_drawdown": max(0, 100000 - self.portfolio_state.total_value)
            }
            
            # Store in Redis for monitoring
            self.redis_client.hset("risk_metrics", mapping=metrics_data)
            
        except Exception as e:
            logger.error(f"❌ Error pushing risk metrics: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        
        return {
            "risk_limits": {
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_total_drawdown": self.risk_limits.max_total_drawdown,
                "max_position_size": self.risk_limits.max_position_size
            },
            "portfolio_state": {
                "total_value": self.portfolio_state.total_value,
                "positions": self.portfolio_state.positions,
                "daily_pnl": self.portfolio_state.daily_pnl,
                "trades_today": self.portfolio_state.trades_today
            },
            "statistics": self.risk_stats,
            "last_updated": datetime.now().isoformat()
        }

async def main():
    """Main function to run risk guard service"""
    
    risk_guard = RiskGuardService()
    
    try:
        await risk_guard.process_recommendations()
    except KeyboardInterrupt:
        logger.info("🛡️ Risk Guard service stopped")
    except Exception as e:
        logger.error(f"❌ Risk Guard service error: {e}")

if __name__ == "__main__":
    asyncio.run(main())