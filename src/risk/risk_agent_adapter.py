# src/risk/risk_agent_adapter.py
"""
RiskAgentAdapter: Compatibility layer between OrchestratorAgent and RiskAgentV2.
Provides synchronous interface while leveraging the advanced RiskAgentV2 capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .risk_agent_v2 import RiskAgentV2
from .event_bus import RiskEvent, EventType, EventPriority
from .calculators import DrawdownCalculator, TurnoverCalculator
from .rules_engine import RulesEngine, RuleAction, RiskPolicy, ThresholdRule


class RiskAgentAdapter:
    """
    Adapter that provides the old RiskAgent interface while using RiskAgentV2 internally.
    This allows the OrchestratorAgent to use the new risk system without major refactoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with RiskAgentV2 and required components.
        
        Args:
            config: Risk configuration dictionary (same format as old RiskAgent)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration
        self.max_daily_drawdown_pct = float(config.get('max_daily_drawdown_pct', 0.02))
        self.max_hourly_turnover_ratio = float(config.get('max_hourly_turnover_ratio', 5.0))
        self.max_daily_turnover_ratio = float(config.get('max_daily_turnover_ratio', 20.0))
        self.halt_on_breach = bool(config.get('halt_on_breach', True))
        self.liquidate_on_halt = bool(config.get('liquidate_on_halt', False))
        
        # State variables (compatible with old RiskAgent)
        self.start_of_day_portfolio_value = None
        self.current_portfolio_value = None
        self.daily_traded_value = 0.0
        self.hourly_traded_value = 0.0
        self.last_event_timestamp = None
        
        # Initialize RiskAgentV2 components
        self._init_risk_agent_v2()
        
        self.logger.info("RiskAgentAdapter initialized with RiskAgentV2 backend")
    
    def _init_risk_agent_v2(self):
        """Initialize RiskAgentV2 with appropriate calculators and rules."""
        # Create calculators
        calculators = [
            DrawdownCalculator(config={
                'max_drawdown_pct': self.max_daily_drawdown_pct,
                'lookback_window': 1440  # 1 day in minutes
            }),
            TurnoverCalculator(config={
                'max_hourly_turnover_ratio': self.max_hourly_turnover_ratio,
                'max_daily_turnover_ratio': self.max_daily_turnover_ratio,
                'capital_base_method': 'start_of_day'
            })
        ]
        
        # Create rules engine with basic policies
        rules_engine = RulesEngine()
        
        # Create basic risk policy (compatible with old behavior)
        basic_policy = RiskPolicy(
            policy_id='basic_risk_limits',
            policy_name='Basic Risk Limits',
            version='1.0.0'
        )
        
        # Add rules to the policy
        drawdown_rule = ThresholdRule(
            rule_id='daily_drawdown_limit',
            rule_name='Daily Drawdown Limit',
            config={
                'field': 'daily_drawdown',
                'threshold': -self.max_daily_drawdown_pct,
                'operator': 'lt',
                'action': 'halt' if self.halt_on_breach else 'warn',
                'severity': 'critical',
                'message': f'Daily drawdown {{value:.2%}} exceeded {self.max_daily_drawdown_pct:.1%} limit'
            }
        )
        
        hourly_turnover_rule = ThresholdRule(
            rule_id='hourly_turnover_limit',
            rule_name='Hourly Turnover Limit',
            config={
                'field': 'hourly_turnover_ratio',
                'threshold': self.max_hourly_turnover_ratio,
                'operator': 'gt',
                'action': 'block',
                'severity': 'high',
                'message': f'Hourly turnover {{value:.2f}}x exceeded {self.max_hourly_turnover_ratio}x limit'
            }
        )
        
        daily_turnover_rule = ThresholdRule(
            rule_id='daily_turnover_limit',
            rule_name='Daily Turnover Limit',
            config={
                'field': 'daily_turnover_ratio',
                'threshold': self.max_daily_turnover_ratio,
                'operator': 'gt',
                'action': 'halt' if self.halt_on_breach else 'warn',
                'severity': 'high',
                'message': f'Daily turnover {{value:.2f}}x exceeded {self.max_daily_turnover_ratio}x limit'
            }
        )
        
        basic_policy.add_rule(drawdown_rule)
        basic_policy.add_rule(hourly_turnover_rule)
        basic_policy.add_rule(daily_turnover_rule)
        
        rules_engine.register_policy(basic_policy)
        
        # Create limits config
        limits_config = {
            'active_policy': 'basic_risk_limits',
            **self.config
        }
        
        # Initialize RiskAgentV2
        self.risk_agent_v2 = RiskAgentV2(
            calculators=calculators,
            rules_engine=rules_engine,
            limits_config=limits_config
        )
    
    def reset_daily_limits(self, current_portfolio_value: float, timestamp: datetime):
        """Reset daily limits (compatible with old RiskAgent interface)."""
        self.start_of_day_portfolio_value = current_portfolio_value
        self.current_portfolio_value = current_portfolio_value
        self.daily_traded_value = 0.0
        self.hourly_traded_value = 0.0
        self.last_event_timestamp = timestamp
        
        # Update RiskAgentV2 state
        self.risk_agent_v2.start_of_day_value = current_portfolio_value
        self.risk_agent_v2.last_portfolio_value = current_portfolio_value
        self.risk_agent_v2.trade_history = []
        
        self.logger.info(f"Daily limits reset with portfolio value: ${current_portfolio_value:,.2f}")
    
    def update_portfolio_value(self, portfolio_value: float, timestamp: datetime):
        """Update current portfolio value (compatible with old RiskAgent interface)."""
        self.current_portfolio_value = portfolio_value
        self.last_event_timestamp = timestamp
        
        # Update RiskAgentV2 state
        self.risk_agent_v2.last_portfolio_value = portfolio_value
        
        self.logger.debug(f"Portfolio value updated: ${portfolio_value:,.2f}")
    
    def record_trade(self, trade_value: float, timestamp: datetime):
        """Record a trade for turnover calculations (compatible with old RiskAgent interface)."""
        abs_trade_value = abs(trade_value)
        
        # Update local state
        self.daily_traded_value += abs_trade_value
        self.hourly_traded_value += abs_trade_value
        
        # Update RiskAgentV2 state
        trade_record = {
            'value': abs_trade_value,
            'timestamp': timestamp
        }
        self.risk_agent_v2.trade_history.append(trade_record)
        
        self.logger.debug(f"Trade recorded: ${abs_trade_value:,.2f}")
    
    def assess_trade_risk(self, trade_value: float, timestamp: datetime) -> Tuple[bool, str]:
        """
        Assess if a proposed trade violates risk limits.
        
        Args:
            trade_value: Absolute value of proposed trade
            timestamp: Current timestamp
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Calculate risk metrics for assessment
            proposed_daily_traded = self.daily_traded_value + abs(trade_value)
            proposed_hourly_traded = self.hourly_traded_value + abs(trade_value)
            
            # Calculate ratios
            daily_drawdown = 0.0
            daily_turnover_ratio = 0.0
            hourly_turnover_ratio = 0.0
            
            if self.start_of_day_portfolio_value and self.start_of_day_portfolio_value > 0:
                if self.current_portfolio_value is not None:
                    daily_drawdown = (self.current_portfolio_value - self.start_of_day_portfolio_value) / self.start_of_day_portfolio_value
                
                daily_turnover_ratio = proposed_daily_traded / self.start_of_day_portfolio_value
                hourly_turnover_ratio = proposed_hourly_traded / self.start_of_day_portfolio_value
            
            # Create event for risk assessment
            event_data = {
                'trade_value': abs(trade_value),
                'timestamp': timestamp,
                'current_portfolio_value': self.current_portfolio_value,
                'start_of_day_value': self.start_of_day_portfolio_value,
                'daily_traded_value': proposed_daily_traded,
                'hourly_traded_value': proposed_hourly_traded,
                'daily_drawdown': daily_drawdown,
                'daily_turnover_ratio': daily_turnover_ratio,
                'hourly_turnover_ratio': hourly_turnover_ratio
            }
            
            event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                data=event_data,
                priority=EventPriority.HIGH,
                timestamp_ns=int(timestamp.timestamp() * 1_000_000_000)  # Convert to nanoseconds
            )
            
            # Run risk assessment directly through rules engine
            policy_result = self.risk_agent_v2.rules_engine.evaluate_policy('basic_risk_limits', event_data)
            
            if policy_result is None:
                return True, "No risk assessment result"
            
            # Check if any blocking actions were triggered
            if policy_result.overall_action in [RuleAction.BLOCK, RuleAction.HALT]:
                triggered_rule_names = [rule.rule_name for rule in policy_result.rule_results if rule.triggered]
                reason = f"{policy_result.overall_action.value.upper()}: {', '.join(triggered_rule_names)}"
                return False, reason
            elif policy_result.overall_action == RuleAction.WARN:
                triggered_rule_names = [rule.rule_name for rule in policy_result.rule_results if rule.triggered]
                reason = f"WARNING: {', '.join(triggered_rule_names)}"
                return True, reason
            else:
                return True, "Trade within risk limits"
                
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            # Fail safe - block trade if assessment fails
            return False, f"Risk assessment error: {str(e)}"
    
    def _run_risk_assessment_sync(self, event: RiskEvent):
        """Run risk assessment synchronously by creating a temporary event loop."""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # Create a new thread with its own event loop
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.risk_agent_v2.handle(event))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=1.0)  # 1 second timeout
            else:
                # No running loop, we can use it directly
                return loop.run_until_complete(self.risk_agent_v2.handle(event))
                
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self.risk_agent_v2.handle(event))
        except Exception as e:
            self.logger.error(f"Async risk assessment failed: {e}")
            return None
    
    def get_current_drawdown(self) -> float:
        """Get current daily drawdown percentage."""
        if self.start_of_day_portfolio_value is None or self.current_portfolio_value is None:
            return 0.0
        
        return (self.current_portfolio_value - self.start_of_day_portfolio_value) / self.start_of_day_portfolio_value
    
    def get_daily_turnover_ratio(self) -> float:
        """Get current daily turnover ratio."""
        if self.start_of_day_portfolio_value is None or self.start_of_day_portfolio_value == 0:
            return 0.0
        
        return self.daily_traded_value / self.start_of_day_portfolio_value
    
    def get_hourly_turnover_ratio(self) -> float:
        """Get current hourly turnover ratio."""
        if self.start_of_day_portfolio_value is None or self.start_of_day_portfolio_value == 0:
            return 0.0
        
        return self.hourly_traded_value / self.start_of_day_portfolio_value
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics."""
        return {
            'current_portfolio_value': self.current_portfolio_value,
            'start_of_day_value': self.start_of_day_portfolio_value,
            'daily_drawdown': self.get_current_drawdown(),
            'daily_turnover_ratio': self.get_daily_turnover_ratio(),
            'hourly_turnover_ratio': self.get_hourly_turnover_ratio(),
            'daily_traded_value': self.daily_traded_value,
            'hourly_traded_value': self.hourly_traded_value,
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'max_hourly_turnover_ratio': self.max_hourly_turnover_ratio,
            'max_daily_turnover_ratio': self.max_daily_turnover_ratio,
            'halt_on_breach': self.halt_on_breach
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from RiskAgentV2."""
        if hasattr(self.risk_agent_v2, '_evaluation_count'):
            return {
                'evaluation_count': self.risk_agent_v2._evaluation_count,
                'total_evaluation_time_ns': self.risk_agent_v2._total_evaluation_time_ns,
                'avg_evaluation_time_us': (
                    self.risk_agent_v2._total_evaluation_time_ns / self.risk_agent_v2._evaluation_count / 1000.0
                    if self.risk_agent_v2._evaluation_count > 0 else 0.0
                ),
                'action_counts': dict(self.risk_agent_v2._action_counts)
            }
        else:
            return {'evaluation_count': 0}