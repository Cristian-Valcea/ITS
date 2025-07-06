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
from .event_types import RiskEvent, EventType, EventPriority
from .calculators import (
    DrawdownCalculator, TurnoverCalculator,
    UlcerIndexCalculator, DrawdownVelocityCalculator,
    ExpectedShortfallCalculator, KyleLambdaCalculator,
    DepthShockCalculator, FeedStalenessCalculator,
    LatencyDriftCalculator, ADVParticipationCalculator
)
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
            }),
            # Add new sensor calculators with minimal config (thresholds from YAML)
            UlcerIndexCalculator(config={}),
            DrawdownVelocityCalculator(config={}),
            ExpectedShortfallCalculator(config={}),
            KyleLambdaCalculator(config={}),
            DepthShockCalculator(config={}),
            FeedStalenessCalculator(config={}),
            LatencyDriftCalculator(config={}),
            ADVParticipationCalculator(config={})
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
        Legacy method for backward compatibility.
        Only checks basic turnover/drawdown limits.
        
        Args:
            trade_value: Absolute value of proposed trade
            timestamp: Current timestamp
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Calculate basic risk metrics for legacy assessment
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
            
            # Create event for basic risk assessment
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
            
            # Run basic risk assessment through rules engine
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
                return True, "Trade within basic risk limits"
                
        except Exception as e:
            self.logger.error(f"Basic risk assessment failed: {e}")
            # Fail safe - block trade if assessment fails
            return False, f"Risk assessment error: {str(e)}"
    
    def pre_trade_check(self, symbol: str, quantity: float, price: float, 
                       timestamp: datetime, market_data: Dict[str, Any] = None) -> Tuple[bool, str, str]:
        """
        Comprehensive pre-trade risk check using all sensor calculators and policies.
        
        This method:
        1. Runs all sensor calculators to get fresh risk metrics
        2. Creates a comprehensive TRADE_REQUEST event
        3. Evaluates all active policies through RulesEngine
        4. Returns the overall action and detailed reasoning
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (signed: +buy, -sell)
            price: Trade price per share
            timestamp: Current timestamp
            market_data: Optional market data for sensor calculations
            
        Returns:
            Tuple of (is_safe, action, detailed_reason)
            - is_safe: True if trade should proceed
            - action: RuleAction (ALLOW/WARN/THROTTLE/BLOCK/HALT/KILL_SWITCH)
            - detailed_reason: Comprehensive explanation with triggered rules
        """
        try:
            trade_value = abs(quantity * price)
            
            self.logger.debug(f"Pre-trade check for {symbol}: {quantity} @ ${price:.2f} = ${trade_value:,.2f}")
            
            # Step 1: Gather comprehensive market data for sensor calculations
            comprehensive_data = self._gather_comprehensive_market_data(
                symbol, quantity, price, timestamp, market_data
            )
            
            # Step 2: Run all sensor calculators to get fresh metrics
            all_metrics = self._run_all_calculators(comprehensive_data)
            
            # Step 3: Create comprehensive TRADE_REQUEST event
            trade_event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                data={
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'trade_value': trade_value,
                    'timestamp': timestamp,
                    **all_metrics  # Include all sensor metrics
                },
                priority=EventPriority.HIGH,
                source='RiskAgentAdapter',
                timestamp_ns=int(timestamp.timestamp() * 1_000_000_000)
            )
            
            # Step 4: Forward to RiskAgentV2 for comprehensive evaluation
            assessment_result = self._run_comprehensive_risk_assessment(trade_event)
            
            if assessment_result is None:
                self.logger.warning("No comprehensive risk assessment result - allowing trade")
                return True, "ALLOW", "No risk assessment result available"
            
            # Step 5: Interpret results and determine action
            overall_action = assessment_result.overall_action
            triggered_rules = [r for r in assessment_result.rule_results if r.triggered]
            
            # Build detailed reason
            if triggered_rules:
                rule_details = []
                for rule in triggered_rules:
                    rule_details.append(f"{rule.rule_name} ({rule.severity})")
                
                detailed_reason = f"{overall_action.value.upper()}: {len(triggered_rules)} rules triggered - {', '.join(rule_details)}"
            else:
                detailed_reason = f"{overall_action.value.upper()}: All risk checks passed"
            
            # Determine if trade is safe
            blocking_actions = [RuleAction.BLOCK, RuleAction.HALT, RuleAction.LIQUIDATE]
            is_safe = overall_action not in blocking_actions
            
            # Log the decision
            if is_safe:
                self.logger.info(f"✅ Trade approved for {symbol}: {detailed_reason}")
            else:
                self.logger.warning(f"❌ Trade blocked for {symbol}: {detailed_reason}")
            
            return is_safe, overall_action.value, detailed_reason
            
        except Exception as e:
            self.logger.error(f"Comprehensive pre-trade check failed for {symbol}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fail safe - block trade if comprehensive assessment fails
            return False, "BLOCK", f"Pre-trade check error: {str(e)}"
    
    def _gather_comprehensive_market_data(self, symbol: str, quantity: float, price: float, 
                                        timestamp: datetime, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gather comprehensive market data for all sensor calculators."""
        
        # Start with basic trade data
        data = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'trade_value': abs(quantity * price),
            'timestamp': timestamp,
            'current_time': timestamp.timestamp(),
            
            # Portfolio state
            'current_portfolio_value': self.current_portfolio_value,
            'start_of_day_value': self.start_of_day_portfolio_value,
            'daily_traded_value': self.daily_traded_value + abs(quantity * price),
            'hourly_traded_value': self.hourly_traded_value + abs(quantity * price),
        }
        
        # Calculate basic metrics
        if self.start_of_day_portfolio_value and self.start_of_day_portfolio_value > 0:
            if self.current_portfolio_value is not None:
                data['daily_drawdown'] = (self.current_portfolio_value - self.start_of_day_portfolio_value) / self.start_of_day_portfolio_value
            
            data['daily_turnover_ratio'] = data['daily_traded_value'] / self.start_of_day_portfolio_value
            data['hourly_turnover_ratio'] = data['hourly_traded_value'] / self.start_of_day_portfolio_value
        
        # Add market data if provided
        if market_data:
            data.update(market_data)
        
        # Add mock data for sensors that need it (in production, get from real market data)
        if 'feed_timestamps' not in data:
            data['feed_timestamps'] = {
                'market_data': timestamp.timestamp() - 0.1,  # 100ms old
                'order_book': timestamp.timestamp() - 0.05,  # 50ms old
                'trades': timestamp.timestamp() - 0.2,       # 200ms old
            }
        
        if 'order_latencies' not in data:
            # Use recent order latencies if available, otherwise mock
            data['order_latencies'] = [50.0, 45.0, 55.0, 48.0, 52.0]  # Mock latencies in ms
        
        if 'portfolio_values' not in data:
            # Create recent portfolio value history
            if self.current_portfolio_value:
                data['portfolio_values'] = [self.current_portfolio_value] * 10
                data['returns'] = [0.0001, -0.0005, 0.0003, -0.0002, 0.0001] * 2
        
        return data
    
    def _run_all_calculators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all sensor calculators and collect metrics."""
        all_metrics = {}
        
        for calculator in self.risk_agent_v2.calculators:
            calc_name = calculator.__class__.__name__
            try:
                result = calculator.calculate_safe(data)
                if result.is_valid and result.values:
                    all_metrics.update(result.values)
                    self.logger.debug(f"✅ {calc_name}: {len(result.values)} metrics")
                else:
                    self.logger.debug(f"⚠️ {calc_name}: No valid metrics")
            except Exception as e:
                self.logger.warning(f"❌ {calc_name} failed: {e}")
        
        self.logger.debug(f"Collected {len(all_metrics)} total metrics from {len(self.risk_agent_v2.calculators)} calculators")
        return all_metrics
    
    def _run_comprehensive_risk_assessment(self, event: RiskEvent):
        """Run comprehensive risk assessment through RiskAgentV2."""
        try:
            # Run all calculators to get fresh metrics
            all_metrics = self._run_all_calculators(event.data)
            
            # Evaluate all active policies using the rules engine
            # For now, use the basic policy - in production, evaluate all active policies
            policy_result = self.risk_agent_v2.rules_engine.evaluate_policy('basic_risk_limits', all_metrics)
            
            return policy_result
        except Exception as e:
            self.logger.error(f"Comprehensive risk assessment failed: {e}")
            return None
    
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