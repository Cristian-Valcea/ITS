# src/risk/risk_agent_v2.py
"""
RiskAgentV2: Enterprise-grade risk management orchestrator.
Subscribes to events, runs calculators, evaluates policies, and enforces actions.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .event_bus import EventHandler, RiskEvent, EventType, EventPriority
from .calculators.base_calculator import BaseRiskCalculator, RiskCalculationResult
from .rules_engine import RulesEngine, RuleAction, PolicyEvaluationResult
from .calculators import DrawdownCalculator, TurnoverCalculator


class RiskAgentV2(EventHandler):
    """
    Enterprise risk management orchestrator that:
    1. Subscribes to trade/market events
    2. Runs risk calculators in parallel
    3. Evaluates policies through rules engine
    4. Enforces risk actions (block/halt/liquidate)
    5. Publishes monitoring events
    """
    
    def __init__(self, calculators: List[BaseRiskCalculator], 
                 rules_engine: RulesEngine, 
                 limits_config: Dict[str, Any]):
        """
        Initialize RiskAgentV2.
        
        Args:
            calculators: List of risk calculators to run
            rules_engine: Rules engine for policy evaluation
            limits_config: Configuration including active policy
        """
        self.calculators = calculators
        self.rules_engine = rules_engine
        self.limits_config = limits_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self._evaluation_count = 0
        self._total_evaluation_time_ns = 0
        self._action_counts = {action: 0 for action in RuleAction}
        
        # State tracking
        self.last_portfolio_value = None
        self.start_of_day_value = None
        self.trade_history = []
        
        self.logger.info(f"RiskAgentV2 initialized with {len(calculators)} calculators")
    
    @property
    def supported_event_types(self):
        """Events this handler processes."""
        return [
            EventType.TRADE_REQUEST,
            EventType.MARKET_DATA,
            EventType.POSITION_UPDATE
        ]
    
    @property
    def priority_filter(self):
        """Process high and critical priority events."""
        return [EventPriority.CRITICAL, EventPriority.HIGH]
    
    async def handle(self, event: RiskEvent) -> Optional[RiskEvent]:
        """
        Main event handler - orchestrates the complete risk evaluation pipeline.
        
        Args:
            event: Incoming risk event
            
        Returns:
            Risk action event if enforcement needed, monitoring event otherwise
        """
        start_time = time.time_ns()
        
        try:
            # Update internal state from event
            self._update_state_from_event(event)
            
            # Step 1: Run all calculators
            calc_results = await self._run_calculators(event.data)
            
            # Step 2: Flatten results into dict for rules engine
            flat_data = self._flatten_calculation_results(calc_results, event.data)
            
            # Step 3: Evaluate active policy
            policy_result = await self._evaluate_policy(flat_data)
            
            # Step 4: Enforce action if needed
            response_event = await self._enforce_action(policy_result, event)
            
            # Update performance metrics
            evaluation_time = time.time_ns() - start_time
            self._evaluation_count += 1
            self._total_evaluation_time_ns += evaluation_time
            
            if policy_result:
                self._action_counts[policy_result.overall_action] += 1
            
            # Log performance for critical events
            if event.priority == EventPriority.CRITICAL:
                evaluation_time_us = evaluation_time / 1000.0
                self.logger.info(f"Risk evaluation completed in {evaluation_time_us:.2f}µs")
            
            return response_event
            
        except Exception as e:
            self.logger.error(f"Risk evaluation failed: {e}")
            
            # Return emergency halt on evaluation failure
            return RiskEvent(
                event_type=EventType.KILL_SWITCH,
                priority=EventPriority.CRITICAL,
                source='RiskAgentV2',
                data={
                    'action': RuleAction.HALT.value,
                    'reason': f'Risk evaluation error: {str(e)}',
                    'original_event_id': event.event_id,
                    'emergency': True
                }
            )
    
    def _update_state_from_event(self, event: RiskEvent) -> None:
        """Update internal state from incoming event."""
        data = event.data
        
        # Update portfolio value
        if 'portfolio_value' in data:
            self.last_portfolio_value = data['portfolio_value']
        elif 'current_portfolio_value' in data:
            self.last_portfolio_value = data['current_portfolio_value']
        
        # Set start of day value if not set
        if 'start_of_day_value' in data and self.start_of_day_value is None:
            self.start_of_day_value = data['start_of_day_value']
        
        # Track trades
        if event.event_type == EventType.TRADE_REQUEST and 'trade_value' in data:
            self.trade_history.append({
                'value': data['trade_value'],
                'timestamp': datetime.now(),
                'event_id': event.event_id
            })
    
    async def _run_calculators(self, event_data: Dict[str, Any]) -> Dict[str, RiskCalculationResult]:
        """
        Run all calculators against event data.
        
        Args:
            event_data: Data from the event
            
        Returns:
            Dictionary mapping calculator metric type to results
        """
        calc_results = {}
        
        # Prepare enhanced data for calculators
        enhanced_data = self._prepare_calculator_data(event_data)
        
        for calculator in self.calculators:
            try:
                # Check if calculator has required inputs
                required_inputs = calculator.get_required_inputs()
                if not all(key in enhanced_data for key in required_inputs):
                    self.logger.debug(f"Skipping {calculator.__class__.__name__}: missing required inputs")
                    continue
                
                # Run calculation
                result = calculator.calculate_safe(enhanced_data)
                calc_results[calculator.metric_type.value] = result
                
                if result.is_valid:
                    self.logger.debug(f"{calculator.__class__.__name__} completed in {result.get_calculation_time_us():.2f}µs")
                else:
                    self.logger.warning(f"{calculator.__class__.__name__} failed: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Calculator {calculator.__class__.__name__} failed: {e}")
                # Continue with other calculators
        
        return calc_results
    
    def _prepare_calculator_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare enhanced data for calculators by combining event data with state.
        
        Args:
            event_data: Raw event data
            
        Returns:
            Enhanced data dictionary
        """
        enhanced_data = event_data.copy()
        
        # Add state information
        if self.last_portfolio_value is not None:
            enhanced_data['current_portfolio_value'] = self.last_portfolio_value
        
        if self.start_of_day_value is not None:
            enhanced_data['start_of_day_value'] = self.start_of_day_value
            enhanced_data['capital_base'] = self.start_of_day_value
        
        # Add trade history for turnover calculations
        if self.trade_history:
            trade_values = [t['value'] for t in self.trade_history]
            trade_timestamps = [t['timestamp'] for t in self.trade_history]
            
            enhanced_data['trade_values'] = trade_values
            enhanced_data['trade_timestamps'] = trade_timestamps
        
        # Build portfolio values array for drawdown calculations
        if 'portfolio_values' not in enhanced_data and self.last_portfolio_value is not None:
            # Simple case: use current value as numpy array
            import numpy as np
            enhanced_data['portfolio_values'] = np.array([self.last_portfolio_value])
        
        # Calculate daily drawdown if we have both values
        if (self.start_of_day_value is not None and 
            self.last_portfolio_value is not None and 
            'daily_drawdown' not in enhanced_data):
            daily_drawdown = (self.last_portfolio_value - self.start_of_day_value) / self.start_of_day_value
            enhanced_data['daily_drawdown'] = daily_drawdown
        
        return enhanced_data
    
    def _flatten_calculation_results(self, calc_results: Dict[str, RiskCalculationResult], 
                                   event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten calculation results into a single dict for rules engine.
        
        Args:
            calc_results: Results from calculators
            event_data: Original event data
            
        Returns:
            Flattened dictionary for rules evaluation
        """
        flat_data = event_data.copy()  # Start with original event data
        
        # Flatten calculator results
        for calc_type, result in calc_results.items():
            if result.is_valid:
                # Add all values from the result
                for key, value in result.values.items():
                    try:
                        # Avoid numpy arrays in rules - use scalar values
                        if hasattr(value, 'item'):  # numpy scalar
                            flat_data[key] = value.item()
                        elif hasattr(value, '__len__') and not isinstance(value, str):  # array-like
                            if len(value) > 0:
                                # Handle different array types
                                if hasattr(value, 'iloc'):  # pandas
                                    flat_data[key] = float(value.iloc[-1])
                                else:  # numpy or list
                                    flat_data[key] = float(value[-1])
                        else:
                            flat_data[key] = float(value) if isinstance(value, (int, float)) else value
                    except (ValueError, TypeError, IndexError) as e:
                        self.logger.debug(f"Could not flatten {key}={value}: {e}")
                        # Skip problematic values
                        continue
            else:
                # Mark failed calculations
                flat_data[f'{calc_type}_failed'] = True
                flat_data[f'{calc_type}_error'] = result.error_message
        
        return flat_data
    
    async def _evaluate_policy(self, flat_data: Dict[str, Any]) -> Optional[PolicyEvaluationResult]:
        """
        Evaluate the active policy against flattened data.
        
        Args:
            flat_data: Flattened data for rules evaluation
            
        Returns:
            Policy evaluation result or None if no active policy
        """
        active_policy = self.limits_config.get('active_policy')
        if not active_policy:
            self.logger.warning("No active policy configured")
            return None
        
        try:
            result = self.rules_engine.evaluate_policy(active_policy, flat_data)
            
            if result and result.triggered_rules:
                self.logger.info(f"Policy {active_policy} triggered: {result.overall_action.value} "
                               f"({len(result.triggered_rules)} rules)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Policy evaluation failed: {e}")
            return None
    
    async def _enforce_action(self, policy_result: Optional[PolicyEvaluationResult], 
                            original_event: RiskEvent) -> Optional[RiskEvent]:
        """
        Enforce risk action based on policy result.
        
        Args:
            policy_result: Result from policy evaluation
            original_event: Original event that triggered evaluation
            
        Returns:
            Risk action event if enforcement needed, monitoring event otherwise
        """
        if not policy_result:
            # No policy result - publish monitoring event
            return self._create_monitoring_event(original_event, None)
        
        action = policy_result.overall_action
        
        # Critical actions that require immediate enforcement
        if action in (RuleAction.BLOCK, RuleAction.HALT, RuleAction.LIQUIDATE):
            self.logger.critical(f"RISK ACTION: {action.value} - {policy_result.triggered_rules}")
            
            return RiskEvent(
                event_type=EventType.KILL_SWITCH,
                priority=EventPriority.CRITICAL,
                source='RiskAgentV2',
                data={
                    'action': action.value,
                    'reason': policy_result.triggered_rules,
                    'policy_id': policy_result.policy_id,
                    'evaluation_time_us': policy_result.get_evaluation_time_us(),
                    'original_event_id': original_event.event_id,
                    'triggered_rules_count': len(policy_result.triggered_rules),
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        # Warning actions
        elif action == RuleAction.WARN:
            self.logger.warning(f"RISK WARNING: {policy_result.triggered_rules}")
            
            return RiskEvent(
                event_type=EventType.ALERT,
                priority=EventPriority.HIGH,
                source='RiskAgentV2',
                data={
                    'alert_type': 'risk_warning',
                    'action': action.value,
                    'reason': policy_result.triggered_rules,
                    'policy_id': policy_result.policy_id,
                    'original_event_id': original_event.event_id
                }
            )
        
        # Position management actions
        elif action in (RuleAction.REDUCE_POSITION, RuleAction.HEDGE):
            self.logger.info(f"RISK MANAGEMENT: {action.value} - {policy_result.triggered_rules}")
            
            return RiskEvent(
                event_type=EventType.POSITION_MANAGEMENT,
                priority=EventPriority.HIGH,
                source='RiskAgentV2',
                data={
                    'action': action.value,
                    'reason': policy_result.triggered_rules,
                    'policy_id': policy_result.policy_id,
                    'original_event_id': original_event.event_id
                }
            )
        
        # Default: monitoring event
        else:
            return self._create_monitoring_event(original_event, policy_result)
    
    def _create_monitoring_event(self, original_event: RiskEvent, 
                               policy_result: Optional[PolicyEvaluationResult]) -> RiskEvent:
        """Create monitoring event for normal operations."""
        return RiskEvent(
            event_type=EventType.RISK_MONITORING,
            priority=EventPriority.LOW,
            source='RiskAgentV2',
            data={
                'status': 'normal',
                'policy_evaluated': policy_result is not None,
                'policy_id': policy_result.policy_id if policy_result else None,
                'triggered_rules': policy_result.triggered_rules if policy_result else [],
                'evaluation_time_us': policy_result.get_evaluation_time_us() if policy_result else 0,
                'original_event_id': original_event.event_id,
                'calculators_run': len(self.calculators),
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this risk agent."""
        if self._evaluation_count == 0:
            return {
                'evaluation_count': 0,
                'avg_evaluation_time_us': 0.0,
                'action_counts': dict(self._action_counts)
            }
        
        avg_time_ns = self._total_evaluation_time_ns / self._evaluation_count
        
        return {
            'evaluation_count': self._evaluation_count,
            'avg_evaluation_time_us': avg_time_ns / 1000.0,
            'action_counts': {action.value: count for action, count in self._action_counts.items()},
            'calculators_count': len(self.calculators),
            'last_portfolio_value': self.last_portfolio_value,
            'start_of_day_value': self.start_of_day_value,
            'trade_history_count': len(self.trade_history)
        }
    
    def reset_daily_state(self, start_of_day_value: float) -> None:
        """Reset daily state for new trading day."""
        self.start_of_day_value = start_of_day_value
        self.last_portfolio_value = start_of_day_value
        self.trade_history.clear()
        
        self.logger.info(f"Daily state reset - SOD value: {start_of_day_value:.2f}")
    
    def add_calculator(self, calculator: BaseRiskCalculator) -> None:
        """Add a new calculator to the agent."""
        self.calculators.append(calculator)
        self.logger.info(f"Added calculator: {calculator.__class__.__name__}")
    
    def remove_calculator(self, calculator_type: str) -> bool:
        """Remove a calculator by type."""
        original_count = len(self.calculators)
        self.calculators = [c for c in self.calculators if c.metric_type.value != calculator_type]
        
        if len(self.calculators) < original_count:
            self.logger.info(f"Removed calculator: {calculator_type}")
            return True
        return False
    
    def update_limits_config(self, new_config: Dict[str, Any]) -> None:
        """Update limits configuration (hot-swap)."""
        old_policy = self.limits_config.get('active_policy')
        self.limits_config.update(new_config)
        new_policy = self.limits_config.get('active_policy')
        
        if old_policy != new_policy:
            self.logger.info(f"Active policy changed: {old_policy} -> {new_policy}")


# Factory function for easy setup
def create_risk_agent_v2(config: Dict[str, Any]) -> RiskAgentV2:
    """
    Factory function to create a fully configured RiskAgentV2.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RiskAgentV2 instance
    """
    # Create calculators
    calculators = []
    
    calc_configs = config.get('calculators', {})
    
    if calc_configs.get('drawdown', {}).get('enabled', True):
        drawdown_calc = DrawdownCalculator(
            config=calc_configs.get('drawdown', {}).get('config', {})
        )
        calculators.append(drawdown_calc)
    
    if calc_configs.get('turnover', {}).get('enabled', True):
        turnover_calc = TurnoverCalculator(
            config=calc_configs.get('turnover', {}).get('config', {})
        )
        calculators.append(turnover_calc)
    
    # Create rules engine
    rules_engine = RulesEngine()
    
    # Setup policies from config
    from .rules_engine import RiskPolicy, ThresholdRule, PolicyValidator
    
    policies_config = config.get('policies', [])
    for policy_config in policies_config:
        # Validate policy
        is_valid, errors = PolicyValidator.validate_policy_config(policy_config)
        if not is_valid:
            raise ValueError(f"Invalid policy configuration: {errors}")
        
        # Create policy
        policy = RiskPolicy(
            policy_config['policy_id'],
            policy_config['policy_name']
        )
        
        # Add rules
        for rule_config in policy_config.get('rules', []):
            if rule_config['rule_type'] == 'threshold':
                rule = ThresholdRule(
                    rule_config['rule_id'],
                    rule_config.get('rule_name', rule_config['rule_id']),
                    rule_config
                )
                policy.add_rule(rule)
        
        rules_engine.register_policy(policy)
    
    # Create limits config
    limits_config = {
        'active_policy': config.get('active_policy', 'default_policy'),
        **config.get('limits', {})
    }
    
    return RiskAgentV2(calculators, rules_engine, limits_config)