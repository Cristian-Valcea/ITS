# RiskAgent Refactoring Plan

## Overview

This document outlines the plan to refactor the existing monolithic `RiskAgent` into the new enterprise-grade modular risk management system while maintaining backward compatibility.

## Current State Analysis

### Existing RiskAgent (`src/agents/risk_agent.py`)

**Current Features**:
- Daily drawdown monitoring
- Hourly/daily turnover tracking
- Portfolio value updates
- Trade recording
- Risk limit breach detection
- Simple halt/liquidate logic

**Current Issues**:
- Monolithic design with mixed responsibilities
- No microsecond-level performance optimization
- Limited extensibility
- Basic error handling
- No hot-swappable configuration
- No comprehensive audit trail

## Refactoring Strategy

### Phase 1: Backward-Compatible Wrapper

Create a new `RiskAgent` that wraps the enterprise system while maintaining the existing API.

```python
# New RiskAgent implementation
class RiskAgent(BaseAgent):
    """
    Refactored RiskAgent using enterprise risk management system.
    Maintains backward compatibility while providing enhanced capabilities.
    """
    
    def __init__(self, config: dict):
        super().__init__(agent_name="RiskAgent", config=config)
        
        # Initialize enterprise risk system
        self._setup_enterprise_system()
        
        # Maintain backward compatibility
        self._setup_legacy_interface()
    
    def _setup_enterprise_system(self):
        """Initialize the enterprise risk management components."""
        # Event bus
        self.event_bus = RiskEventBus()
        
        # Risk calculators
        self.drawdown_calc = DrawdownCalculator(self._get_drawdown_config())
        self.turnover_calc = TurnoverCalculator(self._get_turnover_config())
        
        # Rules engine with policies from config
        self.rules_engine = RulesEngine()
        self._setup_risk_policies()
        
        # Enforcement handler
        self.enforcement = EnforcementHandler()
    
    # Legacy API methods (unchanged interface)
    def update_portfolio_value(self, portfolio_value: float, timestamp: datetime = None):
        """Legacy method - now uses enterprise system internally."""
        
    def record_trade(self, trade_value: float, timestamp: datetime = None):
        """Legacy method - now uses enterprise system internally."""
        
    def assess_trade_risk(self, proposed_trade_value: float, current_time: datetime = None):
        """Legacy method - now uses enterprise system internally."""
```

### Phase 2: Enhanced Configuration

Extend the configuration system to support both legacy and enterprise features.

```yaml
# Enhanced risk_limits.yaml
# Legacy configuration (maintained for backward compatibility)
max_daily_drawdown_pct: 0.02
max_hourly_turnover_ratio: 5.0
max_daily_turnover_ratio: 20.0
halt_on_breach: true
liquidate_on_halt: false

# Enterprise configuration (new features)
enterprise_features:
  enabled: true
  
  # Performance settings
  latency_targets:
    critical_us: 20
    high_us: 150
    medium_us: 100
  
  # Risk calculators
  calculators:
    drawdown:
      enabled: true
      lookback_periods: [1, 5, 20, 60]
      config:
        min_periods: 1
        annualization_factor: 252
    
    turnover:
      enabled: true
      config:
        hourly_window_minutes: 60
        daily_window_hours: 24
        use_absolute_values: true
    
    volatility:
      enabled: false  # Future enhancement
    
    concentration:
      enabled: false  # Future enhancement
  
  # Risk policies
  policies:
    - policy_id: "drawdown_policy"
      policy_name: "Drawdown Risk Management"
      rules:
        - rule_id: "daily_drawdown_limit"
          rule_type: "threshold"
          field: "daily_drawdown"
          threshold: -0.02
          operator: "lt"
          action: "halt"
          severity: "critical"
        
        - rule_id: "max_drawdown_warning"
          rule_type: "threshold"
          field: "max_drawdown"
          threshold: -0.05
          operator: "lt"
          action: "warn"
          severity: "medium"
    
    - policy_id: "turnover_policy"
      policy_name: "Turnover Risk Management"
      rules:
        - rule_id: "hourly_turnover_limit"
          rule_type: "threshold"
          field: "hourly_turnover_ratio"
          threshold: 5.0
          operator: "gt"
          action: "block"
          severity: "high"
        
        - rule_id: "daily_turnover_limit"
          rule_type: "threshold"
          field: "daily_turnover_ratio"
          threshold: 20.0
          operator: "gt"
          action: "halt"
          severity: "critical"
  
  # Monitoring and alerting
  monitoring:
    latency_slo_monitoring: true
    performance_tracking: true
    audit_trail: true
    
  # Circuit breakers
  circuit_breakers:
    enabled: true
    max_errors_per_type: 10
    recovery_time_seconds: 60
```

### Phase 3: Migration Implementation

#### Step 1: Create Refactored RiskAgent

```python
# src/agents/risk_agent_v2.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from .base_agent import BaseAgent
from ..risk import (
    RiskEventBus, RiskEvent, EventType, EventPriority,
    DrawdownCalculator, TurnoverCalculator,
    RulesEngine, RiskPolicy, ThresholdRule,
    PolicyValidator
)

class RiskAgentV2(BaseAgent):
    """
    Enterprise-grade RiskAgent with backward compatibility.
    """
    
    def __init__(self, config: dict):
        super().__init__(agent_name="RiskAgent", config=config)
        
        # Legacy state variables (for backward compatibility)
        self.start_of_day_portfolio_value = None
        self.current_portfolio_value = None
        self.daily_traded_value = 0.0
        self.hourly_traded_value = 0.0
        self.last_event_timestamp = None
        
        # Legacy configuration
        self.max_daily_drawdown_pct = float(self.config.get('max_daily_drawdown_pct', 0.02))
        self.max_hourly_turnover_ratio = float(self.config.get('max_hourly_turnover_ratio', 5.0))
        self.max_daily_turnover_ratio = float(self.config.get('max_daily_turnover_ratio', 20.0))
        self.halt_on_breach = bool(self.config.get('halt_on_breach', True))
        self.liquidate_on_halt = bool(self.config.get('liquidate_on_halt', False))
        
        # Enterprise system
        self.enterprise_enabled = self.config.get('enterprise_features', {}).get('enabled', False)
        
        if self.enterprise_enabled:
            self._setup_enterprise_system()
        else:
            self.logger.info("Running in legacy mode")
    
    def _setup_enterprise_system(self):
        """Initialize enterprise risk management system."""
        self.logger.info("Initializing enterprise risk management system")
        
        # Event bus
        latency_config = self.config.get('enterprise_features', {}).get('latency_targets', {})
        self.event_bus = RiskEventBus(
            latency_slo_us={
                EventPriority.CRITICAL: latency_config.get('critical_us', 20),
                EventPriority.HIGH: latency_config.get('high_us', 150),
                EventPriority.MEDIUM: latency_config.get('medium_us', 100)
            }
        )
        
        # Risk calculators
        calc_config = self.config.get('enterprise_features', {}).get('calculators', {})
        
        if calc_config.get('drawdown', {}).get('enabled', True):
            self.drawdown_calc = DrawdownCalculator(
                config=calc_config.get('drawdown', {}).get('config', {})
            )
        
        if calc_config.get('turnover', {}).get('enabled', True):
            self.turnover_calc = TurnoverCalculator(
                config=calc_config.get('turnover', {}).get('config', {})
            )
        
        # Rules engine
        self.rules_engine = RulesEngine()
        self._setup_risk_policies()
        
        # Start event bus
        asyncio.create_task(self.event_bus.start())
        
        self.logger.info("Enterprise risk management system initialized")
    
    def _setup_risk_policies(self):
        """Setup risk policies from configuration."""
        policies_config = self.config.get('enterprise_features', {}).get('policies', [])
        
        for policy_config in policies_config:
            # Validate policy configuration
            is_valid, errors = PolicyValidator.validate_policy_config(policy_config)
            if not is_valid:
                self.logger.error(f"Invalid policy configuration: {errors}")
                continue
            
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
            
            self.rules_engine.register_policy(policy)
            self.logger.info(f"Registered policy: {policy.policy_name}")
    
    # Legacy API methods (backward compatibility)
    
    def update_portfolio_value(self, portfolio_value: float, timestamp: datetime = None):
        """Update portfolio value - enhanced with enterprise features."""
        self.current_portfolio_value = portfolio_value
        self.last_event_timestamp = timestamp or datetime.now()
        
        if self.start_of_day_portfolio_value is None:
            self.start_of_day_portfolio_value = self.current_portfolio_value
            self.logger.info(f"Start of day portfolio value set to: {self.start_of_day_portfolio_value:.2f}")
        
        # Enterprise system processing
        if self.enterprise_enabled:
            asyncio.create_task(self._process_portfolio_update(portfolio_value, timestamp))
    
    async def _process_portfolio_update(self, portfolio_value: float, timestamp: datetime):
        """Process portfolio update through enterprise system."""
        event = RiskEvent(
            event_type=EventType.POSITION_UPDATE,
            priority=EventPriority.HIGH,
            source="RiskAgent",
            data={
                'portfolio_value': portfolio_value,
                'start_of_day_value': self.start_of_day_portfolio_value,
                'timestamp': timestamp.isoformat() if timestamp else None
            }
        )
        
        await self.event_bus.publish(event)
    
    def record_trade(self, trade_value: float, timestamp: datetime = None):
        """Record trade - enhanced with enterprise features."""
        if self.start_of_day_portfolio_value is None:
            self.logger.error("Cannot record trade: start_of_day_portfolio_value is not set")
            return
        
        current_time = timestamp or datetime.now()
        self.last_event_timestamp = current_time
        
        abs_trade_value = abs(trade_value)
        self.daily_traded_value += abs_trade_value
        
        # Legacy hourly tracking (simplified)
        self.hourly_traded_value += abs_trade_value
        
        # Enterprise system processing
        if self.enterprise_enabled:
            asyncio.create_task(self._process_trade_record(trade_value, timestamp))
        
        self.logger.debug(f"Trade recorded: Value={abs_trade_value:.2f}")
    
    async def _process_trade_record(self, trade_value: float, timestamp: datetime):
        """Process trade record through enterprise system."""
        event = RiskEvent(
            event_type=EventType.TRADE_REQUEST,
            priority=EventPriority.MEDIUM,
            source="RiskAgent",
            data={
                'trade_value': trade_value,
                'daily_traded_value': self.daily_traded_value,
                'capital_base': self.start_of_day_portfolio_value,
                'timestamp': timestamp.isoformat() if timestamp else None
            }
        )
        
        await self.event_bus.publish(event)
    
    def assess_trade_risk(self, proposed_trade_value: float, current_time: datetime = None) -> Tuple[bool, str]:
        """Assess trade risk - enhanced with enterprise features."""
        if self.enterprise_enabled:
            return asyncio.run(self._assess_trade_risk_enterprise(proposed_trade_value, current_time))
        else:
            return self._assess_trade_risk_legacy(proposed_trade_value, current_time)
    
    def _assess_trade_risk_legacy(self, proposed_trade_value: float, current_time: datetime = None) -> Tuple[bool, str]:
        """Legacy risk assessment logic."""
        # Original implementation logic here
        # ... (existing assess_trade_risk logic)
        pass
    
    async def _assess_trade_risk_enterprise(self, proposed_trade_value: float, current_time: datetime = None) -> Tuple[bool, str]:
        """Enterprise risk assessment using the new system."""
        # Create assessment event
        event = RiskEvent(
            event_type=EventType.TRADE_REQUEST,
            priority=EventPriority.CRITICAL,
            source="RiskAgent",
            data={
                'proposed_trade_value': proposed_trade_value,
                'current_portfolio_value': self.current_portfolio_value,
                'start_of_day_value': self.start_of_day_portfolio_value,
                'daily_traded_value': self.daily_traded_value,
                'capital_base': self.start_of_day_portfolio_value,
                'assessment_mode': True
            }
        )
        
        # Process through enterprise system
        await self.event_bus.publish(event)
        
        # Wait for result (simplified for demo)
        await asyncio.sleep(0.001)  # 1ms timeout
        
        # Return assessment result
        return True, "Enterprise assessment completed"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from enterprise system."""
        if not self.enterprise_enabled:
            return {"enterprise_enabled": False}
        
        metrics = {
            "enterprise_enabled": True,
            "event_bus_metrics": self.event_bus.get_metrics(),
        }
        
        if hasattr(self, 'drawdown_calc'):
            metrics["drawdown_calculator"] = self.drawdown_calc.get_performance_stats()
        
        if hasattr(self, 'turnover_calc'):
            metrics["turnover_calculator"] = self.turnover_calc.get_performance_stats()
        
        if hasattr(self, 'rules_engine'):
            metrics["rules_engine"] = self.rules_engine.get_performance_stats()
        
        return metrics
    
    def reset_daily_limits(self, current_portfolio_value: float = None, timestamp: datetime = None):
        """Reset daily limits - enhanced with enterprise features."""
        # Legacy reset logic
        sod_portfolio_val = current_portfolio_value if current_portfolio_value is not None else self.current_portfolio_value
        if sod_portfolio_val is None:
            self.logger.warning("Cannot reset daily limits: current portfolio value unknown")
            return
        
        self.start_of_day_portfolio_value = sod_portfolio_val
        self.daily_traded_value = 0.0
        self.hourly_traded_value = 0.0
        self.last_event_timestamp = timestamp or datetime.now()
        
        # Enterprise system reset
        if self.enterprise_enabled:
            asyncio.create_task(self._reset_enterprise_system())
        
        self.logger.info(f"Daily risk limits reset. SOD portfolio value: {self.start_of_day_portfolio_value:.2f}")
    
    async def _reset_enterprise_system(self):
        """Reset enterprise system for new day."""
        # Reset performance stats
        if hasattr(self, 'drawdown_calc'):
            self.drawdown_calc.reset_performance_stats()
        
        if hasattr(self, 'turnover_calc'):
            self.turnover_calc.reset_performance_stats()
        
        # Publish reset event
        event = RiskEvent(
            event_type=EventType.CONFIG_UPDATE,
            priority=EventPriority.LOW,
            source="RiskAgent",
            data={
                'action': 'daily_reset',
                'start_of_day_value': self.start_of_day_portfolio_value,
                'timestamp': self.last_event_timestamp.isoformat()
            }
        )
        
        await self.event_bus.publish(event)
    
    async def shutdown(self):
        """Graceful shutdown of enterprise system."""
        if self.enterprise_enabled and hasattr(self, 'event_bus'):
            await self.event_bus.stop()
            self.logger.info("Enterprise risk management system shutdown complete")
```

### Phase 4: Testing and Validation

#### Backward Compatibility Tests
```python
# tests/test_risk_agent_compatibility.py
def test_legacy_api_compatibility():
    """Test that legacy API still works."""
    config = {
        'max_daily_drawdown_pct': 0.02,
        'max_hourly_turnover_ratio': 5.0,
        'enterprise_features': {'enabled': False}
    }
    
    agent = RiskAgentV2(config)
    
    # Test legacy methods
    agent.update_portfolio_value(100000.0)
    agent.record_trade(10000.0)
    safe, reason = agent.assess_trade_risk(5000.0)
    
    assert isinstance(safe, bool)
    assert isinstance(reason, str)

def test_enterprise_features():
    """Test enterprise features work correctly."""
    config = {
        'enterprise_features': {
            'enabled': True,
            'calculators': {
                'drawdown': {'enabled': True},
                'turnover': {'enabled': True}
            }
        }
    }
    
    agent = RiskAgentV2(config)
    metrics = agent.get_performance_metrics()
    
    assert metrics['enterprise_enabled'] is True
    assert 'event_bus_metrics' in metrics
```

#### Performance Tests
```python
def test_latency_requirements():
    """Test that latency requirements are met."""
    agent = RiskAgentV2(enterprise_config)
    
    # Benchmark critical operations
    start_time = time.time_ns()
    agent.assess_trade_risk(10000.0)
    end_time = time.time_ns()
    
    latency_us = (end_time - start_time) / 1000.0
    assert latency_us < 100.0  # Sub-100µs requirement
```

### Phase 5: Deployment Strategy

#### Step 1: Feature Flag Deployment
- Deploy with `enterprise_features.enabled: false` by default
- Existing systems continue using legacy mode
- No disruption to current operations

#### Step 2: Gradual Rollout
- Enable enterprise features in development environment
- A/B testing with subset of strategies
- Performance monitoring and validation

#### Step 3: Full Migration
- Enable enterprise features in production
- Monitor performance metrics
- Fallback to legacy mode if issues detected

#### Step 4: Legacy Cleanup
- Remove legacy code paths after validation period
- Simplify configuration
- Update documentation

## Benefits of Refactored System

### 1. Performance Improvements
- **Microsecond-level latency** for critical operations
- **Vectorized calculations** for better throughput
- **Event-driven architecture** for scalability

### 2. Enhanced Capabilities
- **Hot-swappable policies** for dynamic risk management
- **Comprehensive audit trail** for compliance
- **Advanced risk metrics** beyond basic drawdown/turnover

### 3. Operational Excellence
- **Circuit breakers** for graceful degradation
- **Performance monitoring** with SLO tracking
- **Configuration validation** to prevent errors

### 4. Future-Proof Architecture
- **Plugin-based calculators** for easy extension
- **Modular design** for independent scaling
- **Enterprise-grade features** for institutional requirements

## Migration Timeline

- **Week 1-2**: Implement RiskAgentV2 with backward compatibility
- **Week 3**: Deploy with feature flags (enterprise disabled)
- **Week 4**: Enable enterprise features in development
- **Week 5-6**: A/B testing and performance validation
- **Week 7**: Production rollout with monitoring
- **Week 8**: Full migration and legacy cleanup

This refactoring plan ensures a smooth transition from the monolithic RiskAgent to the enterprise-grade system while maintaining all existing functionality and adding powerful new capabilities.