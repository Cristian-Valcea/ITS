# examples/risk_agent_v2_demo.py
"""
Demonstration of RiskAgentV2 - the complete orchestrating component.
Shows how RiskAgentV2 subscribes to events, runs calculators, evaluates policies, and enforces actions.
"""

import asyncio
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from src.risk import (
    RiskEventBus, RiskEvent, EventType, EventPriority,
    RiskAgentV2, create_risk_agent_v2
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MarketDataSimulator:
    """Simulates market data and trade events."""
    
    def __init__(self, event_bus: RiskEventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Simulation state
        self.initial_capital = 1_000_000.0
        self.current_portfolio_value = self.initial_capital
        self.trade_count = 0
    
    async def simulate_trading_session(self, duration_minutes: int = 60):
        """Simulate a trading session with various scenarios."""
        self.logger.info(f"Starting {duration_minutes}-minute trading simulation")
        
        # Send initial portfolio update
        await self._send_portfolio_update()
        
        # Simulate different trading phases
        for minute in range(duration_minutes):
            await self._simulate_minute(minute)
            await asyncio.sleep(0.01)  # 10ms between minutes (accelerated)
        
        self.logger.info("Trading simulation completed")
    
    async def _simulate_minute(self, minute: int):
        """Simulate one minute of trading."""
        
        # Phase 1: Normal trading (0-20 minutes)
        if minute < 20:
            # Small portfolio changes
            change_pct = np.random.normal(0, 0.001)
            self.current_portfolio_value *= (1 + change_pct)
            
            # Moderate trading
            if minute % 5 == 0:  # Trade every 5 minutes
                trade_value = np.random.uniform(10000, 50000)
                await self._send_trade_request(trade_value)
        
        # Phase 2: Volatile period (20-40 minutes)
        elif minute < 40:
            # Larger portfolio swings
            change_pct = np.random.normal(-0.002, 0.003)  # Slight downward bias
            self.current_portfolio_value *= (1 + change_pct)
            
            # Increased trading
            if minute % 2 == 0:  # Trade every 2 minutes
                trade_value = np.random.uniform(50000, 150000)
                await self._send_trade_request(trade_value)
        
        # Phase 3: Crisis scenario (40-50 minutes)
        elif minute < 50:
            # Sharp decline
            change_pct = np.random.normal(-0.005, 0.002)
            self.current_portfolio_value *= (1 + change_pct)
            
            # Heavy trading (panic)
            trade_value = np.random.uniform(100000, 300000)
            await self._send_trade_request(trade_value)
        
        # Phase 4: Recovery (50-60 minutes)
        else:
            # Gradual recovery
            change_pct = np.random.normal(0.002, 0.001)
            self.current_portfolio_value *= (1 + change_pct)
            
            # Reduced trading
            if minute % 3 == 0:
                trade_value = np.random.uniform(20000, 80000)
                await self._send_trade_request(trade_value)
        
        # Send portfolio update
        await self._send_portfolio_update()
        
        # Log status every 10 minutes
        if minute % 10 == 0:
            drawdown = (self.initial_capital - self.current_portfolio_value) / self.initial_capital
            self.logger.info(f"Minute {minute}: Portfolio=${self.current_portfolio_value:,.0f}, "
                           f"Drawdown={drawdown:.2%}, Trades={self.trade_count}")
    
    async def _send_portfolio_update(self):
        """Send portfolio value update event."""
        event = RiskEvent(
            event_type=EventType.POSITION_UPDATE,
            priority=EventPriority.HIGH,
            source="MarketDataSimulator",
            data={
                'portfolio_value': self.current_portfolio_value,
                'start_of_day_value': self.initial_capital,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _send_trade_request(self, trade_value: float):
        """Send trade request event."""
        self.trade_count += 1
        
        event = RiskEvent(
            event_type=EventType.TRADE_REQUEST,
            priority=EventPriority.CRITICAL,  # Trade requests are critical
            source="MarketDataSimulator",
            data={
                'trade_value': trade_value,
                'portfolio_value': self.current_portfolio_value,
                'start_of_day_value': self.initial_capital,
                'trade_id': f"TRADE_{self.trade_count:04d}",
                'timestamp': datetime.now().isoformat()
            }
        )
        
        await self.event_bus.publish(event)


class RiskMonitor:
    """Monitors risk events and actions."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.kill_switches = []
        self.alerts = []
        self.monitoring_events = []
    
    async def monitor_events(self, event_bus: RiskEventBus, duration_seconds: int):
        """Monitor events for specified duration."""
        self.logger.info(f"Starting risk monitoring for {duration_seconds} seconds")
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # In a real system, we'd subscribe to events
            # For demo, we'll just sleep and check metrics
            await asyncio.sleep(1)
            
            # Get latest metrics
            metrics = event_bus.get_metrics()
            
            # Check for circuit breakers
            if any(metrics.get('circuit_breakers', {}).values()):
                self.logger.warning("Circuit breakers activated!")
            
            # Check for latency violations
            violations = metrics.get('latency_violations', {})
            if violations:
                total_violations = sum(violations.values())
                if total_violations > 0:
                    self.logger.warning(f"Latency violations: {total_violations}")
        
        self.logger.info("Risk monitoring completed")


async def demonstrate_risk_agent_v2():
    """Main demonstration of RiskAgentV2 capabilities."""
    print("=== RiskAgentV2 Complete Orchestration Demo ===")
    print("Demonstrating the full risk management pipeline\n")
    
    # Configuration for RiskAgentV2
    risk_config = {
        'calculators': {
            'drawdown': {
                'enabled': True,
                'config': {
                    'lookback_periods': [1, 5, 20],
                    'min_periods': 1
                }
            },
            'turnover': {
                'enabled': True,
                'config': {
                    'hourly_window_minutes': 60,
                    'daily_window_hours': 24,
                    'use_absolute_values': True
                }
            }
        },
        'policies': [
            {
                'policy_id': 'main_risk_policy',
                'policy_name': 'Main Risk Policy',
                'rules': [
                    {
                        'rule_id': 'daily_drawdown_limit',
                        'rule_name': 'Daily Drawdown Limit',
                        'rule_type': 'threshold',
                        'field': 'daily_drawdown',
                        'threshold': -0.03,  # -3%
                        'operator': 'lt',
                        'action': 'halt',
                        'severity': 'critical'
                    },
                    {
                        'rule_id': 'max_drawdown_warning',
                        'rule_name': 'Max Drawdown Warning',
                        'rule_type': 'threshold',
                        'field': 'max_drawdown',
                        'threshold': -0.08,  # -8%
                        'operator': 'lt',
                        'action': 'warn',
                        'severity': 'medium'
                    },
                    {
                        'rule_id': 'turnover_limit',
                        'rule_name': 'Daily Turnover Limit',
                        'rule_type': 'threshold',
                        'field': 'total_turnover_ratio',
                        'threshold': 15.0,  # 15x capital
                        'operator': 'gt',
                        'action': 'block',
                        'severity': 'high'
                    }
                ]
            }
        ],
        'active_policy': 'main_risk_policy',
        'limits': {
            'max_daily_drawdown': 0.03,
            'max_turnover_ratio': 15.0
        }
    }
    
    # Initialize event bus
    event_bus = RiskEventBus(
        max_workers=4,
        enable_latency_monitoring=True,
        latency_slo_us={
            EventPriority.CRITICAL: 50.0,
            EventPriority.HIGH: 200.0,
            EventPriority.MEDIUM: 150.0
        }
    )
    
    # Create RiskAgentV2 using factory function
    print("Creating RiskAgentV2...")
    risk_agent = create_risk_agent_v2(risk_config)
    
    # Register RiskAgentV2 with event bus
    event_bus.register_handler(risk_agent)
    
    # Create market data simulator
    simulator = MarketDataSimulator(event_bus)
    
    # Create risk monitor
    monitor = RiskMonitor()
    
    # Start event bus
    await event_bus.start()
    
    try:
        print("Starting trading simulation with RiskAgentV2 monitoring...")
        
        # Run simulation and monitoring concurrently
        simulation_task = asyncio.create_task(
            simulator.simulate_trading_session(duration_minutes=60)
        )
        
        monitoring_task = asyncio.create_task(
            monitor.monitor_events(event_bus, duration_seconds=65)
        )
        
        # Wait for both to complete
        await asyncio.gather(simulation_task, monitoring_task)
        
        # Wait for final event processing
        await asyncio.sleep(2)
        
        # Display results
        print("\n=== Final Results ===")
        
        # Event bus metrics
        bus_metrics = event_bus.get_metrics()
        print(f"Total events processed: {sum(bus_metrics['event_counts'].values())}")
        print(f"Event counts by type: {bus_metrics['event_counts']}")
        print(f"Latency violations: {bus_metrics['latency_violations']}")
        
        # RiskAgentV2 performance
        agent_stats = risk_agent.get_performance_stats()
        print(f"\n=== RiskAgentV2 Performance ===")
        print(f"Risk evaluations: {agent_stats['evaluation_count']}")
        print(f"Average evaluation time: {agent_stats['avg_evaluation_time_us']:.2f}µs")
        print(f"Action counts: {agent_stats['action_counts']}")
        print(f"Final portfolio value: ${agent_stats['last_portfolio_value']:,.0f}")
        
        # Calculate final metrics
        if agent_stats['start_of_day_value'] and agent_stats['last_portfolio_value']:
            final_drawdown = (agent_stats['start_of_day_value'] - agent_stats['last_portfolio_value']) / agent_stats['start_of_day_value']
            print(f"Final drawdown: {final_drawdown:.2%}")
        
        print(f"Total trades processed: {agent_stats['trade_history_count']}")
        
        # Latency statistics
        if 'latency_stats' in bus_metrics:
            print(f"\n=== Latency Performance ===")
            for priority, stats in bus_metrics['latency_stats'].items():
                print(f"{priority}: P50={stats['p50']:.1f}µs, P95={stats['p95']:.1f}µs, "
                      f"P99={stats['p99']:.1f}µs, Max={stats['max']:.1f}µs")
        
        # Check if risk limits were breached
        halt_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
        alert_events = bus_metrics['event_counts'].get(EventType.ALERT, 0)
        
        print(f"\n=== Risk Actions ===")
        print(f"Kill switch events: {halt_events}")
        print(f"Alert events: {alert_events}")
        
        if halt_events > 0:
            print("⚠️  RISK HALT WAS ACTIVATED during simulation")
        else:
            print("✅ No risk halts triggered")
        
    finally:
        await event_bus.stop()
    
    print("\n=== Demo Complete ===")
    print("✅ RiskAgentV2 successfully orchestrated:")
    print("  - Event subscription and processing")
    print("  - Risk calculator execution")
    print("  - Policy evaluation through rules engine")
    print("  - Risk action enforcement")
    print("  - Performance monitoring and metrics")


async def demonstrate_hot_swap():
    """Demonstrate hot-swapping of risk policies."""
    print("\n=== Hot-Swap Policy Demo ===")
    
    # Create a simple risk agent
    config = {
        'calculators': {
            'drawdown': {'enabled': True, 'config': {}}
        },
        'policies': [
            {
                'policy_id': 'lenient_policy',
                'policy_name': 'Lenient Policy',
                'rules': [
                    {
                        'rule_id': 'lenient_drawdown',
                        'rule_type': 'threshold',
                        'field': 'daily_drawdown',
                        'threshold': -0.10,  # -10% (very lenient)
                        'operator': 'lt',
                        'action': 'warn',
                        'severity': 'low'
                    }
                ]
            }
        ],
        'active_policy': 'lenient_policy'
    }
    
    risk_agent = create_risk_agent_v2(config)
    
    print("Initial policy: Lenient (10% drawdown warning)")
    
    # Simulate hot-swap to strict policy
    new_config = {
        'active_policy': 'strict_policy'
    }
    
    # Add strict policy to rules engine
    from src.risk import RiskPolicy, ThresholdRule, RuleAction, RuleSeverity
    
    strict_policy = RiskPolicy('strict_policy', 'Strict Policy')
    strict_rule = ThresholdRule(
        'strict_drawdown',
        'Strict Drawdown Limit',
        {
            'field': 'daily_drawdown',
            'threshold': -0.02,  # -2% (very strict)
            'operator': 'lt',
            'action': 'halt',
            'severity': 'critical'
        }
    )
    strict_policy.add_rule(strict_rule)
    risk_agent.rules_engine.register_policy(strict_policy)
    
    # Hot-swap configuration
    risk_agent.update_limits_config(new_config)
    
    print("Policy hot-swapped to: Strict (2% drawdown halt)")
    print("✅ Hot-swap completed without downtime")


if __name__ == "__main__":
    print("RiskAgentV2 - Complete Risk Management Orchestrator")
    print("=" * 60)
    
    try:
        # Run main demonstration
        asyncio.run(demonstrate_risk_agent_v2())
        
        # Run hot-swap demo
        asyncio.run(demonstrate_hot_swap())
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()