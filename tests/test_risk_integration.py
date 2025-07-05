# tests/test_risk_integration.py
"""
Integration tests for event-bus ↔ rules engine with latency benchmarks.
Tests the complete risk management pipeline end-to-end.
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk import (
    RiskEventBus, RiskEvent, EventType, EventPriority,
    RulesEngine, RiskPolicy, ThresholdRule, RuleAction, RuleSeverity,
    DrawdownCalculator, TurnoverCalculator,
    RiskAgentV2, create_risk_agent_v2
)


class TestEventBusRulesEngineIntegration:
    """Integration tests for event bus and rules engine."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create and start event bus."""
        bus = RiskEventBus(
            max_workers=2,
            enable_latency_monitoring=True,
            latency_slo_us={
                EventPriority.CRITICAL: 50.0,
                EventPriority.HIGH: 200.0,
                EventPriority.MEDIUM: 150.0
            }
        )
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def rules_engine(self):
        """Create rules engine with test policies."""
        engine = RulesEngine()
        
        # Create test policy
        policy = RiskPolicy("test_policy", "Test Policy")
        
        # Add drawdown rule
        drawdown_rule = ThresholdRule(
            "drawdown_limit",
            "Drawdown Limit",
            {
                'field': 'daily_drawdown',
                'threshold': -0.05,  # -5%
                'operator': 'lt',
                'action': 'halt',
                'severity': 'critical'
            }
        )
        policy.add_rule(drawdown_rule)
        
        # Add turnover rule
        turnover_rule = ThresholdRule(
            "turnover_limit",
            "Turnover Limit",
            {
                'field': 'total_turnover_ratio',
                'threshold': 10.0,  # 10x capital
                'operator': 'gt',
                'action': 'block',
                'severity': 'high'
            }
        )
        policy.add_rule(turnover_rule)
        
        engine.register_policy(policy)
        return engine
    
    @pytest.fixture
    def risk_agent_config(self):
        """Configuration for RiskAgentV2."""
        return {
            'calculators': {
                'drawdown': {
                    'enabled': True,
                    'config': {'lookback_periods': [1, 5]}
                },
                'turnover': {
                    'enabled': True,
                    'config': {'hourly_window_minutes': 60}
                }
            },
            'policies': [
                {
                    'policy_id': 'integration_test_policy',
                    'policy_name': 'Integration Test Policy',
                    'rules': [
                        {
                            'rule_id': 'test_drawdown_limit',
                            'rule_type': 'threshold',
                            'field': 'daily_drawdown',
                            'threshold': -0.03,  # -3%
                            'operator': 'lt',
                            'action': 'halt',
                            'severity': 'critical'
                        },
                        {
                            'rule_id': 'test_turnover_limit',
                            'rule_type': 'threshold',
                            'field': 'total_turnover_ratio',
                            'threshold': 5.0,  # 5x capital
                            'operator': 'gt',
                            'action': 'block',
                            'severity': 'high'
                        }
                    ]
                }
            ],
            'active_policy': 'integration_test_policy'
        }
    
    async def test_end_to_end_risk_pipeline(self, event_bus, risk_agent_config):
        """Test complete end-to-end risk pipeline with latency measurement."""
        # Create RiskAgentV2
        risk_agent = create_risk_agent_v2(risk_agent_config)
        event_bus.register_handler(risk_agent)
        
        # Test data
        initial_capital = 1000000.0
        current_portfolio = 970000.0  # -3% drawdown (should trigger halt)
        
        # Create test event
        test_event = RiskEvent(
            event_type=EventType.TRADE_REQUEST,
            priority=EventPriority.CRITICAL,
            source="IntegrationTest",
            data={
                'trade_value': 50000.0,
                'portfolio_value': current_portfolio,
                'start_of_day_value': initial_capital,
                'capital_base': initial_capital,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Measure end-to-end latency
        start_time = time.time_ns()
        
        # Publish event
        await event_bus.publish(test_event)
        
        # Wait for processing
        await asyncio.sleep(0.01)  # 10ms should be enough
        
        end_time = time.time_ns()
        total_latency_us = (end_time - start_time) / 1000.0
        
        print(f"\nEnd-to-end pipeline latency: {total_latency_us:.2f}µs")
        
        # Get metrics
        bus_metrics = event_bus.get_metrics()
        agent_stats = risk_agent.get_performance_stats()
        
        # Verify processing
        assert bus_metrics['events_processed'] > 0, "No events were processed"
        assert agent_stats['evaluation_count'] > 0, "No risk evaluations performed"
        
        # Check for kill switch events (drawdown should trigger halt)
        kill_switch_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
        assert kill_switch_events > 0, "Expected kill switch activation for -3% drawdown"
        
        # Verify latency targets
        assert total_latency_us < 10000.0, f"End-to-end latency {total_latency_us:.2f}µs exceeds 10ms"
        
        print(f"✅ End-to-end test passed:")
        print(f"   Events processed: {bus_metrics['events_processed']}")
        print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
        print(f"   Kill switches: {kill_switch_events}")
        print(f"   Average evaluation time: {agent_stats['avg_evaluation_time_us']:.2f}µs")
    
    async def test_high_frequency_event_processing(self, event_bus, risk_agent_config):
        """Test high-frequency event processing with latency benchmarks."""
        risk_agent = create_risk_agent_v2(risk_agent_config)
        event_bus.register_handler(risk_agent)
        
        # Generate high-frequency events
        num_events = 100
        events = []
        
        for i in range(num_events):
            event = RiskEvent(
                event_type=EventType.POSITION_UPDATE,
                priority=EventPriority.HIGH,
                source="HighFrequencyTest",
                data={
                    'portfolio_value': 1000000 + np.random.normal(0, 5000),
                    'start_of_day_value': 1000000,
                    'timestamp': datetime.now().isoformat(),
                    'sequence': i
                }
            )
            events.append(event)
        
        # Measure batch processing latency
        start_time = time.time_ns()
        
        # Publish all events
        for event in events:
            await event_bus.publish(event)
        
        # Wait for all processing to complete
        await asyncio.sleep(0.1)  # 100ms should be enough for 100 events
        
        end_time = time.time_ns()
        total_time_us = (end_time - start_time) / 1000.0
        avg_latency_per_event = total_time_us / num_events
        
        print(f"\nHigh-frequency processing:")
        print(f"  Total time: {total_time_us:.2f}µs")
        print(f"  Average per event: {avg_latency_per_event:.2f}µs")
        print(f"  Throughput: {num_events / (total_time_us / 1_000_000):.0f} events/sec")
        
        # Get final metrics
        bus_metrics = event_bus.get_metrics()
        agent_stats = risk_agent.get_performance_stats()
        
        # Verify all events were processed
        assert bus_metrics['events_processed'] >= num_events, \
            f"Not all events processed: {bus_metrics['events_processed']} < {num_events}"
        
        # Verify performance targets
        assert avg_latency_per_event < 1000.0, \
            f"Average latency per event {avg_latency_per_event:.2f}µs exceeds 1ms"
        
        print(f"✅ High-frequency test passed:")
        print(f"   Events processed: {bus_metrics['events_processed']}")
        print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
    
    async def test_policy_evaluation_latency(self, rules_engine):
        """Test policy evaluation latency with various data scenarios."""
        # Test data scenarios
        test_scenarios = [
            {
                'name': 'normal_operation',
                'data': {
                    'daily_drawdown': -0.01,  # -1% (normal)
                    'total_turnover_ratio': 2.0  # 2x (normal)
                },
                'expected_action': RuleAction.ALLOW
            },
            {
                'name': 'drawdown_breach',
                'data': {
                    'daily_drawdown': -0.06,  # -6% (breach)
                    'total_turnover_ratio': 2.0
                },
                'expected_action': RuleAction.HALT
            },
            {
                'name': 'turnover_breach',
                'data': {
                    'daily_drawdown': -0.01,
                    'total_turnover_ratio': 12.0  # 12x (breach)
                },
                'expected_action': RuleAction.BLOCK
            },
            {
                'name': 'multiple_breaches',
                'data': {
                    'daily_drawdown': -0.06,  # Breach
                    'total_turnover_ratio': 12.0  # Breach
                },
                'expected_action': RuleAction.HALT  # Most severe action
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Warm up
            for _ in range(10):
                rules_engine.evaluate_policy("test_policy", scenario['data'])
            
            # Benchmark
            latencies = []
            for _ in range(100):
                start_time = time.time_ns()
                result = rules_engine.evaluate_policy("test_policy", scenario['data'])
                end_time = time.time_ns()
                
                latencies.append((end_time - start_time) / 1000.0)
                
                # Verify expected action
                if result:
                    assert result.overall_action == scenario['expected_action'], \
                        f"Expected {scenario['expected_action']}, got {result.overall_action}"
            
            # Statistics
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            
            print(f"  Policy evaluation latency:")
            print(f"    P50: {p50:.2f}µs")
            print(f"    P95: {p95:.2f}µs")
            print(f"    P99: {p99:.2f}µs")
            
            # Assert performance targets
            assert p50 < 100.0, f"P50 latency {p50:.2f}µs exceeds 100µs target"
            assert p95 < 200.0, f"P95 latency {p95:.2f}µs exceeds 200µs target"
    
    async def test_event_bus_latency_monitoring(self, event_bus):
        """Test event bus latency monitoring and SLO tracking."""
        # Create events with different priorities
        events = [
            RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="LatencyTest",
                data={'test': 'critical'}
            ),
            RiskEvent(
                event_type=EventType.POSITION_UPDATE,
                priority=EventPriority.HIGH,
                source="LatencyTest",
                data={'test': 'high'}
            ),
            RiskEvent(
                event_type=EventType.RISK_MONITORING,
                priority=EventPriority.MEDIUM,
                source="LatencyTest",
                data={'test': 'medium'}
            )
        ]
        
        # Publish events
        for event in events:
            await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.01)
        
        # Get latency metrics
        metrics = event_bus.get_metrics()
        
        print(f"\nEvent bus latency metrics:")
        print(f"  Events processed: {metrics['events_processed']}")
        print(f"  Latency violations: {metrics['latency_violations']}")
        
        if 'latency_stats' in metrics:
            for priority, stats in metrics['latency_stats'].items():
                print(f"  {priority}:")
                print(f"    P50: {stats['p50']:.2f}µs")
                print(f"    P95: {stats['p95']:.2f}µs")
                print(f"    P99: {stats['p99']:.2f}µs")
        
        # Verify monitoring is working
        assert metrics['events_processed'] >= len(events), "Not all events processed"
        assert 'latency_violations' in metrics, "Latency monitoring not working"
    
    async def test_circuit_breaker_functionality(self, event_bus):
        """Test circuit breaker functionality under load."""
        # Create a handler that will fail
        class FailingHandler:
            def __init__(self):
                self.call_count = 0
            
            @property
            def supported_event_types(self):
                return [EventType.TRADE_REQUEST]
            
            @property
            def priority_filter(self):
                return [EventPriority.CRITICAL]
            
            async def handle(self, event):
                self.call_count += 1
                if self.call_count <= 5:
                    raise Exception("Simulated failure")
                return None
        
        failing_handler = FailingHandler()
        event_bus.register_handler(failing_handler)
        
        # Send events that will cause failures
        for i in range(10):
            event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="CircuitBreakerTest",
                data={'sequence': i}
            )
            await event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.05)
        
        # Get metrics
        metrics = event_bus.get_metrics()
        
        print(f"\nCircuit breaker test:")
        print(f"  Events processed: {metrics['events_processed']}")
        print(f"  Circuit breakers: {metrics.get('circuit_breakers', {})}")
        
        # Verify circuit breaker activated
        circuit_breakers = metrics.get('circuit_breakers', {})
        assert any(circuit_breakers.values()), "Circuit breaker should have activated"
        
        print("✅ Circuit breaker test passed")


class TestRiskSystemGoldenFiles:
    """Golden file tests for complete risk system scenarios."""
    
    @pytest.fixture
    def golden_scenarios(self):
        """Load golden test scenarios."""
        return {
            'market_crash_scenario': {
                'description': 'Simulated market crash with 10% drawdown',
                'initial_capital': 1000000,
                'portfolio_values': [1000000, 950000, 900000, 850000, 800000],
                'trade_values': [100000, 150000, 200000, 250000],
                'expected_outcomes': {
                    'halt_triggered': True,
                    'max_drawdown': -0.20,
                    'total_turnover_ratio': 0.70,
                    'kill_switch_events': 1
                }
            },
            'high_frequency_trading': {
                'description': 'High frequency trading with turnover limits',
                'initial_capital': 1000000,
                'portfolio_values': [1000000] * 100,  # Stable portfolio
                'trade_values': [50000] * 200,  # Many small trades
                'expected_outcomes': {
                    'halt_triggered': True,  # Turnover limit breach
                    'max_drawdown': 0.0,
                    'total_turnover_ratio': 10.0,
                    'block_events': 1
                }
            },
            'normal_trading_day': {
                'description': 'Normal trading day within all limits',
                'initial_capital': 1000000,
                'portfolio_values': [1000000, 1005000, 998000, 1002000, 1001000],
                'trade_values': [25000, 30000, 20000, 15000],
                'expected_outcomes': {
                    'halt_triggered': False,
                    'max_drawdown': -0.002,
                    'total_turnover_ratio': 0.09,
                    'kill_switch_events': 0
                }
            }
        }
    
    async def test_golden_scenario_market_crash(self, golden_scenarios):
        """Test market crash scenario against golden expectations."""
        scenario = golden_scenarios['market_crash_scenario']
        
        # Setup
        config = {
            'calculators': {
                'drawdown': {'enabled': True, 'config': {}},
                'turnover': {'enabled': True, 'config': {}}
            },
            'policies': [{
                'policy_id': 'crash_test_policy',
                'policy_name': 'Crash Test Policy',
                'rules': [{
                    'rule_id': 'drawdown_limit',
                    'rule_type': 'threshold',
                    'field': 'daily_drawdown',
                    'threshold': -0.05,  # -5% limit
                    'operator': 'lt',
                    'action': 'halt',
                    'severity': 'critical'
                }]
            }],
            'active_policy': 'crash_test_policy'
        }
        
        # Create event bus and risk agent
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(config)
            event_bus.register_handler(risk_agent)
            
            # Simulate market crash
            for i, portfolio_value in enumerate(scenario['portfolio_values']):
                event = RiskEvent(
                    event_type=EventType.POSITION_UPDATE,
                    priority=EventPriority.HIGH,
                    source="GoldenTest",
                    data={
                        'portfolio_value': portfolio_value,
                        'start_of_day_value': scenario['initial_capital'],
                        'timestamp': datetime.now().isoformat(),
                        'sequence': i
                    }
                )
                await event_bus.publish(event)
                
                # Add trades
                if i < len(scenario['trade_values']):
                    trade_event = RiskEvent(
                        event_type=EventType.TRADE_REQUEST,
                        priority=EventPriority.CRITICAL,
                        source="GoldenTest",
                        data={
                            'trade_value': scenario['trade_values'][i],
                            'portfolio_value': portfolio_value,
                            'start_of_day_value': scenario['initial_capital'],
                            'capital_base': scenario['initial_capital']
                        }
                    )
                    await event_bus.publish(trade_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Verify outcomes
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            # Check kill switch activation
            kill_switch_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
            expected_halt = scenario['expected_outcomes']['halt_triggered']
            
            if expected_halt:
                assert kill_switch_events > 0, "Expected halt was not triggered"
            else:
                assert kill_switch_events == 0, "Unexpected halt was triggered"
            
            print(f"✅ Market crash scenario passed:")
            print(f"   Kill switch events: {kill_switch_events}")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            
        finally:
            await event_bus.stop()
    
    async def test_golden_scenario_normal_trading(self, golden_scenarios):
        """Test normal trading day scenario."""
        scenario = golden_scenarios['normal_trading_day']
        
        # Setup with more lenient limits
        config = {
            'calculators': {
                'drawdown': {'enabled': True, 'config': {}},
                'turnover': {'enabled': True, 'config': {}}
            },
            'policies': [{
                'policy_id': 'normal_policy',
                'policy_name': 'Normal Trading Policy',
                'rules': [
                    {
                        'rule_id': 'drawdown_limit',
                        'rule_type': 'threshold',
                        'field': 'daily_drawdown',
                        'threshold': -0.10,  # -10% limit (lenient)
                        'operator': 'lt',
                        'action': 'halt',
                        'severity': 'critical'
                    },
                    {
                        'rule_id': 'turnover_limit',
                        'rule_type': 'threshold',
                        'field': 'total_turnover_ratio',
                        'threshold': 5.0,  # 5x limit
                        'operator': 'gt',
                        'action': 'block',
                        'severity': 'high'
                    }
                ]
            }],
            'active_policy': 'normal_policy'
        }
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(config)
            event_bus.register_handler(risk_agent)
            
            # Simulate normal trading
            for i, portfolio_value in enumerate(scenario['portfolio_values']):
                event = RiskEvent(
                    event_type=EventType.POSITION_UPDATE,
                    priority=EventPriority.HIGH,
                    source="GoldenTest",
                    data={
                        'portfolio_value': portfolio_value,
                        'start_of_day_value': scenario['initial_capital']
                    }
                )
                await event_bus.publish(event)
                
                if i < len(scenario['trade_values']):
                    trade_event = RiskEvent(
                        event_type=EventType.TRADE_REQUEST,
                        priority=EventPriority.CRITICAL,
                        source="GoldenTest",
                        data={
                            'trade_value': scenario['trade_values'][i],
                            'portfolio_value': portfolio_value,
                            'start_of_day_value': scenario['initial_capital'],
                            'capital_base': scenario['initial_capital']
                        }
                    )
                    await event_bus.publish(trade_event)
            
            await asyncio.sleep(0.1)
            
            # Verify no halts occurred
            bus_metrics = event_bus.get_metrics()
            kill_switch_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
            
            expected_kill_switches = scenario['expected_outcomes']['kill_switch_events']
            assert kill_switch_events == expected_kill_switches, \
                f"Expected {expected_kill_switches} kill switches, got {kill_switch_events}"
            
            print(f"✅ Normal trading scenario passed:")
            print(f"   No halts triggered as expected")
            print(f"   Events processed: {bus_metrics['events_processed']}")
            
        finally:
            await event_bus.stop()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])