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
    FeedStalenessCalculator, LiquidityCalculator, ConcentrationCalculator,
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


class TestSensorBasedRiskScenarios:
    """Test specific sensor-based risk scenarios with expected actions."""
    
    @pytest.fixture
    def comprehensive_risk_config(self):
        """Configuration with all sensor calculators enabled."""
        return {
            'calculators': {
                'drawdown': {
                    'enabled': True,
                    'config': {'lookback_periods': [1, 5]}
                },
                'turnover': {
                    'enabled': True,
                    'config': {'hourly_window_minutes': 60}
                },
                'feed_staleness': {
                    'enabled': True,
                    'config': {'max_staleness_seconds': 1.0}
                },
                'liquidity': {
                    'enabled': True,
                    'config': {'min_depth_ratio': 0.1}
                },
                'concentration': {
                    'enabled': True,
                    'config': {'max_position_ratio': 0.25}
                }
            },
            'policies': [{
                'policy_id': 'comprehensive_sensor_policy',
                'policy_name': 'Comprehensive Sensor Policy',
                'rules': [
                    {
                        'rule_id': 'stale_feed_kill_switch',
                        'rule_type': 'threshold',
                        'field': 'feed_staleness_seconds',
                        'threshold': 1.0,  # 1 second staleness limit
                        'operator': 'gt',
                        'action': 'liquidate',  # KILL_SWITCH equivalent
                        'severity': 'critical'
                    },
                    {
                        'rule_id': 'liquidity_throttle',
                        'rule_type': 'threshold',
                        'field': 'order_book_depth_ratio',
                        'threshold': 0.1,  # Deep orderbook sweep threshold
                        'operator': 'lt',
                        'action': 'reduce_position',  # THROTTLE equivalent
                        'severity': 'medium'
                    },
                    {
                        'rule_id': 'concentration_block',
                        'rule_type': 'threshold',
                        'field': 'position_concentration_ratio',
                        'threshold': 0.25,  # 25% of portfolio (4x ADV equivalent)
                        'operator': 'gt',
                        'action': 'block',
                        'severity': 'high'
                    },
                    {
                        'rule_id': 'drawdown_halt',
                        'rule_type': 'threshold',
                        'field': 'daily_drawdown',
                        'threshold': -0.05,  # -5%
                        'operator': 'lt',
                        'action': 'halt',
                        'severity': 'critical'
                    }
                ]
            }],
            'active_policy': 'comprehensive_sensor_policy'
        }
    
    async def test_stale_tick_timestamp_kill_switch(self, comprehensive_risk_config):
        """Test that stale tick timestamp triggers KILL_SWITCH (LIQUIDATE) action."""
        print("\n🧪 Testing Stale Tick Timestamp → KILL_SWITCH")
        print("=" * 60)
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(comprehensive_risk_config)
            event_bus.register_handler(risk_agent)
            
            # Create trade request with stale feed timestamp
            current_time = datetime.now()
            stale_timestamp = current_time - timedelta(seconds=2.5)  # 2.5 seconds stale
            
            trade_event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="StaleFeedTest",
                data={
                    'symbol': 'AAPL',
                    'trade_value': 100000.0,
                    'portfolio_value': 1000000.0,
                    'start_of_day_value': 1000000.0,
                    'capital_base': 1000000.0,
                    'timestamp': current_time.isoformat(),
                    
                    # Feed staleness data - CRITICAL: feeds are stale
                    'feed_timestamps': {
                        'market_data': stale_timestamp.timestamp(),  # 2.5s stale
                        'order_book': stale_timestamp.timestamp(),   # 2.5s stale
                        'trades': stale_timestamp.timestamp()        # 2.5s stale
                    },
                    
                    # Other sensor data (normal)
                    'order_book_depth': {
                        'AAPL': {
                            'bids': [(150.00, 10000), (149.95, 5000)],
                            'asks': [(150.05, 8000), (150.10, 6000)]
                        }
                    },
                    'positions': {'AAPL': 1000},  # Normal position
                    'daily_volumes': {'AAPL': [50000000] * 20}
                }
            )
            
            # Publish event
            await event_bus.publish(trade_event)
            await asyncio.sleep(0.05)  # Wait for processing
            
            # Verify KILL_SWITCH (LIQUIDATE) was triggered
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            # Check for liquidation events
            liquidate_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
            
            print(f"📊 Results:")
            print(f"   Feed staleness: 2.5 seconds (threshold: 1.0s)")
            print(f"   Kill switch events: {liquidate_events}")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            
            # Verify kill switch was triggered
            assert liquidate_events > 0, f"Expected KILL_SWITCH for stale feeds, got {liquidate_events} events"
            
            print("✅ PASS: Stale tick timestamp correctly triggered KILL_SWITCH")
            
        finally:
            await event_bus.stop()
    
    async def test_deep_orderbook_sweep_throttle(self, comprehensive_risk_config):
        """Test that deep orderbook sweep triggers THROTTLE (REDUCE_POSITION) action."""
        print("\n🧪 Testing Deep Orderbook Sweep → THROTTLE")
        print("=" * 60)
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(comprehensive_risk_config)
            event_bus.register_handler(risk_agent)
            
            current_time = datetime.now()
            
            trade_event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="LiquidityTest",
                data={
                    'symbol': 'AAPL',
                    'trade_value': 500000.0,  # Large trade
                    'portfolio_value': 1000000.0,
                    'start_of_day_value': 1000000.0,
                    'capital_base': 1000000.0,
                    'timestamp': current_time.isoformat(),
                    
                    # Fresh feed timestamps (good)
                    'feed_timestamps': {
                        'market_data': current_time.timestamp() - 0.1,  # 100ms fresh
                        'order_book': current_time.timestamp() - 0.05,  # 50ms fresh
                        'trades': current_time.timestamp() - 0.2        # 200ms fresh
                    },
                    
                    # CRITICAL: Thin order book - deep sweep required
                    'order_book_depth': {
                        'AAPL': {
                            'bids': [(150.00, 100), (149.95, 50)],      # Very thin bids
                            'asks': [(150.05, 80), (150.10, 60)]       # Very thin asks
                        }
                    },
                    
                    # Calculate depth ratio: trade_value / available_liquidity
                    # Trade: $500k, Available liquidity: ~$22.5k → ratio = 0.045 < 0.1 threshold
                    'order_book_depth_ratio': 0.045,  # Below 0.1 threshold
                    
                    'positions': {'AAPL': 1000},  # Normal position
                    'daily_volumes': {'AAPL': [50000000] * 20}
                }
            )
            
            # Publish event
            await event_bus.publish(trade_event)
            await asyncio.sleep(0.05)
            
            # Verify THROTTLE (REDUCE_POSITION) was triggered
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            # Check for throttle/reduce position events
            throttle_events = bus_metrics['event_counts'].get(EventType.RISK_ALERT, 0)
            
            print(f"📊 Results:")
            print(f"   Order book depth ratio: 0.045 (threshold: 0.1)")
            print(f"   Trade size: $500,000")
            print(f"   Available liquidity: ~$22,500")
            print(f"   Throttle events: {throttle_events}")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            
            # Verify throttle was triggered
            assert throttle_events > 0, f"Expected THROTTLE for thin order book, got {throttle_events} events"
            
            print("✅ PASS: Deep orderbook sweep correctly triggered THROTTLE")
            
        finally:
            await event_bus.stop()
    
    async def test_4x_adv_position_block(self, comprehensive_risk_config):
        """Test that 4x ADV position triggers BLOCK action."""
        print("\n🧪 Testing 4x ADV Position → BLOCK")
        print("=" * 60)
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(comprehensive_risk_config)
            event_bus.register_handler(risk_agent)
            
            current_time = datetime.now()
            
            # Calculate 4x ADV scenario
            # ADV (Average Daily Volume) = 50M shares
            # 4x ADV = 200M shares
            # At $150/share = $30B position (unrealistic but for testing)
            # Let's use more realistic numbers: 25% of portfolio = 4x "normal" position
            
            trade_event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="ConcentrationTest",
                data={
                    'symbol': 'AAPL',
                    'trade_value': 300000.0,  # This trade would create concentration
                    'portfolio_value': 1000000.0,
                    'start_of_day_value': 1000000.0,
                    'capital_base': 1000000.0,
                    'timestamp': current_time.isoformat(),
                    
                    # Fresh feed timestamps (good)
                    'feed_timestamps': {
                        'market_data': current_time.timestamp() - 0.1,
                        'order_book': current_time.timestamp() - 0.05,
                        'trades': current_time.timestamp() - 0.2
                    },
                    
                    # Good order book depth
                    'order_book_depth': {
                        'AAPL': {
                            'bids': [(150.00, 10000), (149.95, 8000)],
                            'asks': [(150.05, 12000), (150.10, 9000)]
                        }
                    },
                    'order_book_depth_ratio': 0.5,  # Good liquidity
                    
                    # CRITICAL: High position concentration
                    # Current position: $200k, New trade: $300k, Total: $500k
                    # Concentration: $500k / $1M portfolio = 50% > 25% threshold
                    'positions': {'AAPL': 1333},  # Current position ~$200k at $150/share
                    'position_concentration_ratio': 0.50,  # 50% > 25% threshold
                    
                    'daily_volumes': {'AAPL': [50000000] * 20}  # Normal volume
                }
            )
            
            # Publish event
            await event_bus.publish(trade_event)
            await asyncio.sleep(0.05)
            
            # Verify BLOCK was triggered
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            # Check for block events
            block_events = bus_metrics['event_counts'].get(EventType.RISK_ALERT, 0)
            
            print(f"📊 Results:")
            print(f"   Current position: ~$200,000 (1,333 shares)")
            print(f"   New trade: $300,000")
            print(f"   Total position: ~$500,000")
            print(f"   Portfolio concentration: 50% (threshold: 25%)")
            print(f"   Block events: {block_events}")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            
            # Verify block was triggered
            assert block_events > 0, f"Expected BLOCK for high concentration, got {block_events} events"
            
            print("✅ PASS: 4x ADV position correctly triggered BLOCK")
            
        finally:
            await event_bus.stop()
    
    async def test_combined_sensor_scenarios(self, comprehensive_risk_config):
        """Test multiple sensor conditions simultaneously."""
        print("\n🧪 Testing Combined Sensor Scenarios")
        print("=" * 60)
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(comprehensive_risk_config)
            event_bus.register_handler(risk_agent)
            
            current_time = datetime.now()
            stale_timestamp = current_time - timedelta(seconds=3.0)  # Very stale
            
            # Scenario: Multiple risk factors triggered simultaneously
            trade_event = RiskEvent(
                event_type=EventType.TRADE_REQUEST,
                priority=EventPriority.CRITICAL,
                source="CombinedRiskTest",
                data={
                    'symbol': 'AAPL',
                    'trade_value': 400000.0,
                    'portfolio_value': 900000.0,  # Portfolio down 10%
                    'start_of_day_value': 1000000.0,
                    'capital_base': 1000000.0,
                    'timestamp': current_time.isoformat(),
                    
                    # RISK 1: Stale feeds (should trigger KILL_SWITCH)
                    'feed_timestamps': {
                        'market_data': stale_timestamp.timestamp(),  # 3s stale
                        'order_book': stale_timestamp.timestamp(),
                        'trades': stale_timestamp.timestamp()
                    },
                    
                    # RISK 2: Thin order book (should trigger THROTTLE)
                    'order_book_depth': {
                        'AAPL': {
                            'bids': [(150.00, 50), (149.95, 30)],
                            'asks': [(150.05, 40), (150.10, 25)]
                        }
                    },
                    'order_book_depth_ratio': 0.03,  # Very thin
                    
                    # RISK 3: High concentration (should trigger BLOCK)
                    'positions': {'AAPL': 2000},  # Large existing position
                    'position_concentration_ratio': 0.60,  # 60% concentration
                    
                    'daily_volumes': {'AAPL': [50000000] * 20}
                }
            )
            
            # Publish event
            await event_bus.publish(trade_event)
            await asyncio.sleep(0.05)
            
            # Verify most severe action was taken (KILL_SWITCH should win)
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            kill_switch_events = bus_metrics['event_counts'].get(EventType.KILL_SWITCH, 0)
            risk_alerts = bus_metrics['event_counts'].get(EventType.RISK_ALERT, 0)
            
            print(f"📊 Results:")
            print(f"   Feed staleness: 3.0 seconds (KILL_SWITCH trigger)")
            print(f"   Order book depth: 0.03 (THROTTLE trigger)")
            print(f"   Position concentration: 60% (BLOCK trigger)")
            print(f"   Kill switch events: {kill_switch_events}")
            print(f"   Risk alert events: {risk_alerts}")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            
            # Most severe action (KILL_SWITCH) should be taken
            assert kill_switch_events > 0, "Expected KILL_SWITCH as most severe action"
            
            print("✅ PASS: Combined sensor scenarios handled correctly")
            print("   Most severe action (KILL_SWITCH) was triggered")
            
        finally:
            await event_bus.stop()
    
    async def test_sensor_performance_benchmarks(self, comprehensive_risk_config):
        """Benchmark sensor-based risk evaluation performance."""
        print("\n🧪 Testing Sensor Performance Benchmarks")
        print("=" * 60)
        
        event_bus = RiskEventBus(max_workers=2)
        await event_bus.start()
        
        try:
            risk_agent = create_risk_agent_v2(comprehensive_risk_config)
            event_bus.register_handler(risk_agent)
            
            # Generate test events with full sensor data
            num_events = 50
            latencies = []
            
            for i in range(num_events):
                current_time = datetime.now()
                
                event = RiskEvent(
                    event_type=EventType.TRADE_REQUEST,
                    priority=EventPriority.CRITICAL,
                    source="PerformanceTest",
                    data={
                        'symbol': 'AAPL',
                        'trade_value': 100000.0 + (i * 1000),
                        'portfolio_value': 1000000.0,
                        'start_of_day_value': 1000000.0,
                        'capital_base': 1000000.0,
                        'timestamp': current_time.isoformat(),
                        'sequence': i,
                        
                        # Full sensor data
                        'feed_timestamps': {
                            'market_data': current_time.timestamp() - 0.1,
                            'order_book': current_time.timestamp() - 0.05,
                            'trades': current_time.timestamp() - 0.2
                        },
                        'order_book_depth': {
                            'AAPL': {
                                'bids': [(150.00, 10000), (149.95, 8000)],
                                'asks': [(150.05, 12000), (150.10, 9000)]
                            }
                        },
                        'order_book_depth_ratio': 0.5,
                        'positions': {'AAPL': 1000 + i},
                        'position_concentration_ratio': 0.15 + (i * 0.001),
                        'daily_volumes': {'AAPL': [50000000] * 20}
                    }
                )
                
                # Measure individual event latency
                start_time = time.time_ns()
                await event_bus.publish(event)
                await asyncio.sleep(0.001)  # Small delay for processing
                end_time = time.time_ns()
                
                latencies.append((end_time - start_time) / 1000.0)  # Convert to microseconds
            
            # Wait for all processing to complete
            await asyncio.sleep(0.1)
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # Get final metrics
            bus_metrics = event_bus.get_metrics()
            agent_stats = risk_agent.get_performance_stats()
            
            print(f"📊 Sensor Performance Results:")
            print(f"   Events processed: {num_events}")
            print(f"   Average latency: {avg_latency:.2f}µs")
            print(f"   P50 latency: {p50_latency:.2f}µs")
            print(f"   P95 latency: {p95_latency:.2f}µs")
            print(f"   P99 latency: {p99_latency:.2f}µs")
            print(f"   Risk evaluations: {agent_stats['evaluation_count']}")
            print(f"   Avg evaluation time: {agent_stats.get('avg_evaluation_time_us', 0):.2f}µs")
            
            # Performance assertions
            assert avg_latency < 5000.0, f"Average latency {avg_latency:.2f}µs exceeds 5ms target"
            assert p95_latency < 10000.0, f"P95 latency {p95_latency:.2f}µs exceeds 10ms target"
            
            print("✅ PASS: Sensor performance within acceptable limits")
            
        finally:
            await event_bus.stop()


class TestSensorBasedRiskScenarios:
    """Test specific sensor-based risk scenarios with expected actions."""
    
    @pytest.fixture
    def sensor_risk_config(self):
        """Configuration with sensor-based rules for testing specific scenarios."""
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
            'policies': [{
                'policy_id': 'sensor_test_policy',
                'policy_name': 'Sensor Test Policy',
                'rules': [
                    {
                        'rule_id': 'stale_feed_kill_switch',
                        'rule_name': 'Stale Feed Kill Switch',
                        'rule_type': 'threshold',
                        'field': 'feed_staleness_seconds',
                        'threshold': 1.0,  # 1 second staleness limit
                        'operator': 'gt',
                        'action': 'liquidate',  # KILL_SWITCH equivalent
                        'severity': 'critical'
                    },
                    {
                        'rule_id': 'liquidity_throttle',
                        'rule_name': 'Liquidity Throttle',
                        'rule_type': 'threshold',
                        'field': 'order_book_depth_ratio',
                        'threshold': 0.1,  # Deep orderbook sweep threshold
                        'operator': 'lt',
                        'action': 'reduce_position',  # THROTTLE equivalent
                        'severity': 'medium'
                    },
                    {
                        'rule_id': 'concentration_block',
                        'rule_name': 'Concentration Block',
                        'rule_type': 'threshold',
                        'field': 'position_concentration_ratio',
                        'threshold': 0.25,  # 25% of portfolio (4x ADV equivalent)
                        'operator': 'gt',
                        'action': 'block',
                        'severity': 'high'
                    }
                ]
            }],
            'active_policy': 'sensor_test_policy'
        }
    
    async def test_stale_tick_timestamp_kill_switch(self, sensor_risk_config):
        """Test that stale tick timestamp triggers KILL_SWITCH (LIQUIDATE) action."""
        print("\n🧪 Testing Stale Tick Timestamp → KILL_SWITCH")
        
        # Create rules engine directly for precise testing
        from src.risk import RulesEngine, RiskPolicy, ThresholdRule
        
        engine = RulesEngine()
        policy = RiskPolicy("test_policy", "Test Policy")
        
        stale_feed_rule = ThresholdRule(
            "stale_feed_kill_switch",
            "Stale Feed Kill Switch",
            {
                'field': 'feed_staleness_seconds',
                'threshold': 1.0,
                'operator': 'gt',
                'action': 'liquidate',
                'severity': 'critical'
            }
        )
        policy.add_rule(stale_feed_rule)
        engine.register_policy(policy)
        
        # Test data with stale feed timestamp
        test_data = {
            'symbol': 'AAPL',
            'feed_staleness_seconds': 2.5,  # 2.5 seconds stale > 1.0 threshold
            'order_book_depth_ratio': 0.5,  # Normal liquidity
            'position_concentration_ratio': 0.15,  # Normal concentration
        }
        
        # Evaluate policy
        result = engine.evaluate_policy("test_policy", test_data)
        
        print(f"   Feed staleness: {test_data['feed_staleness_seconds']} seconds (threshold: 1.0s)")
        print(f"   Overall action: {result.overall_action}")
        print(f"   Triggered rules: {len(result.triggered_rules)}")
        
        # Verify LIQUIDATE action was triggered
        assert result.overall_action.value == 'liquidate', f"Expected LIQUIDATE, got {result.overall_action}"
        assert len(result.triggered_rules) > 0, "No rules were triggered"
        
        print("✅ PASS: Stale tick timestamp correctly triggered KILL_SWITCH")
    
    async def test_deep_orderbook_sweep_throttle(self, sensor_risk_config):
        """Test that deep orderbook sweep triggers THROTTLE (REDUCE_POSITION) action."""
        print("\n🧪 Testing Deep Orderbook Sweep → THROTTLE")
        
        # Create rules engine directly
        from src.risk import RulesEngine, RiskPolicy, ThresholdRule
        
        engine = RulesEngine()
        policy = RiskPolicy("test_policy", "Test Policy")
        
        liquidity_rule = ThresholdRule(
            "liquidity_throttle",
            "Liquidity Throttle",
            {
                'field': 'order_book_depth_ratio',
                'threshold': 0.1,
                'operator': 'lt',
                'action': 'reduce_position',
                'severity': 'medium'
            }
        )
        policy.add_rule(liquidity_rule)
        engine.register_policy(policy)
        
        # Test data with thin order book
        test_data = {
            'symbol': 'AAPL',
            'feed_staleness_seconds': 0.1,  # Fresh feeds
            'order_book_depth_ratio': 0.045,  # Thin liquidity < 0.1 threshold
            'position_concentration_ratio': 0.15,  # Normal concentration
        }
        
        # Evaluate policy
        result = engine.evaluate_policy("test_policy", test_data)
        
        print(f"   Order book depth ratio: {test_data['order_book_depth_ratio']} (threshold: 0.1)")
        print(f"   Overall action: {result.overall_action}")
        print(f"   Triggered rules: {len(result.triggered_rules)}")
        
        # Verify REDUCE_POSITION action was triggered
        assert result.overall_action.value == 'reduce_position', f"Expected REDUCE_POSITION, got {result.overall_action}"
        assert len(result.triggered_rules) > 0, "No rules were triggered"
        
        print("✅ PASS: Deep orderbook sweep correctly triggered THROTTLE")
    
    async def test_4x_adv_position_block(self, sensor_risk_config):
        """Test that 4x ADV position triggers BLOCK action."""
        print("\n🧪 Testing 4x ADV Position → BLOCK")
        
        # Create rules engine directly
        from src.risk import RulesEngine, RiskPolicy, ThresholdRule
        
        engine = RulesEngine()
        policy = RiskPolicy("test_policy", "Test Policy")
        
        concentration_rule = ThresholdRule(
            "concentration_block",
            "Concentration Block",
            {
                'field': 'position_concentration_ratio',
                'threshold': 0.25,
                'operator': 'gt',
                'action': 'block',
                'severity': 'high'
            }
        )
        policy.add_rule(concentration_rule)
        engine.register_policy(policy)
        
        # Test data with high position concentration
        test_data = {
            'symbol': 'AAPL',
            'feed_staleness_seconds': 0.1,  # Fresh feeds
            'order_book_depth_ratio': 0.5,  # Good liquidity
            'position_concentration_ratio': 0.50,  # 50% concentration > 25% threshold
        }
        
        # Evaluate policy
        result = engine.evaluate_policy("test_policy", test_data)
        
        print(f"   Position concentration: {test_data['position_concentration_ratio']*100:.0f}% (threshold: 25%)")
        print(f"   Overall action: {result.overall_action}")
        print(f"   Triggered rules: {len(result.triggered_rules)}")
        
        # Verify BLOCK action was triggered
        assert result.overall_action.value == 'block', f"Expected BLOCK, got {result.overall_action}"
        assert len(result.triggered_rules) > 0, "No rules were triggered"
        
        print("✅ PASS: 4x ADV position correctly triggered BLOCK")
    
    async def test_combined_sensor_scenarios(self, sensor_risk_config):
        """Test multiple sensor conditions simultaneously to verify priority handling."""
        print("\n🧪 Testing Combined Sensor Scenarios")
        
        # Create rules engine with all sensor rules
        from src.risk import RulesEngine, RiskPolicy, ThresholdRule
        
        engine = RulesEngine()
        policy = RiskPolicy("test_policy", "Test Policy")
        
        # Add all sensor rules
        rules = [
            ThresholdRule("stale_feed", "Stale Feed", {
                'field': 'feed_staleness_seconds', 'threshold': 1.0, 'operator': 'gt',
                'action': 'liquidate', 'severity': 'critical'
            }),
            ThresholdRule("liquidity", "Liquidity", {
                'field': 'order_book_depth_ratio', 'threshold': 0.1, 'operator': 'lt',
                'action': 'reduce_position', 'severity': 'medium'
            }),
            ThresholdRule("concentration", "Concentration", {
                'field': 'position_concentration_ratio', 'threshold': 0.25, 'operator': 'gt',
                'action': 'block', 'severity': 'high'
            })
        ]
        
        for rule in rules:
            policy.add_rule(rule)
        engine.register_policy(policy)
        
        # Test data with multiple risk factors
        test_data = {
            'symbol': 'AAPL',
            'feed_staleness_seconds': 3.0,  # LIQUIDATE trigger
            'order_book_depth_ratio': 0.03,  # REDUCE_POSITION trigger
            'position_concentration_ratio': 0.60,  # BLOCK trigger
        }
        
        # Evaluate policy
        result = engine.evaluate_policy("test_policy", test_data)
        
        print(f"   Feed staleness: {test_data['feed_staleness_seconds']}s (LIQUIDATE)")
        print(f"   Order book depth: {test_data['order_book_depth_ratio']} (REDUCE_POSITION)")
        print(f"   Position concentration: {test_data['position_concentration_ratio']*100:.0f}% (BLOCK)")
        print(f"   Overall action: {result.overall_action}")
        print(f"   Triggered rules: {len(result.triggered_rules)}")
        
        # Most severe action (LIQUIDATE) should be taken
        assert result.overall_action.value == 'liquidate', f"Expected LIQUIDATE as most severe, got {result.overall_action}"
        assert len(result.triggered_rules) == 3, f"Expected 3 triggered rules, got {len(result.triggered_rules)}"
        
        print("✅ PASS: Combined sensor scenarios handled correctly")
        print("   Most severe action (LIQUIDATE) was triggered")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])