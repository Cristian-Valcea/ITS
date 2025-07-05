# src/risk/__init__.py
"""
Enterprise Risk Management System

A modular, high-performance risk management system designed for intraday trading
with microsecond-level latency requirements and enterprise-grade features.

Architecture:
1. Data Ingest - Tick normalization and market data enrichment
2. Risk Calculators - Stateless, vectorized risk computations  
3. Rules Engine - Micro-kernel policy evaluation system
4. Risk Enforcement - Pre-trade gates and kill switches
5. Risk Oversight - Real-time monitoring and alerting
6. Analytics - Scenario analysis and risk attribution

Key Features:
- Sub-millisecond latency for critical paths
- Plugin-based calculator system
- Hierarchical risk limits (Firm → Desk → Account → Strategy → Symbol)
- Hot-swappable configuration
- Comprehensive audit trail
- Circuit breakers and graceful degradation
"""

from .calculators import (
    BaseRiskCalculator, DrawdownCalculator, TurnoverCalculator
)
from .rules_engine import RulesEngine, RiskPolicy, PolicyValidator, ThresholdRule, RuleAction
from .event_bus import RiskEventBus, RiskEvent, EventType, EventPriority
from .risk_agent_v2 import RiskAgentV2, create_risk_agent_v2

__all__ = [
    # Calculators
    'BaseRiskCalculator', 'DrawdownCalculator', 'TurnoverCalculator',
    
    # Rules & Enforcement
    'RulesEngine', 'RiskPolicy', 'PolicyValidator', 'ThresholdRule', 'RuleAction',
    
    # Infrastructure
    'RiskEventBus', 'RiskEvent', 'EventType', 'EventPriority',
    
    # Risk Agent
    'RiskAgentV2', 'create_risk_agent_v2'
]

# Version info
__version__ = '1.0.0'
__author__ = 'IntradayJules Risk Engineering'