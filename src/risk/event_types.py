from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import uuid
import time

class EventPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    ANALYTICS = 4

class EventType(Enum):
    MARKET_DATA = "market_data"
    POSITION_UPDATE = "position_update"
    TRADE_REQUEST = "trade_request"
    RISK_CALCULATION = "risk_calculation"
    RULE_EVALUATION = "rule_evaluation"
    LIMIT_BREACH = "limit_breach"
    KILL_SWITCH = "kill_switch"
    ALERT = "alert"
    CONFIG_UPDATE = "config_update"
    RISK_MONITORING = "risk_monitoring"
    POSITION_MANAGEMENT = "position_management"
    HEARTBEAT = "heartbeat"         

@dataclass
class RiskEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.MARKET_DATA
    priority: EventPriority = EventPriority.MEDIUM
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata.setdefault('created_at_ns', self.timestamp_ns)
        self.metadata.setdefault('processing_start_ns', None)
        self.metadata.setdefault('processing_end_ns', None)

    def start_processing(self) -> None:
        self.metadata['processing_start_ns'] = time.time_ns()

    def end_processing(self) -> None:
        self.metadata['processing_end_ns'] = time.time_ns()

    def get_processing_latency_us(self) -> Optional[float]:
        start = self.metadata.get('processing_start_ns')
        end = self.metadata.get('processing_end_ns')
        if start and end:
            return (end - start) / 1000.0
        return None

    def get_total_latency_us(self) -> float:
        return (time.time_ns() - self.timestamp_ns) / 1000.0
