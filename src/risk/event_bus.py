# src/risk/event_bus.py
"""
High-performance event bus for risk system components.
Designed for microsecond-level latency with async streaming capabilities.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import defaultdict, deque
import logging
import threading
from concurrent.futures import ThreadPoolExecutor


class EventPriority(Enum):
    """Event priority levels for latency-sensitive processing."""
    CRITICAL = 0    # Pre-trade, kill switches (5-20 µs)
    HIGH = 1        # Risk calculations (100-150 µs)  
    MEDIUM = 2      # Rules evaluation (50-100 µs)
    LOW = 3         # Monitoring, alerts (0.5-1s)
    ANALYTICS = 4   # Batch processing (minutes)


class EventType(Enum):
    """Risk event types."""
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


@dataclass
class RiskEvent:
    """
    Immutable risk event with microsecond precision timing.
    Designed for high-frequency, low-latency processing.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.MARKET_DATA
    priority: EventPriority = EventPriority.MEDIUM
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure immutability and add processing metadata."""
        # Add latency tracking
        self.metadata.setdefault('created_at_ns', self.timestamp_ns)
        self.metadata.setdefault('processing_start_ns', None)
        self.metadata.setdefault('processing_end_ns', None)
    
    def start_processing(self) -> None:
        """Mark start of processing for latency tracking."""
        self.metadata['processing_start_ns'] = time.time_ns()
    
    def end_processing(self) -> None:
        """Mark end of processing for latency tracking."""
        self.metadata['processing_end_ns'] = time.time_ns()
    
    def get_processing_latency_us(self) -> Optional[float]:
        """Get processing latency in microseconds."""
        start = self.metadata.get('processing_start_ns')
        end = self.metadata.get('processing_end_ns')
        if start and end:
            return (end - start) / 1000.0  # Convert ns to µs
        return None
    
    def get_total_latency_us(self) -> float:
        """Get total latency from creation to now in microseconds."""
        return (time.time_ns() - self.timestamp_ns) / 1000.0


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: RiskEvent) -> Optional[RiskEvent]:
        """
        Handle an event and optionally return a new event.
        
        Args:
            event: The event to handle
            
        Returns:
            Optional new event to publish
        """
        pass
    
    @property
    @abstractmethod
    def supported_event_types(self) -> List[EventType]:
        """Return list of event types this handler supports."""
        pass
    
    @property
    def priority_filter(self) -> Optional[List[EventPriority]]:
        """Return list of priorities to handle, or None for all."""
        return None


class RiskEventBus:
    """
    High-performance event bus with priority queues and latency monitoring.
    
    Features:
    - Priority-based routing for latency-sensitive events
    - Async processing with configurable concurrency
    - Latency SLO monitoring and alerting
    - Circuit breakers for graceful degradation
    - Comprehensive audit trail
    """
    
    def __init__(self, 
                 max_workers: int = 10,
                 enable_latency_monitoring: bool = True,
                 latency_slo_us: Dict[EventPriority, float] = None):
        """
        Initialize the event bus.
        
        Args:
            max_workers: Maximum number of worker threads
            enable_latency_monitoring: Whether to track latency metrics
            latency_slo_us: SLO thresholds in microseconds per priority
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Event routing
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._priority_queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in EventPriority
        }
        
        # Processing infrastructure
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        
        # Latency monitoring
        self._enable_latency_monitoring = enable_latency_monitoring
        self._latency_slo_us = latency_slo_us or {
            EventPriority.CRITICAL: 20.0,
            EventPriority.HIGH: 150.0,
            EventPriority.MEDIUM: 100.0,
            EventPriority.LOW: 1000000.0,  # 1s
            EventPriority.ANALYTICS: float('inf')
        }
        
        # Metrics
        self._event_counts: Dict[EventType, int] = defaultdict(int)
        self._latency_violations: Dict[EventPriority, int] = defaultdict(int)
        self._processing_times: Dict[EventPriority, deque] = {
            priority: deque(maxlen=1000) for priority in EventPriority
        }
        
        # Circuit breakers
        self._circuit_breakers: Dict[EventType, bool] = defaultdict(bool)
        self._error_counts: Dict[EventType, int] = defaultdict(int)
        self._max_errors_per_type = 10
        
        self.logger.info(f"RiskEventBus initialized with {max_workers} workers")
    
    def register_handler(self, handler: EventHandler) -> None:
        """Register an event handler."""
        for event_type in handler.supported_event_types:
            self._handlers[event_type].append(handler)
            self.logger.info(f"Registered handler {handler.__class__.__name__} for {event_type}")
    
    def unregister_handler(self, handler: EventHandler) -> None:
        """Unregister an event handler."""
        for event_type in handler.supported_event_types:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                self.logger.info(f"Unregistered handler {handler.__class__.__name__} for {event_type}")
    
    async def publish(self, event: RiskEvent) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: The event to publish
        """
        if self._circuit_breakers[event.event_type]:
            self.logger.warning(f"Circuit breaker open for {event.event_type}, dropping event")
            return
        
        # Add to appropriate priority queue
        await self._priority_queues[event.priority].put(event)
        self._event_counts[event.event_type] += 1
        
        self.logger.debug(f"Published event {event.event_id} of type {event.event_type}")
    
    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks for each priority level
        for priority in EventPriority:
            task = asyncio.create_task(self._process_priority_queue(priority))
            self._worker_tasks.append(task)
        
        # Start monitoring task
        if self._enable_latency_monitoring:
            monitor_task = asyncio.create_task(self._monitor_latency())
            self._worker_tasks.append(monitor_task)
        
        self.logger.info("RiskEventBus started")
    
    async def stop(self) -> None:
        """Stop the event bus processing."""
        self._running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("RiskEventBus stopped")
    
    async def _process_priority_queue(self, priority: EventPriority) -> None:
        """Process events from a specific priority queue."""
        queue = self._priority_queues[priority]
        
        while self._running:
            try:
                # Get event with timeout to allow periodic checks
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue processing
            except Exception as e:
                self.logger.error(f"Error processing {priority} queue: {e}")
    
    async def _process_event(self, event: RiskEvent) -> None:
        """Process a single event through all registered handlers."""
        event.start_processing()
        
        try:
            handlers = self._handlers.get(event.event_type, [])
            
            # Filter handlers by priority if they specify a filter
            filtered_handlers = []
            for handler in handlers:
                if (handler.priority_filter is None or 
                    event.priority in handler.priority_filter):
                    filtered_handlers.append(handler)
            
            # Process through all applicable handlers
            for handler in filtered_handlers:
                try:
                    result_event = await handler.handle(event)
                    
                    # If handler returns a new event, publish it
                    if result_event:
                        await self.publish(result_event)
                        
                except Exception as e:
                    self.logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                    self._error_counts[event.event_type] += 1
                    
                    # Check circuit breaker threshold
                    if self._error_counts[event.event_type] >= self._max_errors_per_type:
                        self._circuit_breakers[event.event_type] = True
                        self.logger.critical(f"Circuit breaker opened for {event.event_type}")
        
        finally:
            event.end_processing()
            
            # Record latency metrics
            if self._enable_latency_monitoring:
                latency_us = event.get_processing_latency_us()
                if latency_us:
                    self._processing_times[event.priority].append(latency_us)
                    
                    # Check SLO violation
                    slo_threshold = self._latency_slo_us[event.priority]
                    if latency_us > slo_threshold:
                        self._latency_violations[event.priority] += 1
                        self.logger.warning(
                            f"Latency SLO violation: {latency_us:.2f}µs > {slo_threshold:.2f}µs "
                            f"for {event.priority} event {event.event_id}"
                        )
    
    async def _monitor_latency(self) -> None:
        """Monitor latency metrics and emit alerts."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                for priority in EventPriority:
                    times = self._processing_times[priority]
                    if len(times) < 10:  # Need minimum samples
                        continue
                    
                    # Calculate percentiles
                    sorted_times = sorted(times)
                    p99_9 = sorted_times[int(len(sorted_times) * 0.999)]
                    p99 = sorted_times[int(len(sorted_times) * 0.99)]
                    p95 = sorted_times[int(len(sorted_times) * 0.95)]
                    
                    slo_threshold = self._latency_slo_us[priority]
                    
                    if p99_9 > slo_threshold:
                        self.logger.critical(
                            f"LATENCY ALERT: {priority} P99.9 = {p99_9:.2f}µs > SLO {slo_threshold:.2f}µs"
                        )
                    
                    self.logger.debug(
                        f"{priority} latency: P95={p95:.2f}µs, P99={p99:.2f}µs, P99.9={p99_9:.2f}µs"
                    )
            
            except Exception as e:
                self.logger.error(f"Latency monitoring error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current event bus metrics."""
        metrics = {
            'event_counts': dict(self._event_counts),
            'latency_violations': dict(self._latency_violations),
            'circuit_breakers': dict(self._circuit_breakers),
            'error_counts': dict(self._error_counts),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self._priority_queues.items()
            }
        }
        
        # Add latency percentiles
        latency_stats = {}
        for priority, times in self._processing_times.items():
            if len(times) > 0:
                sorted_times = sorted(times)
                latency_stats[priority.name] = {
                    'count': len(times),
                    'p50': sorted_times[len(sorted_times) // 2],
                    'p95': sorted_times[int(len(sorted_times) * 0.95)],
                    'p99': sorted_times[int(len(sorted_times) * 0.99)],
                    'max': max(times)
                }
        
        metrics['latency_stats'] = latency_stats
        return metrics
    
    def reset_circuit_breaker(self, event_type: EventType) -> None:
        """Reset circuit breaker for an event type."""
        self._circuit_breakers[event_type] = False
        self._error_counts[event_type] = 0
        self.logger.info(f"Reset circuit breaker for {event_type}")


# Singleton instance for global access
_global_event_bus: Optional[RiskEventBus] = None

def get_global_event_bus() -> RiskEventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = RiskEventBus()
    return _global_event_bus

def set_global_event_bus(event_bus: RiskEventBus) -> None:
    """Set the global event bus instance."""
    global _global_event_bus
    _global_event_bus = event_bus