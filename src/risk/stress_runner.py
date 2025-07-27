"""
Flash-Crash Lite Intraday Stress Engine

Runs hourly synthetic liquidity-shock scenarios during live trading.
Auto-pages via PagerDuty and triggers KILL_SWITCH on risk limit breaches.
Zero impact on microsecond trade path.
"""

import asyncio
import datetime as dt
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import numpy as np

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
    
    # Prometheus metrics
    STRESS_RUN_TOTAL = Counter('stress_runs_total', 'Total stress test runs')
    STRESS_BREACH_TOTAL = Counter('stress_breaches_total', 'Total stress test breaches')
    STRESS_RUNTIME_SECONDS = Histogram('stress_runtime_seconds', 'Stress test runtime')
    STRESS_SYMBOLS_TESTED = Gauge('stress_symbols_tested', 'Number of symbols in last stress test')
    
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import risk system components
try:
    from ..risk.event_bus import RiskEventBus, RiskEvent, EventType, EventPriority
    from ..risk.agent.risk_agent_v2 import RiskAgentV2
    RISK_SYSTEM_AVAILABLE = True
except ImportError:
    RISK_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)


class StressScenario:
    """Represents a stress testing scenario configuration."""
    
    def __init__(self, config_path: Path):
        """Initialize stress scenario from YAML configuration."""
        self.config_path = config_path
        self.config = yaml.safe_load(config_path.read_text())
        self._validate_config()
    
    def _validate_config(self):
        """Validate scenario configuration."""
        required_fields = [
            'scenario_name', 'symbol_set', 'price_shock_pct', 
            'spread_mult', 'duration_sec'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if not -0.20 <= self.config['price_shock_pct'] <= 0.20:
            raise ValueError("price_shock_pct must be between -20% and +20%")
        
        if not 1.0 <= self.config['spread_mult'] <= 10.0:
            raise ValueError("spread_mult must be between 1.0 and 10.0")
        
        if not 10 <= self.config['duration_sec'] <= 300:
            raise ValueError("duration_sec must be between 10 and 300 seconds")
    
    @property
    def name(self) -> str:
        return self.config['scenario_name']
    
    @property
    def symbol_set(self) -> str:
        return self.config['symbol_set']
    
    @property
    def price_shock_pct(self) -> float:
        return self.config['price_shock_pct']
    
    @property
    def spread_mult(self) -> float:
        return self.config['spread_mult']
    
    @property
    def duration_sec(self) -> int:
        return self.config['duration_sec']
    
    @property
    def max_runtime_ms(self) -> int:
        return self.config.get('max_runtime_ms', 50)
    
    @property
    def max_symbols(self) -> int:
        return self.config.get('max_symbols', 100)
    
    @property
    def alert_on_breach(self) -> bool:
        return self.config.get('alert_on_breach', True)
    
    @property
    def halt_on_breach(self) -> bool:
        return self.config.get('halt_on_breach', True)


class PagerDutyAlerter:
    """PagerDuty integration for stress test alerts."""
    
    def __init__(self, routing_key: str):
        """Initialize PagerDuty alerter with routing key."""
        self.routing_key = routing_key
        self.enabled = bool(routing_key)
    
    async def send_alert(self, symbols: List[str], scenario_name: str):
        """Send PagerDuty alert for stress test breach."""
        if not self.enabled:
            logger.warning("PagerDuty alerter disabled - no routing key")
            return
        
        try:
            import httpx
            
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"Stress test breach: {scenario_name} on {', '.join(symbols)}",
                    "severity": "critical",
                    "source": "stress_runner",
                    "component": "risk_management",
                    "group": "trading_system",
                    "custom_details": {
                        "scenario": scenario_name,
                        "symbols": symbols,
                        "breach_count": len(symbols),
                        "timestamp": dt.datetime.utcnow().isoformat()
                    }
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=5.0
                )
                response.raise_for_status()
                
            logger.info(f"PagerDuty alert sent for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")


class StressRunner:
    """Main stress testing engine."""
    
    def __init__(self, 
                 scenario: StressScenario,
                 bus: Optional[Any] = None,
                 risk_agent: Optional[Any] = None,
                 alerter: Optional[PagerDutyAlerter] = None):
        """Initialize stress runner."""
        self.scenario = scenario
        self.bus = bus
        self.risk_agent = risk_agent
        self.alerter = alerter or PagerDutyAlerter("")
        
        logger.info(f"StressRunner initialized for scenario: {scenario.name}")
    
    async def run_once(self) -> Dict[str, Any]:
        """Run stress test once and return results."""
        start_time = dt.datetime.utcnow()
        
        try:
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                STRESS_RUN_TOTAL.inc()
            
            # Get symbols to test
            symbols = self._resolve_symbols(self.scenario.symbol_set)
            
            if len(symbols) > self.scenario.max_symbols:
                symbols = symbols[:self.scenario.max_symbols]
                logger.warning(f"Limited symbols to {self.scenario.max_symbols}")
            
            if PROMETHEUS_AVAILABLE:
                STRESS_SYMBOLS_TESTED.set(len(symbols))
            
            logger.info(f"Running stress test on {len(symbols)} symbols")
            
            # Run stress test on each symbol
            breaches = []
            results = {}
            
            for symbol in symbols:
                try:
                    breach_detected = await self._test_symbol(symbol)
                    results[symbol] = {
                        'breach': breach_detected,
                        'timestamp': dt.datetime.utcnow().isoformat()
                    }
                    
                    if breach_detected:
                        breaches.append(symbol)
                        
                except Exception as e:
                    logger.error(f"Error testing symbol {symbol}: {e}")
                    results[symbol] = {'error': str(e)}
            
            # Handle breaches
            if breaches:
                await self._handle_breaches(breaches)
                
                if PROMETHEUS_AVAILABLE:
                    STRESS_BREACH_TOTAL.inc(len(breaches))
            
            # Calculate runtime
            runtime = (dt.datetime.utcnow() - start_time).total_seconds()
            
            if PROMETHEUS_AVAILABLE:
                STRESS_RUNTIME_SECONDS.observe(runtime)
            
            # Check performance constraint
            runtime_ms = runtime * 1000
            if runtime_ms > self.scenario.max_runtime_ms:
                logger.warning(f"Stress test exceeded max runtime: {runtime_ms:.1f}ms > {self.scenario.max_runtime_ms}ms")
            
            return {
                'scenario': self.scenario.name,
                'symbols_tested': len(symbols),
                'breaches': breaches,
                'breach_count': len(breaches),
                'runtime_ms': runtime_ms,
                'results': results,
                'timestamp': start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return {
                'scenario': self.scenario.name,
                'error': str(e),
                'timestamp': start_time.isoformat()
            }
    
    def _resolve_symbols(self, symbol_set: str) -> List[str]:
        """Resolve symbol set to actual symbol list."""
        if symbol_set == "active_book":
            # Get symbols from active positions
            if self.bus and hasattr(self.bus, 'get_metrics'):
                try:
                    metrics = self.bus.get_metrics()
                    if 'positions' in metrics:
                        return list(metrics['positions'].keys())
                except Exception as e:
                    logger.warning(f"Could not get active positions: {e}")
            
            # Fallback to common CME micro futures
            return ['MES', 'MNQ', 'M2K', 'MCL']
        
        elif symbol_set == "cme_micros":
            return ['MES', 'MNQ', 'M2K', 'MCL']
        
        elif symbol_set == "test_symbols":
            return ['MES', 'MNQ']
        
        else:
            raise ValueError(f"Unknown symbol_set: {symbol_set}")
    
    async def _test_symbol(self, symbol: str) -> bool:
        """Test a single symbol for stress scenario breach."""
        try:
            # Build synthetic market event
            synthetic_event = self._build_synthetic_event(symbol)
            
            # Test with risk agent if available
            if self.risk_agent and hasattr(self.risk_agent, 'handle'):
                result = await self.risk_agent.handle(synthetic_event)
                
                # Check if result indicates a breach
                if result and hasattr(result, 'event_type'):
                    return result.event_type == EventType.LIMIT_BREACH
            
            # Fallback: simulate basic risk check
            return self._simulate_risk_check(symbol)
            
        except Exception as e:
            logger.error(f"Error testing symbol {symbol}: {e}")
            return False
    
    def _build_synthetic_event(self, symbol: str) -> Any:
        """Build synthetic market event for stress testing."""
        # Generate price path for stress scenario
        num_points = 10
        t = np.linspace(0, self.scenario.duration_sec, num=num_points)
        
        # Price shock with linear recovery
        if self.scenario.config.get('recovery_type', 'linear') == 'linear':
            # Start at shock, recover linearly to original
            price_multipliers = 1.0 + self.scenario.price_shock_pct * (1 - t / self.scenario.duration_sec)
        else:
            # Exponential recovery
            price_multipliers = 1.0 + self.scenario.price_shock_pct * np.exp(-t / (self.scenario.duration_sec / 3))
        
        # Build event data
        event_data = {
            'symbol': symbol,
            'price_path': price_multipliers.tolist(),
            'spread_mult': self.scenario.spread_mult,
            'duration_sec': self.scenario.duration_sec,
            'scenario': self.scenario.name
        }
        
        # Create risk event if available
        if RISK_SYSTEM_AVAILABLE:
            return RiskEvent(
                event_type=EventType.MARKET_DATA,
                priority=EventPriority.HIGH,
                source="stress_runner",
                data=event_data
            )
        else:
            return event_data
    
    def _simulate_risk_check(self, symbol: str) -> bool:
        """Simulate basic risk check when risk agent not available."""
        # Simple simulation: randomly breach 5% of the time for testing
        import random
        return random.random() < 0.05
    
    async def _handle_breaches(self, breaches: List[str]):
        """Handle detected breaches."""
        logger.warning(f"Stress test breaches detected: {breaches}")
        
        # Send PagerDuty alert
        if self.scenario.alert_on_breach:
            await self.alerter.send_alert(breaches, self.scenario.name)
        
        # Trigger KILL_SWITCH if configured
        if self.scenario.halt_on_breach and self.bus:
            try:
                if RISK_SYSTEM_AVAILABLE:
                    kill_event = RiskEvent(
                        event_type=EventType.KILL_SWITCH,
                        priority=EventPriority.CRITICAL,
                        source="stress_runner",
                        data={
                            'reason': 'stress_test_breach',
                            'symbols': breaches,
                            'scenario': self.scenario.name
                        }
                    )
                    
                    if hasattr(self.bus, 'publish'):
                        await self.bus.publish(kill_event)
                        logger.critical(f"KILL_SWITCH triggered due to stress test breach on {breaches}")
                
            except Exception as e:
                logger.error(f"Failed to trigger KILL_SWITCH: {e}")


class HourlyStressScheduler:
    """Scheduler for hourly stress testing."""
    
    def __init__(self, stress_runner: StressRunner):
        """Initialize hourly scheduler."""
        self.stress_runner = stress_runner
        self.running = False
        self.task = None
        
        logger.info("HourlyStressScheduler initialized")
    
    async def start(self):
        """Start the hourly stress testing scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        logger.info("Hourly stress scheduler started")
    
    async def stop(self):
        """Stop the hourly stress testing scheduler."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info("Hourly stress scheduler stopped")
    
    async def _run_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Calculate next run time (top of next hour + 5 seconds)
                now = dt.datetime.utcnow()
                next_hour = (now + dt.timedelta(hours=1)).replace(
                    minute=0, second=5, microsecond=0
                )
                
                sleep_seconds = (next_hour - now).total_seconds()
                
                logger.info(f"Next stress test scheduled for {next_hour} UTC ({sleep_seconds:.0f}s)")
                
                # Sleep until next run time
                await asyncio.sleep(sleep_seconds)
                
                # Check if we should run (market hours, etc.)
                if self._should_run():
                    logger.info("Running hourly stress test")
                    result = await self.stress_runner.run_once()
                    
                    if result.get('breach_count', 0) > 0:
                        logger.warning(f"Stress test completed with {result['breach_count']} breaches")
                    else:
                        logger.info(f"Stress test completed successfully - {result.get('symbols_tested', 0)} symbols tested")
                else:
                    logger.info("Skipping stress test (outside market hours or disabled)")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stress scheduler: {e}")
                # Sleep 5 minutes before retrying
                await asyncio.sleep(300)
    
    def _should_run(self) -> bool:
        """Check if stress test should run now."""
        # Check if market hours only is configured
        if self.stress_runner.scenario.config.get('market_hours_only', True):
            now = dt.datetime.utcnow()
            
            # Simple market hours check (9:30 AM - 4:00 PM ET = 14:30 - 21:00 UTC)
            # This is a simplified check - in production you'd want more sophisticated logic
            hour_utc = now.hour
            
            # Skip weekends
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check market hours (approximate)
            if not (14 <= hour_utc < 21):  # 9:30 AM - 4:00 PM ET
                return False
        
        return True


# Factory function for easy initialization
def create_stress_system(config_path: str = "risk/stress_packs/flash_crash.yaml",
                        pagerduty_key: Optional[str] = None) -> tuple:
    """Create and configure the stress testing system."""
    
    # Load scenario
    scenario = StressScenario(Path(config_path))
    
    # Create alerter
    alerter = PagerDutyAlerter(pagerduty_key or "")
    
    # Create stress runner
    stress_runner = StressRunner(
        scenario=scenario,
        alerter=alerter
    )
    
    # Create scheduler
    scheduler = HourlyStressScheduler(stress_runner)
    
    return stress_runner, scheduler


# Async context manager for easy usage
class StressTestingSystem:
    """Context manager for the complete stress testing system."""
    
    def __init__(self, config_path: str = "risk/stress_packs/flash_crash.yaml",
                 pagerduty_key: Optional[str] = None):
        self.config_path = config_path
        self.pagerduty_key = pagerduty_key
        self.stress_runner = None
        self.scheduler = None
    
    async def __aenter__(self):
        """Start the stress testing system."""
        self.stress_runner, self.scheduler = create_stress_system(
            self.config_path, self.pagerduty_key
        )
        await self.scheduler.start()
        return self.stress_runner, self.scheduler
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the stress testing system."""
        if self.scheduler:
            await self.scheduler.stop()


# Example usage function
async def run_stress_test_example():
    """Example of how to use the stress testing system."""
    
    # Create stress system
    stress_runner, scheduler = create_stress_system()
    
    # Run once manually
    result = await stress_runner.run_once()
    print(f"Stress test result: {result}")
    
    # Or start hourly scheduler
    # await scheduler.start()
    # # Keep running...
    # await asyncio.sleep(3600)  # Run for 1 hour
    # await scheduler.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(run_stress_test_example())