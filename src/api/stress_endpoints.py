"""
Stress Testing API Endpoints

Provides REST API access to the Flash-Crash Lite stress testing system.
"""

import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime, timedelta

try:
    from ..risk.stress_runner import (
        create_stress_system, StressTestingSystem,
        StressScenario, StressRunner
    )
    STRESS_SYSTEM_AVAILABLE = True
except ImportError:
    STRESS_SYSTEM_AVAILABLE = False

router = APIRouter(prefix="/api/v1/stress", tags=["stress"])

# Global stress system instance
_stress_runner = None
_stress_scheduler = None


class StressTestRequest(BaseModel):
    """Request model for manual stress test."""
    scenario: Optional[str] = "flash_crash_lite"
    symbols: Optional[List[str]] = None
    force_run: Optional[bool] = False


class StressTestResponse(BaseModel):
    """Response model for stress test results."""
    scenario: str
    symbols_tested: int
    breach_count: int
    breaches: List[str]
    runtime_ms: float
    timestamp: str
    results: Dict[str, Any]


class StressStatusResponse(BaseModel):
    """Response model for stress system status."""
    enabled: bool
    scheduler_running: bool
    last_run: Optional[str]
    next_run: Optional[str]
    total_runs: int
    total_breaches: int
    current_scenario: str


async def get_stress_system():
    """Get or create the global stress system."""
    global _stress_runner, _stress_scheduler
    
    if not STRESS_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stress testing system not available")
    
    if _stress_runner is None:
        # Get PagerDuty key from environment
        pagerduty_key = os.getenv('PD_ROUTING_KEY', '')
        
        # Create stress system
        _stress_runner, _stress_scheduler = create_stress_system(
            config_path="risk/stress_packs/flash_crash.yaml",
            pagerduty_key=pagerduty_key
        )
        
        # Start scheduler if enabled
        if os.getenv('STRESS_ENABLED', 'false').lower() == 'true':
            await _stress_scheduler.start()
    
    return _stress_runner, _stress_scheduler


@router.get("/health")
async def stress_system_health():
    """Check stress testing system health."""
    if not STRESS_SYSTEM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Stress testing system not available")
    
    try:
        stress_runner, scheduler = await get_stress_system()
        
        return {
            "status": "healthy",
            "system_available": True,
            "scheduler_running": scheduler.running if scheduler else False,
            "scenario": stress_runner.scenario.name if stress_runner else None,
            "enabled": os.getenv('STRESS_ENABLED', 'false').lower() == 'true'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress system error: {str(e)}")


@router.get("/status")
async def get_stress_status():
    """Get detailed stress testing system status."""
    try:
        stress_runner, scheduler = await get_stress_system()
        
        # Calculate next run time (simplified)
        now = datetime.utcnow()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        
        return StressStatusResponse(
            enabled=os.getenv('STRESS_ENABLED', 'false').lower() == 'true',
            scheduler_running=scheduler.running if scheduler else False,
            last_run=None,  # Would need to track this in production
            next_run=next_hour.isoformat() + "Z",
            total_runs=0,   # Would need to track this in production
            total_breaches=0,  # Would need to track this in production
            current_scenario=stress_runner.scenario.name if stress_runner else "unknown"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.post("/run")
async def run_stress_test(request: StressTestRequest = StressTestRequest()):
    """Run a manual stress test."""
    try:
        stress_runner, _ = await get_stress_system()
        
        # Override symbols if provided
        if request.symbols:
            original_resolve = stress_runner._resolve_symbols
            stress_runner._resolve_symbols = lambda _: request.symbols
        
        # Run stress test
        result = await stress_runner.run_once()
        
        # Restore original symbol resolution
        if request.symbols:
            stress_runner._resolve_symbols = original_resolve
        
        return StressTestResponse(
            scenario=result.get('scenario', 'unknown'),
            symbols_tested=result.get('symbols_tested', 0),
            breach_count=result.get('breach_count', 0),
            breaches=result.get('breaches', []),
            runtime_ms=result.get('runtime_ms', 0),
            timestamp=result.get('timestamp', datetime.utcnow().isoformat()),
            results=result.get('results', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running stress test: {str(e)}")


@router.post("/start")
async def start_stress_scheduler():
    """Start the hourly stress testing scheduler."""
    try:
        _, scheduler = await get_stress_system()
        
        if scheduler.running:
            return {"message": "Scheduler already running", "status": "running"}
        
        await scheduler.start()
        
        return {"message": "Stress scheduler started", "status": "running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting scheduler: {str(e)}")


@router.post("/stop")
async def stop_stress_scheduler():
    """Stop the hourly stress testing scheduler."""
    try:
        _, scheduler = await get_stress_system()
        
        if not scheduler.running:
            return {"message": "Scheduler not running", "status": "stopped"}
        
        await scheduler.stop()
        
        return {"message": "Stress scheduler stopped", "status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping scheduler: {str(e)}")


@router.get("/results")
async def get_stress_results(hours: int = 24):
    """Get recent stress test results."""
    # In production, this would query a database or log store
    # For now, return a placeholder response
    
    return {
        "time_range_hours": hours,
        "total_runs": 0,
        "total_breaches": 0,
        "results": [],
        "note": "Historical results not implemented - would query database in production"
    }


@router.get("/scenarios")
async def list_stress_scenarios():
    """List available stress testing scenarios."""
    try:
        # In production, this would scan the stress_packs directory
        scenarios = [
            {
                "name": "flash_crash_lite",
                "description": "60-second down-spike with recovery",
                "price_shock_pct": -0.03,
                "duration_sec": 60,
                "file": "risk/stress_packs/flash_crash.yaml"
            }
        ]
        
        return {
            "scenarios": scenarios,
            "total_scenarios": len(scenarios)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing scenarios: {str(e)}")


@router.get("/config")
async def get_stress_config():
    """Get current stress testing configuration."""
    try:
        stress_runner, scheduler = await get_stress_system()
        
        config = {
            "scenario_name": stress_runner.scenario.name,
            "price_shock_pct": stress_runner.scenario.price_shock_pct,
            "spread_mult": stress_runner.scenario.spread_mult,
            "duration_sec": stress_runner.scenario.duration_sec,
            "max_runtime_ms": stress_runner.scenario.max_runtime_ms,
            "max_symbols": stress_runner.scenario.max_symbols,
            "alert_on_breach": stress_runner.scenario.alert_on_breach,
            "halt_on_breach": stress_runner.scenario.halt_on_breach,
            "enabled": os.getenv('STRESS_ENABLED', 'false').lower() == 'true',
            "pagerduty_enabled": bool(os.getenv('PD_ROUTING_KEY')),
            "market_hours_only": stress_runner.scenario.config.get('market_hours_only', True)
        }
        
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@router.get("/metrics")
async def get_stress_metrics():
    """Get stress testing metrics for monitoring."""
    try:
        # In production, this would query Prometheus or metrics store
        metrics = {
            "stress_runs_total": 0,
            "stress_breaches_total": 0,
            "stress_runtime_seconds_avg": 0.025,  # 25ms average
            "stress_symbols_tested_avg": 15,
            "last_runtime_ms": 0,
            "scheduler_uptime_hours": 0,
            "note": "Metrics not implemented - would query Prometheus in production"
        }
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


# Background task for system initialization
async def initialize_stress_system():
    """Initialize the stress testing system on startup."""
    if os.getenv('STRESS_ENABLED', 'false').lower() == 'true':
        try:
            await get_stress_system()
            print("✅ Stress testing system initialized and started")
        except Exception as e:
            print(f"⚠️  Failed to initialize stress testing system: {e}")


# Example usage endpoints
@router.get("/examples/flash-crash")
async def flash_crash_example():
    """Example of flash crash stress test configuration."""
    return {
        "scenario": "flash_crash_lite",
        "description": "3% down-spike with 60-second linear recovery",
        "parameters": {
            "price_shock_pct": -0.03,
            "spread_mult": 3.0,
            "duration_sec": 60,
            "recovery_type": "linear"
        },
        "expected_impact": {
            "symbols_tested": "10-50 (active positions)",
            "runtime_target": "<50ms",
            "breach_threshold": "5% drawdown or 2x VaR"
        },
        "response": {
            "breach_detected": "PagerDuty alert + KILL_SWITCH",
            "no_breach": "Continue normal operations"
        }
    }


@router.get("/examples/custom-scenario")
async def custom_scenario_example():
    """Example of custom stress scenario configuration."""
    return {
        "yaml_config": {
            "scenario_name": "custom_stress",
            "symbol_set": "active_book",
            "price_shock_pct": -0.05,  # 5% shock
            "spread_mult": 2.0,
            "duration_sec": 120,       # 2 minutes
            "recovery_type": "exponential",
            "max_runtime_ms": 100,
            "alert_on_breach": True,
            "halt_on_breach": True
        },
        "usage": "Save as risk/stress_packs/custom_stress.yaml and restart system"
    }