#!/usr/bin/env python3
"""
🚀 PHASE 2B SERVICE LAUNCHER
Orchestrates the complete live trading pipeline
"""

import asyncio
import subprocess
import time
import logging
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any
import redis
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase2BOrchestrator:
    """Orchestrator for Phase 2B live trading services"""
    
    def __init__(self):
        self.services = []
        self.redis_client = None
        self.shutdown_event = asyncio.Event()
        
        # Service definitions
        self.service_configs = [
            {
                "name": "pushgateway",
                "type": "docker",
                "command": ["docker", "compose", "-f", "docker-compose.timescale.yml", "--profile", "live", "up", "-d", "pushgateway"],
                "health_check": self.check_pushgateway_health,
                "startup_delay": 5
            },
            {
                "name": "inference_api",
                "type": "python",
                "command": ["python", "inference_api.py"],
                "health_check": self.check_inference_api_health,
                "startup_delay": 10
            },
            {
                "name": "risk_guard",
                "type": "python", 
                "command": ["python", "risk_guard.py"],
                "health_check": self.check_risk_guard_health,
                "startup_delay": 5
            },
            {
                "name": "ib_executor",
                "type": "python",
                "command": ["python", "ib_executor.py"],
                "health_check": self.check_ib_executor_health,
                "startup_delay": 10
            },
            {
                "name": "pnl_tracker",
                "type": "python",
                "command": ["python", "pnl_tracker.py"],
                "health_check": self.check_pnl_tracker_health,
                "startup_delay": 5
            }
        ]
    
    async def initialize_redis(self):
        """Initialize Redis connection"""
        
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def check_prerequisites(self):
        """Check system prerequisites"""
        
        logger.info("🔍 Checking Phase 2B prerequisites")
        
        checks = []
        
        # 1. Check production model exists
        model_path = Path("deploy_models/dual_ticker_prod_20250731_step201k_stable.zip")
        checks.append(("Production Model", model_path.exists(), f"Model file: {model_path}"))
        
        # 2. Check Redis connection
        redis_ok = await self.initialize_redis()
        checks.append(("Redis Connection", redis_ok, "localhost:6379"))
        
        # 3. Check TimescaleDB connection
        try:
            from secrets_helper import SecretsHelper
            db_url = SecretsHelper.get_database_url()
            checks.append(("Database Config", bool(db_url), "TimescaleDB credentials"))
        except Exception:
            checks.append(("Database Config", False, "TimescaleDB credentials missing"))
        
        # 4. Check required Python modules
        required_modules = ["fastapi", "uvicorn", "redis", "psycopg2", "stable_baselines3"]
        for module in required_modules:
            try:
                __import__(module)
                checks.append((f"Module {module}", True, "Available"))
            except ImportError:
                checks.append((f"Module {module}", False, "Missing"))
        
        # Report results
        logger.info("📋 Prerequisite Check Results:")
        all_passed = True
        for name, passed, details in checks:
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {name}: {details}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            logger.error("❌ Prerequisites not met - cannot start Phase 2B")
            return False
        
        logger.info("🎉 All prerequisites met!")
        return True
    
    async def start_service(self, config: Dict[str, Any]):
        """Start a single service"""
        
        logger.info(f"🚀 Starting {config['name']}")
        
        try:
            if config["type"] == "docker":
                # Docker service
                process = subprocess.Popen(
                    config["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                # Python service
                process = subprocess.Popen(
                    config["command"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=Path.cwd()
                )
            
            # Wait for startup
            await asyncio.sleep(config.get("startup_delay", 5))
            
            # Check if process is still running
            if process.poll() is None:
                service_info = {
                    "name": config["name"],
                    "process": process,
                    "config": config,
                    "start_time": time.time(),
                    "status": "starting"
                }
                self.services.append(service_info)
                logger.info(f"✅ {config['name']} started (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ {config['name']} failed to start")
                logger.error(f"   stdout: {stdout}")
                logger.error(f"   stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting {config['name']}: {e}")
            return False
    
    async def check_service_health(self, service_info: Dict[str, Any]):
        """Check health of a running service"""
        
        try:
            # Check if process is still running
            process = service_info["process"]
            if process.poll() is not None:
                service_info["status"] = "failed"
                return False
            
            # Run service-specific health check
            health_check = service_info["config"].get("health_check")
            if health_check:
                health_ok = await health_check()
                service_info["status"] = "healthy" if health_ok else "unhealthy"
                return health_ok
            else:
                service_info["status"] = "running"
                return True
                
        except Exception as e:
            logger.error(f"❌ Health check error for {service_info['name']}: {e}")
            service_info["status"] = "error"
            return False
    
    # Health check functions
    async def check_pushgateway_health(self):
        try:
            response = requests.get("http://localhost:9091/metrics", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def check_inference_api_health(self):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def check_risk_guard_health(self):
        try:
            if not self.redis_client:
                return False
            # Check if risk metrics are being updated
            metrics = self.redis_client.hgetall("risk_metrics")
            return len(metrics) > 0
        except:
            return False
    
    async def check_ib_executor_health(self):
        try:
            if not self.redis_client:
                return False
            # Check if execution metrics are being updated
            metrics = self.redis_client.hgetall("execution_metrics")
            return len(metrics) > 0
        except:
            return False
    
    async def check_pnl_tracker_health(self):
        try:
            if not self.redis_client:
                return False
            # Check if portfolio metrics are being updated
            metrics = self.redis_client.hgetall("portfolio_metrics")
            return len(metrics) > 0
        except:
            return False
    
    async def monitor_services(self):
        """Monitor all running services"""
        
        logger.info("🔍 Starting service monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                for service_info in self.services:
                    await self.check_service_health(service_info)
                
                # Log service status
                if self.services:
                    healthy_count = sum(1 for s in self.services if s["status"] == "healthy")
                    total_count = len(self.services)
                    logger.info(f"📊 Services: {healthy_count}/{total_count} healthy")
                    
                    # Detailed status
                    for service_info in self.services:
                        name = service_info["name"]
                        status = service_info["status"]
                        uptime = time.time() - service_info["start_time"]
                        logger.debug(f"   {name}: {status} (uptime: {uptime:.0f}s)")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in service monitoring: {e}")
                await asyncio.sleep(10)
    
    async def display_status_dashboard(self):
        """Display real-time status dashboard"""
        
        logger.info("📊 Starting status dashboard")
        
        while not self.shutdown_event.is_set():
            try:
                if self.redis_client:
                    # Get key metrics
                    risk_metrics = self.redis_client.hgetall("risk_metrics")
                    portfolio_metrics = self.redis_client.hgetall("portfolio_metrics")
                    execution_metrics = self.redis_client.hgetall("execution_metrics")
                    
                    # Display dashboard
                    print("\n" + "="*60)
                    print("🚀 PHASE 2B LIVE TRADING DASHBOARD")
                    print("="*60)
                    
                    # Service status
                    print("📊 SERVICE STATUS:")
                    for service_info in self.services:
                        name = service_info["name"].ljust(15)
                        status = service_info["status"]
                        status_icon = {"healthy": "✅", "running": "🟡", "unhealthy": "❌", "failed": "💥"}.get(status, "❓")
                        print(f"   {status_icon} {name} {status}")
                    
                    # Portfolio metrics
                    if portfolio_metrics:
                        print("\n💰 PORTFOLIO:")
                        total_value = float(portfolio_metrics.get("portfolio_total_value", 0))
                        daily_pnl = float(portfolio_metrics.get("portfolio_daily_pnl", 0))
                        print(f"   Total Value: ${total_value:,.2f}")
                        print(f"   Daily P&L:   ${daily_pnl:+,.2f}")
                        print(f"   NVDA Pos:    {portfolio_metrics.get('portfolio_nvda_position', 0)}")
                        print(f"   MSFT Pos:    {portfolio_metrics.get('portfolio_msft_position', 0)}")
                    
                    # Risk metrics
                    if risk_metrics:
                        print("\n🛡️ RISK STATUS:")
                        print(f"   Decisions:   {risk_metrics.get('risk_guard_decisions_total', 0)}")
                        print(f"   Approved:    {risk_metrics.get('risk_guard_approved_total', 0)}")
                        print(f"   Rejected:    {risk_metrics.get('risk_guard_rejected_total', 0)}")
                    
                    # Execution metrics
                    if execution_metrics:
                        print("\n📊 EXECUTION:")
                        print(f"   Orders:      {execution_metrics.get('ib_executor_orders_total', 0)}")
                        print(f"   Fills:       {execution_metrics.get('ib_executor_fills_total', 0)}")
                        connected = execution_metrics.get('ib_connected', '0') == '1'
                        print(f"   IB Status:   {'✅ Connected' if connected else '❌ Disconnected'}")
                    
                    print("="*60)
                
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in status dashboard: {e}")
                await asyncio.sleep(10)
    
    async def shutdown_services(self):
        """Gracefully shutdown all services"""
        
        logger.info("🛑 Shutting down Phase 2B services")
        
        for service_info in self.services:
            try:
                name = service_info["name"]
                process = service_info["process"]
                
                logger.info(f"🛑 Stopping {name}")
                
                # Send termination signal
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10)
                    logger.info(f"✅ {name} stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ {name} didn't stop gracefully, forcing...")
                    process.kill()
                    await asyncio.to_thread(process.wait)
                    logger.info(f"💥 {name} force stopped")
                    
            except Exception as e:
                logger.error(f"❌ Error stopping {service_info['name']}: {e}")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📨 Received signal {signum}")
        self.shutdown_event.set()
    
    async def run_phase2b(self):
        """Main function to run Phase 2B services"""
        
        logger.info("🚀 Starting Phase 2B Live Trading Pipeline")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        try:
            # Check prerequisites
            if not await self.check_prerequisites():
                return False
            
            # Start services in order
            logger.info("🚀 Starting services...")
            for config in self.service_configs:
                success = await self.start_service(config)
                if not success:
                    logger.error(f"❌ Failed to start {config['name']} - aborting")
                    await self.shutdown_services()
                    return False
            
            # Wait for all services to become healthy
            logger.info("🔍 Waiting for services to become healthy...")
            await asyncio.sleep(30)
            
            # Check final health
            all_healthy = True
            for service_info in self.services:
                health_ok = await self.check_service_health(service_info)
                if not health_ok:
                    logger.warning(f"⚠️ {service_info['name']} is not healthy")
                    all_healthy = False
            
            if all_healthy:
                logger.info("🎉 Phase 2B pipeline is fully operational!")
            else:
                logger.warning("⚠️ Some services are not healthy - proceeding with monitoring")
            
            # Start monitoring tasks
            monitoring_task = asyncio.create_task(self.monitor_services())
            dashboard_task = asyncio.create_task(self.display_status_dashboard())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cleanup
            monitoring_task.cancel()
            dashboard_task.cancel()
            await self.shutdown_services()
            
            logger.info("✅ Phase 2B shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 2B orchestration error: {e}")
            await self.shutdown_services()
            return False

async def main():
    """Main entry point"""
    
    orchestrator = Phase2BOrchestrator()
    
    try:
        success = await orchestrator.run_phase2b()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        logger.info("🛑 Phase 2B stopped by user")
        exit_code = 0
    except Exception as e:
        logger.error(f"❌ Phase 2B error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())