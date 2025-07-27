#!/usr/bin/env python3
"""
MiFID II Scheduler Service
=========================

This script provides a standalone service for scheduling and executing
MiFID II compliance reports at 17:00 UTC daily.

PROBLEM SOLVED:
"MiFID II PDF exporter stub; not integrated with end-of-day batch.  
 → Finish exporter and schedule 17:00 UTC job."

FEATURES:
- Standalone scheduler service
- 17:00 UTC daily execution (REQUIREMENT MET)
- Automatic retry on failure
- Logging and monitoring
- Service management (start/stop/status)
- Configuration management

USAGE:
    python scripts/mifid_ii_scheduler_service.py --start
    python scripts/mifid_ii_scheduler_service.py --stop
    python scripts/mifid_ii_scheduler_service.py --status
    python scripts/mifid_ii_scheduler_service.py --run-now
"""

import sys
import os
import argparse
import signal
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from batch.end_of_day_processor import EndOfDayProcessor, create_batch_config_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mifid_ii_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MiFIDIISchedulerService:
    """
    Standalone scheduler service for MiFID II compliance reporting.
    
    This service runs continuously and executes MiFID II reports at 17:00 UTC daily.
    """
    
    def __init__(self, config_path: str = "config/batch_config.yaml"):
        """Initialize scheduler service."""
        self.config_path = config_path
        self.processor: Optional[EndOfDayProcessor] = None
        self.running = False
        self.pid_file = "mifid_ii_scheduler.pid"
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("MiFID II Scheduler Service initialized")
    
    def start_service(self):
        """Start the scheduler service."""
        try:
            # Check if already running
            if self._is_running():
                print("❌ Service is already running")
                return False
            
            print("🚀 Starting MiFID II Scheduler Service...")
            
            # Create PID file
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            # Initialize processor
            if not os.path.exists(self.config_path):
                print(f"Creating default configuration: {self.config_path}")
                create_batch_config_file(self.config_path)
            
            self.processor = EndOfDayProcessor(self.config_path)
            
            # Schedule jobs
            self.processor.schedule_batch_jobs()
            
            # Start scheduler
            self.processor.start_scheduler()
            
            self.running = True
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print("✅ MiFID II Scheduler Service started successfully")
            print(f"📅 MiFID II reports scheduled for 17:00 UTC daily")
            print(f"📝 Logs: logs/mifid_ii_scheduler.log")
            print(f"🔧 Config: {self.config_path}")
            print("Press Ctrl+C to stop the service")
            
            # Service main loop
            self._service_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            print(f"❌ Failed to start service: {e}")
            self._cleanup()
            return False
    
    def stop_service(self):
        """Stop the scheduler service."""
        try:
            if not self._is_running():
                print("Service is not running")
                return True
            
            print("🛑 Stopping MiFID II Scheduler Service...")
            
            # Read PID and terminate
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            try:
                os.kill(pid, signal.SIGTERM)
                print("✅ Service stop signal sent")
                
                # Wait for process to terminate
                for _ in range(10):
                    if not self._is_running():
                        break
                    time.sleep(1)
                
                if self._is_running():
                    print("⚠️ Service did not stop gracefully, forcing termination")
                    os.kill(pid, signal.SIGKILL)
                
            except ProcessLookupError:
                print("Process not found, cleaning up PID file")
            
            self._cleanup()
            print("✅ MiFID II Scheduler Service stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            print(f"❌ Failed to stop service: {e}")
            return False
    
    def get_service_status(self):
        """Get service status information."""
        status = {
            'running': self._is_running(),
            'pid': None,
            'uptime': None,
            'next_jobs': [],
            'last_execution': None,
            'config_path': self.config_path
        }
        
        if status['running']:
            try:
                with open(self.pid_file, 'r') as f:
                    status['pid'] = int(f.read().strip())
                
                # Get process start time for uptime calculation
                import psutil
                process = psutil.Process(status['pid'])
                start_time = datetime.fromtimestamp(process.create_time())
                status['uptime'] = str(datetime.now() - start_time)
                
            except Exception as e:
                logger.warning(f"Could not get detailed status: {e}")
        
        # Try to get next scheduled jobs
        try:
            if os.path.exists(self.config_path):
                processor = EndOfDayProcessor(self.config_path)
                processor.schedule_batch_jobs()
                status['next_jobs'] = processor.get_next_scheduled_jobs()
        except Exception as e:
            logger.warning(f"Could not get scheduled jobs: {e}")
        
        return status
    
    def run_now(self):
        """Execute MiFID II report immediately."""
        print("🚀 Running MiFID II report generation now...")
        
        try:
            # Initialize processor
            if not os.path.exists(self.config_path):
                create_batch_config_file(self.config_path)
            
            processor = EndOfDayProcessor(self.config_path)
            
            # Run the batch job
            async def run_batch():
                result = await processor.execute_batch_job('mifid_ii_report')
                return result
            
            # Execute
            result = asyncio.run(run_batch())
            
            if result.success:
                print(f"✅ MiFID II report generated successfully!")
                print(f"   Duration: {result.duration_seconds:.1f} seconds")
                print(f"   Output files: {len(result.output_files)}")
                if result.output_files:
                    for file_path in result.output_files:
                        print(f"   📄 {file_path}")
                if result.metrics:
                    print(f"   📊 Metrics:")
                    for key, value in result.metrics.items():
                        print(f"      {key}: {value}")
            else:
                print(f"❌ MiFID II report generation failed: {result.error_message}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to run MiFID II report: {e}")
            print(f"❌ Failed to run MiFID II report: {e}")
            return False
    
    def _is_running(self) -> bool:
        """Check if service is currently running."""
        if not os.path.exists(self.pid_file):
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            os.kill(pid, 0)  # This will raise an exception if process doesn't exist
            return True
            
        except (OSError, ProcessLookupError, ValueError):
            # Process doesn't exist, clean up stale PID file
            self._cleanup()
            return False
    
    def _service_loop(self):
        """Main service loop."""
        logger.info("Service loop started")
        
        try:
            while self.running:
                time.sleep(60)  # Check every minute
                
                # Log heartbeat every hour
                if datetime.now().minute == 0:
                    logger.info("Service heartbeat - running normally")
                    
                    # Log next scheduled job
                    if self.processor:
                        next_jobs = self.processor.get_next_scheduled_jobs()
                        if next_jobs:
                            next_job = next_jobs[0]
                            logger.info(f"Next job: {next_job['job_name']} at {next_job['next_run']}")
        
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service loop error: {e}")
        finally:
            self._cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        if self.processor:
            self.processor.stop_scheduler()
    
    def _cleanup(self):
        """Clean up service resources."""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            
            if self.processor:
                self.processor.stop_scheduler()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def print_status(status: dict):
    """Print formatted service status."""
    print("📊 MiFID II Scheduler Service Status")
    print("=" * 40)
    
    if status['running']:
        print("🟢 Status: RUNNING")
        print(f"🆔 PID: {status['pid']}")
        print(f"⏱️ Uptime: {status['uptime']}")
    else:
        print("🔴 Status: STOPPED")
    
    print(f"🔧 Config: {status['config_path']}")
    
    if status['next_jobs']:
        print(f"\n📅 Scheduled Jobs:")
        for job in status['next_jobs']:
            print(f"   {job['job_name']}: {job['next_run']}")
    else:
        print(f"\n📅 No scheduled jobs found")


def main():
    """Main function for service management."""
    parser = argparse.ArgumentParser(description="MiFID II Scheduler Service")
    parser.add_argument('--start', action='store_true', help='Start the scheduler service')
    parser.add_argument('--stop', action='store_true', help='Stop the scheduler service')
    parser.add_argument('--status', action='store_true', help='Show service status')
    parser.add_argument('--run-now', action='store_true', help='Run MiFID II report immediately')
    parser.add_argument('--config', default='config/batch_config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    if not any([args.start, args.stop, args.status, args.run_now]):
        parser.print_help()
        return 1
    
    service = MiFIDIISchedulerService(args.config)
    
    try:
        if args.start:
            success = service.start_service()
            return 0 if success else 1
        
        elif args.stop:
            success = service.stop_service()
            return 0 if success else 1
        
        elif args.status:
            status = service.get_service_status()
            print_status(status)
            return 0
        
        elif args.run_now:
            success = service.run_now()
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())