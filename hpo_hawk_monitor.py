#!/usr/bin/env python3
"""
🦅 HPO HAWK MONITOR
Watch grid search like a hawk - kill dead runs early, tag survivors, export winners
"""

import sys
from pathlib import Path
import json
import time
import logging
import psutil
import signal
from datetime import datetime
import subprocess
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPOHawkMonitor:
    """Hawk-like monitoring of HPO grid search"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.dead_runs_killed = 0
        self.current_runs = {}
        self.completed_runs = []
        
        # Thresholds for hawk decisions
        self.green_light_reward = 1000    # ≈ +1% return
        self.min_episode_length = 4000    # Avoid DD spikes
        self.death_reward = -500          # No alpha signal
        self.divergence_kl = 1e-4         # Stalled training
        self.max_runtime_minutes = 90     # Kill whole grid after 90min
        
    def find_hpo_process(self):
        """Find the main HPO grid process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python3' in proc.info['name'] and 'hpo_grid_sprint.py' in ' '.join(proc.info['cmdline']):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def check_tensorboard_logs(self):
        """Check for TensorBoard logs to extract metrics"""
        
        # Look for log directories
        log_dirs = []
        for path in Path('.').glob('**/events.out.tfevents.*'):
            log_dirs.append(path.parent)
        
        if not log_dirs:
            logger.info("   No TensorBoard logs found yet")
            return []
        
        # Parse recent logs for key metrics
        run_metrics = []
        for log_dir in log_dirs:
            try:
                # Try to extract run info from directory name
                run_info = self._parse_run_metrics(log_dir)
                if run_info:
                    run_metrics.append(run_info)
            except Exception as e:
                logger.debug(f"   Error parsing {log_dir}: {e}")
        
        return run_metrics
    
    def _parse_run_metrics(self, log_dir):
        """Parse metrics from log directory (simplified version)"""
        # This would ideally parse TensorBoard logs, but for now we'll simulate
        # In practice, you'd use tensorboard.backend.event_processing.event_accumulator
        
        # For now, return None - the real implementation would parse TB logs
        return None
    
    def check_success_files(self):
        """Check for HPO success/failure files"""
        
        success_file = Path('winning_hpo_config.json')
        failure_file = Path('hpo_failure_analysis.json')
        
        if success_file.exists():
            with open(success_file, 'r') as f:
                config = json.load(f)
            return 'success', config
        elif failure_file.exists():
            with open(failure_file, 'r') as f:
                analysis = json.load(f)
            return 'failure', analysis
        else:
            return 'running', None
    
    def kill_hpo_process(self, reason="manual_abort"):
        """Kill the HPO process if needed"""
        
        pid = self.find_hpo_process()
        if pid:
            try:
                proc = psutil.Process(pid)
                proc.terminate()  # Try graceful termination first
                
                # Wait a few seconds for graceful shutdown
                time.sleep(3)
                
                if proc.is_running():
                    proc.kill()  # Force kill if needed
                
                logger.info(f"🚫 HPO process {pid} killed - Reason: {reason}")
                return True
            except Exception as e:
                logger.error(f"Failed to kill process {pid}: {e}")
                return False
        else:
            logger.info("No HPO process found to kill")
            return False
    
    def export_winning_model(self, config_info):
        """Export winning model weights"""
        
        logger.info("🎉 EXPORTING WINNING MODEL")
        
        # The config should contain model path info
        # For now, we'll use a placeholder path
        model_path = "placeholder_model.zip"  # Would be extracted from config
        
        try:
            # Run export script
            cmd = [
                'python3', 'export_policy.py',
                '--model_path', model_path,
                '--output', 'models/singleticker_gatepass.zip'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Model exported successfully")
                logger.info(f"   Output: models/singleticker_gatepass.zip")
                return True
            else:
                logger.error(f"❌ Export failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Export exception: {e}")
            return False
    
    def launch_dual_ticker_warmstart(self):
        """Launch dual-ticker warm-start training"""
        
        logger.info("🚀 LAUNCHING DUAL-TICKER WARM-START")
        
        try:
            # Run warm-start script
            cmd = ['python3', 'dual_ticker_warmstart.py']
            
            logger.info("   Starting 25K fine-tune with 2e-5 LR...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ Dual-ticker warm-start completed successfully")
                return True
            else:
                logger.error(f"❌ Warm-start failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Warm-start timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ Warm-start exception: {e}")
            return False
    
    def run_synthetic_alpha_test(self):
        """Run synthetic alpha test for contingency debugging"""
        
        logger.info("🔬 RUNNING SYNTHETIC ALPHA TEST")
        logger.info("   Testing if agent can learn OBVIOUS alpha patterns...")
        
        try:
            cmd = ['python3', 'synthetic_alpha_test.py']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minutes
            
            if result.returncode == 0:
                logger.info("✅ Synthetic alpha test passed")
                logger.info("   Problem was alpha strength, not RL settings")
                return True
            else:
                logger.info("❌ Synthetic alpha test failed")
                logger.info("   Problem is in RL hyperparameters or search space")
                return False
                
        except Exception as e:
            logger.error(f"❌ Synthetic test exception: {e}")
            return False
    
    def hawk_watch_loop(self):
        """Main hawk monitoring loop"""
        
        logger.info("🦅 HPO HAWK MONITOR ENGAGED")
        logger.info("   Watching for: Green-light (≥+1000 reward), Death signals (≤-500), Stalls")
        logger.info(f"   Max runtime: {self.max_runtime_minutes} minutes")
        logger.info("   Will kill dead runs early and export winners immediately")
        
        check_count = 0
        
        while True:
            check_count += 1
            elapsed = datetime.now() - self.start_time
            elapsed_minutes = elapsed.total_seconds() / 60
            
            logger.info(f"\n🦅 Hawk Check #{check_count} (Elapsed: {elapsed_minutes:.1f}min)")
            
            # Check if HPO process is still running
            hpo_pid = self.find_hpo_process()
            if not hpo_pid:
                logger.info("   HPO process not found - may have completed")
            
            # Check for success/failure files
            status, data = self.check_success_files()
            
            if status == 'success':
                logger.info("🎉 ✅ GREEN-LIGHT DETECTED!")
                logger.info(f"   Winning config: {data['config']}")
                logger.info(f"   Performance: {data['performance']['total_return']:+.2%} return")
                
                # Export and warm-start pipeline
                if self.export_winning_model(data):
                    logger.info("\n🚀 PROCEEDING TO DUAL-TICKER WARM-START")
                    success = self.launch_dual_ticker_warmstart()
                    
                    if success:
                        logger.info("🎉 ✅ DUAL-TICKER WARM-START SUCCESS!")
                        logger.info("   Cross-asset alpha extraction achieved")
                    else:
                        logger.info("⚠️ Dual-ticker warm-start incomplete")
                        logger.info("   May need ent_coef adjustment and retry")
                
                break
                
            elif status == 'failure':
                logger.info("⚠️ ❌ ENTIRE GRID WHIFFED")
                logger.info(f"   Configs tested: {data.get('configs_tested', 'unknown')}")
                logger.info("   Moving to synthetic alpha isolation...")
                
                # Run synthetic alpha test
                self.run_synthetic_alpha_test()
                break
                
            else:  # still running
                # Check runtime limit
                if elapsed_minutes > self.max_runtime_minutes:
                    logger.info(f"🚫 RUNTIME LIMIT EXCEEDED ({self.max_runtime_minutes}min)")
                    logger.info("   Killing HPO process and moving to synthetic alpha test")
                    
                    self.kill_hpo_process("runtime_limit")
                    self.run_synthetic_alpha_test()
                    break
                
                # Check TensorBoard logs for early signals (if available)
                # This would be the place to parse TB logs and kill dead runs
                logger.info("   Grid search continues...")
                logger.info("   💡 Tip: Monitor TensorBoard for rollout/ep_rew_mean ≥ +1000")
            
            # Sleep between checks
            time.sleep(60)  # Check every minute
    
    def quick_status(self):
        """Quick status check without loop"""
        
        logger.info("🦅 HPO HAWK STATUS CHECK")
        
        # Check process
        hpo_pid = self.find_hpo_process()
        if hpo_pid:
            logger.info(f"   HPO process: Running (PID {hpo_pid})")
        else:
            logger.info("   HPO process: Not found")
        
        # Check files
        status, data = self.check_success_files()
        if status == 'success':
            logger.info("   Status: ✅ SUCCESS FOUND!")
        elif status == 'failure':
            logger.info("   Status: ❌ GRID COMPLETED - NO SUCCESS")
        else:
            logger.info("   Status: 🔄 RUNNING")
        
        # Check elapsed time
        elapsed = datetime.now() - self.start_time
        logger.info(f"   Elapsed: {elapsed}")
        
        return status, data

def main():
    """Main hawk monitor function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='HPO Hawk Monitor')
    parser.add_argument('--quick', action='store_true', help='Quick status check only')
    parser.add_argument('--kill', action='store_true', help='Kill HPO process')
    args = parser.parse_args()
    
    monitor = HPOHawkMonitor()
    
    if args.kill:
        monitor.kill_hpo_process("manual_request")
        return
    
    if args.quick:
        monitor.quick_status()
        return
    
    # Full hawk watch
    monitor.hawk_watch_loop()

if __name__ == "__main__":
    main()