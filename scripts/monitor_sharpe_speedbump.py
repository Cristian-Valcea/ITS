#!/usr/bin/env python3
"""
🚨 SPEED-BUMP MONITORING FOR PHASE 1B
Real-time Sharpe ratio monitoring with auto-abort capability
"""

import argparse
import time
import logging
import numpy as np
from collections import deque
from pathlib import Path
import signal
import sys
import os

class SpeedBumpMonitor:
    """Real-time training monitor with auto-abort on Sharpe degradation"""
    
    def __init__(self, variant: str, log_file: str, abort_threshold: float = -0.5, 
                 min_steps: int = 30000, check_interval: int = 1000):
        self.variant = variant
        self.abort_threshold = abort_threshold
        self.min_steps = min_steps
        self.check_interval = check_interval
        
        # Setup logging
        self.logger = logging.getLogger(f"SpeedBump-{variant}")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Monitoring state
        self.rewards_window = deque(maxlen=1000)  # Rolling 1K window
        self.steps_completed = 0
        self.should_abort = False
        self.abort_reason = None
        
        self.logger.info(f"🚨 Speed-bump monitor initialized for Variant {variant}")
        self.logger.info(f"   Abort threshold: {abort_threshold}")
        self.logger.info(f"   Min steps before abort: {min_steps}")
        self.logger.info(f"   Check interval: {check_interval}")
        
    def update_reward(self, reward: float, step: int):
        """Update reward tracking and check for abort conditions"""
        self.rewards_window.append(reward)
        self.steps_completed = step
        
        # Only check abort conditions after minimum steps and sufficient data
        if step >= self.min_steps and len(self.rewards_window) >= 100:
            if step % self.check_interval == 0:
                self._check_abort_conditions(step)
    
    def _check_abort_conditions(self, step: int):
        """Check if training should be aborted based on Sharpe ratio"""
        if len(self.rewards_window) < 50:
            return
            
        rewards = np.array(self.rewards_window)
        
        # Calculate rolling Sharpe ratio
        if np.std(rewards) > 0:
            sharpe = np.mean(rewards) / np.std(rewards)
        else:
            sharpe = 0.0
            
        self.logger.info(f"📊 Step {step}: Rolling Sharpe = {sharpe:.4f} (window={len(rewards)})")
        
        # Check abort condition
        if sharpe < self.abort_threshold:
            self.should_abort = True
            self.abort_reason = f"Sharpe ratio {sharpe:.4f} below threshold {self.abort_threshold}"
            
            self.logger.error(f"🚨 SPEED-BUMP ABORT TRIGGERED at step {step}")
            self.logger.error(f"   Reason: {self.abort_reason}")
            self.logger.error(f"   Rolling Sharpe: {sharpe:.4f}")
            self.logger.error(f"   Mean reward: {np.mean(rewards):.4f}")
            self.logger.error(f"   Reward std: {np.std(rewards):.4f}")
            
            # Signal parent process to abort
            self._trigger_abort()
    
    def _trigger_abort(self):
        """Trigger training abort by signaling parent process"""
        try:
            # Find parent training process and send SIGTERM
            parent_pid = os.getppid()
            self.logger.info(f"🚨 Sending abort signal to parent process {parent_pid}")
            os.kill(parent_pid, signal.SIGTERM)
        except Exception as e:
            self.logger.error(f"Failed to signal parent process: {e}")
    
    def monitor_log_file(self, log_pattern: str):
        """Monitor training log file for reward updates"""
        self.logger.info(f"🔍 Starting log file monitoring for pattern: {log_pattern}")
        
        # This would be implemented to tail the training log and extract rewards
        # For now, we'll use a simple polling approach
        last_position = 0
        
        while not self.should_abort:
            try:
                # Look for training log files
                log_files = list(Path("diagnostic_runs").glob("**/logs/*.log"))
                
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    
                    with open(latest_log, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        # Parse reward information from log lines
                        for line in new_lines:
                            if "ep_rew_mean" in line:
                                try:
                                    # Extract reward from stable-baselines3 log format
                                    parts = line.split("|")
                                    for part in parts:
                                        if "ep_rew_mean" in part:
                                            reward_str = part.split()[-1]
                                            reward = float(reward_str)
                                            
                                            # Extract step count
                                            step = self._extract_step_from_log(line)
                                            if step:
                                                self.update_reward(reward, step)
                                            break
                                except (ValueError, IndexError) as e:
                                    continue
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring log file: {e}")
                time.sleep(10)
    
    def _extract_step_from_log(self, log_line: str) -> int:
        """Extract step count from training log line"""
        try:
            # Look for total_timesteps in the log line
            if "total_timesteps" in log_line:
                parts = log_line.split("|")
                for part in parts:
                    if "total_timesteps" in part:
                        step_str = part.split()[-1]
                        return int(step_str)
        except:
            pass
        return None

def main():
    parser = argparse.ArgumentParser(description="Speed-bump monitoring for Phase 1B")
    parser.add_argument("--variant", required=True, help="Variant name (A, B, or C)")
    parser.add_argument("--log-file", required=True, help="Monitor log file path")
    parser.add_argument("--abort-threshold", type=float, default=-0.5, 
                       help="Sharpe threshold for abort")
    parser.add_argument("--min-steps", type=int, default=30000,
                       help="Minimum steps before abort can trigger")
    parser.add_argument("--check-interval", type=int, default=1000,
                       help="Steps between abort checks")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SpeedBumpMonitor(
        variant=args.variant,
        log_file=args.log_file,
        abort_threshold=args.abort_threshold,
        min_steps=args.min_steps,
        check_interval=args.check_interval
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        monitor.logger.info(f"🛑 Monitor received signal {signum}, shutting down")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    try:
        monitor.monitor_log_file("ep_rew_mean")
    except KeyboardInterrupt:
        monitor.logger.info("🛑 Monitor stopped by user")
    except Exception as e:
        monitor.logger.error(f"Monitor failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()