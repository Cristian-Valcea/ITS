#!/usr/bin/env python3
"""
📊 REAL-TIME TRAINING MONITOR
Track training progress, metrics, and partial results every 20K steps
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

class TrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.start_time = time.time()
        self.checkpoints = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Monitoring paths
        self.monitor_file = f"train_runs/{run_name}/monitor.csv"
        self.progress_file = f"train_runs/{run_name}/progress.log"
        
    def log_progress(self, current_step: int, total_steps: int, phase: str, metrics: dict = None):
        """Log current training progress"""
        
        elapsed = time.time() - self.start_time
        progress_pct = (current_step / total_steps) * 100
        
        # Estimate remaining time
        if current_step > 0:
            time_per_step = elapsed / current_step
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps * time_per_step
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "calculating..."
        
        # Progress message
        msg = f"🎯 {phase} | Step {current_step:,}/{total_steps:,} ({progress_pct:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}"
        
        if metrics:
            msg += f" | Metrics: {metrics}"
        
        self.logger.info(msg)
        
        # Save to progress file
        with open(self.progress_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")
    
    def analyze_episode_rewards(self) -> dict:
        """Analyze recent episode rewards from monitor.csv"""
        
        if not os.path.exists(self.monitor_file):
            return {"status": "no_data", "message": "Monitor file not found"}
        
        try:
            df = pd.read_csv(self.monitor_file)
            
            if len(df) == 0:
                return {"status": "no_episodes", "message": "No episodes completed yet"}
            
            # Get recent episodes (last 10)
            recent_df = df.tail(10)
            
            metrics = {
                "status": "active",
                "total_episodes": len(df),
                "recent_episodes": len(recent_df),
                "avg_reward": recent_df['r'].mean(),
                "avg_length": recent_df['l'].mean(),
                "best_reward": df['r'].max(),
                "worst_reward": df['r'].min(),
                "reward_trend": "improving" if recent_df['r'].iloc[-3:].mean() > recent_df['r'].iloc[:3].mean() else "declining"
            }
            
            return metrics
            
        except Exception as e:
            return {"status": "error", "message": f"Analysis failed: {e}"}
    
    def check_tensorboard_logs(self) -> dict:
        """Check if TensorBoard logs are being generated"""
        
        tb_path = f"tensorboard_logs/{self.run_name}"
        
        if not os.path.exists(tb_path):
            return {"status": "no_logs", "message": "TensorBoard logs not found"}
        
        # Count log files
        log_files = list(Path(tb_path).rglob("events.out.tfevents.*"))
        
        if not log_files:
            return {"status": "no_events", "message": "No TensorBoard events found"}
        
        # Get latest log file info
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        file_age = time.time() - latest_log.stat().st_mtime
        
        return {
            "status": "active",
            "log_files": len(log_files),
            "latest_log": str(latest_log.name),
            "last_updated": f"{file_age:.0f}s ago",
            "tensorboard_cmd": f"tensorboard --logdir {tb_path}"
        }
    
    def generate_progress_report(self, current_step: int, total_steps: int, phase: str) -> str:
        """Generate comprehensive progress report"""
        
        # Basic progress
        progress_pct = (current_step / total_steps) * 100
        elapsed = time.time() - self.start_time
        
        report = []
        report.append("=" * 60)
        report.append(f"🎯 STAIRWAYS V4 TRAINING PROGRESS REPORT")
        report.append("=" * 60)
        report.append(f"Run: {self.run_name}")
        report.append(f"Current Phase: {phase}")
        report.append(f"Progress: {current_step:,}/{total_steps:,} steps ({progress_pct:.1f}%)")
        report.append(f"Elapsed Time: {elapsed/3600:.1f} hours")
        report.append("")
        
        # Episode analysis
        episode_metrics = self.analyze_episode_rewards()
        report.append("📊 EPISODE METRICS:")
        if episode_metrics["status"] == "active":
            report.append(f"  Total Episodes: {episode_metrics['total_episodes']}")
            report.append(f"  Recent Avg Reward: {episode_metrics['avg_reward']:.3f}")
            report.append(f"  Recent Avg Length: {episode_metrics['avg_length']:.0f} steps")
            report.append(f"  Best/Worst Reward: {episode_metrics['best_reward']:.3f} / {episode_metrics['worst_reward']:.3f}")
            report.append(f"  Reward Trend: {episode_metrics['reward_trend']}")
        else:
            report.append(f"  Status: {episode_metrics['message']}")
        report.append("")
        
        # TensorBoard status
        tb_status = self.check_tensorboard_logs()
        report.append("📈 TENSORBOARD STATUS:")
        if tb_status["status"] == "active":
            report.append(f"  Log Files: {tb_status['log_files']}")
            report.append(f"  Last Updated: {tb_status['last_updated']}")
            report.append(f"  View Command: {tb_status['tensorboard_cmd']}")
        else:
            report.append(f"  Status: {tb_status['message']}")
        report.append("")
        
        # Checkpoints
        report.append("💾 CHECKPOINTS:")
        checkpoint_dir = Path(f"train_runs/{self.run_name}")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.rglob("*_model_*.zip"))
            if checkpoints:
                for cp in sorted(checkpoints)[-3:]:  # Show last 3
                    size_mb = cp.stat().st_size / (1024*1024)
                    report.append(f"  {cp.name} ({size_mb:.1f}MB)")
            else:
                report.append("  No checkpoints saved yet")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_milestone_report(self, current_step: int, total_steps: int, phase: str):
        """Save milestone report to file"""
        
        report = self.generate_progress_report(current_step, total_steps, phase)
        
        milestone_file = f"train_runs/{self.run_name}/milestone_{current_step//1000}k.txt"
        with open(milestone_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📄 Milestone report saved: {milestone_file}")
        
        # Also print to console
        print(report)

def monitor_training(run_name: str):
    """Monitor an active training run"""
    
    monitor = TrainingMonitor(run_name)
    
    print(f"🔍 Monitoring training run: {run_name}")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        step_counter = 0
        while True:
            # Check if training is still active
            progress_file = f"train_runs/{run_name}/progress.log"
            
            if os.path.exists(progress_file):
                # Read latest progress
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"Latest: {lines[-1].strip()}")
            
            # Generate report every 60 seconds
            if step_counter % 12 == 0:  # Every 12 * 5s = 60s
                episode_metrics = monitor.analyze_episode_rewards()
                if episode_metrics["status"] == "active":
                    print(f"📊 Episodes: {episode_metrics['total_episodes']}, "
                          f"Avg Reward: {episode_metrics['avg_reward']:.3f}, "
                          f"Trend: {episode_metrics['reward_trend']}")
            
            time.sleep(5)  # Check every 5 seconds
            step_counter += 1
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        monitor_training(run_name)
    else:
        print("Usage: python training_monitor.py <run_name>")
        print("Example: python training_monitor.py stairways_v4_200k_20250804_123456")