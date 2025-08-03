#!/usr/bin/env python3
"""Monitor V3 training progress"""

import os
import time
import psutil
from datetime import datetime

def monitor_training():
    """Monitor the training process"""
    
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'chunk_driver_v3.py' in ' '.join(proc.info['cmdline']):
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if not training_pid:
        print("❌ Training process not found")
        return
    
    print(f"📊 MONITORING TRAINING PROCESS (PID: {training_pid})")
    print("=" * 60)
    
    try:
        process = psutil.Process(training_pid)
        start_time = datetime.now()
        
        while True:
            # Get process stats
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Check if process is still running
            if not process.is_running():
                print("❌ Training process has stopped")
                break
            
            # Runtime
            runtime = datetime.now() - start_time
            runtime_minutes = runtime.total_seconds() / 60
            
            # Check for checkpoint files
            checkpoint_dir = "train_runs/v3_gold_standard_400k_20250802_202736/checkpoints"
            checkpoint_files = []
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
            
            # Display status
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\r🕐 {current_time} | Runtime: {runtime_minutes:.1f}m | CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.0f}MB | Checkpoints: {len(checkpoint_files)}", end="", flush=True)
            
            # Check for completion indicators
            if len(checkpoint_files) > 0:
                print(f"\n✅ Checkpoints found: {checkpoint_files}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor_training()