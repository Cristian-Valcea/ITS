#!/usr/bin/env python3
"""
🖥️ REAL-TIME TRAINING MONITOR DASHBOARD
Live monitoring dashboard for 8-cycle training progress

FEATURES:
- Real-time step counter and progress bar
- Cycle-by-cycle performance tracking
- ETA calculations and speed monitoring
- Memory and system resource tracking
- Live checkpoint validation
"""

import time
import sys
import os
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_system_stats():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3)
    }

def monitor_training_logs():
    """Monitor training log files for progress updates"""
    log_pattern = "complete_8cycle_training_*.log"
    log_files = list(Path(".").glob(log_pattern))
    
    if not log_files:
        return None
        
    # Get the most recent log file
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            
        # Parse key information from log
        training_info = {
            'current_cycle': 0,
            'current_step': 0,
            'total_steps': 48000,
            'status': 'Starting...',
            'last_update': datetime.now()
        }
        
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if 'STARTING CYCLE' in line:
                try:
                    cycle_num = int(line.split('CYCLE ')[1].split('/')[0])
                    training_info['current_cycle'] = cycle_num
                    training_info['status'] = f'Training Cycle {cycle_num}'
                except:
                    pass
                    
            elif 'Progress:' in line and '/48,000' in line:
                try:
                    step_part = line.split('Progress: ')[1].split('/48,000')[0]
                    current_step = int(step_part.replace(',', ''))
                    training_info['current_step'] = current_step
                except:
                    pass
                    
            elif 'COMPLETED' in line:
                training_info['status'] = 'Completed'
                
        return training_info
        
    except Exception as e:
        return None

def display_dashboard():
    """Display real-time training dashboard"""
    start_time = time.time()
    
    while True:
        clear_screen()
        
        # Header
        print("🚀 STAIRWAYS TO HEAVEN V3 - TRAINING MONITOR")
        print("=" * 80)
        print(f"📅 Monitor Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Monitor Uptime: {str(timedelta(seconds=int(time.time() - start_time)))}")
        print("=" * 80)
        
        # Training Progress
        training_info = monitor_training_logs()
        
        if training_info:
            current_step = training_info['current_step']
            total_steps = training_info['total_steps']
            current_cycle = training_info['current_cycle']
            
            # Progress calculations
            progress_pct = (current_step / total_steps) * 100
            
            # Progress bar
            bar_length = 50
            filled_length = int(bar_length * progress_pct / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"🎯 TRAINING PROGRESS")
            print(f"   Current Cycle: {current_cycle}/8")
            print(f"   Training Steps: {current_step:,}/{total_steps:,}")
            print(f"   Progress: [{bar}] {progress_pct:.1f}%")
            print(f"   Status: {training_info['status']}")
            
            # ETA calculation
            if current_step > 0:
                elapsed = time.time() - start_time
                steps_per_second = current_step / elapsed
                if steps_per_second > 0:
                    eta_seconds = (total_steps - current_step) / steps_per_second
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    print(f"   ETA: {eta_str}")
                    print(f"   Speed: {steps_per_second:.1f} steps/sec")
            
        else:
            print("🔍 WAITING FOR TRAINING TO START...")
            print("   No training log detected yet")
            print("   Run: python start_complete_8cycle_training.py")
        
        print("=" * 80)
        
        # System Resources
        sys_stats = get_system_stats()
        print(f"💻 SYSTEM RESOURCES")
        print(f"   CPU Usage: {sys_stats['cpu']:.1f}%")
        print(f"   Memory: {sys_stats['memory_used_gb']:.1f}GB / {sys_stats['memory_total_gb']:.1f}GB ({sys_stats['memory_percent']:.1f}%)")
        
        # Check for training directories
        training_dirs = list(Path("train_runs").glob("stairways_8cycle_*")) if Path("train_runs").exists() else []
        if training_dirs:
            latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
            print(f"   Training Dir: {latest_dir.name}")
            
            # Check for checkpoints
            checkpoints = list(latest_dir.rglob("*.zip"))
            if checkpoints:
                print(f"   Checkpoints: {len(checkpoints)} models saved")
        
        print("=" * 80)
        
        # Control Information
        print(f"🎛️ MONITOR CONTROLS")
        print(f"   Press Ctrl+C to exit monitor")
        print(f"   Training controls available in training terminal")
        
        print("=" * 80)
        
        # Live Updates
        print(f"🔄 Last Update: {datetime.now().strftime('%H:%M:%S')} (refreshing every 5 seconds)")
        
        try:
            time.sleep(5)  # Update every 5 seconds
        except KeyboardInterrupt:
            print("\n\n👋 Training monitor stopped by user")
            break

if __name__ == "__main__":
    print("🖥️ Starting Training Monitor Dashboard...")
    print("📊 This will show live progress of 8-cycle training")
    print("⚡ Updates every 5 seconds")
    print("\nPress Ctrl+C to exit\n")
    
    time.sleep(2)
    display_dashboard()