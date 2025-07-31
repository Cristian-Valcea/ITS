#!/usr/bin/env python3
"""
⚡ Quick Training Status Check
Fast overview of training progress without full dashboard
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

def check_session():
    """Quick tmux session check"""
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', 'training200k'], 
                              capture_output=True)
        return result.returncode == 0
    except:
        return False

def get_latest_checkpoint():
    """Get latest checkpoint info"""
    checkpoint_dir = "models/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith('dual_ticker_200k_') and f.endswith('.zip')]
    
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    
    # Extract steps from filename
    steps = 0
    try:
        parts = latest.split('_')
        for part in parts:
            if part.endswith('steps.zip'):
                steps = int(part.replace('steps.zip', ''))
                break
    except:
        pass
    
    return {
        'filename': latest,
        'steps': steps,
        'progress_pct': (steps / 200000) * 100 if steps > 0 else 0
    }

def get_log_tail():
    """Get last few lines from training log"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return []
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('200k_training_')]
    if not log_files:
        return []
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(log_dir, f)))
    log_path = os.path.join(log_dir, latest_log)
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines[-3:]]  # Last 3 lines
    except:
        return []

def main():
    print("⚡ QUICK TRAINING STATUS")
    print("=" * 30)
    
    # Session status
    session_active = check_session()
    print(f"🔄 Training Session: {'✅ ACTIVE' if session_active else '❌ INACTIVE'}")
    
    # Checkpoint status
    checkpoint = get_latest_checkpoint()
    if checkpoint:
        print(f"💾 Latest Checkpoint: {checkpoint['filename']}")
        print(f"📊 Progress: {checkpoint['steps']:,} steps ({checkpoint['progress_pct']:.1f}%)")
        
        # Estimate remaining time
        if checkpoint['steps'] > 0:
            remaining_steps = 200000 - checkpoint['steps']
            estimated_hours = remaining_steps / 25000  # ~25K steps/hour on RTX 3060
            print(f"⏰ Estimated Remaining: ~{estimated_hours:.1f} hours")
    else:
        print("💾 No checkpoints found yet")
    
    # Recent logs
    recent_logs = get_log_tail()
    if recent_logs:
        print(f"\n📋 Recent Log Entries:")
        for log_line in recent_logs:
            if log_line.strip():
                print(f"   {log_line}")
    
    # Quick commands
    print(f"\n🛠️  Quick Commands:")
    print(f"   Full dashboard:  python training_control_dashboard.py")
    print(f"   Monitor options: ./monitor_training.sh")
    print(f"   Attach session:  tmux attach -t training200k")
    print(f"   TensorBoard:     tensorboard --logdir runs/")

if __name__ == "__main__":
    main()