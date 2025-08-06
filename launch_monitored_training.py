#!/usr/bin/env python3
"""
🚀 LAUNCH STAIRWAYS V4 TRAINING WITH MONITORING
Start training with real-time progress tracking and monitoring instructions
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def main():
    print("🎯 STAIRWAYS V4 - 200K REAL DATA TRAINING LAUNCHER")
    print("=" * 60)
    
    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"stairways_v4_200k_{timestamp}"
    
    print(f"🎪 Run Name: {run_name}")
    print(f"📊 Expected Duration: 2-3 hours")
    print(f"🔧 Policy Updates: ~390 (vs failed 48K with only 23)")
    print()
    
    # Monitoring instructions
    print("📊 REAL-TIME MONITORING OPTIONS:")
    print("=" * 40)
    print("1. 📈 TensorBoard (live metrics):")
    print(f"   tensorboard --logdir tensorboard_logs/{run_name}")
    print()
    print("2. 🔍 Progress Monitor (detailed status):")
    print(f"   python training_monitor.py {run_name}")
    print()
    print("3. 📄 Milestone Reports (saved every 20K steps):")
    print(f"   tail -f train_runs/{run_name}/progress.log")
    print()
    print("4. 💾 Checkpoints (saved every 25K steps):")
    print(f"   ls -la train_runs/{run_name}/*/")
    print()
    
    # Progress schedule
    print("⏰ TRAINING SCHEDULE:")
    print("=" * 40)
    print("Phase 1 (Warm-up):     0K →  50K steps (high exploration)")
    print("Phase 2 (Core):       50K → 170K steps (balanced learning)")  
    print("Phase 3 (Refinement): 170K → 200K steps (fine-tuning)")
    print()
    print("📊 Progress Reports Every:")
    print("  • 5K steps: Quick progress update")
    print("  • 10K steps: Detailed metrics log")
    print("  • 20K steps: Full milestone report")
    print("  • 25K steps: Model checkpoint saved")
    print()
    
    # Ask for confirmation
    response = input("🚀 Ready to launch training? (y/n): ").lower().strip()
    
    if response != 'y':
        print("❌ Training cancelled by user")
        return False
    
    print("\n🎪 LAUNCHING TRAINING...")
    print("💡 Open another terminal for monitoring!")
    print("=" * 60)
    
    # Start training in the background with output capture
    try:
        # Create log file for training output
        log_file = f"train_runs/{run_name}/training_output.log"
        os.makedirs(f"train_runs/{run_name}", exist_ok=True)
        
        # Launch training
        cmd = ["python", "train_200k_stairways_v4.py"]
        
        print(f"🎯 Training started! Output logged to: {log_file}")
        print(f"📊 Monitor with: python training_monitor.py {run_name}")
        
        # Start the training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and file
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line.rstrip())  # Print to console
                f.write(line)         # Write to log file
                f.flush()             # Ensure immediate write
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ STAIRWAYS V4 TRAINING: COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved in: train_runs/{run_name}")
        else:
            print(f"\n❌ STAIRWAYS V4 TRAINING: FAILED (exit code {return_code})")
            
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Training launch failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)