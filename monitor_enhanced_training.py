#!/usr/bin/env python3
"""
📊 ENHANCED TRAINING MONITOR
Real-time monitoring of enhanced training progress
"""

import time
import os
import subprocess
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_training_status():
    """Check if enhanced training is still running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train_50k_ENHANCED.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_latest_log_file():
    """Find the latest enhanced training log file"""
    try:
        log_files = [f for f in os.listdir('.') if f.startswith('enhanced_training_') and f.endswith('.log')]
        if log_files:
            return max(log_files, key=os.path.getctime)
    except:
        pass
    return None

def monitor_progress():
    """Monitor training progress in real-time"""
    logger.info("🔍 ENHANCED TRAINING MONITOR STARTED")
    
    log_file = get_latest_log_file()
    if not log_file:
        logger.error("❌ No enhanced training log file found")
        return
    
    logger.info(f"📊 Monitoring log file: {log_file}")
    
    # Track last position in file
    last_position = 0
    
    while True:
        # Check if training is still running
        if not check_training_status():
            logger.info("🏁 Training process completed!")
            break
        
        try:
            # Read new content from log file
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_content = f.read()
                last_position = f.tell()
            
            # Look for important progress indicators
            lines = new_content.split('\n')
            for line in lines:
                if line.strip():
                    # Portfolio updates
                    if "💰 Portfolio:" in line and "AFTER CHUNK" in line:
                        logger.info(f"📊 {line.strip()}")
                    
                    # Chunk completions
                    elif "✅ Completed chunk:" in line:
                        logger.info(f"🎯 {line.strip()}")
                    
                    # Drawdown warnings
                    elif "Episode terminated (TRAINING): Drawdown" in line:
                        logger.warning(f"⚠️ {line.strip()}")
                    
                    # Final completion
                    elif "🎉 ENHANCED PROFIT-MAXIMIZING TRAINING COMPLETED!" in line:
                        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                        return
                        
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
        
        # Wait before next check
        time.sleep(30)  # Check every 30 seconds
    
    # Final status check
    logger.info("📊 FINAL STATUS CHECK:")
    
    # Check for final model files
    if os.path.exists('models/dual_ticker_enhanced_50k_final.zip'):
        model_size = os.path.getsize('models/dual_ticker_enhanced_50k_final.zip')
        model_time = datetime.fromtimestamp(os.path.getmtime('models/dual_ticker_enhanced_50k_final.zip'))
        logger.info(f"✅ Final model saved: {model_size/1024/1024:.1f}MB at {model_time}")
    else:
        logger.warning("❌ Final model file not found")
    
    # Check checkpoints
    checkpoint_dir = 'models/checkpoints'
    if os.path.exists(checkpoint_dir):
        enhanced_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                              if f.startswith('dual_ticker_enhanced_') and f.endswith('.zip')]
        logger.info(f"📁 Checkpoints found: {len(enhanced_checkpoints)}")
        for checkpoint in sorted(enhanced_checkpoints)[-3:]:  # Show last 3
            logger.info(f"   📂 {checkpoint}")
    
    logger.info("🔍 MONITORING COMPLETED")

if __name__ == "__main__":
    monitor_progress()