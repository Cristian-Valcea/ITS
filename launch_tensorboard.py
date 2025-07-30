#!/usr/bin/env python3
"""
📊 LAUNCH TENSORBOARD
Quick launcher for TensorBoard to monitor HPO grid search
"""

import sys
import subprocess
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_log_directories():
    """Find TensorBoard log directories"""
    
    log_dirs = []
    
    # Look for common log directory patterns
    search_patterns = [
        'logs/**/events.out.tfevents.*',
        '**/events.out.tfevents.*',
        'runs/**/events.out.tfevents.*',
        'tensorboard_logs/**/events.out.tfevents.*'
    ]
    
    for pattern in search_patterns:
        for path in Path('.').glob(pattern):
            log_dir = path.parent
            if log_dir not in log_dirs:
                log_dirs.append(log_dir)
    
    return log_dirs

def main():
    """Main TensorBoard launcher"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Launch TensorBoard for HPO monitoring')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard port')
    parser.add_argument('--scan', action='store_true', help='Scan for log directories only')
    
    args = parser.parse_args()
    
    if args.scan:
        # Just scan and report
        log_dirs = find_log_directories()
        if log_dirs:
            logger.info(f"📊 Found {len(log_dirs)} TensorBoard log directories:")
            for i, log_dir in enumerate(log_dirs):
                logger.info(f"   {i+1}. {log_dir}")
        else:
            logger.info("📊 No TensorBoard log directories found yet")
            logger.info("   HPO training may not have started logging")
        return
    
    # Try to launch TensorBoard (simplified version)
    try:
        cmd = ['tensorboard', '--logdir', '.', '--port', str(args.port)]
        logger.info(f"📊 Launching TensorBoard: {' '.join(cmd)}")
        subprocess.run(cmd)
    except FileNotFoundError:
        logger.error("❌ TensorBoard not found - install with: pip install tensorboard")

if __name__ == "__main__":
    main()