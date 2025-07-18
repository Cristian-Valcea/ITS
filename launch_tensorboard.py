#!/usr/bin/env python3
"""
TensorBoard Launch Utility for IntradayJules

This script provides easy TensorBoard launching with automatic browser opening
and comprehensive logging visualization for the turnover penalty system.

Usage:
    python launch_tensorboard.py                    # Launch with defaults
    python launch_tensorboard.py --port 6007        # Custom port
    python launch_tensorboard.py --no-browser       # Don't open browser
    python launch_tensorboard.py --list-runs        # List available runs
"""

import argparse
import subprocess
import webbrowser
import logging
from pathlib import Path
from time import sleep
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_tensorboard_runs(log_dir: Path) -> None:
    """List available TensorBoard runs."""
    
    print("📊 AVAILABLE TENSORBOARD RUNS")
    print("=" * 50)
    
    if not log_dir.exists():
        print(f"❌ Log directory does not exist: {log_dir}")
        return
    
    runs = list(log_dir.iterdir())
    if not runs:
        print("📭 No TensorBoard runs found")
        return
    
    print(f"📁 Found {len(runs)} runs in {log_dir}:")
    print()
    
    for i, run_dir in enumerate(sorted(runs), 1):
        if run_dir.is_dir():
            # Extract run information
            run_name = run_dir.name
            
            # Check for event files
            event_files = list(run_dir.rglob("events.out.tfevents.*"))
            has_data = len(event_files) > 0
            
            # Get creation time
            try:
                creation_time = run_dir.stat().st_ctime
                import datetime
                creation_str = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            except:
                creation_str = "Unknown"
            
            status = "✅ Has data" if has_data else "❌ No data"
            print(f"{i:2d}. {run_name}")
            print(f"    Created: {creation_str}")
            print(f"    Status: {status}")
            print(f"    Events: {len(event_files)} files")
            print()

def launch_tensorboard(log_dir: Path, port: int = 6006, open_browser: bool = True) -> None:
    """Launch TensorBoard server."""
    
    print("🚀 LAUNCHING TENSORBOARD")
    print("=" * 50)
    
    if not log_dir.exists():
        logger.error(f"Log directory does not exist: {log_dir}")
        return
    
    # Check if there are any runs
    runs = list(log_dir.iterdir())
    if not runs:
        logger.warning(f"No runs found in {log_dir}")
        logger.info("Run some training first to generate TensorBoard logs")
        return
    
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Port: {port}")
    logger.info(f"Found {len(runs)} runs")
    
    try:
        # Build TensorBoard command
        cmd = ["tensorboard", "--logdir", str(log_dir), "--port", str(port)]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Launch TensorBoard
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        logger.info("Starting TensorBoard server...")
        sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error("TensorBoard failed to start!")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            return
        
        # Open browser
        url = f"http://localhost:{port}"
        logger.info(f"TensorBoard running at: {url}")
        
        if open_browser:
            logger.info("Opening browser...")
            webbrowser.open(url)
        
        print()
        print("🎯 TENSORBOARD FEATURES AVAILABLE:")
        print("=" * 50)
        print("📊 SCALARS:")
        print("   • episode/total_reward - Episode rewards over time")
        print("   • episode/portfolio_value - Portfolio value evolution")
        print("   • turnover/penalty - Turnover penalty tracking")
        print("   • turnover/normalized - Normalized turnover ratio")
        print("   • turnover/excess - Excess over target turnover")
        print("   • performance/win_rate - Win rate percentage")
        print("   • performance/sharpe_ratio - Risk-adjusted returns")
        print("   • performance/max_drawdown - Maximum drawdown")
        print("   • training/loss - Training loss convergence")
        print("   • training/q_value_variance - Q-value spread")
        print("   • risk/volatility - Portfolio volatility")
        print()
        print("📈 HISTOGRAMS:")
        print("   • Model weights and gradients")
        print("   • Q-value distributions")
        print()
        print("📝 TEXT:")
        print("   • Experiment descriptions")
        print("   • System information")
        print("   • Hyperparameter settings")
        print()
        print("🔧 HPARAMS:")
        print("   • Hyperparameter comparison")
        print("   • Performance correlation analysis")
        print()
        
        logger.info("TensorBoard launched successfully!")
        logger.info("Press Ctrl+C to stop the server")
        
        # Wait for process or keyboard interrupt
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping TensorBoard...")
            process.terminate()
            process.wait()
            logger.info("TensorBoard stopped")
        
    except FileNotFoundError:
        logger.error("TensorBoard not found! Install with: pip install tensorboard")
    except KeyboardInterrupt:
        logger.info("Stopping TensorBoard...")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")

def main():
    """Main CLI interface."""
    
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for IntradayJules training visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_tensorboard.py                    # Launch with defaults
  python launch_tensorboard.py --port 6007        # Custom port
  python launch_tensorboard.py --no-browser       # Don't open browser
  python launch_tensorboard.py --list-runs        # List available runs
  python launch_tensorboard.py --logdir logs      # Custom log directory

TensorBoard Features:
  • Real-time training metrics visualization
  • Turnover penalty evolution tracking
  • Performance metrics (Sharpe, drawdown, win rate)
  • Risk management indicators
  • Hyperparameter comparison
        """
    )
    
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="Directory containing TensorBoard logs (default: runs)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for TensorBoard server (default: 6006)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available TensorBoard runs and exit"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.logdir)
    
    if args.list_runs:
        list_tensorboard_runs(log_dir)
        return
    
    launch_tensorboard(
        log_dir=log_dir,
        port=args.port,
        open_browser=not args.no_browser
    )

if __name__ == "__main__":
    main()