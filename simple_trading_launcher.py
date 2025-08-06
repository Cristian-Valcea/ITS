#!/usr/bin/env python3
"""
🚀 Simple Paper Trading Launcher
Simplified interface for IBKR paper trading with clean slate functionality
"""

import os
import sys
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SimpleTradingLauncher:
    """
    Simple command-line interface for paper trading with clean slate
    """
    
    def __init__(self):
        self.trading_process = None
        self.trading_active = False
    
    def check_ibkr_status(self):
        """Check IBKR connection status"""
        try:
            result = subprocess.run([
                'python', 'src/brokers/ibkr_account_manager.py', '--positions'
            ], capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                return "✅ Connected", result.stdout
            else:
                return "❌ Failed", result.stderr
        except Exception as e:
            return f"❌ Error: {str(e)}", ""
    
    def check_governor_status(self):
        """Check Risk Governor status"""
        try:
            result = subprocess.run([
                'python', 'operator_docs/governor_state_manager.py', '--status'
            ], capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                # Parse output for state
                if "State: RUNNING" in result.stdout:
                    return "✅ RUNNING"
                elif "State: PAUSED" in result.stdout:
                    return "⚠️ PAUSED"
                else:
                    return "❓ Unknown"
            else:
                return "❌ Error"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def show_status(self):
        """Display current system status"""
        print("\n🎯 INTRADAYJULES TRADING SYSTEM STATUS")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # IBKR Status
        print("📡 IBKR Connection:")
        ibkr_status, ibkr_details = self.check_ibkr_status()
        print(f"   {ibkr_status}")
        if "No positions" in ibkr_details:
            print("   🧹 Account is clean (no positions)")
        elif "positions" in ibkr_details.lower():
            print("   ⚠️ Account has positions")
        
        # Risk Governor Status
        print("\n🛡️ Risk Governor:")
        governor_status = self.check_governor_status()
        print(f"   {governor_status}")
        
        # Trading Model Status
        print("\n🤖 Trading Model:")
        model_paths = [
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
        ]
        
        model_found = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"   ✅ Available: {os.path.basename(model_path)}")
                model_found = True
                break
        
        if not model_found:
            print("   ❌ Model not found")
        
        # Trading Status
        print("\n💹 Trading Status:")
        if self.trading_active:
            print("   🟢 Active")
        else:
            print("   ⚪ Inactive")
        
        print("=" * 50)
    
    def reset_account(self):
        """Reset IBKR paper trading account (clean slate)"""
        print("\n🧹 RESETTING PAPER TRADING ACCOUNT")
        print("=" * 40)
        
        if input("Are you sure you want to reset the account? (y/N): ").lower() != 'y':
            print("Reset cancelled.")
            return False
        
        try:
            print("🔄 Executing account reset...")
            result = subprocess.run([
                'python', 'src/brokers/ibkr_account_manager.py', '--reset'
            ], cwd=str(project_root), text=True)
            
            if result.returncode == 0:
                print("✅ Account reset successful - clean slate achieved")
                return True
            else:
                print("❌ Account reset failed")
                return False
                
        except Exception as e:
            print(f"❌ Reset error: {e}")
            return False
    
    def start_trading(self, symbols=None, position_size=10, clean_slate=False):
        """Start paper trading"""
        
        if symbols is None:
            symbols = ["NVDA", "MSFT"]
        
        print(f"\n🚀 STARTING PAPER TRADING")
        print("=" * 40)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Position Size: ${position_size} per symbol")
        print(f"Clean Slate: {'Yes' if clean_slate else 'No'}")
        print()
        
        # Step 1: Clean slate if requested
        if clean_slate:
            print("Step 1: Performing clean slate reset...")
            if not self.reset_account():
                print("❌ Cannot start trading - clean slate failed")
                return False
            print()
        
        # Step 2: Check if already running
        if self.trading_active:
            print("⚠️ Trading is already active")
            return False
        
        # Step 3: Start trading
        print("Step 2: Starting production trading...")
        
        try:
            # Start production deployment in background
            cmd = [
                'bash', '-c',
                'source venv/bin/activate && python production_deployment.py'
            ]
            
            self.trading_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.trading_active = True
            
            print("✅ Paper trading started successfully!")
            print(f"📊 Process ID: {self.trading_process.pid}")
            print("📝 Monitor logs: tail -f logs/production/*.log")
            print("🛑 Stop with: Ctrl+C or call stop_trading()")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start trading: {e}")
            return False
    
    def stop_trading(self):
        """Stop paper trading"""
        
        print("\n🛑 STOPPING PAPER TRADING")
        print("=" * 40)
        
        if not self.trading_active or not self.trading_process:
            print("⚠️ Trading is not currently active")
            return False
        
        try:
            print("🔄 Terminating trading process...")
            self.trading_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.trading_process.wait(timeout=10)
                print("✅ Trading stopped gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️ Forcing process termination...")
                self.trading_process.kill()
                self.trading_process.wait()
                print("✅ Trading stopped (forced)")
            
            self.trading_active = False
            self.trading_process = None
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop trading: {e}")
            return False
    
    def interactive_menu(self):
        """Interactive command-line menu"""
        
        while True:
            print("\n🎯 INTRADAYJULES PAPER TRADING LAUNCHER")
            print("=" * 50)
            print("1. 📊 Show System Status")
            print("2. 🧹 Reset Account (Clean Slate)")
            print("3. 🚀 Start Trading")
            print("4. 🛑 Stop Trading")
            print("5. 📈 Start with Clean Slate")
            print("6. 📋 Show Positions")
            print("7. 📜 Show Recent Logs")
            print("0. 🚪 Exit")
            print("=" * 50)
            
            try:
                choice = input("Choose option (0-7): ").strip()
                
                if choice == '1':
                    self.show_status()
                
                elif choice == '2':
                    self.reset_account()
                
                elif choice == '3':
                    symbols_input = input("Enter symbols (default: NVDA,MSFT): ").strip()
                    if symbols_input:
                        symbols = [s.strip().upper() for s in symbols_input.split(',')]
                    else:
                        symbols = ["NVDA", "MSFT"]
                    
                    position_input = input("Enter position size (default: 10): ").strip()
                    position_size = int(position_input) if position_input.isdigit() else 10
                    
                    self.start_trading(symbols, position_size, clean_slate=False)
                
                elif choice == '4':
                    self.stop_trading()
                
                elif choice == '5':
                    symbols_input = input("Enter symbols (default: NVDA,MSFT): ").strip()
                    if symbols_input:
                        symbols = [s.strip().upper() for s in symbols_input.split(',')]
                    else:
                        symbols = ["NVDA", "MSFT"]
                    
                    position_input = input("Enter position size (default: 10): ").strip()
                    position_size = int(position_input) if position_input.isdigit() else 10
                    
                    self.start_trading(symbols, position_size, clean_slate=True)
                
                elif choice == '6':
                    print("\n📊 Current Positions:")
                    result = subprocess.run([
                        'python', 'src/brokers/ibkr_account_manager.py', '--positions'
                    ], cwd=str(project_root))
                
                elif choice == '7':
                    print("\n📜 Recent Logs:")
                    log_dir = Path('logs/production')
                    if log_dir.exists():
                        log_files = list(log_dir.glob('*.log'))
                        if log_files:
                            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                            print(f"Showing last 20 lines from: {latest_log}")
                            subprocess.run(['tail', '-20', str(latest_log)])
                        else:
                            print("No production logs found")
                    else:
                        print("No log directory found")
                
                elif choice == '0':
                    if self.trading_active:
                        print("⚠️ Trading is still active. Stopping...")
                        self.stop_trading()
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                if self.trading_active:
                    print("Stopping trading...")
                    self.stop_trading()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    
    print("🎯 INTRADAYJULES SIMPLE TRADING LAUNCHER")
    print("Enhanced IBKR Integration with Clean Slate Support") 
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('src/brokers/ibkr_account_manager.py'):
        print("❌ Error: Please run from the IntradayTrading/ITS directory")
        return 1
    
    # Create launcher
    launcher = SimpleTradingLauncher()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            launcher.show_status()
        elif command == 'reset':
            launcher.reset_account()
        elif command == 'start':
            launcher.start_trading(clean_slate=False)
        elif command == 'start-clean':
            launcher.start_trading(clean_slate=True)
        elif command == 'stop':
            launcher.stop_trading()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: status, reset, start, start-clean, stop")
            return 1
    else:
        # Interactive mode
        launcher.interactive_menu()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())