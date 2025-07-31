#!/usr/bin/env python3
"""
🎛️ Training Control Dashboard
Real-time monitoring and control for 200K dual-ticker training
"""

import os
import json
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class TrainingController:
    """Control and monitor training sessions"""
    
    def __init__(self):
        self.training_session = "training200k"
        self.log_dir = "logs"
        self.checkpoint_dir = "models/checkpoints"
        self.dashboard_file = "reports/executive_dashboard.json"
        
    def get_training_status(self) -> Dict:
        """Get comprehensive training status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'session_active': self.is_session_active(),
            'gpu_status': self.get_gpu_status(),
            'training_progress': self.get_training_progress(),
            'recent_logs': self.get_recent_logs(),
            'checkpoints': self.get_checkpoint_status(),
            'system_resources': self.get_system_resources()
        }
        return status
    
    def is_session_active(self) -> bool:
        """Check if tmux training session is active"""
        try:
            result = subprocess.run(
                ['tmux', 'has-session', '-t', self.training_session],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def get_gpu_status(self) -> Dict:
        """Get GPU utilization and memory"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)
                
                return {
                    'available': True,
                    'name': torch.cuda.get_device_name(0),
                    'memory_total_gb': gpu_memory / 1e9,
                    'memory_allocated_gb': gpu_allocated / 1e9,
                    'memory_cached_gb': gpu_cached / 1e9,
                    'utilization_pct': (gpu_allocated / gpu_memory) * 100
                }
            else:
                return {'available': False}
        except:
            return {'available': False, 'error': 'Could not query GPU'}
    
    def get_training_progress(self) -> Dict:
        """Get training progress from logs and checkpoints"""
        progress = {
            'total_steps': 200000,
            'current_steps': 0,
            'progress_pct': 0,
            'estimated_completion': 'Unknown',
            'last_checkpoint': None,
            'episodes_completed': 0,
            'current_reward': None
        }
        
        # Check latest log file
        latest_log = self.get_latest_log_file()
        if latest_log:
            progress.update(self.parse_training_progress(latest_log))
        
        # Check checkpoints
        checkpoints = self.get_checkpoint_files()
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            progress['last_checkpoint'] = os.path.basename(latest_checkpoint)
            
            # Extract steps from checkpoint name
            if 'steps' in latest_checkpoint:
                try:
                    steps_str = latest_checkpoint.split('_')
                    for part in steps_str:
                        if part.endswith('steps.zip'):
                            steps = int(part.replace('steps.zip', ''))
                            progress['current_steps'] = steps
                            progress['progress_pct'] = (steps / 200000) * 100
                            break
                except:
                    pass
        
        return progress
    
    def get_latest_log_file(self) -> Optional[str]:
        """Get the most recent training log file"""
        if not os.path.exists(self.log_dir):
            return None
        
        log_files = [f for f in os.listdir(self.log_dir) if f.startswith('200k_training_')]
        if not log_files:
            return None
        
        latest = max(log_files, key=lambda f: os.path.getctime(os.path.join(self.log_dir, f)))
        return os.path.join(self.log_dir, latest)
    
    def parse_training_progress(self, log_file: str) -> Dict:
        """Parse training progress from log file"""
        progress = {}
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Get last 50 lines for recent progress
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
            for line in reversed(recent_lines):
                # Look for step information
                if 'timesteps' in line.lower() and 'total' in line.lower():
                    # Extract current steps
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'timesteps' in part.lower() and i > 0:
                                steps = int(parts[i-1].replace(',', ''))
                                progress['current_steps'] = steps
                                progress['progress_pct'] = (steps / 200000) * 100
                                break
                    except:
                        pass
                
                # Look for episode information
                if 'episode' in line.lower() and 'reward' in line.lower():
                    try:
                        # Extract reward value
                        if 'reward:' in line.lower():
                            reward_part = line.lower().split('reward:')[1].split()[0]
                            progress['current_reward'] = float(reward_part)
                    except:
                        pass
                
                if progress.get('current_steps') and progress.get('current_reward'):
                    break
                    
        except Exception as e:
            progress['parse_error'] = str(e)
        
        return progress
    
    def get_recent_logs(self, lines: int = 20) -> List[str]:
        """Get recent log entries"""
        latest_log = self.get_latest_log_file()
        if not latest_log:
            return ["No training logs found"]
        
        try:
            with open(latest_log, 'r') as f:
                all_lines = f.readlines()
            
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return [line.strip() for line in recent]
        except:
            return ["Could not read log file"]
    
    def get_checkpoint_files(self) -> List[str]:
        """Get list of checkpoint files"""
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith('dual_ticker_200k_') and file.endswith('.zip'):
                checkpoints.append(os.path.join(self.checkpoint_dir, file))
        
        return checkpoints
    
    def get_checkpoint_status(self) -> Dict:
        """Get checkpoint status and timing"""
        checkpoints = self.get_checkpoint_files()
        
        status = {
            'total_checkpoints': len(checkpoints),
            'checkpoints': [],
            'next_checkpoint_at': None
        }
        
        for checkpoint in checkpoints:
            stat = os.stat(checkpoint)
            size_mb = stat.st_size / 1e6
            created = datetime.fromtimestamp(stat.st_ctime)
            
            status['checkpoints'].append({
                'filename': os.path.basename(checkpoint),
                'size_mb': round(size_mb, 1),
                'created': created.isoformat(),
                'age_hours': (datetime.now() - created).total_seconds() / 3600
            })
        
        # Estimate next checkpoint
        progress = self.get_training_progress()
        current_steps = progress.get('current_steps', 0)
        if current_steps > 0:
            next_checkpoint_steps = ((current_steps // 25000) + 1) * 25000
            if next_checkpoint_steps <= 200000:
                status['next_checkpoint_at'] = f"{next_checkpoint_steps:,} steps"
        
        return status
    
    def get_system_resources(self) -> Dict:
        """Get system resource utilization"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('.').percent,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def get_executive_dashboard(self) -> Dict:
        """Get executive dashboard data if available"""
        if os.path.exists(self.dashboard_file):
            try:
                with open(self.dashboard_file, 'r') as f:
                    return json.load(f)
            except:
                return {'error': 'Could not load dashboard data'}
        return {'status': 'Dashboard not yet available'}
    
    def control_training(self, action: str) -> Dict:
        """Control training session"""
        result = {'action': action, 'success': False, 'message': ''}
        
        try:
            if action == 'pause':
                # Send Ctrl+Z to pause training
                subprocess.run(['tmux', 'send-keys', '-t', self.training_session, 'C-z'])
                result['success'] = True
                result['message'] = 'Training paused (Ctrl+Z sent)'
                
            elif action == 'resume':
                # Send 'fg' to resume training
                subprocess.run(['tmux', 'send-keys', '-t', self.training_session, 'fg', 'Enter'])
                result['success'] = True
                result['message'] = 'Training resumed (fg sent)'
                
            elif action == 'stop':
                # Send Ctrl+C to stop training
                subprocess.run(['tmux', 'send-keys', '-t', self.training_session, 'C-c'])
                result['success'] = True
                result['message'] = 'Stop signal sent (Ctrl+C)'
                
            elif action == 'kill_session':
                # Kill the entire tmux session
                subprocess.run(['tmux', 'kill-session', '-t', self.training_session])
                result['success'] = True
                result['message'] = 'Training session killed'
                
            else:
                result['message'] = f'Unknown action: {action}'
                
        except Exception as e:
            result['message'] = f'Control action failed: {str(e)}'
        
        return result

def print_status_dashboard(controller: TrainingController):
    """Print formatted status dashboard"""
    status = controller.get_training_status()
    
    print("🎛️  200K DUAL-TICKER TRAINING DASHBOARD")
    print("=" * 60)
    print(f"⏰ Timestamp: {status['timestamp']}")
    print(f"🔄 Session Active: {'✅ YES' if status['session_active'] else '❌ NO'}")
    
    # Training Progress
    progress = status['training_progress']
    print(f"\n📊 TRAINING PROGRESS:")
    print(f"   Steps: {progress['current_steps']:,} / {progress['total_steps']:,}")
    print(f"   Progress: {progress['progress_pct']:.1f}%")
    if progress['current_reward']:
        print(f"   Current Reward: {progress['current_reward']:.2f}")
    if progress['last_checkpoint']:
        print(f"   Last Checkpoint: {progress['last_checkpoint']}")
    
    # GPU Status
    gpu = status['gpu_status']
    if gpu['available']:
        print(f"\n🚀 GPU STATUS:")
        print(f"   Name: {gpu['name']}")
        print(f"   Memory: {gpu['memory_allocated_gb']:.1f} / {gpu['memory_total_gb']:.1f} GB")
        print(f"   Utilization: {gpu['utilization_pct']:.1f}%")
    
    # System Resources
    resources = status['system_resources']
    print(f"\n💻 SYSTEM RESOURCES:")
    print(f"   CPU: {resources['cpu_percent']:.1f}%")
    print(f"   Memory: {resources['memory_percent']:.1f}%")
    print(f"   Disk: {resources['disk_usage_percent']:.1f}%")
    
    # Checkpoints
    checkpoints = status['checkpoints']
    print(f"\n💾 CHECKPOINTS ({checkpoints['total_checkpoints']} total):")
    for cp in checkpoints['checkpoints'][-3:]:  # Show last 3
        print(f"   {cp['filename']} ({cp['size_mb']} MB, {cp['age_hours']:.1f}h ago)")
    if checkpoints['next_checkpoint_at']:
        print(f"   Next: {checkpoints['next_checkpoint_at']}")
    
    # Recent Logs
    print(f"\n📋 RECENT LOGS (last 5 lines):")
    for log_line in status['recent_logs'][-5:]:
        print(f"   {log_line}")

def main():
    parser = argparse.ArgumentParser(description='Training Control Dashboard')
    parser.add_argument('--action', choices=['status', 'pause', 'resume', 'stop', 'kill'], 
                       default='status', help='Control action')
    parser.add_argument('--watch', action='store_true', help='Watch mode (refresh every 30s)')
    parser.add_argument('--logs', type=int, default=20, help='Number of log lines to show')
    
    args = parser.parse_args()
    
    controller = TrainingController()
    
    if args.action == 'status':
        if args.watch:
            print("👀 WATCH MODE - Press Ctrl+C to exit")
            try:
                while True:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    print_status_dashboard(controller)
                    print(f"\n🔄 Refreshing in 30 seconds... (Ctrl+C to exit)")
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\n👋 Exiting watch mode")
        else:
            print_status_dashboard(controller)
    
    else:
        # Control actions
        result = controller.control_training(args.action)
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ {result['message']}")

if __name__ == "__main__":
    main()