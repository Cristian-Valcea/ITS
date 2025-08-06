#!/usr/bin/env python3
"""
🔍 ENHANCED PHASE 2 MONITORING
Real-time tracking with safety tweaks and tax penalty analytics
"""

import re
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

class Phase2Monitor:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.tax_counts = defaultdict(int)  # Tax penalties per 1K step window
        self.episode_history = deque(maxlen=100)  # Recent episode data
        self.step_windows = {}  # Track metrics per 1K windows
        self.current_step = 0
        self.abort_threshold = -40  # Start with buffer, will change to -25 at 5K
        self.current_tax = 5.0
        self.tax_adjustments = []
        
    def parse_log_metrics(self):
        """Extract latest metrics from training log"""
        if not Path(self.log_file).exists():
            return None
            
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Extract latest metrics
            ep_lens = re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)
            ep_rews = re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)
            steps = re.findall(r'total_timesteps\s+\|\s+(\d+)', content)
            
            # Count tax penalties in recent content (last 2000 chars)
            recent_content = content[-2000:]
            tax_hits = len(re.findall(r'Early-exit tax applied', recent_content))
            
            if ep_lens and ep_rews and steps:
                return {
                    'ep_len': float(ep_lens[-1]),
                    'ep_rew': float(ep_rews[-1]),
                    'steps': int(steps[-1]),
                    'tax_hits_recent': tax_hits
                }
        except Exception as e:
            print(f"⚠️  Error parsing log: {e}")
            
        return None
    
    def count_tax_penalties_in_window(self, start_step: int, end_step: int):
        """Count tax penalties in a specific step window"""
        if not Path(self.log_file).exists():
            return 0
            
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            # Find all tax penalties with context
            tax_pattern = r'Early-exit tax applied: Episode length (\d+) < 80'
            matches = re.findall(tax_pattern, content)
            return len(matches)
        except:
            return 0
    
    def should_adjust_tax(self, metrics: dict):
        """Check if tax should be ramped down"""
        if metrics['steps'] < 2500:  # Wait for first eval
            return False
            
        # Check conditions: ep_len > 90 and tax fired < 30% of episodes
        if metrics['ep_len'] > 90:
            # Estimate tax rate (rough approximation)
            recent_episodes = max(1, (metrics['steps'] - self.current_step) // 70)  # Approx episodes
            tax_rate = metrics['tax_hits_recent'] / recent_episodes if recent_episodes > 0 else 0
            
            if tax_rate < 0.3 and self.current_tax > 3.0:
                return True
        return False
    
    def apply_safety_tweaks(self, metrics: dict):
        """Apply dynamic safety tweaks"""
        changes = []
        
        # 1. Dynamic tax ramp-down
        if self.should_adjust_tax(metrics):
            self.current_tax = 3.0
            self.tax_adjustments.append((metrics['steps'], 3.0))
            changes.append(f"🔧 Tax reduced to 3.0 (ep_len={metrics['ep_len']:.1f}, low tax rate)")
        
        # 2. Abort guard buffer adjustment
        if metrics['steps'] >= 5000 and self.abort_threshold == -40:
            self.abort_threshold = -25
            changes.append(f"🔧 Abort threshold restored to -25 (past 5K steps)")
        
        return changes
    
    def check_abort_conditions(self, metrics: dict):
        """Check if training should be aborted"""
        abort_reasons = []
        
        if metrics['ep_len'] < 50:
            abort_reasons.append(f"Episode length {metrics['ep_len']:.1f} < 50")
            
        if metrics['ep_rew'] < self.abort_threshold:
            abort_reasons.append(f"Episode reward {metrics['ep_rew']:.1f} < {self.abort_threshold}")
            
        return abort_reasons
    
    def assess_learning_progress(self, metrics: dict):
        """Assess if agent is learning as expected"""
        step = metrics['steps']
        assessments = []
        
        # Expected progress milestones
        if 3000 <= step <= 4000:
            assessments.append("📚 TUITION PHASE: Expect ugly rewards as agent pays tax penalties")
            
        elif 5000 <= step <= 8000:
            if metrics['ep_len'] >= 90:
                assessments.append("✅ LEARNING PHASE: Episode length stabilizing well!")
            else:
                assessments.append("⚠️  LEARNING PHASE: Episode length should be ≥90 by now")
                
            if metrics['ep_rew'] >= -20:
                assessments.append("✅ RECOVERY PHASE: Reward climbing back nicely!")
            elif metrics['ep_rew'] >= -25:
                assessments.append("🎯 RECOVERY PHASE: Reward improving, on track")
            else:
                assessments.append("⚠️  RECOVERY PHASE: Reward recovery slower than expected")
                
        elif step >= 20000:
            # Phase transition assessment
            if metrics['ep_len'] >= 90 and metrics['ep_rew'] >= -18:
                assessments.append("🎉 PHASE TRANSITION READY: Excellent progress!")
            else:
                assessments.append("⚠️  PHASE TRANSITION: May need adjustment before 60% DD")
        
        return assessments
    
    def generate_report(self, metrics: dict):
        """Generate comprehensive monitoring report"""
        step = metrics['steps']
        progress_pct = (step / 100000) * 100
        
        print(f"\n🔍 PHASE 2 ENHANCED MONITORING REPORT")
        print(f"=====================================")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Progress: {step:,}/100,000 steps ({progress_pct:.1f}%)")
        print(f"")
        
        # Current metrics
        print(f"📊 CURRENT METRICS:")
        print(f"   Episode Length: {metrics['ep_len']:.1f} steps")
        print(f"   Episode Reward: {metrics['ep_rew']:.1f}")
        print(f"   Recent Tax Hits: {metrics['tax_hits_recent']}")
        print(f"")
        
        # Safety status
        print(f"🛡️  SAFETY STATUS:")
        print(f"   Current Tax: {self.current_tax}")
        print(f"   Abort Threshold: {self.abort_threshold}")
        print(f"   Tax Adjustments: {len(self.tax_adjustments)}")
        print(f"")
        
        # Learning assessment
        assessments = self.assess_learning_progress(metrics)
        if assessments:
            print(f"🧠 LEARNING ASSESSMENT:")
            for assessment in assessments:
                print(f"   {assessment}")
            print(f"")
        
        # Safety tweaks
        changes = self.apply_safety_tweaks(metrics)
        if changes:
            print(f"🔧 SAFETY TWEAKS APPLIED:")
            for change in changes:
                print(f"   {change}")
            print(f"")
        
        # Abort check
        abort_reasons = self.check_abort_conditions(metrics)
        if abort_reasons:
            print(f"🚨 ABORT CONDITIONS:")
            for reason in abort_reasons:
                print(f"   ❌ {reason}")
            print(f"")
            return True  # Signal abort
        else:
            print(f"✅ NO ABORT CONDITIONS")
            print(f"")
        
        # Phase-specific expectations
        if step < 5000:
            print(f"📈 NEXT EXPECTATIONS (First 5K):")
            print(f"   • Reward will look ugly (-30 to -40 range)")
            print(f"   • Tax penalties frequent (learning phase)")
            print(f"   • Episode length should stay >100")
        elif step < 8000:
            print(f"📈 NEXT EXPECTATIONS (5K-8K):")
            print(f"   • Tax frequency should drop")
            print(f"   • Episode length stabilize ≥90")
            print(f"   • Reward climb toward -20/-18 zone")
        elif step < 20000:
            print(f"📈 NEXT EXPECTATIONS (8K-20K):")
            print(f"   • Target: ≤-18 reward with ≥90 episodes")
            print(f"   • Prepare for 60% DD transition")
        
        self.current_step = step
        return False  # No abort
    
    def run_monitoring(self, check_interval: int = 120):
        """Run continuous monitoring"""
        print(f"🔍 Starting Enhanced Phase 2 Monitoring")
        print(f"Log file: {self.log_file}")
        print(f"Check interval: {check_interval} seconds")
        print(f"")
        
        while True:
            metrics = self.parse_log_metrics()
            if metrics:
                should_abort = self.generate_report(metrics)
                if should_abort:
                    print(f"🛑 ABORT CONDITIONS MET - STOPPING MONITORING")
                    break
            else:
                print(f"⏳ Waiting for training data...")
            
            time.sleep(check_interval)

if __name__ == "__main__":
    import sys
    
    log_file = "train_runs/phase2_full/training.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    monitor = Phase2Monitor(log_file)
    try:
        monitor.run_monitoring()
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")