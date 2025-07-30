#!/usr/bin/env python3
"""
📅 TIMELINE TRACKER
Track progress against the 6-hour tactical plan
"""

import sys
from pathlib import Path
import json
import time
import logging
from datetime import datetime, timedelta
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineTracker:
    """Track progress against tactical timeline"""
    
    def __init__(self, start_time=None):
        self.start_time = start_time or datetime.now()
        
        # Timeline milestones (from your plan)
        self.milestones = {
            60: "1h: Likely 6-8 HPO runs finished; kill dead loops manually",
            120: "2h: If survivor appears → export, warm-start dual-ticker",
            240: "4h: Dual-ticker 25K fine-tune + gate evaluation complete", 
            360: "6h: Either celebrating or iterating on knob tweaks"
        }
    
    def elapsed_minutes(self):
        """Get elapsed minutes since start"""
        return (datetime.now() - self.start_time).total_seconds() / 60
    
    def find_hpo_process(self):
        """Check if HPO is still running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python3' in proc.info['name'] and 'hpo_grid_sprint.py' in ' '.join(proc.info['cmdline']):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def check_status_files(self):
        """Check current status from files"""
        
        status = {}
        
        # HPO results
        if Path('winning_hpo_config.json').exists():
            status['hpo'] = 'success'
        elif Path('hpo_failure_analysis.json').exists():
            status['hpo'] = 'failure'
        else:
            status['hpo'] = 'running'
        
        # Export status
        if Path('models/singleticker_gatepass.zip').exists():
            status['export'] = 'complete'
        else:
            status['export'] = 'pending'
        
        # Dual-ticker warm-start
        if Path('models/dual_ticker_success.zip').exists():
            status['dual_ticker'] = 'success'
        elif Path('dual_ticker_warmstart.log').exists():
            status['dual_ticker'] = 'running'
        else:
            status['dual_ticker'] = 'pending'
        
        # Knob tuning fallback
        if Path('knob_tuning_success.json').exists():
            status['knob_tuning'] = 'success'
        elif Path('knob_tuning_analysis.json').exists():
            status['knob_tuning'] = 'failure'
        else:
            status['knob_tuning'] = 'pending'
        
        return status
    
    def get_current_phase(self, elapsed_min):
        """Determine current phase based on elapsed time"""
        
        if elapsed_min < 60:
            return "Phase 1: HPO Grid Search"
        elif elapsed_min < 120:
            return "Phase 2: Early Results Analysis"
        elif elapsed_min < 240:
            return "Phase 3: Export & Dual-Ticker Warm-Start"
        elif elapsed_min < 360:
            return "Phase 4: Dual-Ticker Evaluation"
        else:
            return "Phase 5: Iteration & Refinement"
    
    def get_next_milestone(self, elapsed_min):
        """Get next upcoming milestone"""
        
        for milestone_time, description in self.milestones.items():
            if elapsed_min < milestone_time:
                return milestone_time, description
        return None, "All milestones passed"
    
    def display_timeline_status(self):
        """Display current timeline status"""
        
        elapsed_min = self.elapsed_minutes()
        current_phase = self.get_current_phase(elapsed_min)
        next_milestone_time, next_milestone = self.get_next_milestone(elapsed_min)
        
        logger.info(f"📅 TIMELINE TRACKER")
        logger.info(f"   Start time: {self.start_time.strftime('%H:%M:%S')}")
        logger.info(f"   Elapsed: {elapsed_min:.1f} minutes")
        logger.info(f"   Current phase: {current_phase}")
        
        if next_milestone_time:
            remaining = next_milestone_time - elapsed_min
            logger.info(f"   Next milestone: {remaining:.1f}min ({next_milestone})")
        else:
            logger.info(f"   Status: {next_milestone}")
        
        # Check current status
        hpo_pid = self.find_hpo_process()
        status = self.check_status_files()
        
        logger.info(f"\\n🔍 CURRENT STATUS:")
        logger.info(f"   HPO Process: {'Running (PID ' + str(hpo_pid) + ')' if hpo_pid else 'Not found'}")
        logger.info(f"   HPO Result: {status['hpo'].upper()}")
        logger.info(f"   Export: {status['export'].upper()}")
        logger.info(f"   Dual-Ticker: {status['dual_ticker'].upper()}")
        logger.info(f"   Knob Tuning: {status['knob_tuning'].upper()}")
        
        # Phase-specific recommendations
        self._phase_recommendations(elapsed_min, status)
    
    def _phase_recommendations(self, elapsed_min, status):
        """Provide phase-specific recommendations"""
        
        logger.info(f"\\n💡 RECOMMENDATIONS:")
        
        if elapsed_min < 60:
            # Phase 1: Watch for early signals
            logger.info("   - Monitor HPO for early success signals")
            logger.info("   - Watch TensorBoard for rollout/ep_rew_mean ≥ +1000")
            logger.info("   - Kill obviously dead runs to free GPU")
            
        elif elapsed_min < 120:
            # Phase 2: Early results analysis
            if status['hpo'] == 'success':
                logger.info("   ✅ HPO SUCCESS! Export weights and start dual-ticker")
                logger.info("   → python export_policy.py")
                logger.info("   → python dual_ticker_warmstart.py")
            elif status['hpo'] == 'failure':
                logger.info("   ❌ HPO FAILED! Run synthetic alpha test")
                logger.info("   → python synthetic_alpha_test.py")
            else:
                logger.info("   - Continue monitoring HPO progress")
                logger.info("   - Consider manual kill of dead runs")
            
        elif elapsed_min < 240:
            # Phase 3: Export & dual-ticker
            if status['export'] == 'complete' and status['dual_ticker'] == 'pending':
                logger.info("   🚀 Export complete! Launch dual-ticker warm-start")
                logger.info("   → python dual_ticker_warmstart.py")
            elif status['dual_ticker'] == 'running':
                logger.info("   ⏳ Dual-ticker training in progress...")
                logger.info("   - Monitor for 25K fine-tune completion")
            else:
                logger.info("   - Complete missing steps from phases 1-2")
            
        elif elapsed_min < 360:
            # Phase 4: Evaluation
            if status['dual_ticker'] == 'success':
                logger.info("   🎉 DUAL-TICKER SUCCESS! Extend to 100K steps")
                logger.info("   - Consider production deployment preparation")
            elif status['dual_ticker'] == 'failure':
                logger.info("   ⚠️ Dual-ticker incomplete - try ent_coef adjustment")
                logger.info("   → Retry with higher entropy (0.003-0.005)")
            else:
                logger.info("   - Complete dual-ticker evaluation")
            
        else:
            # Phase 5: Iteration
            logger.info("   - Analyze results and plan next iteration")
            if status['knob_tuning'] == 'pending':
                logger.info("   → Consider knob tuning fallback")
                logger.info("   → python knob_tuning_fallback.py")
    
    def wait_and_update(self, check_interval=300):
        """Wait and periodically update timeline"""
        
        logger.info(f"📅 Timeline tracking started (checking every {check_interval/60:.1f}min)")
        
        try:
            while True:
                self.display_timeline_status()
                
                # Check if we should stop
                status = self.check_status_files()
                if status['dual_ticker'] == 'success':
                    logger.info("\\n🎉 SUCCESS ACHIEVED! Dual-ticker training complete")
                    break
                elif self.elapsed_minutes() > 480:  # 8 hours max
                    logger.info("\\n⏰ Maximum timeline reached (8 hours)")
                    break
                
                logger.info(f"\\n⏳ Next check in {check_interval/60:.1f} minutes...")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\\n🛑 Timeline tracking stopped by user")

def main():
    """Main timeline tracker function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Timeline Tracker')
    parser.add_argument('--start_time', type=str, help='Start time (HH:MM:SS format)')
    parser.add_argument('--watch', action='store_true', help='Watch and update periodically')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            time_obj = datetime.strptime(args.start_time, '%H:%M:%S').time()
            today = datetime.now().date()
            start_time = datetime.combine(today, time_obj)
        except ValueError:
            logger.error(f"Invalid time format: {args.start_time} (use HH:MM:SS)")
            sys.exit(1)
    
    tracker = TimelineTracker(start_time)
    
    if args.watch:
        tracker.wait_and_update(args.interval)
    else:
        tracker.display_timeline_status()

if __name__ == "__main__":
    main()