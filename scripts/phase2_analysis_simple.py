#!/usr/bin/env python3
"""
📊 PHASE 2 ANALYSIS - Simple Version
Complete curriculum assessment without external dependencies
"""

import re
import json
from pathlib import Path
from datetime import datetime

class Phase2Analyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.data = self.parse_training_log()
        
    def parse_training_log(self):
        """Parse complete training log into structured data"""
        print("📊 Parsing Phase 2 training log...")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Extract all metrics
        ep_lens = [float(x) for x in re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)]
        ep_rews = [float(x) for x in re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)]
        steps = [int(x) for x in re.findall(r'total_timesteps\s+\|\s+(\d+)', content)]
        
        # Extract tax penalties
        tax_penalties = re.findall(r'Early-exit tax applied: Episode length (\d+) < 80', content)
        tax_episodes = [int(x) for x in tax_penalties]
        
        # Extract drawdown terminations
        dd_terminations = re.findall(r'Daily drawdown -([\d.]+)% exceeded limit', content)
        dd_values = [float(x) for x in dd_terminations]
        
        # Align data lengths
        min_len = min(len(ep_lens), len(ep_rews), len(steps))
        
        data = {
            'steps': steps[:min_len],
            'ep_len': ep_lens[:min_len],
            'ep_rew': ep_rews[:min_len],
            'tax_episodes': tax_episodes,
            'dd_terminations': dd_values
        }
        
        print(f"✅ Parsed {min_len} training checkpoints")
        print(f"✅ Found {len(tax_episodes)} tax penalties")
        print(f"✅ Found {len(dd_values)} drawdown terminations")
        
        return data
    
    def mean(self, values):
        """Calculate mean without numpy"""
        return sum(values) / len(values) if values else 0
    
    def analyze_curriculum_phases(self):
        """Analyze performance across curriculum phases"""
        print("\n🎯 CURRICULUM PHASE ANALYSIS")
        print("=" * 50)
        
        phases = {
            'Warm-keep (0-20K)': (0, 20000),
            'Risk-tighten (20K-60K)': (20000, 60000), 
            'Profit-polish (60K-100K)': (60000, 100000)
        }
        
        results = {}
        
        for phase_name, (start, end) in phases.items():
            # Find data points in this phase
            phase_indices = [i for i, step in enumerate(self.data['steps']) 
                           if start <= step <= end]
            
            if not phase_indices:
                continue
                
            phase_ep_lens = [self.data['ep_len'][i] for i in phase_indices]
            phase_ep_rews = [self.data['ep_rew'][i] for i in phase_indices]
            phase_steps = [self.data['steps'][i] for i in phase_indices]
            
            # Calculate phase statistics
            results[phase_name] = {
                'start_step': phase_steps[0] if phase_steps else start,
                'end_step': phase_steps[-1] if phase_steps else end,
                'start_ep_len': phase_ep_lens[0] if phase_ep_lens else 0,
                'end_ep_len': phase_ep_lens[-1] if phase_ep_lens else 0,
                'start_ep_rew': phase_ep_rews[0] if phase_ep_rews else 0,
                'end_ep_rew': phase_ep_rews[-1] if phase_ep_rews else 0,
                'avg_ep_len': self.mean(phase_ep_lens),
                'avg_ep_rew': self.mean(phase_ep_rews),
                'ep_len_improvement': (phase_ep_lens[-1] - phase_ep_lens[0]) if len(phase_ep_lens) > 1 else 0,
                'ep_rew_improvement': (phase_ep_rews[-1] - phase_ep_rews[0]) if len(phase_ep_rews) > 1 else 0,
                'checkpoints': len(phase_indices)
            }
            
            print(f"\n📋 {phase_name}")
            print(f"   Steps: {results[phase_name]['start_step']:,} → {results[phase_name]['end_step']:,}")
            print(f"   Episode Length: {results[phase_name]['start_ep_len']:.1f} → {results[phase_name]['end_ep_len']:.1f} (Δ{results[phase_name]['ep_len_improvement']:+.1f})")
            print(f"   Episode Reward: {results[phase_name]['start_ep_rew']:.1f} → {results[phase_name]['end_ep_rew']:.1f} (Δ{results[phase_name]['ep_rew_improvement']:+.1f})")
            print(f"   Averages: len={results[phase_name]['avg_ep_len']:.1f}, rew={results[phase_name]['avg_ep_rew']:.1f}")
            print(f"   Checkpoints: {results[phase_name]['checkpoints']}")
        
        return results
    
    def analyze_tax_effectiveness(self):
        """Analyze early-exit tax effectiveness"""
        print("\n💰 EARLY-EXIT TAX ANALYSIS")
        print("=" * 50)
        
        total_episodes_approx = len(self.data['steps']) * 10  # Rough estimate
        tax_count = len(self.data['tax_episodes'])
        tax_rate = (tax_count / total_episodes_approx) * 100 if total_episodes_approx > 0 else 0
        
        print(f"📊 Tax Penalty Statistics:")
        print(f"   Total tax penalties: {tax_count}")
        print(f"   Estimated tax rate: {tax_rate:.1f}% of episodes")
        
        if self.data['tax_episodes']:
            avg_penalized_length = self.mean(self.data['tax_episodes'])
            print(f"   Average penalized episode length: {avg_penalized_length:.1f} steps")
            print(f"   Tax episode length range: {min(self.data['tax_episodes'])}-{max(self.data['tax_episodes'])} steps")
            
            # Tax effectiveness assessment
            if tax_rate < 10:
                print(f"   ✅ Tax highly effective (low penalty rate)")
            elif tax_rate < 30:
                print(f"   🎯 Tax moderately effective")
            else:
                print(f"   ⚠️  Tax may be too harsh (high penalty rate)")
        
        return {
            'tax_count': tax_count,
            'tax_rate': tax_rate,
            'avg_penalized_length': self.mean(self.data['tax_episodes']) if self.data['tax_episodes'] else 0
        }
    
    def analyze_drawdown_behavior(self):
        """Analyze drawdown termination patterns"""
        print("\n📉 DRAWDOWN ANALYSIS")
        print("=" * 50)
        
        if not self.data['dd_terminations']:
            print("   No drawdown terminations found")
            return {}
        
        dd_stats = {
            'count': len(self.data['dd_terminations']),
            'avg_dd': self.mean(self.data['dd_terminations']),
            'max_dd': max(self.data['dd_terminations']),
            'min_dd': min(self.data['dd_terminations'])
        }
        
        print(f"📊 Drawdown Termination Statistics:")
        print(f"   Total DD terminations: {dd_stats['count']}")
        print(f"   Average DD at termination: {dd_stats['avg_dd']:.1f}%")
        print(f"   DD range: {dd_stats['min_dd']:.1f}% - {dd_stats['max_dd']:.1f}%")
        
        # Assess DD behavior
        if dd_stats['avg_dd'] > 70:
            print(f"   ⚠️  High average DD suggests aggressive trading")
        elif dd_stats['avg_dd'] > 60:
            print(f"   🎯 Moderate DD levels")
        else:
            print(f"   ✅ Conservative DD management")
        
        return dd_stats
    
    def evaluate_success_gates(self):
        """Evaluate against Phase 2 success criteria"""
        print("\n🎯 SUCCESS GATE EVALUATION")
        print("=" * 50)
        
        final_ep_len = self.data['ep_len'][-1] if self.data['ep_len'] else 0
        final_ep_rew = self.data['ep_rew'][-1] if self.data['ep_rew'] else 0
        final_steps = self.data['steps'][-1] if self.data['steps'] else 0
        
        gates = {
            'Episode Length ≥ 70': final_ep_len >= 70,
            'Final Reward ≥ -5': final_ep_rew >= -5,
            'Completion ≥ 90K steps': final_steps >= 90000,
            'Reward at 20K ≥ -18': self._get_reward_at_step(20000) >= -18,
            'Reward at 60K ≥ -10': self._get_reward_at_step(60000) >= -10
        }
        
        passed_gates = sum(gates.values())
        total_gates = len(gates)
        
        print(f"📊 Gate Results ({passed_gates}/{total_gates} passed):")
        for gate_name, passed in gates.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {gate_name}")
        
        # Overall assessment
        if passed_gates == total_gates:
            assessment = "🎉 COMPLETE SUCCESS"
        elif passed_gates >= 4:
            assessment = "🎯 STRONG SUCCESS"
        elif passed_gates >= 3:
            assessment = "⚠️  PARTIAL SUCCESS"
        else:
            assessment = "❌ NEEDS REVISION"
        
        print(f"\n🏆 OVERALL ASSESSMENT: {assessment}")
        
        return {
            'gates': gates,
            'passed': passed_gates,
            'total': total_gates,
            'assessment': assessment,
            'final_metrics': {
                'ep_len': final_ep_len,
                'ep_rew': final_ep_rew,
                'steps': final_steps
            }
        }
    
    def _get_reward_at_step(self, target_step):
        """Get reward value closest to target step"""
        if not self.data['steps']:
            return -999
        
        # Find closest step
        closest_idx = min(range(len(self.data['steps'])), 
                         key=lambda i: abs(self.data['steps'][i] - target_step))
        
        return self.data['ep_rew'][closest_idx] if closest_idx < len(self.data['ep_rew']) else -999
    
    def generate_improvement_metrics(self):
        """Calculate key improvement metrics"""
        print("\n📈 IMPROVEMENT METRICS")
        print("=" * 50)
        
        if len(self.data['ep_len']) < 2 or len(self.data['ep_rew']) < 2:
            print("   Insufficient data for improvement analysis")
            return {}
        
        # Calculate improvements
        start_ep_len = self.data['ep_len'][0]
        end_ep_len = self.data['ep_len'][-1]
        start_ep_rew = self.data['ep_rew'][0]
        end_ep_rew = self.data['ep_rew'][-1]
        
        len_improvement = end_ep_len - start_ep_len
        rew_improvement = end_ep_rew - start_ep_rew
        len_improvement_pct = (len_improvement / start_ep_len) * 100 if start_ep_len > 0 else 0
        rew_improvement_pct = (rew_improvement / abs(start_ep_rew)) * 100 if start_ep_rew != 0 else 0
        
        metrics = {
            'episode_length': {
                'start': start_ep_len,
                'end': end_ep_len,
                'absolute_change': len_improvement,
                'percent_change': len_improvement_pct
            },
            'episode_reward': {
                'start': start_ep_rew,
                'end': end_ep_rew,
                'absolute_change': rew_improvement,
                'percent_change': rew_improvement_pct
            }
        }
        
        print(f"📊 Episode Length:")
        print(f"   Start: {start_ep_len:.1f} steps")
        print(f"   End: {end_ep_len:.1f} steps")
        print(f"   Change: {len_improvement:+.1f} steps ({len_improvement_pct:+.1f}%)")
        
        print(f"\n📊 Episode Reward:")
        print(f"   Start: {start_ep_rew:.1f}")
        print(f"   End: {end_ep_rew:.1f}")
        print(f"   Change: {rew_improvement:+.1f} ({rew_improvement_pct:+.1f}%)")
        
        return metrics
    
    def recommend_next_steps(self, success_eval):
        """Generate recommendations for next steps"""
        print("\n🚀 NEXT STEPS RECOMMENDATIONS")
        print("=" * 50)
        
        final_metrics = success_eval['final_metrics']
        
        recommendations = []
        
        # Episode length recommendations
        if final_metrics['ep_len'] < 70:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Episode Length',
                'issue': f"Final length {final_metrics['ep_len']:.1f} < 70 target",
                'actions': [
                    "Increase early-exit tax to 8.0-10.0",
                    "Add per-step time bonus (+0.01-0.02 per step)",
                    "Consider tighter drawdown limits (55-50%)"
                ]
            })
        elif final_metrics['ep_len'] < 80:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': 'Episode Length',
                'issue': f"Length {final_metrics['ep_len']:.1f} below optimal 80+",
                'actions': [
                    "Fine-tune early-exit tax to 6.0-7.0",
                    "Monitor in live-bar fine-tune phase"
                ]
            })
        
        # Reward recommendations
        if final_metrics['ep_rew'] >= -5:
            recommendations.append({
                'priority': 'LOW',
                'area': 'Reward Performance',
                'issue': "Excellent reward achievement",
                'actions': [
                    "Maintain current reward structure",
                    "Focus on consistency in live trading"
                ]
            })
        elif final_metrics['ep_rew'] >= -10:
            recommendations.append({
                'priority': 'LOW',
                'area': 'Reward Performance', 
                'issue': "Good reward progress",
                'actions': [
                    "Continue current approach",
                    "Minor optimizations in live-bar phase"
                ]
            })
        
        # Next phase recommendations
        if success_eval['passed'] >= 4:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Phase Transition',
                'issue': "Ready for live-bar fine-tune",
                'actions': [
                    "Proceed with 20K live-bar fine-tune",
                    "Maintain early-exit tax during live phase",
                    "Prepare IBKR paper-trade setup"
                ]
            })
        else:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Phase Transition',
                'issue': "Consider additional training",
                'actions': [
                    "Run focused 20K extension on weak areas",
                    "Adjust parameters based on analysis",
                    "Re-evaluate before live trading"
                ]
            })
        
        # Print recommendations
        for rec in recommendations:
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[rec['priority']]
            print(f"\n{priority_emoji} {rec['priority']} PRIORITY: {rec['area']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Actions:")
            for action in rec['actions']:
                print(f"     • {action}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run comprehensive Phase 2 analysis"""
        print("🎯 PHASE 2 COMPLETE ANALYSIS")
        print("=" * 60)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {self.log_file}")
        
        # Run all analyses
        curriculum_results = self.analyze_curriculum_phases()
        tax_results = self.analyze_tax_effectiveness()
        dd_results = self.analyze_drawdown_behavior()
        success_eval = self.evaluate_success_gates()
        improvement_metrics = self.generate_improvement_metrics()
        recommendations = self.recommend_next_steps(success_eval)
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📋 EXECUTIVE SUMMARY")
        print("=" * 60)
        
        final_metrics = success_eval['final_metrics']
        print(f"🎯 Final Results:")
        print(f"   • Episode Length: {final_metrics['ep_len']:.1f} steps")
        print(f"   • Episode Reward: {final_metrics['ep_rew']:.1f}")
        print(f"   • Training Steps: {final_metrics['steps']:,}/100,000")
        print(f"   • Success Gates: {success_eval['passed']}/{success_eval['total']} passed")
        print(f"   • Overall Assessment: {success_eval['assessment']}")
        
        if improvement_metrics:
            print(f"\n📈 Key Improvements:")
            ep_rew_change = improvement_metrics['episode_reward']['absolute_change']
            ep_len_change = improvement_metrics['episode_length']['absolute_change']
            print(f"   • Reward: {ep_rew_change:+.1f} points")
            print(f"   • Episode Length: {ep_len_change:+.1f} steps")
        
        print(f"\n🚀 Ready for Next Phase:")
        high_priority_recs = [r for r in recommendations if r['priority'] == 'HIGH']
        if any('live-bar fine-tune' in str(r['actions']) for r in high_priority_recs):
            print(f"   ✅ PROCEED TO LIVE-BAR FINE-TUNE")
        else:
            print(f"   ⚠️  ADDITIONAL TRAINING RECOMMENDED")
        
        return {
            'curriculum': curriculum_results,
            'tax': tax_results,
            'drawdown': dd_results,
            'success': success_eval,
            'improvements': improvement_metrics,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    log_file = "train_runs/phase2_full/training.log"
    
    analyzer = Phase2Analyzer(log_file)
    results = analyzer.run_complete_analysis()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results available in analysis object")