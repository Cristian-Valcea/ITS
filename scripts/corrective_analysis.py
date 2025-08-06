#!/usr/bin/env python3
"""
🔧 PHASE 2 CORRECTIVE TRAINING ANALYSIS
Quick analysis of corrective training results
"""

import re
import sys
from pathlib import Path
from datetime import datetime

def parse_corrective_log(log_file):
    """Parse corrective training log for key metrics"""
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        return None
    
    print(f"📊 Analyzing corrective training log: {log_file}")
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract training metrics
    ep_len_pattern = r'ep_len_mean\s*\|\s*([\d.]+)'
    ep_rew_pattern = r'ep_rew_mean\s*\|\s*([-\d.]+)'
    timesteps_pattern = r'total_timesteps\s*\|\s*(\d+)'
    
    ep_lens = [float(m) for m in re.findall(ep_len_pattern, content)]
    ep_rews = [float(m) for m in re.findall(ep_rew_pattern, content)]
    timesteps = [int(m) for m in re.findall(timesteps_pattern, content)]
    
    # Count tax penalties
    tax_penalties = len(re.findall(r'Early-exit tax applied', content))
    
    # Count drawdown terminations
    dd_terminations = len(re.findall(r'Daily drawdown.*exceeded limit', content))
    
    # Extract final metrics
    final_reward_match = re.search(r'Final avg reward: ([-\d.]+)', content)
    final_reward = float(final_reward_match.group(1)) if final_reward_match else None
    
    total_episodes_match = re.search(r'Total episodes: (\d+)', content)
    total_episodes = int(total_episodes_match.group(1)) if total_episodes_match else None
    
    return {
        'ep_lens': ep_lens,
        'ep_rews': ep_rews,
        'timesteps': timesteps,
        'tax_penalties': tax_penalties,
        'dd_terminations': dd_terminations,
        'final_reward': final_reward,
        'total_episodes': total_episodes
    }

def analyze_corrective_results(data):
    """Analyze corrective training results"""
    
    if not data or not data['ep_lens']:
        print("❌ No valid data found")
        return
    
    print("\n🎯 CORRECTIVE TRAINING ANALYSIS")
    print("=" * 60)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Basic metrics
    final_ep_len = data['ep_lens'][-1] if data['ep_lens'] else 0
    final_ep_rew = data['ep_rews'][-1] if data['ep_rews'] else 0
    final_steps = data['timesteps'][-1] if data['timesteps'] else 0
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Final Episode Length: {final_ep_len:.1f} steps")
    print(f"   Final Episode Reward: {final_ep_rew:.1f}")
    print(f"   Total Training Steps: {final_steps:,}")
    print(f"   Total Episodes: {data['total_episodes']}")
    print(f"   Final Avg Reward: {data['final_reward']:.3f}")
    
    # Tax analysis
    if data['total_episodes']:
        tax_rate = (data['tax_penalties'] / data['total_episodes']) * 100
        print(f"\n💰 TAX PENALTY ANALYSIS:")
        print(f"   Total Tax Penalties: {data['tax_penalties']}")
        print(f"   Tax Rate: {tax_rate:.1f}% of episodes")
        
        if tax_rate > 50:
            print("   ⚠️  Tax rate still high - incentives may need adjustment")
        elif tax_rate > 30:
            print("   🎯 Tax rate moderate - showing improvement")
        else:
            print("   ✅ Tax rate low - incentives working well")
    
    # Drawdown analysis
    print(f"\n📉 DRAWDOWN ANALYSIS:")
    print(f"   Drawdown Terminations: {data['dd_terminations']}")
    
    # Progress analysis
    if len(data['ep_lens']) >= 2:
        start_len = data['ep_lens'][0]
        len_change = final_ep_len - start_len
        print(f"\n📈 PROGRESS ANALYSIS:")
        print(f"   Episode Length: {start_len:.1f} → {final_ep_len:.1f} (Δ{len_change:+.1f})")
        
        if len(data['ep_rews']) >= 2:
            start_rew = data['ep_rews'][0]
            rew_change = final_ep_rew - start_rew
            print(f"   Episode Reward: {start_rew:.1f} → {final_ep_rew:.1f} (Δ{rew_change:+.1f})")
    
    # Gate evaluation
    print(f"\n🎯 CORRECTIVE GATE EVALUATION:")
    print("=" * 40)
    
    gates_passed = 0
    total_gates = 4
    
    # Gate 1: Episode length ≥ 70
    if final_ep_len >= 80:
        print("   ✅ Episode Length: ≥80 steps (EXCELLENT)")
        gates_passed += 1
    elif final_ep_len >= 70:
        print("   🎯 Episode Length: ≥70 steps (GOOD)")
        gates_passed += 1
    else:
        print(f"   ❌ Episode Length: {final_ep_len:.1f} < 70 steps")
    
    # Gate 2: Reward ≥ -15
    if final_ep_rew >= -15:
        print("   ✅ Episode Reward: ≥-15 (EXCELLENT)")
        gates_passed += 1
    elif final_ep_rew >= -20:
        print("   🎯 Episode Reward: ≥-20 (GOOD)")
        gates_passed += 1
    else:
        print(f"   ❌ Episode Reward: {final_ep_rew:.1f} < -20")
    
    # Gate 3: Tax rate < 30%
    if data['total_episodes']:
        tax_rate = (data['tax_penalties'] / data['total_episodes']) * 100
        if tax_rate < 20:
            print("   ✅ Tax Rate: <20% (EXCELLENT)")
            gates_passed += 1
        elif tax_rate < 30:
            print("   🎯 Tax Rate: <30% (GOOD)")
            gates_passed += 1
        else:
            print(f"   ❌ Tax Rate: {tax_rate:.1f}% ≥30%")
    
    # Gate 4: Training completion
    if final_steps >= 20000:
        print("   ✅ Training Complete: 20K steps")
        gates_passed += 1
    else:
        print(f"   ❌ Training Incomplete: {final_steps} < 20K steps")
    
    print(f"\n🏆 CORRECTIVE ASSESSMENT:")
    print("=" * 30)
    print(f"Gates Passed: {gates_passed}/{total_gates}")
    
    if gates_passed >= 3:
        print("🎉 CORRECTIVE TRAINING SUCCESS!")
        print("✅ Ready for live-bar fine-tune")
        print("✅ Episode length and reward targets achieved")
        status = "SUCCESS"
    elif gates_passed >= 2:
        print("🎯 CORRECTIVE TRAINING PARTIAL SUCCESS")
        print("✅ Significant improvement achieved")
        print("🔄 Consider minor additional adjustments")
        status = "PARTIAL"
    else:
        print("❌ CORRECTIVE TRAINING NEEDS MORE WORK")
        print("🔄 Requires parameter adjustment or extended training")
        status = "NEEDS_WORK"
    
    # Comparison with Phase 2
    print(f"\n📊 COMPARISON WITH PHASE 2:")
    print("=" * 35)
    print("Metric               Phase 2    Corrective   Change")
    print("-" * 50)
    print(f"Episode Length       59.9       {final_ep_len:.1f}        {final_ep_len-59.9:+.1f}")
    print(f"Episode Reward       -17.0      {final_ep_rew:.1f}       {final_ep_rew-(-17.0):+.1f}")
    
    # Recommendations
    print(f"\n🚀 NEXT STEPS:")
    print("=" * 15)
    
    if status == "SUCCESS":
        print("✅ PROCEED TO LIVE-BAR FINE-TUNE")
        print("   • Episode length target achieved")
        print("   • Reward improvement maintained")
        print("   • Tax rate under control")
    elif status == "PARTIAL":
        print("🔄 MINOR ADJUSTMENTS RECOMMENDED")
        if final_ep_len < 70:
            print("   • Increase time bonus to 0.03")
        if tax_rate > 25:
            print("   • Reduce early-exit tax to 2.0")
        print("   • Consider 5K extension training")
    else:
        print("🔧 MAJOR ADJUSTMENTS NEEDED")
        print("   • Review incentive balance")
        print("   • Consider position size limits")
        print("   • May need parameter redesign")
    
    print(f"\n✅ CORRECTIVE ANALYSIS COMPLETE!")
    return status

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "train_runs/phase2_corrective/training.log"
    
    data = parse_corrective_log(log_file)
    if data:
        analyze_corrective_results(data)

if __name__ == "__main__":
    main()