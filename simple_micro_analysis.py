#!/usr/bin/env python3
"""
🔬 SIMPLE MICRO-OVERFIT ANALYSIS
Compare 5K success model vs 8K micro-overfit model using existing evaluation script
"""

import subprocess
import pandas as pd
from pathlib import Path
import numpy as np

def run_evaluation(checkpoint_path, model_name, episodes=10):
    """Run evaluation using the existing evaluate_checkpoint.py script"""
    print(f"🔍 Evaluating {model_name}...")
    
    cmd = [
        "python", "evaluate_checkpoint.py",
        "--checkpoint", checkpoint_path,
        "--config", "config/reward_shim.yaml",
        "--eval-episodes", str(episodes)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/cristian/IntradayTrading/ITS")
        
        if result.returncode == 0:
            output = result.stdout
            print(f"✅ {model_name} evaluation completed")
            return parse_evaluation_output(output, model_name)
        else:
            print(f"❌ {model_name} evaluation failed:")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error running evaluation for {model_name}: {e}")
        return None

def parse_evaluation_output(output, model_name):
    """Parse the evaluation output to extract key metrics"""
    lines = output.split('\n')
    
    metrics = {
        'model': model_name,
        'total_return': None,
        'avg_reward_per_step': None,
        'episodes_completed': None,
        'final_portfolio_value': None,
        'max_drawdown': None,
        'action_distribution': {}
    }
    
    for line in lines:
        line = line.strip()
        
        if 'Total return:' in line:
            # Extract percentage: "Total return: -1.69%"
            try:
                pct_str = line.split('Total return:')[1].strip()
                metrics['total_return'] = float(pct_str.replace('%', ''))
            except:
                pass
                
        elif 'Average reward per step:' in line:
            try:
                reward_str = line.split('Average reward per step:')[1].strip()
                metrics['avg_reward_per_step'] = float(reward_str)
            except:
                pass
                
        elif 'Episodes completed:' in line:
            try:
                episodes_str = line.split('Episodes completed:')[1].strip()
                metrics['episodes_completed'] = int(episodes_str)
            except:
                pass
                
        elif 'Final portfolio value:' in line:
            try:
                value_str = line.split('Final portfolio value:')[1].strip()
                value_str = value_str.replace('$', '').replace(',', '')
                metrics['final_portfolio_value'] = float(value_str)
            except:
                pass
                
        elif 'Maximum drawdown:' in line:
            try:
                # Extract the second percentage: "Maximum drawdown: 0.02% (1.7%)"
                dd_str = line.split('(')[1].split(')')[0]
                metrics['max_drawdown'] = float(dd_str.replace('%', ''))
            except:
                pass
                
        elif 'Action' in line and '(' in line and '%' in line:
            # Parse action distribution: "Action 0 (SELL_BOTH): 22 (4.4%)"
            try:
                action_num = int(line.split('Action')[1].split('(')[0].strip())
                action_name = line.split('(')[1].split(')')[0]
                count_pct = line.split('):')[1].strip()
                count = int(count_pct.split('(')[0].strip())
                pct = float(count_pct.split('(')[1].split('%')[0])
                
                metrics['action_distribution'][action_num] = {
                    'name': action_name,
                    'count': count,
                    'percentage': pct
                }
            except:
                pass
    
    return metrics

def analyze_degradation(success_metrics, overfit_metrics):
    """Analyze the degradation between success and overfit models"""
    print(f"\n🔍 DEGRADATION ANALYSIS")
    print("=" * 50)
    
    if not success_metrics or not overfit_metrics:
        print("❌ Missing metrics for comparison")
        return
    
    # Performance comparison
    print(f"📊 PERFORMANCE COMPARISON:")
    print(f"   Model                 | Return  | Avg Reward | Episodes | Drawdown")
    print(f"   --------------------- | ------- | ---------- | -------- | --------")
    print(f"   Success (5K steps)    | {success_metrics['total_return']:6.2f}% | {success_metrics['avg_reward_per_step']:9.3f} | {success_metrics['episodes_completed']:8d} | {success_metrics['max_drawdown']:6.2f}%")
    print(f"   Overfit (8K steps)    | {overfit_metrics['total_return']:6.2f}% | {overfit_metrics['avg_reward_per_step']:9.3f} | {overfit_metrics['episodes_completed']:8d} | {overfit_metrics['max_drawdown']:6.2f}%")
    
    # Calculate degradation
    return_degradation = overfit_metrics['total_return'] - success_metrics['total_return']
    reward_degradation = overfit_metrics['avg_reward_per_step'] - success_metrics['avg_reward_per_step']
    episode_degradation = overfit_metrics['episodes_completed'] - success_metrics['episodes_completed']
    drawdown_degradation = overfit_metrics['max_drawdown'] - success_metrics['max_drawdown']
    
    print(f"\n🚨 DEGRADATION METRICS:")
    print(f"   Return Change:     {return_degradation:+6.2f}% ({return_degradation/abs(success_metrics['total_return'])*100:+5.1f}% relative)")
    print(f"   Reward Change:     {reward_degradation:+9.3f} ({reward_degradation/abs(success_metrics['avg_reward_per_step'])*100:+5.1f}% relative)")
    print(f"   Episode Change:    {episode_degradation:+8d} ({episode_degradation/max(success_metrics['episodes_completed'],1)*100:+5.1f}% relative)")
    print(f"   Drawdown Change:   {drawdown_degradation:+6.2f}% ({drawdown_degradation/max(success_metrics['max_drawdown'],0.1)*100:+5.1f}% relative)")
    
    # Analyze action patterns
    print(f"\n🎯 ACTION PATTERN ANALYSIS:")
    
    action_names = ['SELL_BOTH', 'SELL_NVDA_HOLD_MSFT', 'SELL_NVDA_BUY_MSFT', 
                   'HOLD_NVDA_SELL_MSFT', 'HOLD_BOTH', 'HOLD_NVDA_BUY_MSFT',
                   'BUY_NVDA_SELL_MSFT', 'BUY_NVDA_HOLD_MSFT', 'BUY_BOTH']
    
    print(f"   Action                | Success | Overfit | Change")
    print(f"   --------------------- | ------- | ------- | ------")
    
    for i, name in enumerate(action_names):
        success_pct = success_metrics['action_distribution'].get(i, {}).get('percentage', 0)
        overfit_pct = overfit_metrics['action_distribution'].get(i, {}).get('percentage', 0)
        change = overfit_pct - success_pct
        
        print(f"   {name:20s} | {success_pct:6.1f}% | {overfit_pct:6.1f}% | {change:+5.1f}%")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if overfit_metrics['episodes_completed'] == 0:
        print(f"   🚨 CRITICAL: Overfit model completes 0 episodes - hitting termination conditions immediately")
        print(f"   🔍 This suggests the model learned a strategy that triggers risk limits")
    
    if abs(reward_degradation) > 1.0:
        print(f"   ⚠️  SIGNIFICANT reward degradation: {reward_degradation:.3f}")
        print(f"   🔍 Model performance collapsed between 5K→8K steps")
    
    if abs(return_degradation) > 2.0:
        print(f"   ⚠️  SIGNIFICANT return degradation: {return_degradation:.2f}%")
        print(f"   🔍 Portfolio performance deteriorated substantially")
    
    # Hypothesis generation
    print(f"\n🧠 DEGRADATION HYPOTHESES:")
    
    if overfit_metrics['episodes_completed'] == 0 and overfit_metrics['max_drawdown'] > 20:
        print(f"   1. 🎯 REWARD LOOPHOLE: Model found actions that give short-term reward but trigger drawdown limits")
        print(f"   2. 🔄 POLICY INSTABILITY: Training became unstable around 6K-7K steps")
        print(f"   3. 🎲 EXPLORATION COLLAPSE: Model stopped exploring and converged to risky strategy")
    
    if reward_degradation < -1.0:
        print(f"   4. 📉 REWARD SYSTEM ISSUE: RefinedRewardSystem may have conflicting signals")
        print(f"   5. 🎛️ HYPERPARAMETER MISMATCH: Learning rate or other params may be too aggressive")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print(f"   1. 📊 Analyze Phase 1B results to see if all reward variants show same degradation")
    print(f"   2. 🔍 Inspect action sequences from 8K model to identify problematic patterns")
    print(f"   3. 📈 Plot training curves to find exact degradation point (likely 6K-7K steps)")
    print(f"   4. 🎛️ Test with lower learning rate or tighter KL divergence constraints")
    print(f"   5. 🔄 Try warm-starting from 5K model with frozen value network")

def main():
    print("🔬 MICRO-OVERFIT ANALYSIS")
    print("=" * 50)
    print("Comparing 5K success model vs 8K micro-overfit model")
    print()
    
    # Model paths
    success_model = "preserved_models/phase1a_5k_success_model.zip"
    overfit_model = "diagnostic_runs/micro_overfit_probe/micro_probe_final.zip"
    
    # Check if models exist
    if not Path(success_model).exists():
        print(f"❌ Success model not found: {success_model}")
        return
        
    if not Path(overfit_model).exists():
        print(f"❌ Overfit model not found: {overfit_model}")
        return
    
    # Run evaluations
    success_metrics = run_evaluation(success_model, "Success (5K)", episodes=5)
    overfit_metrics = run_evaluation(overfit_model, "Overfit (8K)", episodes=5)
    
    # Analyze results
    if success_metrics and overfit_metrics:
        analyze_degradation(success_metrics, overfit_metrics)
        
        # Save results
        results_file = Path("diagnostic_runs/micro_overfit_probe/analysis_results.txt")
        with open(results_file, 'w') as f:
            f.write("MICRO-OVERFIT ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Success Model Return: {success_metrics['total_return']:.2f}%\n")
            f.write(f"Overfit Model Return: {overfit_metrics['total_return']:.2f}%\n")
            f.write(f"Degradation: {overfit_metrics['total_return'] - success_metrics['total_return']:.2f}%\n")
            f.write(f"Success Episodes: {success_metrics['episodes_completed']}\n")
            f.write(f"Overfit Episodes: {overfit_metrics['episodes_completed']}\n")
        
        print(f"\n✅ Analysis complete! Results saved to: {results_file}")
    else:
        print(f"❌ Analysis failed - missing evaluation results")

if __name__ == "__main__":
    main()