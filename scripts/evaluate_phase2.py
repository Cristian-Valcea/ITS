#!/usr/bin/env python3
"""
Phase 2 OOS Evaluation Script
Calculates success metrics for curriculum training validation
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from episode returns"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def load_monitor_data(monitor_csv_path: str) -> pd.DataFrame:
    """Load and parse monitor.csv training data"""
    try:
        df = pd.read_csv(monitor_csv_path)
        
        # Standard monitor.csv columns from stable-baselines3
        expected_cols = ['r', 'l', 't']  # reward, length, time
        
        if not all(col in df.columns for col in expected_cols):
            print(f"Warning: Missing expected columns in {monitor_csv_path}")
            print(f"Available columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading {monitor_csv_path}: {e}")
        return pd.DataFrame()

def evaluate_oos_performance(train_run_path: str, 
                           test_period_start: str = "2024-01-01",
                           min_episodes: int = 100) -> Dict:
    """
    Evaluate out-of-sample performance for Phase 2
    
    Success Criteria:
    - Sharpe ratio ≥ 0.3
    - ep_rew_mean ≥ 0.1 on 2024 test data
    """
    
    monitor_path = Path(train_run_path) / "monitor.csv"
    
    if not monitor_path.exists():
        return {
            "success": False,
            "error": f"Monitor file not found: {monitor_path}",
            "sharpe_ratio": 0.0,
            "ep_rew_mean": 0.0,
            "episode_count": 0
        }
    
    # Load training data
    df = load_monitor_data(str(monitor_path))
    
    if df.empty:
        return {
            "success": False,
            "error": "Empty or invalid monitor data",
            "sharpe_ratio": 0.0,
            "ep_rew_mean": 0.0,
            "episode_count": 0
        }
    
    # Filter for test period (2024 data)
    # Note: 't' column contains timestamps
    if 't' in df.columns:
        # Convert timestamp to datetime for filtering
        df['datetime'] = pd.to_datetime(df['t'], unit='s')
        test_start = pd.to_datetime(test_period_start)
        
        # Filter for 2024 test period
        test_df = df[df['datetime'] >= test_start].copy()
    else:
        # If no timestamp filtering possible, use all data
        print("Warning: No timestamp column found, using all episodes")
        test_df = df.copy()
    
    if len(test_df) < min_episodes:
        return {
            "success": False,
            "error": f"Insufficient test episodes: {len(test_df)} < {min_episodes}",
            "sharpe_ratio": 0.0,
            "ep_rew_mean": 0.0,
            "episode_count": len(test_df)
        }
    
    # Calculate metrics
    episode_rewards = test_df['r'].values
    episode_lengths = test_df['l'].values if 'l' in test_df.columns else []
    
    # Core metrics
    ep_rew_mean = np.mean(episode_rewards)
    sharpe_ratio = calculate_sharpe_ratio(episode_rewards)
    
    # Additional statistics
    ep_rew_std = np.std(episode_rewards)
    ep_len_mean = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
    win_rate = np.mean(episode_rewards > 0) * 100
    
    # Success evaluation
    sharpe_success = sharpe_ratio >= 0.3
    reward_success = ep_rew_mean >= 0.1
    overall_success = sharpe_success and reward_success
    
    results = {
        "success": overall_success,
        "sharpe_ratio": float(sharpe_ratio),
        "ep_rew_mean": float(ep_rew_mean),
        "episode_count": len(test_df),
        "evaluation_period": test_period_start,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        # Success gates
        "sharpe_success": sharpe_success,
        "reward_success": reward_success,
        "sharpe_threshold": 0.3,
        "reward_threshold": 0.1,
        
        # Additional metrics
        "ep_rew_std": float(ep_rew_std),
        "ep_len_mean": float(ep_len_mean),
        "win_rate_pct": float(win_rate),
        "total_episodes": len(df),
        "test_episodes": len(test_df),
        
        # Run metadata
        "train_run_path": train_run_path,
        "monitor_file": str(monitor_path)
    }
    
    return results

def evaluate_multiple_runs(run_paths: List[str], output_file: Optional[str] = None) -> Dict:
    """Evaluate multiple training runs and aggregate results"""
    
    all_results = []
    successful_runs = 0
    
    for run_path in run_paths:
        print(f"\n📊 Evaluating: {run_path}")
        
        result = evaluate_oos_performance(run_path)
        all_results.append(result)
        
        if result["success"]:
            successful_runs += 1
            print(f"✅ SUCCESS - Sharpe: {result['sharpe_ratio']:.3f}, "
                  f"Reward: {result['ep_rew_mean']:.3f}")
        else:
            print(f"❌ FAILED - Sharpe: {result['sharpe_ratio']:.3f}, "
                  f"Reward: {result['ep_rew_mean']:.3f}")
            if "error" in result:
                print(f"   Error: {result['error']}")
    
    # Aggregate statistics
    valid_results = [r for r in all_results if not r.get("error")]
    
    if valid_results:
        aggregate_sharpe = np.mean([r["sharpe_ratio"] for r in valid_results])
        aggregate_reward = np.mean([r["ep_rew_mean"] for r in valid_results])
        std_sharpe = np.std([r["sharpe_ratio"] for r in valid_results])
        std_reward = np.std([r["ep_rew_mean"] for r in valid_results])
    else:
        aggregate_sharpe = aggregate_reward = std_sharpe = std_reward = 0.0
    
    summary = {
        "phase2_evaluation_summary": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(run_paths),
            "successful_runs": successful_runs,
            "success_rate": successful_runs / len(run_paths) if run_paths else 0.0,
            
            "aggregate_metrics": {
                "mean_sharpe": float(aggregate_sharpe),
                "mean_reward": float(aggregate_reward),
                "std_sharpe": float(std_sharpe),
                "std_reward": float(std_reward)
            },
            
            "overall_success": successful_runs > 0,
            "phase2_gate_passed": successful_runs >= len(run_paths) * 0.67,  # 2/3 success rate
        },
        
        "individual_results": all_results
    }
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
    
    return summary

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 OOS Evaluation")
    parser.add_argument('--run-path', type=str, 
                       help='Single training run path to evaluate')
    parser.add_argument('--run-pattern', type=str, 
                       default='train_runs/phase2_oos_seed*',
                       help='Glob pattern for multiple runs')
    parser.add_argument('--output', type=str,
                       default='phase2_evaluation_results.json',
                       help='Output JSON file')
    parser.add_argument('--test-start', type=str,
                       default='2024-01-01',
                       help='Test period start date')
    
    args = parser.parse_args()
    
    print("🔍 Phase 2 OOS Evaluation Starting...")
    print("=" * 50)
    
    if args.run_path:
        # Single run evaluation
        result = evaluate_oos_performance(args.run_path, args.test_start)
        
        print(f"\n📊 Results for {args.run_path}:")
        
        if result.get('success', False):
            print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f} (≥0.3: {'✅' if result.get('sharpe_success', False) else '❌'})")
            print(f"   Episode Reward: {result['ep_rew_mean']:.3f} (≥0.1: {'✅' if result.get('reward_success', False) else '❌'})")
            print(f"   Overall Success: ✅")
        else:
            print(f"   Status: ❌ FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"   Episodes Found: {result.get('episode_count', 0)}")
            if 'sharpe_ratio' in result:
                print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            if 'ep_rew_mean' in result:
                print(f"   Episode Reward: {result['ep_rew_mean']:.3f}")
        
        # Save single result
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
    else:
        # Multiple runs evaluation
        import glob
        run_paths = glob.glob(args.run_pattern)
        
        if not run_paths:
            print(f"❌ No runs found matching pattern: {args.run_pattern}")
            return 1
        
        print(f"📂 Found {len(run_paths)} runs to evaluate")
        
        summary = evaluate_multiple_runs(run_paths, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("📋 PHASE 2 EVALUATION SUMMARY")
        print("="*50)
        
        agg = summary["phase2_evaluation_summary"]
        print(f"Success Rate: {agg['successful_runs']}/{agg['total_runs']} ({agg['success_rate']:.1%})")
        print(f"Mean Sharpe: {agg['aggregate_metrics']['mean_sharpe']:.3f}")
        print(f"Mean Reward: {agg['aggregate_metrics']['mean_reward']:.3f}")
        print(f"Phase 2 Gate: {'✅ PASSED' if agg['phase2_gate_passed'] else '❌ FAILED'}")
    
    print("\n🎯 Evaluation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())