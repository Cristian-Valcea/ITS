#!/usr/bin/env python3
"""
CI Validation Script for Filtering Ablation Study
Auto-generated script that fails CI if earnings exclusion doesn't show improvement
"""

import sys
import json
from pathlib import Path

def main():
    """Validate filtering ablation results show earnings exclusion improvement"""
    
    results_dir = Path("studies/filtering_ablation_results")
    
    # Load actual results
    earnings_included_file = results_dir / "config_earnings_included" / "performance_summary.json"
    earnings_excluded_file = results_dir / "config_earnings_excluded" / "performance_summary.json"
    
    if not earnings_included_file.exists() or not earnings_excluded_file.exists():
        print("❌ CI FAILURE: Ablation study results not found")
        print("   Run: python studies/filtering_ablation_study.py")
        sys.exit(1)
    
    # Load results
    with open(earnings_included_file) as f:
        included_results = json.load(f)
    
    with open(earnings_excluded_file) as f:
        excluded_results = json.load(f)
    
    # Get performance metrics
    included_sharpe = included_results['performance']['sharpe_ratio']
    included_dd = included_results['performance']['max_drawdown_pct']
    
    excluded_sharpe = excluded_results['performance']['sharpe_ratio']
    excluded_dd = excluded_results['performance']['max_drawdown_pct']
    
    # Validate improvement direction (at least one metric should improve)
    sharpe_improvement = excluded_sharpe >= included_sharpe
    dd_improvement = excluded_dd <= included_dd
    
    if not (sharpe_improvement or dd_improvement):
        print("❌ CI FAILURE: Earnings exclusion shows no performance improvement")
        print(f"   Sharpe: {included_sharpe:.4f} → {excluded_sharpe:.4f}")
        print(f"   Max DD: {included_dd:.2f}% → {excluded_dd:.2f}%")
        sys.exit(1)
    
    print("✅ CI SUCCESS: Earnings exclusion shows performance improvement")
    print(f"   Sharpe improvement: {'✅' if sharpe_improvement else '❌'}")
    print(f"   Drawdown improvement: {'✅' if dd_improvement else '❌'}")
    print(f"   Earnings impact: Sharpe {included_sharpe:.4f} → {excluded_sharpe:.4f}")
    print(f"   Drawdown impact: {included_dd:.2f}% → {excluded_dd:.2f}%")
    
    # Store lock-box hashes for audit
    from datetime import datetime
    lockbox_hashes = {
        'earnings_included_hash': included_results['lock_box_hash'],
        'earnings_excluded_hash': excluded_results['lock_box_hash'],
        'validation_timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / "lockbox_audit_hashes.json", 'w') as f:
        json.dump(lockbox_hashes, f, indent=2)
    
    print("🔒 Lock-box hashes stored for audit compliance")

if __name__ == "__main__":
    main()
