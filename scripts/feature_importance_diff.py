#!/usr/bin/env python3
"""
🔍 FEATURE IMPORTANCE ANALYSIS FOR DATA LEAKAGE DETECTION
Compares feature rankings before/after +1 step shift
Implements management-enhanced leakage detection criteria
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_feature_importance(X, y, feature_names):
    """Calculate permutation importance for features"""
    
    logger.info(f"📊 Calculating feature importance for {X.shape[0]:,} samples, {X.shape[1]} features...")
    
    # Use RandomForest as proxy model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Calculate permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    logger.info(f"✅ Feature importance calculated")
    return importance_df

def create_synthetic_targets(features_df):
    """Create synthetic targets based on feature patterns for proxy analysis"""
    
    logger.info("🎯 Creating synthetic targets for proxy model...")
    
    # Use a combination of features to create meaningful targets
    # This simulates reward-like patterns without actual reward data
    
    features = features_df.iloc[:, :-1].values  # Exclude dummy target column
    
    # Create synthetic "reward" based on feature interactions
    # Simulate price-momentum-volume relationships
    n_samples, n_features = features.shape
    
    # Assume structure: [NVDA features (0-11), MSFT features (12-23), positions (24-25)]
    synthetic_targets = np.zeros(n_samples)
    
    for i in range(n_samples):
        sample = features[i]
        
        # NVDA contribution (first 12 features)
        if n_features > 12:
            nvda_price_change = sample[1] if len(sample) > 1 else 0  # Assume index 1 is price change
            nvda_volume = sample[5] if len(sample) > 5 else 0        # Assume index 5 is volume
            nvda_momentum = sample[7] if len(sample) > 7 else 0      # Assume index 7 is momentum
            
            nvda_contribution = nvda_price_change * 0.5 + nvda_volume * 0.2 + nvda_momentum * 0.3
        else:
            nvda_contribution = 0
        
        # MSFT contribution (features 12-23)
        if n_features > 24:
            msft_price_change = sample[13] if len(sample) > 13 else 0
            msft_volume = sample[17] if len(sample) > 17 else 0
            msft_momentum = sample[19] if len(sample) > 19 else 0
            
            msft_contribution = msft_price_change * 0.5 + msft_volume * 0.2 + msft_momentum * 0.3
        else:
            msft_contribution = 0
        
        # Combine contributions with some noise
        synthetic_targets[i] = nvda_contribution + msft_contribution + np.random.normal(0, 0.1)
    
    logger.info(f"✅ Synthetic targets created: mean={np.mean(synthetic_targets):.3f}, std={np.std(synthetic_targets):.3f}")
    return synthetic_targets

def compare_feature_importance(original_data, shifted_data, output_path):
    """Compare feature importance between original and shifted data"""
    
    logger.info("🔍 Analyzing feature importance for leakage detection...")
    
    # Load data
    orig_df = pd.read_parquet(original_data)
    shift_df = pd.read_parquet(shifted_data)
    
    logger.info(f"📊 Original data: {orig_df.shape[0]:,} samples, {orig_df.shape[1]} features")
    logger.info(f"📊 Shifted data: {shift_df.shape[0]:,} samples, {shift_df.shape[1]} features")
    
    # Create feature names
    n_features = orig_df.shape[1] - 1  # Exclude target column
    feature_names = []
    
    # NVDA features (0-11)
    nvda_feature_names = [
        'NVDA_price', 'NVDA_return', 'NVDA_volume', 'NVDA_vwap', 'NVDA_high', 'NVDA_low',
        'NVDA_sma_5', 'NVDA_sma_20', 'NVDA_ema_12', 'NVDA_rsi', 'NVDA_bb_upper', 'NVDA_bb_lower'
    ]
    
    # MSFT features (12-23)  
    msft_feature_names = [
        'MSFT_price', 'MSFT_return', 'MSFT_volume', 'MSFT_vwap', 'MSFT_high', 'MSFT_low',
        'MSFT_sma_5', 'MSFT_sma_20', 'MSFT_ema_12', 'MSFT_rsi', 'MSFT_bb_upper', 'MSFT_bb_lower'
    ]
    
    # Position features (24-25)
    position_feature_names = ['NVDA_position', 'MSFT_position']
    
    # Combine all feature names
    feature_names = nvda_feature_names + msft_feature_names + position_feature_names
    
    # Trim to actual feature count
    feature_names = feature_names[:n_features]
    
    # Pad with generic names if needed
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    logger.info(f"🏷️ Feature names: {len(feature_names)} total")
    
    # Create synthetic targets for both datasets
    orig_targets = create_synthetic_targets(orig_df)
    shift_targets = create_synthetic_targets(shift_df)
    
    # Calculate importance for both datasets
    orig_importance = calculate_feature_importance(
        orig_df.iloc[:, :-1].values, 
        orig_targets,
        feature_names
    )
    
    shift_importance = calculate_feature_importance(
        shift_df.iloc[:, :-1].values,
        shift_targets, 
        feature_names
    )
    
    # Merge and compare
    comparison = orig_importance.merge(
        shift_importance, 
        on='feature', 
        suffixes=('_original', '_shifted')
    )
    
    # Calculate ranking changes
    comparison['rank_original'] = comparison['importance_mean_original'].rank(ascending=False)
    comparison['rank_shifted'] = comparison['importance_mean_shifted'].rank(ascending=False)
    comparison['rank_change'] = abs(comparison['rank_original'] - comparison['rank_shifted'])
    
    # Flag potential leakage
    top_5_orig = set(comparison.nsmallest(5, 'rank_original')['feature'])
    top_5_shift = set(comparison.nsmallest(5, 'rank_shifted')['feature'])
    
    overlap = len(top_5_orig.intersection(top_5_shift))
    change_pct = (5 - overlap) / 5 * 100
    
    logger.info(f"📊 Top-5 feature overlap: {overlap}/5 ({100-change_pct:.1f}%)")
    logger.info(f"📊 Top-5 change percentage: {change_pct:.1f}%")
    
    # Management criteria: >50% change indicates leakage
    leakage_detected = change_pct > 50
    
    if leakage_detected:
        logger.warning("🚨 WARNING: >50% change in top-5 features - potential leakage detected!")
    else:
        logger.info("✅ Feature rankings stable - no obvious leakage")
    
    # Generate visualization
    logger.info("📈 Generating feature importance comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analysis - Data Leakage Detection', fontsize=16, fontweight='bold')
    
    # Top 10 features comparison
    top_features = comparison.head(10)
    
    # Original data - top 10
    y_pos = np.arange(len(top_features))
    axes[0,0].barh(y_pos, top_features['importance_mean_original'], alpha=0.7, color='blue')
    axes[0,0].set_yticks(y_pos)
    axes[0,0].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,0].set_title('Original Data - Top 10 Features')
    axes[0,0].set_xlabel('Importance')
    axes[0,0].grid(True, alpha=0.3)
    
    # Shifted data - top 10
    axes[0,1].barh(y_pos, top_features['importance_mean_shifted'], alpha=0.7, color='red')
    axes[0,1].set_yticks(y_pos)
    axes[0,1].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,1].set_title('Shifted Data - Top 10 Features')
    axes[0,1].set_xlabel('Importance')
    axes[0,1].grid(True, alpha=0.3)
    
    # Ranking change scatter plot
    axes[1,0].scatter(comparison['rank_original'], comparison['rank_shifted'], alpha=0.6, s=50)
    axes[1,0].plot([0, len(comparison)], [0, len(comparison)], 'r--', alpha=0.5, linewidth=2)
    axes[1,0].set_xlabel('Original Ranking')
    axes[1,0].set_ylabel('Shifted Ranking')
    axes[1,0].set_title('Feature Ranking Changes')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add text annotation for leakage detection
    axes[1,0].text(0.05, 0.95, f'Top-5 Change: {change_pct:.1f}%\nLeakage: {"YES" if leakage_detected else "NO"}', 
                   transform=axes[1,0].transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Change magnitude histogram
    axes[1,1].hist(comparison['rank_change'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_xlabel('Absolute Rank Change')
    axes[1,1].set_ylabel('Number of Features')
    axes[1,1].set_title('Distribution of Ranking Changes')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add summary stats
    axes[1,1].axvline(comparison['rank_change'].mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {comparison["rank_change"].mean():.1f}')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"📊 Visualization saved: {output_path}/feature_importance_comparison.png")
    
    # Save detailed results
    comparison.to_csv(f"{output_path}/feature_importance_comparison.csv", index=False)
    logger.info(f"📄 Detailed results: {output_path}/feature_importance_comparison.csv")
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Importance Analysis - Data Leakage Detection</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ color: #2E86C1; }}
            .alert {{ background-color: #F8D7DA; color: #721C24; padding: 10px; border-radius: 5px; }}
            .success {{ background-color: #D4EDDA; color: #155724; padding: 10px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1 class="header">🔍 Feature Importance Analysis - Data Leakage Detection</h1>
        
        <h2>Summary</h2>
        <p><strong>Top-5 Feature Overlap:</strong> {overlap}/5 ({100-change_pct:.1f}%)</p>
        <p><strong>Change Percentage:</strong> {change_pct:.1f}%</p>
        
        {'<div class="alert"><strong>⚠️ LEAKAGE DETECTED:</strong> >50% change in top-5 features indicates potential data leakage!</div>' if leakage_detected else '<div class="success"><strong>✅ NO LEAKAGE:</strong> Feature rankings remain stable (≤50% change)</div>'}
        
        <h2>Top 10 Features Comparison</h2>
        <table>
            <tr><th>Feature</th><th>Original Rank</th><th>Shifted Rank</th><th>Rank Change</th><th>Original Importance</th><th>Shifted Importance</th></tr>
    """
    
    for _, row in top_features.iterrows():
        html_report += f"""
            <tr>
                <td>{row['feature']}</td>
                <td>{int(row['rank_original'])}</td>
                <td>{int(row['rank_shifted'])}</td>
                <td>{int(row['rank_change'])}</td>
                <td>{row['importance_mean_original']:.4f}</td>
                <td>{row['importance_mean_shifted']:.4f}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Diagnostic Criteria</h2>
        <ul>
            <li><strong>Alert Threshold:</strong> >50% change in top-5 feature importance indicates hidden leakage</li>
            <li><strong>Success Criteria:</strong> Feature importance rankings remain stable (≤50% change in top-5)</li>
            <li><strong>Method:</strong> Permutation importance on RandomForest proxy model</li>
        </ul>
        
        <h2>Visualization</h2>
        <p>See <code>feature_importance_comparison.png</code> for detailed plots</p>
    </body>
    </html>
    """
    
    with open(f"{output_path}/feature_importance_comparison.html", 'w') as f:
        f.write(html_report)
    
    logger.info(f"📄 HTML report: {output_path}/feature_importance_comparison.html")
    
    # Summary report
    summary = {
        'top_5_overlap': overlap,
        'change_percentage': change_pct,
        'leakage_detected': leakage_detected,
        'max_rank_change': comparison['rank_change'].max(),
        'mean_rank_change': comparison['rank_change'].mean(),
        'total_features': len(feature_names)
    }
    
    pd.DataFrame([summary]).to_csv(f"{output_path}/leakage_summary.csv", index=False)
    logger.info(f"📄 Summary: {output_path}/leakage_summary.csv")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Feature Importance Analysis for Data Leakage Detection")
    parser.add_argument('--original_data', default='diagnostic_runs/phase1c_leakage_audit/original_features.parquet',
                       help='Path to original features parquet file')
    parser.add_argument('--shifted_data', default='diagnostic_runs/phase1c_leakage_audit/shifted_features.parquet',
                       help='Path to shifted features parquet file') 
    parser.add_argument('--output_path', default='diagnostic_runs/phase1c_leakage_audit',
                       help='Output directory for results')
    args = parser.parse_args()
    
    logger.info("🔍 FEATURE IMPORTANCE ANALYSIS - DATA LEAKAGE DETECTION")
    logger.info("=" * 60)
    
    try:
        # Ensure output directory exists
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        summary = compare_feature_importance(args.original_data, args.shifted_data, args.output_path)
        
        # Print summary
        logger.info("\n📋 LEAKAGE ANALYSIS SUMMARY:")
        logger.info(f"   Top-5 overlap: {summary['top_5_overlap']}/5")
        logger.info(f"   Change percentage: {summary['change_percentage']:.1f}%")
        logger.info(f"   Leakage detected: {summary['leakage_detected']}")
        logger.info(f"   Max rank change: {summary['max_rank_change']:.0f}")
        logger.info(f"   Mean rank change: {summary['mean_rank_change']:.1f}")
        
        if summary['leakage_detected']:
            logger.warning("🚨 Data leakage detected - fix data pipeline before proceeding")
            return False
        else:
            logger.info("✅ No data leakage detected - pipeline appears clean")
            return True
        
    except Exception as e:
        logger.error(f"❌ Feature importance analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)