#!/usr/bin/env python3
"""
Reward System Audit Script - Team B Implementation
Automated quarterly audits of reward system components

Usage:
    python scripts/audit_rewards.py --audit-type full --output-dir results/audits/
    python scripts/audit_rewards.py --audit-type reward_components --lookback-days 90
    python scripts/audit_rewards.py --audit-type risk_metrics --generate-plots
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class RewardSystemAuditor:
    """Comprehensive reward system auditor for quarterly reviews"""
    
    def __init__(self, output_dir: Path, lookback_days: int = 90):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_days = lookback_days
        self.audit_timestamp = datetime.now()
        
        # Audit thresholds (Team A: customize based on your system)
        self.thresholds = {
            'critical': {
                'reward_variance_ratio': 2.0,      # σ/μ > 2.0 is critical
                'negative_reward_rate': 0.7,       # >70% negative episodes
                'extreme_drawdown_rate': 0.1,      # >10% extreme drawdowns
                'reward_component_imbalance': 0.8   # >80% from single component
            },
            'warning': {
                'reward_variance_ratio': 1.0,      # σ/μ > 1.0 is warning
                'negative_reward_rate': 0.5,       # >50% negative episodes
                'extreme_drawdown_rate': 0.05,     # >5% extreme drawdowns
                'reward_component_imbalance': 0.6   # >60% from single component
            }
        }
        
        print(f"🔍 Reward System Auditor initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📅 Lookback period: {self.lookback_days} days")
    
    def find_recent_training_runs(self) -> List[Path]:
        """Find training runs from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        # Search common training run directories
        search_paths = [
            Path('train_runs'),
            Path('results'),
            Path('logs'),
            Path('diagnostic_runs')
        ]
        
        recent_runs = []
        
        for search_path in search_paths:
            if search_path.exists():
                for run_dir in search_path.iterdir():
                    if run_dir.is_dir():
                        # Check if directory was modified recently
                        try:
                            mod_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
                            if mod_time > cutoff_date:
                                # Look for monitor.csv or similar training artifacts
                                if any(run_dir.glob('monitor.csv')) or any(run_dir.glob('*.log')):
                                    recent_runs.append(run_dir)
                        except (OSError, ValueError):
                            continue
        
        print(f"📊 Found {len(recent_runs)} recent training runs")
        return recent_runs
    
    def load_training_data(self, run_paths: List[Path]) -> pd.DataFrame:
        """Load training data from multiple runs"""
        all_data = []
        
        for run_path in run_paths:
            # Try to load monitor.csv
            monitor_file = run_path / 'monitor.csv'
            if monitor_file.exists():
                try:
                    df = pd.read_csv(monitor_file, comment='#')
                    df['run_path'] = str(run_path)
                    df['run_name'] = run_path.name
                    all_data.append(df)
                    print(f"✅ Loaded {len(df)} episodes from {run_path.name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {monitor_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"📈 Total episodes loaded: {len(combined_df)}")
            return combined_df
        else:
            print("❌ No training data found")
            return pd.DataFrame()
    
    def audit_reward_components(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit reward system components"""
        print("\n🔍 Auditing reward components...")
        
        if df.empty:
            return {'error': 'No data available for reward component audit'}
        
        # Standardize column names
        reward_col = 'r' if 'r' in df.columns else 'episode_reward'
        length_col = 'l' if 'l' in df.columns else 'episode_length'
        
        if reward_col not in df.columns:
            return {'error': f'No reward column found. Available: {list(df.columns)}'}
        
        rewards = df[reward_col].values
        
        # Basic reward statistics
        reward_stats = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards)),
            'count': len(rewards)
        }
        
        # Variance ratio (σ/μ)
        variance_ratio = reward_stats['std'] / abs(reward_stats['mean']) if reward_stats['mean'] != 0 else float('inf')
        
        # Negative reward rate
        negative_rate = np.sum(rewards < 0) / len(rewards) if len(rewards) > 0 else 0
        
        # Reward distribution analysis
        percentiles = np.percentile(rewards, [5, 25, 75, 95])
        
        # Issue detection
        issues = []
        if variance_ratio > self.thresholds['critical']['reward_variance_ratio']:
            issues.append({
                'level': 'critical',
                'type': 'high_variance',
                'message': f'Reward variance ratio {variance_ratio:.2f} exceeds critical threshold',
                'value': variance_ratio,
                'threshold': self.thresholds['critical']['reward_variance_ratio']
            })
        elif variance_ratio > self.thresholds['warning']['reward_variance_ratio']:
            issues.append({
                'level': 'warning',
                'type': 'moderate_variance',
                'message': f'Reward variance ratio {variance_ratio:.2f} exceeds warning threshold',
                'value': variance_ratio,
                'threshold': self.thresholds['warning']['reward_variance_ratio']
            })
        
        if negative_rate > self.thresholds['critical']['negative_reward_rate']:
            issues.append({
                'level': 'critical',
                'type': 'high_negative_rate',
                'message': f'Negative reward rate {negative_rate:.1%} exceeds critical threshold',
                'value': negative_rate,
                'threshold': self.thresholds['critical']['negative_reward_rate']
            })
        elif negative_rate > self.thresholds['warning']['negative_reward_rate']:
            issues.append({
                'level': 'warning',
                'type': 'moderate_negative_rate',
                'message': f'Negative reward rate {negative_rate:.1%} exceeds warning threshold',
                'value': negative_rate,
                'threshold': self.thresholds['warning']['negative_reward_rate']
            })
        
        audit_result = {
            'audit_type': 'reward_components',
            'timestamp': self.audit_timestamp.isoformat(),
            'data_summary': {
                'total_episodes': len(rewards),
                'date_range': f"{self.audit_timestamp - timedelta(days=self.lookback_days)} to {self.audit_timestamp}",
                'runs_analyzed': df['run_name'].nunique() if 'run_name' in df.columns else 1
            },
            'reward_statistics': reward_stats,
            'key_metrics': {
                'variance_ratio': float(variance_ratio),
                'negative_reward_rate': float(negative_rate),
                'reward_percentiles': {
                    'p5': float(percentiles[0]),
                    'p25': float(percentiles[1]),
                    'p75': float(percentiles[2]),
                    'p95': float(percentiles[3])
                }
            },
            'issues': issues,
            'recommendations': self.generate_reward_recommendations(reward_stats, variance_ratio, negative_rate)
        }
        
        print(f"📊 Reward audit complete: {len(issues)} issues found")
        return audit_result
    
    def audit_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit risk-related metrics"""
        print("\n🔍 Auditing risk metrics...")
        
        if df.empty:
            return {'error': 'No data available for risk metrics audit'}
        
        reward_col = 'r' if 'r' in df.columns else 'episode_reward'
        length_col = 'l' if 'l' in df.columns else 'episode_length'
        
        if reward_col not in df.columns:
            return {'error': f'No reward column found for risk analysis'}
        
        rewards = df[reward_col].values
        
        # Calculate risk metrics
        # Sharpe ratio (episode-level)
        sharpe_ratio = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0
        
        # Drawdown analysis (simplified)
        cumulative_rewards = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative_rewards)
        drawdowns = cumulative_rewards - running_max
        max_drawdown = np.min(drawdowns)
        
        # Extreme drawdown episodes (bottom 5%)
        extreme_threshold = np.percentile(rewards, 5)
        extreme_drawdown_rate = np.sum(rewards <= extreme_threshold) / len(rewards)
        
        # Volatility analysis
        volatility = np.std(rewards)
        
        # Issue detection
        issues = []
        if extreme_drawdown_rate > self.thresholds['critical']['extreme_drawdown_rate']:
            issues.append({
                'level': 'critical',
                'type': 'high_extreme_drawdown_rate',
                'message': f'Extreme drawdown rate {extreme_drawdown_rate:.1%} exceeds critical threshold',
                'value': extreme_drawdown_rate,
                'threshold': self.thresholds['critical']['extreme_drawdown_rate']
            })
        elif extreme_drawdown_rate > self.thresholds['warning']['extreme_drawdown_rate']:
            issues.append({
                'level': 'warning',
                'type': 'moderate_extreme_drawdown_rate',
                'message': f'Extreme drawdown rate {extreme_drawdown_rate:.1%} exceeds warning threshold',
                'value': extreme_drawdown_rate,
                'threshold': self.thresholds['warning']['extreme_drawdown_rate']
            })
        
        audit_result = {
            'audit_type': 'risk_metrics',
            'timestamp': self.audit_timestamp.isoformat(),
            'risk_metrics': {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'extreme_drawdown_rate': float(extreme_drawdown_rate)
            },
            'issues': issues,
            'recommendations': self.generate_risk_recommendations(sharpe_ratio, max_drawdown, extreme_drawdown_rate)
        }
        
        print(f"📊 Risk audit complete: {len(issues)} issues found")
        return audit_result
    
    def audit_performance_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit overall performance trends"""
        print("\n🔍 Auditing performance trends...")
        
        if df.empty:
            return {'error': 'No data available for performance analysis'}
        
        reward_col = 'r' if 'r' in df.columns else 'episode_reward'
        
        if reward_col not in df.columns:
            return {'error': f'No reward column found for performance analysis'}
        
        # Performance trend analysis
        rewards = df[reward_col].values
        
        # Split into time periods for trend analysis
        n_periods = min(4, len(rewards) // 10)  # At least 10 episodes per period
        if n_periods < 2:
            trend_analysis = {'error': 'Insufficient data for trend analysis'}
        else:
            period_size = len(rewards) // n_periods
            period_means = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(rewards)
                period_rewards = rewards[start_idx:end_idx]
                period_means.append(np.mean(period_rewards))
            
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(period_means))
            trend_slope = np.polyfit(x, period_means, 1)[0] if len(period_means) > 1 else 0
            
            trend_analysis = {
                'periods': n_periods,
                'period_means': [float(x) for x in period_means],
                'trend_slope': float(trend_slope),
                'trend_direction': 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
            }
        
        # Recent performance vs historical
        recent_episodes = min(50, len(rewards) // 4)  # Last 25% or 50 episodes
        if recent_episodes > 10:
            recent_performance = np.mean(rewards[-recent_episodes:])
            historical_performance = np.mean(rewards[:-recent_episodes])
            performance_change = recent_performance - historical_performance
        else:
            recent_performance = np.mean(rewards)
            historical_performance = np.mean(rewards)
            performance_change = 0
        
        audit_result = {
            'audit_type': 'performance_analysis',
            'timestamp': self.audit_timestamp.isoformat(),
            'trend_analysis': trend_analysis,
            'performance_comparison': {
                'recent_performance': float(recent_performance),
                'historical_performance': float(historical_performance),
                'performance_change': float(performance_change),
                'recent_episodes_analyzed': recent_episodes
            },
            'issues': [],  # Performance issues would be detected in other audits
            'recommendations': self.generate_performance_recommendations(trend_analysis, performance_change)
        }
        
        print(f"📊 Performance audit complete")
        return audit_result
    
    def generate_reward_recommendations(self, stats: Dict, variance_ratio: float, negative_rate: float) -> List[str]:
        """Generate recommendations based on reward audit"""
        recommendations = []
        
        if variance_ratio > 1.5:
            recommendations.append("Consider reducing reward system volatility through smoothing or normalization")
        
        if negative_rate > 0.6:
            recommendations.append("High negative reward rate suggests need for reward system rebalancing")
        
        if stats['mean'] < 0:
            recommendations.append("Negative mean reward indicates fundamental reward system issues")
        
        if abs(stats['mean']) < 0.01:
            recommendations.append("Very low mean reward suggests insufficient learning signal")
        
        return recommendations
    
    def generate_risk_recommendations(self, sharpe: float, max_dd: float, extreme_rate: float) -> List[str]:
        """Generate recommendations based on risk audit"""
        recommendations = []
        
        if sharpe < 0.5:
            recommendations.append("Low Sharpe ratio suggests poor risk-adjusted returns")
        
        if extreme_rate > 0.1:
            recommendations.append("High extreme drawdown rate indicates insufficient risk controls")
        
        if max_dd < -100:  # Arbitrary threshold
            recommendations.append("Excessive maximum drawdown suggests need for stricter risk limits")
        
        return recommendations
    
    def generate_performance_recommendations(self, trend: Dict, change: float) -> List[str]:
        """Generate recommendations based on performance audit"""
        recommendations = []
        
        if 'trend_direction' in trend:
            if trend['trend_direction'] == 'declining':
                recommendations.append("Declining performance trend suggests need for model retraining or parameter adjustment")
            elif trend['trend_direction'] == 'stable' and change < 0:
                recommendations.append("Stable but declining recent performance may indicate overfitting")
        
        if change < -0.1:
            recommendations.append("Significant recent performance decline warrants immediate investigation")
        
        return recommendations
    
    def generate_plots(self, df: pd.DataFrame, audit_results: List[Dict]):
        """Generate audit visualization plots"""
        if df.empty:
            print("⚠️ No data available for plot generation")
            return
        
        print("📊 Generating audit plots...")
        
        reward_col = 'r' if 'r' in df.columns else 'episode_reward'
        if reward_col not in df.columns:
            print("⚠️ No reward column found for plotting")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Reward distribution plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df[reward_col], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(df[reward_col].mean(), color='red', linestyle='--', label=f'Mean: {df[reward_col].mean():.3f}')
        plt.title('Episode Reward Distribution')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 2. Reward time series
        plt.subplot(2, 2, 2)
        plt.plot(df[reward_col], alpha=0.7)
        plt.axhline(df[reward_col].mean(), color='red', linestyle='--', label='Mean')
        plt.title('Episode Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # 3. Cumulative reward
        plt.subplot(2, 2, 3)
        cumulative_rewards = df[reward_col].cumsum()
        plt.plot(cumulative_rewards)
        plt.title('Cumulative Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        
        # 4. Rolling statistics
        plt.subplot(2, 2, 4)
        window = min(50, len(df) // 10)
        if window > 1:
            rolling_mean = df[reward_col].rolling(window).mean()
            rolling_std = df[reward_col].rolling(window).std()
            plt.plot(rolling_mean, label=f'Rolling Mean ({window})')
            plt.fill_between(range(len(rolling_mean)), 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.3, label='±1 Std')
            plt.title(f'Rolling Statistics (Window: {window})')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'reward_audit_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to {plots_dir}")
    
    def run_full_audit(self, generate_plots: bool = False) -> Dict[str, Any]:
        """Run comprehensive audit of all components"""
        print(f"\n🎯 Starting full reward system audit...")
        print(f"📅 Audit timestamp: {self.audit_timestamp}")
        
        # Find and load recent training data
        recent_runs = self.find_recent_training_runs()
        training_data = self.load_training_data(recent_runs)
        
        # Run individual audits
        audit_results = []
        
        # Reward components audit
        reward_audit = self.audit_reward_components(training_data)
        audit_results.append(reward_audit)
        
        # Risk metrics audit
        risk_audit = self.audit_risk_metrics(training_data)
        audit_results.append(risk_audit)
        
        # Performance analysis audit
        performance_audit = self.audit_performance_analysis(training_data)
        audit_results.append(performance_audit)
        
        # Generate plots if requested
        if generate_plots:
            self.generate_plots(training_data, audit_results)
        
        # Compile comprehensive summary
        all_issues = []
        for audit in audit_results:
            if 'issues' in audit:
                all_issues.extend(audit['issues'])
        
        critical_issues = [issue for issue in all_issues if issue.get('level') == 'critical']
        warning_issues = [issue for issue in all_issues if issue.get('level') == 'warning']
        
        comprehensive_summary = {
            'audit_timestamp': self.audit_timestamp.isoformat(),
            'audit_type': 'comprehensive',
            'lookback_days': self.lookback_days,
            'data_summary': {
                'training_runs_found': len(recent_runs),
                'total_episodes': len(training_data) if not training_data.empty else 0,
                'date_range': f"{self.audit_timestamp - timedelta(days=self.lookback_days)} to {self.audit_timestamp}"
            },
            'audit_results': audit_results,
            'issue_summary': {
                'total_issues': len(all_issues),
                'critical_issues': len(critical_issues),
                'warning_issues': len(warning_issues),
                'critical_issues_count': len(critical_issues),  # For CI workflow
                'warning_issues_count': len(warning_issues)     # For CI workflow
            },
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'overall_status': 'critical' if critical_issues else 'warning' if warning_issues else 'healthy',
            'recommendations': self.compile_overall_recommendations(audit_results)
        }
        
        # Save comprehensive summary
        summary_file = self.output_dir / 'audit_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        
        print(f"\n📋 AUDIT SUMMARY:")
        print(f"   Total Issues: {len(all_issues)}")
        print(f"   Critical: {len(critical_issues)}")
        print(f"   Warning: {len(warning_issues)}")
        print(f"   Overall Status: {comprehensive_summary['overall_status'].upper()}")
        print(f"   Summary saved: {summary_file}")
        
        return comprehensive_summary
    
    def compile_overall_recommendations(self, audit_results: List[Dict]) -> List[str]:
        """Compile recommendations from all audits"""
        all_recommendations = []
        
        for audit in audit_results:
            if 'recommendations' in audit:
                all_recommendations.extend(audit['recommendations'])
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations

def main():
    """Main CLI interface for reward system auditor"""
    parser = argparse.ArgumentParser(
        description="Automated Reward System Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--audit-type',
        choices=['full', 'reward_components', 'risk_metrics', 'performance_analysis'],
        default='full',
        help='Type of audit to perform'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='results/quarterly_audits',
        help='Output directory for audit results'
    )
    
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=90,
        help='Number of days to look back for analysis'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir / f'audit_{timestamp}'
    
    # Initialize auditor
    auditor = RewardSystemAuditor(output_dir, args.lookback_days)
    
    # Run requested audit
    if args.audit_type == 'full':
        results = auditor.run_full_audit(generate_plots=args.generate_plots)
    else:
        # Run specific audit type
        recent_runs = auditor.find_recent_training_runs()
        training_data = auditor.load_training_data(recent_runs)
        
        if args.audit_type == 'reward_components':
            results = auditor.audit_reward_components(training_data)
        elif args.audit_type == 'risk_metrics':
            results = auditor.audit_risk_metrics(training_data)
        elif args.audit_type == 'performance_analysis':
            results = auditor.audit_performance_analysis(training_data)
        
        # Save individual audit results
        results_file = output_dir / f'{args.audit_type}_audit.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if args.generate_plots:
            auditor.generate_plots(training_data, [results])
    
    # Print summary
    if args.verbose:
        print(f"\n📁 Audit results saved to: {output_dir}")
        if 'issue_summary' in results:
            issue_summary = results['issue_summary']
            print(f"📊 Issues found: {issue_summary['total_issues']} total, {issue_summary['critical_issues']} critical")
    
    # Return exit code based on results
    if 'critical_issues' in results and results['critical_issues']:
        return 2  # Critical issues found
    elif 'warning_issues' in results and results['warning_issues']:
        return 1  # Warning issues found
    else:
        return 0  # No issues found

if __name__ == "__main__":
    exit(main())