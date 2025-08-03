#!/usr/bin/env python3
"""
Comprehensive Gold Standard Checkpoint Evaluation
Tests all checkpoints (100K, 200K, 300K, 409K) to find optimal risk/return trade-off

Usage:
python evaluate_all_checkpoints.py --data raw/polygon_dual_ticker_20250802_131953.json
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest_v3 import V3Backtester
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointEvaluator:
    """Evaluate all Gold Standard checkpoints"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = []
        
    def find_checkpoints(self) -> List[Dict]:
        """Find all available checkpoints"""
        base_path = Path("train_runs/v3_gold_standard_400k_20250802_202736")
        
        checkpoints = [
            {
                'name': 'Gold_Standard_100K',
                'steps': 102400,
                'path': base_path / "chunk2_final_102400steps.zip"
            },
            {
                'name': 'Gold_Standard_200K', 
                'steps': 204800,
                'path': base_path / "chunk4_final_204800steps.zip"
            },
            {
                'name': 'Gold_Standard_300K',
                'steps': 307200, 
                'path': base_path / "chunk6_final_307200steps.zip"
            },
            {
                'name': 'Gold_Standard_409K',
                'steps': 409600,
                'path': base_path / "v3_gold_standard_final_409600steps.zip"
            }
        ]
        
        # Filter to only existing checkpoints
        available = []
        for checkpoint in checkpoints:
            if checkpoint['path'].exists():
                available.append(checkpoint)
                logger.info(f"✅ Found checkpoint: {checkpoint['name']} ({checkpoint['steps']} steps)")
            else:
                logger.warning(f"⚠️  Missing checkpoint: {checkpoint['path']}")
        
        return available
    
    def evaluate_checkpoint(self, checkpoint: Dict) -> Dict:
        """Evaluate a single checkpoint"""
        logger.info(f"\n🎯 Evaluating {checkpoint['name']} ({checkpoint['steps']} steps)")
        logger.info("=" * 60)
        
        # Create backtester
        backtester = V3Backtester(str(checkpoint['path']), self.data_path, verbose=False)
        
        try:
            # Load and test
            if not backtester.load_model():
                return None
                
            combined_features, price_data = backtester.prepare_historic_data()
            backtester.create_v3_environment(combined_features, price_data)
            
            # Run 5 episodes for better statistics
            results = backtester.run_backtest(num_episodes=5)
            
            # Extract key metrics
            summary = {
                'checkpoint_name': checkpoint['name'],
                'training_steps': checkpoint['steps'],
                'avg_return_pct': results['avg_return_pct'],
                'std_return_pct': results['std_return_pct'],
                'avg_episode_reward': results['avg_episode_reward'],
                'avg_sharpe_ratio': results['avg_sharpe_ratio'],
                'max_drawdown_pct': results['max_drawdown_pct'],
                'avg_trades_per_episode': results['avg_trades_per_episode'],
                'total_trades': results['total_trades'],
                'positive_episodes': results['positive_episodes'],
                'episodes_run': results['episodes_run'],
                'trading_activity': self._classify_activity(results['avg_trades_per_episode']),
                'risk_profile': self._classify_risk(results['max_drawdown_pct']),
                'performance_score': self._calculate_score(results)
            }
            
            self.results.append(summary)
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {checkpoint['name']}: {e}")
            return None
    
    def _classify_activity(self, avg_trades: float) -> str:
        """Classify trading activity level"""
        if avg_trades == 0:
            return "Ultra-Conservative"
        elif avg_trades < 5:
            return "Low Activity"
        elif avg_trades < 15:
            return "Moderate Activity"
        elif avg_trades < 30:
            return "High Activity"
        else:
            return "Very High Activity"
    
    def _classify_risk(self, max_drawdown: float) -> str:
        """Classify risk profile"""
        if max_drawdown == 0:
            return "Zero Risk"
        elif max_drawdown < 2:
            return "Very Low Risk"
        elif max_drawdown < 5:
            return "Low Risk"
        elif max_drawdown < 10:
            return "Moderate Risk"
        elif max_drawdown < 20:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _calculate_score(self, results: Dict) -> float:
        """Calculate overall performance score"""
        # Balanced scoring: return/risk ratio with trading efficiency
        return_score = max(0, results['avg_return_pct'])
        risk_penalty = results['max_drawdown_pct'] * 2  # Penalize risk
        activity_bonus = min(2, results['avg_trades_per_episode'] * 0.1)  # Small bonus for activity
        
        return return_score - risk_penalty + activity_bonus
    
    def generate_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        if not self.results:
            logger.error("No results to report")
            return {}
        
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE GOLD STANDARD CHECKPOINT EVALUATION")
        logger.info("="*80)
        
        # Create summary table
        print(f"\n{'Checkpoint':<20} {'Steps':<8} {'Return %':<10} {'Drawdown %':<12} {'Sharpe':<8} {'Trades':<8} {'Activity':<18} {'Risk':<15} {'Score':<8}")
        print("-" * 110)
        
        for result in sorted(self.results, key=lambda x: x['training_steps']):
            print(f"{result['checkpoint_name']:<20} "
                  f"{result['training_steps']:<8} "
                  f"{result['avg_return_pct']:<10.2f} "
                  f"{result['max_drawdown_pct']:<12.2f} "
                  f"{result['avg_sharpe_ratio']:<8.3f} "
                  f"{result['avg_trades_per_episode']:<8.1f} "
                  f"{result['trading_activity']:<18} "
                  f"{result['risk_profile']:<15} "
                  f"{result['performance_score']:<8.2f}")
        
        # Find best performers
        best_return = max(self.results, key=lambda x: x['avg_return_pct'])
        best_sharpe = max(self.results, key=lambda x: x['avg_sharpe_ratio'] if x['avg_sharpe_ratio'] > -1000 else -999)
        lowest_risk = min(self.results, key=lambda x: x['max_drawdown_pct'])
        best_score = max(self.results, key=lambda x: x['performance_score'])
        most_active = max(self.results, key=lambda x: x['avg_trades_per_episode'])
        
        logger.info(f"\n🏆 PERFORMANCE LEADERS:")
        logger.info(f"   📈 Best Return: {best_return['checkpoint_name']} ({best_return['avg_return_pct']:.2f}%)")
        logger.info(f"   📊 Best Sharpe: {best_sharpe['checkpoint_name']} ({best_sharpe['avg_sharpe_ratio']:.3f})")
        logger.info(f"   🛡️  Lowest Risk: {lowest_risk['checkpoint_name']} ({lowest_risk['max_drawdown_pct']:.2f}%)")
        logger.info(f"   🎯 Best Score: {best_score['checkpoint_name']} ({best_score['performance_score']:.2f})")
        logger.info(f"   🔄 Most Active: {most_active['checkpoint_name']} ({most_active['avg_trades_per_episode']:.1f} trades/ep)")
        
        # Training progression analysis
        logger.info(f"\n📈 TRAINING PROGRESSION ANALYSIS:")
        
        returns_trend = [r['avg_return_pct'] for r in sorted(self.results, key=lambda x: x['training_steps'])]
        risk_trend = [r['max_drawdown_pct'] for r in sorted(self.results, key=lambda x: x['training_steps'])]
        activity_trend = [r['avg_trades_per_episode'] for r in sorted(self.results, key=lambda x: x['training_steps'])]
        
        logger.info(f"   Returns Trend: {' → '.join([f'{r:.1f}%' for r in returns_trend])}")
        logger.info(f"   Risk Trend: {' → '.join([f'{r:.1f}%' for r in risk_trend])}")
        logger.info(f"   Activity Trend: {' → '.join([f'{a:.1f}' for a in activity_trend])}")
        
        # Identify optimal checkpoint
        logger.info(f"\n💡 OPTIMAL CHECKPOINT RECOMMENDATION:")
        
        if best_score['avg_return_pct'] > 0:
            logger.info(f"   🎯 RECOMMENDED: {best_score['checkpoint_name']}")
            logger.info(f"      ✅ Best overall score: {best_score['performance_score']:.2f}")
            logger.info(f"      📈 Return: {best_score['avg_return_pct']:.2f}%")
            logger.info(f"      📉 Risk: {best_score['max_drawdown_pct']:.2f}%")
            logger.info(f"      🔄 Activity: {best_score['trading_activity']}")
        else:
            # If no positive returns, recommend based on use case
            logger.info(f"   🔍 NO CLEAR WINNER - Choose by use case:")
            logger.info(f"      💰 For Growth: {best_return['checkpoint_name']} ({best_return['avg_return_pct']:.2f}%)")
            logger.info(f"      🛡️ For Safety: {lowest_risk['checkpoint_name']} ({lowest_risk['max_drawdown_pct']:.2f}% risk)")
            logger.info(f"      ⚖️ For Balance: {best_score['checkpoint_name']} (score: {best_score['performance_score']:.2f})")
        
        # Management demo recommendation
        logger.info(f"\n🎪 MANAGEMENT DEMO RECOMMENDATION:")
        
        # Find checkpoint with good balance for demo
        demo_candidates = [r for r in self.results if r['avg_return_pct'] > 0 and r['max_drawdown_pct'] < 20]
        
        if demo_candidates:
            demo_pick = max(demo_candidates, key=lambda x: x['performance_score'])
            logger.info(f"   🎯 DEMO MODEL: {demo_pick['checkpoint_name']}")
            logger.info(f"      📊 Professional metrics for presentation")
            logger.info(f"      💼 Balanced risk/return for institutional audience")
            logger.info(f"      📈 {demo_pick['avg_return_pct']:.2f}% return, {demo_pick['max_drawdown_pct']:.2f}% max drawdown")
        else:
            logger.info(f"   ⚠️  All models show conservative behavior")
            logger.info(f"   🎯 Consider {most_active['checkpoint_name']} for most trading activity demonstration")
        
        # Summary report
        summary_report = {
            'evaluation_date': datetime.now().isoformat(),
            'data_file': self.data_path,
            'checkpoints_evaluated': len(self.results),
            'results': self.results,
            'leaders': {
                'best_return': best_return,
                'best_sharpe': best_sharpe,
                'lowest_risk': lowest_risk,
                'best_score': best_score,
                'most_active': most_active
            },
            'trends': {
                'returns': returns_trend,
                'risk': risk_trend,
                'activity': activity_trend
            }
        }
        
        return summary_report

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate all Gold Standard checkpoints')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historic data file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CheckpointEvaluator(args.data)
    
    # Find checkpoints
    checkpoints = evaluator.find_checkpoints()
    
    if not checkpoints:
        logger.error("❌ No checkpoints found!")
        return 1
    
    logger.info(f"🎯 Found {len(checkpoints)} checkpoints to evaluate")
    
    # Evaluate each checkpoint
    for checkpoint in checkpoints:
        result = evaluator.evaluate_checkpoint(checkpoint)
        if not result:
            logger.warning(f"⚠️  Skipping failed checkpoint: {checkpoint['name']}")
    
    # Generate report
    report = evaluator.generate_report()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📁 Results saved to: {args.output}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"checkpoint_evaluation_{timestamp}.json"
    with open(default_output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📁 Results also saved to: {default_output}")
    
    return 0

if __name__ == '__main__':
    exit(main())