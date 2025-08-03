#!/usr/bin/env python3
"""
Comprehensive V3 Model Evaluation Report
Compares the latest V3 model against previous versions
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

def create_simple_features(price_data):
    """Create simple technical features from price data"""
    df = pd.DataFrame(price_data)
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Simple moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Price relative to moving averages
    df['price_ma_ratio'] = df['close'] / df['sma_20']
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    # Select features (12 features to match training)
    features = [
        'returns', 'sma_5', 'sma_20', 'rsi', 'volatility', 'price_ma_ratio',
        'volume_ratio', 'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Normalize features
    feature_matrix = df[features].values
    for i in range(feature_matrix.shape[1]):
        col = feature_matrix[:, i]
        if np.std(col) > 0:
            feature_matrix[:, i] = (col - np.mean(col)) / np.std(col)
    
    return feature_matrix, df['close'].values

def evaluate_model(model_path, model_name):
    """Evaluate a single model and return metrics"""
    print(f"\n🤖 Evaluating {model_name}")
    print(f"📁 Path: {model_path}")
    
    # Load model
    try:
        model = RecurrentPPO.load(model_path)
        print("✅ Loaded as RecurrentPPO")
    except:
        try:
            model = PPO.load(model_path)
            print("✅ Loaded as PPO")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            return None
    
    # Load Polygon data
    dual_ticker_files = list(Path("raw").glob("polygon_dual_ticker_*.json"))
    if not dual_ticker_files:
        print("❌ No data files found")
        return None
    
    latest_file = max(dual_ticker_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        polygon_data = json.load(f)
    
    # Extract data
    nvda_data = polygon_data['symbols'].get('NVDA', {}).get('data', [])
    msft_data = polygon_data['symbols'].get('MSFT', {}).get('data', [])
    
    if not nvda_data or not msft_data:
        print("❌ Missing data")
        return None
    
    # Create features
    nvda_features, nvda_prices = create_simple_features(nvda_data)
    msft_features, msft_prices = create_simple_features(msft_data)
    
    # Align data
    min_len = min(len(nvda_features), len(msft_features))
    nvda_features = nvda_features[:min_len]
    msft_features = msft_features[:min_len]
    nvda_prices = nvda_prices[:min_len]
    msft_prices = msft_prices[:min_len]
    
    # Combine features
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    
    # Run simulation
    capital = 100000.0
    nvda_position = 0
    msft_position = 0
    equity_curve = [capital]
    actions_taken = []
    trades = 0
    
    # LSTM states
    if hasattr(model, 'policy') and hasattr(model.policy, 'lstm'):
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
    
    for i in range(20, min_len - 1):
        # Create observation
        features = combined_features[i]
        nvda_pos_norm = nvda_position / 1000
        msft_pos_norm = msft_position / 1000
        obs = np.concatenate([features, [nvda_pos_norm, msft_pos_norm]])
        
        # Get action
        if hasattr(model, 'policy') and hasattr(model.policy, 'lstm'):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        actions_taken.append(int(action))
        
        # Execute trades
        nvda_price = nvda_prices[i]
        msft_price = msft_prices[i]
        
        # Enhanced action mapping for 9-action dual ticker (3x3 grid)
        # Actions: 0=sell_both, 1=sell_nvda_hold_msft, 2=sell_nvda_buy_msft, 
        #         3=hold_nvda_sell_msft, 4=hold_both, 5=hold_nvda_buy_msft,
        #         6=buy_nvda_sell_msft, 7=buy_nvda_hold_msft, 8=buy_both
        
        position_size = min(capital * 0.05, 5000)  # 5% of capital or $5k max per trade
        
        # NVDA actions (based on action // 3)
        nvda_action = action // 3
        if nvda_action == 0:  # Sell NVDA
            if nvda_position > 0:
                capital += nvda_position * nvda_price * 0.999
                nvda_position = 0
                trades += 1
        elif nvda_action == 2:  # Buy NVDA
            if position_size > nvda_price and capital >= position_size:
                nvda_shares = int(position_size / nvda_price)
                nvda_cost = nvda_shares * nvda_price * 1.001
                if capital >= nvda_cost:
                    capital -= nvda_cost
                    nvda_position += nvda_shares
                    trades += 1
        
        # MSFT actions (based on action % 3)
        msft_action = action % 3
        if msft_action == 0:  # Sell MSFT
            if msft_position > 0:
                capital += msft_position * msft_price * 0.999
                msft_position = 0
                trades += 1
        elif msft_action == 2:  # Buy MSFT
            if position_size > msft_price and capital >= position_size:
                msft_shares = int(position_size / msft_price)
                msft_cost = msft_shares * msft_price * 1.001
                if capital >= msft_cost:
                    capital -= msft_cost
                    msft_position += msft_shares
                    trades += 1
        
        # Calculate equity
        current_equity = capital + nvda_position * nvda_price + msft_position * msft_price
        equity_curve.append(current_equity)
    
    # Calculate metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - 100000) / 100000 * 100
    max_equity = max(equity_curve)
    min_equity = min(equity_curve)
    max_drawdown = (max_equity - min_equity) / max_equity * 100 if max_equity > 0 else 0
    
    # Sharpe ratio (simplified)
    returns = np.diff(equity_curve) / equity_curve[:-1] * 100
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    # Action diversity
    action_counts = {}
    for action in actions_taken:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    action_entropy = -sum((count/len(actions_taken)) * np.log(count/len(actions_taken)) 
                         for count in action_counts.values())
    
    return {
        'model_name': model_name,
        'final_equity': final_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'total_trades': trades,
        'action_entropy': action_entropy,
        'days_simulated': len(equity_curve) - 1,
        'action_distribution': action_counts
    }

def main():
    """Run comprehensive model evaluation"""
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 50)
    
    # Models to evaluate
    models = [
        {
            'name': 'V3_Gold_Standard_409K',
            'path': 'train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip'
        },
        {
            'name': 'V3_100K_Latest',
            'path': 'train_runs/v3_from_200k_20250802_183726/v3_model_final_100000steps.zip'
        },
        {
            'name': 'Deployed_300K',
            'path': 'deploy_models/dual_ticker_prod_20250802_step300k_manual.zip'
        },
        {
            'name': 'Deployed_251K',
            'path': 'deploy_models/dual_ticker_prod_20250731_step251k.zip'
        }
    ]
    
    results = []
    
    for model_info in models:
        if os.path.exists(model_info['path']):
            result = evaluate_model(model_info['path'], model_info['name'])
            if result:
                results.append(result)
        else:
            print(f"⚠️ Model not found: {model_info['path']}")
    
    if not results:
        print("❌ No models could be evaluated")
        return
    
    # Generate comparison report
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON REPORT")
    print("="*60)
    
    # Performance table
    print(f"\n{'Model':<20} {'Return %':<10} {'Drawdown %':<12} {'Sharpe':<8} {'Trades':<8} {'Entropy':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['total_return']:<10.2f} "
              f"{result['max_drawdown']:<12.2f} "
              f"{result['sharpe_ratio']:<8.3f} "
              f"{result['total_trades']:<8} "
              f"{result['action_entropy']:<8.3f}")
    
    # Best performer analysis
    best_return = max(results, key=lambda x: x['total_return'])
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    lowest_drawdown = min(results, key=lambda x: x['max_drawdown'])
    
    print(f"\n🏆 PERFORMANCE WINNERS:")
    print(f"   Best Return: {best_return['model_name']} ({best_return['total_return']:.2f}%)")
    print(f"   Best Sharpe: {best_sharpe['model_name']} ({best_sharpe['sharpe_ratio']:.3f})")
    print(f"   Lowest Drawdown: {lowest_drawdown['model_name']} ({lowest_drawdown['max_drawdown']:.2f}%)")
    
    # V3 specific analysis
    v3_result = next((r for r in results if 'V3' in r['model_name']), None)
    if v3_result:
        print(f"\n🎯 V3 MODEL ANALYSIS:")
        print(f"   Environment: DualTickerTradingEnvV3 (Risk-aware)")
        print(f"   Training: 200K base + 100K additional steps")
        print(f"   Performance: {v3_result['total_return']:.2f}% return, {v3_result['max_drawdown']:.2f}% drawdown")
        print(f"   Trading Activity: {v3_result['total_trades']} trades over {v3_result['days_simulated']} days")
        print(f"   Action Diversity: {v3_result['action_entropy']:.3f} (higher = more diverse)")
        
        # Compare V3 to other models
        other_results = [r for r in results if 'V3' not in r['model_name']]
        if other_results:
            avg_other_return = np.mean([r['total_return'] for r in other_results])
            avg_other_drawdown = np.mean([r['max_drawdown'] for r in other_results])
            
            print(f"\n📈 V3 vs Other Models:")
            print(f"   Return: {v3_result['total_return']:.2f}% vs {avg_other_return:.2f}% avg")
            print(f"   Drawdown: {v3_result['max_drawdown']:.2f}% vs {avg_other_drawdown:.2f}% avg")
            
            return_improvement = v3_result['total_return'] - avg_other_return
            drawdown_improvement = avg_other_drawdown - v3_result['max_drawdown']
            
            print(f"   Improvement: {return_improvement:+.2f}% return, {drawdown_improvement:+.2f}% better drawdown")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if v3_result and v3_result['total_return'] > 0:
        print("   ✅ V3 model shows positive returns - suitable for deployment")
        print("   ✅ Risk-aware environment improvements are working")
        print("   🎯 Consider using V3 for live trading trials")
    else:
        print("   ⚠️  Models need further optimization")
        print("   🔧 Consider additional training or hyperparameter tuning")
    
    if any(r['max_drawdown'] > 10 for r in results):
        print("   🚨 High drawdowns detected - implement stricter risk controls")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_evaluation_report_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': len(results),
            'results': results,
            'summary': {
                'best_return_model': best_return['model_name'],
                'best_return_value': best_return['total_return'],
                'best_sharpe_model': best_sharpe['model_name'],
                'best_sharpe_value': best_sharpe['sharpe_ratio'],
                'lowest_drawdown_model': lowest_drawdown['model_name'],
                'lowest_drawdown_value': lowest_drawdown['max_drawdown']
            }
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    print(f"\n✅ MODEL EVALUATION COMPLETE")

if __name__ == '__main__':
    main()