#!/usr/bin/env python3
"""
Simple 300K Model Backtest using fetched Polygon data
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

def simple_backtest():
    """Run simple backtest using fetched Polygon data"""
    print("🚀 V3 GOLD STANDARD MODEL BACKTEST (409K Steps)")
    print("=" * 45)
    
    # Load model - using the latest V3 Gold Standard model
    model_path = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
    print(f"🤖 Loading model: {model_path}")
    
    try:
        model = RecurrentPPO.load(model_path)
        print("✅ Model loaded successfully (RecurrentPPO)")
    except:
        try:
            model = PPO.load(model_path)
            print("✅ Model loaded successfully (PPO)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
    
    # Load Polygon data
    print("📊 Loading Polygon data...")
    
    # Find the dual ticker polygon data file
    dual_ticker_files = list(Path("raw").glob("polygon_dual_ticker_*.json"))
    if not dual_ticker_files:
        print("❌ No dual ticker Polygon data files found in raw/")
        return
    
    latest_file = max(dual_ticker_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using data file: {latest_file}")
    
    with open(latest_file, 'r') as f:
        polygon_data = json.load(f)
    
    # Extract NVDA and MSFT data
    nvda_data = polygon_data['symbols'].get('NVDA', {}).get('data', [])
    msft_data = polygon_data['symbols'].get('MSFT', {}).get('data', [])
    
    if not nvda_data or not msft_data:
        print("❌ Missing NVDA or MSFT data")
        return
    
    print(f"✅ Loaded {len(nvda_data)} NVDA bars, {len(msft_data)} MSFT bars")
    
    # Create features
    print("🔧 Creating features...")
    nvda_features, nvda_prices = create_simple_features(nvda_data)
    msft_features, msft_prices = create_simple_features(msft_data)
    
    # Align data length
    min_len = min(len(nvda_features), len(msft_features))
    nvda_features = nvda_features[:min_len]
    msft_features = msft_features[:min_len]
    nvda_prices = nvda_prices[:min_len]
    msft_prices = msft_prices[:min_len]
    
    # Combine features (24 features total)  
    combined_features = np.concatenate([nvda_features, msft_features], axis=1)
    
    print(f"📐 Combined features shape: {combined_features.shape}")
    print(f"💰 Price data length: {len(nvda_prices)}")
    
    # Simple trading simulation
    print("🚀 Running trading simulation...")
    
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
    
    for i in range(20, min_len - 1):  # Start after warmup period
        # Create observation (26 dims: 24 features + 2 positions)
        features = combined_features[i]
        nvda_pos_norm = nvda_position / 1000  # Normalize position
        msft_pos_norm = msft_position / 1000
        
        obs = np.concatenate([features, [nvda_pos_norm, msft_pos_norm]])
        
        # Get action from model
        if hasattr(model, 'policy') and hasattr(model.policy, 'lstm'):
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        actions_taken.append(int(action))
        
        # Execute trades based on action (simplified)
        nvda_price = nvda_prices[i]
        msft_price = msft_prices[i]
        
        # Debug: Print first few actions
        if i < 25:
            print(f"Step {i}: Action={action}, NVDA=${nvda_price:.2f}, MSFT=${msft_price:.2f}, Capital=${capital:.2f}")
        
        # Simple action mapping (assuming 9 actions for dual-ticker)
        if action == 0:  # Sell both
            if nvda_position > 0:
                capital += nvda_position * nvda_price * 0.999  # Transaction cost
                nvda_position = 0
                trades += 1
            if msft_position > 0:
                capital += msft_position * msft_price * 0.999
                msft_position = 0
                trades += 1
        elif action == 8:  # Buy both
            trade_size = min(capital * 0.1, 10000)  # 10% of capital or $10k max
            if trade_size > nvda_price:
                nvda_shares = int(trade_size / nvda_price / 2)
                msft_shares = int(trade_size / msft_price / 2)
                
                nvda_cost = nvda_shares * nvda_price * 1.001
                msft_cost = msft_shares * msft_price * 1.001
                
                if capital >= nvda_cost + msft_cost:
                    capital -= nvda_cost + msft_cost
                    nvda_position += nvda_shares
                    msft_position += msft_shares
                    trades += 2
        
        # Calculate current equity
        current_equity = capital + nvda_position * nvda_price + msft_position * msft_price
        equity_curve.append(current_equity)
    
    # Final results
    final_equity = equity_curve[-1]
    total_return = (final_equity - 100000) / 100000 * 100
    max_equity = max(equity_curve)
    min_equity = min(equity_curve)
    max_drawdown = (max_equity - min_equity) / max_equity * 100 if max_equity > 0 else 0
    
    print("\n📊 BACKTEST RESULTS")
    print("=" * 30)
    print(f"💰 Starting Capital: $100,000.00")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")
    print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
    print(f"🔄 Total Trades: {trades}")
    print(f"📊 Days Simulated: {len(equity_curve) - 1}")
    
    # Action analysis
    action_counts = {}
    for action in actions_taken:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\n🎯 ACTION DISTRIBUTION:")
    for action, count in sorted(action_counts.items()):
        print(f"   Action {action}: {count} times ({count/len(actions_taken)*100:.1f}%)")
    
    print(f"\n✅ MODEL PERFORMANCE: {'GOOD' if total_return > 0 and max_drawdown < 20 else 'NEEDS IMPROVEMENT'}")

if __name__ == '__main__':
    simple_backtest()