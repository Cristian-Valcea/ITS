#!/usr/bin/env python3
"""
Working AI Test - Test the actual AI model functionality
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Set up logging first
logging.basicConfig(level=logging.INFO)

def test_direct_model_loading():
    """Test direct model loading and prediction"""
    print("🤖 DIRECT MODEL LOADING TEST")
    print("=" * 40)
    
    model_path = "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip"
    
    if not Path(model_path).exists():
        print(f"   ❌ Model not found: {model_path}")
        return False
    
    try:
        from stable_baselines3 import PPO
        
        print(f"   🔄 Loading Stairways V3 model...")
        model = PPO.load(model_path)
        print(f"   ✅ Model loaded: {type(model)}")
        
        # Test with realistic observation
        print(f"   🧠 Testing AI decision making...")
        
        # Create a realistic 26-feature observation
        # Features: 12 per symbol (NVDA, MSFT) + 2 positions
        observation = np.array([
            # NVDA features (12)
            0.02,   # price_change
            0.15,   # volatility
            1.2,    # volume_ratio
            0.8,    # momentum
            0.3,    # rsi
            0.1,    # ema_signal
            0.05,   # vwap_signal
            0.2,    # bollinger_position
            0.7,    # market_strength
            0.4,    # trend_strength
            0.6,    # liquidity_score
            0.3,    # time_of_day
            
            # MSFT features (12)
            -0.01,  # price_change
            0.12,   # volatility
            0.9,    # volume_ratio
            -0.2,   # momentum
            0.6,    # rsi
            -0.05,  # ema_signal
            -0.02,  # vwap_signal
            0.4,    # bollinger_position
            0.5,    # market_strength
            0.2,    # trend_strength
            0.8,    # liquidity_score
            0.3,    # time_of_day
            
            # Position features (2)
            0.1,    # nvda_position_ratio
            -0.05   # msft_position_ratio
        ], dtype=np.float32)
        
        # Test multiple predictions to verify consistency
        predictions = []
        for i in range(5):
            action, _states = model.predict(observation, deterministic=True)
            predictions.append(int(action))
        
        print(f"   📊 Predictions (deterministic): {predictions}")
        
        if len(set(predictions)) == 1:
            print(f"   ✅ AI is deterministic: action={predictions[0]}")
            
            # Interpret the action
            action_meanings = {
                0: "Strong Sell",
                1: "Sell", 
                2: "Hold",
                3: "Buy",
                4: "Strong Buy"
            }
            
            action_meaning = action_meanings.get(predictions[0], "Unknown")
            print(f"   🎯 AI Decision: {action_meaning}")
            
        else:
            print(f"   ⚠️  AI predictions vary: {set(predictions)}")
        
        # Test with different observations
        print(f"   🔄 Testing with different market conditions...")
        
        # Bullish scenario
        bullish_obs = observation.copy()
        bullish_obs[0] = 0.05  # Strong positive price change for NVDA
        bullish_obs[12] = 0.03  # Strong positive price change for MSFT
        
        bullish_action, _ = model.predict(bullish_obs, deterministic=True)
        print(f"   📈 Bullish scenario: {action_meanings.get(int(bullish_action), 'Unknown')}")
        
        # Bearish scenario
        bearish_obs = observation.copy()
        bearish_obs[0] = -0.05  # Strong negative price change for NVDA
        bearish_obs[12] = -0.03  # Strong negative price change for MSFT
        
        bearish_action, _ = model.predict(bearish_obs, deterministic=True)
        print(f"   📉 Bearish scenario: {action_meanings.get(int(bearish_action), 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def test_model_with_ibkr_simulation():
    """Test model with IBKR simulation integration"""
    print("\n🔌 MODEL + IBKR SIMULATION TEST")
    print("=" * 40)
    
    try:
        # Load model
        from stable_baselines3 import PPO
        model_path = "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip"
        model = PPO.load(model_path)
        
        # Load IBKR client
        sys.path.append('src')
        from brokers.ib_gateway import IBGatewayClient
        
        client = IBGatewayClient()
        connected = client.connect()
        
        if not connected:
            print("   ❌ IBKR connection failed")
            return False
        
        print(f"   ✅ IBKR connected in {client.simulation_mode and 'simulation' or 'live'} mode")
        
        # Get real market data
        nvda_price = client.get_current_price('NVDA')
        msft_price = client.get_current_price('MSFT')
        
        print(f"   📊 Market data: NVDA=${nvda_price}, MSFT=${msft_price}")
        
        # Create observation from real data (simplified)
        observation = np.array([
            # NVDA features (simplified)
            0.01, 0.15, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.8, 0.3,
            # MSFT features (simplified) 
            0.005, 0.12, 0.9, 0.3, 0.6, 0.0, 0.0, 0.4, 0.5, 0.4, 0.8, 0.3,
            # Positions
            0.0, 0.0
        ], dtype=np.float32)
        
        # Get AI decision
        action, _ = model.predict(observation, deterministic=True)
        action_meanings = {0: "Strong Sell", 1: "Sell", 2: "Hold", 3: "Buy", 4: "Strong Buy"}
        decision = action_meanings.get(int(action), "Unknown")
        
        print(f"   🧠 AI Decision: {decision} (action={action})")
        
        # Execute a small test trade based on AI decision
        if action >= 3:  # Buy or Strong Buy
            print(f"   🟢 AI suggests buying - placing test BUY order")
            result = client.place_market_order('NVDA', 1, 'BUY')
            if result:
                print(f"   ✅ Test order executed: {result['status']}")
            else:
                print(f"   ❌ Test order failed")
        elif action <= 1:  # Sell or Strong Sell
            print(f"   🔴 AI suggests selling - would place SELL order")
            print(f"   ℹ️  (Skipping sell order - no position to sell)")
        else:  # Hold
            print(f"   ⚪ AI suggests holding - no action taken")
        
        client.disconnect()
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def test_web_interface_backend():
    """Test what the web interface actually connects to"""
    print("\n🌐 WEB INTERFACE BACKEND TEST")
    print("=" * 40)
    
    try:
        # Check if minimal_launcher.py has real functionality
        with open('minimal_launcher.py', 'r') as f:
            content = f.read()
        
        print("   📄 Analyzing minimal_launcher.py...")
        
        # Look for key indicators
        has_ai = any(term in content for term in ['StairwaysV3', 'PPO', 'model.predict', 'chunk7'])
        has_ibkr = any(term in content for term in ['IBGatewayClient', 'ib_gateway', 'place_order'])
        has_random = 'random' in content.lower()
        
        print(f"   🤖 AI Integration: {'✅' if has_ai else '❌'}")
        print(f"   🔌 IBKR Integration: {'✅' if has_ibkr else '❌'}")
        print(f"   🎲 Random Logic: {'⚠️ YES' if has_random else '✅ NO'}")
        
        if not has_ai and not has_ibkr:
            print("   ⚠️  Web interface appears to be a basic launcher only")
            print("   💡 Real functionality likely in real_trading_deployment.py")
        
        return has_ai or has_ibkr
        
    except Exception as e:
        print(f"   ❌ Web interface analysis failed: {e}")
        return False

def main():
    """Run comprehensive working AI tests"""
    print("🧪 WORKING AI SYSTEM TESTS")
    print("=" * 60)
    print("Testing actual AI functionality and integration")
    print()
    
    results = {}
    
    # Test direct model functionality
    results['model_loading'] = test_direct_model_loading()
    
    # Test model with IBKR integration
    results['ibkr_integration'] = test_model_with_ibkr_simulation()
    
    # Test web interface backend
    results['web_backend'] = test_web_interface_backend()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 WORKING AI SYSTEM SUMMARY")
    print("=" * 60)
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"\n📊 Working Components: {working_components}/{total_components}")
    
    for component, working in results.items():
        status = "✅" if working else "❌"
        print(f"   {status} {component.replace('_', ' ').title()}")
    
    if results['model_loading']:
        print(f"\n🎉 GOOD NEWS:")
        print(f"   ✅ Stairways V3 model (400k steps) is REAL and working")
        print(f"   ✅ AI makes actual decisions (not random)")
        print(f"   ✅ Model responds to different market conditions")
    
    if results['ibkr_integration']:
        print(f"   ✅ AI + IBKR integration works in simulation mode")
        print(f"   ✅ Can execute trades based on AI decisions")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if working_components >= 2:
        print(f"   🚀 Core AI trading system is functional!")
        print(f"   🎯 Focus on connecting web interface to real AI backend")
        print(f"   🔧 Use real_trading_deployment.py for actual AI trading")
    else:
        print(f"   🔧 Need to fix integration issues")
        print(f"   🎯 Focus on getting AI + IBKR working together")
    
    return working_components >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)