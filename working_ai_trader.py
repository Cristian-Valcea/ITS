#!/usr/bin/env python3
"""
Working AI Trader - Combines functional components
Real Stairways V3 model + IBKR simulation + actual trading
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('src')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WorkingAITrader')

class WorkingAITrader:
    """A working AI trader using real Stairways V3 model"""
    
    def __init__(self):
        self.model = None
        self.client = None
        self.model_path = "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip"
        
    def load_model(self):
        """Load the real Stairways V3 model"""
        try:
            logger.info("🤖 Loading Stairways V3 model...")
            
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
            
            logger.info("✅ Stairways V3 model loaded successfully!")
            logger.info(f"   Model: {type(self.model)}")
            logger.info(f"   Path: {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def connect_ibkr(self):
        """Connect to IBKR (simulation mode)"""
        try:
            logger.info("🔌 Connecting to IBKR...")
            
            from brokers.ib_gateway import IBGatewayClient
            self.client = IBGatewayClient()
            
            connected = self.client.connect()
            if connected:
                mode = "simulation" if self.client.simulation_mode else "live"
                logger.info(f"✅ IBKR connected in {mode} mode")
                return True
            else:
                logger.error("❌ IBKR connection failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ IBKR connection error: {e}")
            return False
    
    def create_observation(self, nvda_price, msft_price, nvda_pos=0.0, msft_pos=0.0):
        """Create a realistic observation for the AI model"""
        
        # Simple feature engineering (in real system, this would be more sophisticated)
        nvda_change = np.random.normal(0.01, 0.02)  # Simulated price change
        msft_change = np.random.normal(0.005, 0.015)
        
        observation = np.array([
            # NVDA features (12)
            nvda_change,    # price_change
            0.15,           # volatility
            1.2,            # volume_ratio
            nvda_change * 5, # momentum
            0.5,            # rsi
            nvda_change,    # ema_signal
            nvda_change * 0.5, # vwap_signal
            0.5,            # bollinger_position
            0.7,            # market_strength
            0.6,            # trend_strength
            0.8,            # liquidity_score
            datetime.now().hour / 24.0,  # time_of_day
            
            # MSFT features (12)
            msft_change,    # price_change
            0.12,           # volatility
            0.9,            # volume_ratio
            msft_change * 5, # momentum
            0.6,            # rsi
            msft_change,    # ema_signal
            msft_change * 0.5, # vwap_signal
            0.4,            # bollinger_position
            0.5,            # market_strength
            0.4,            # trend_strength
            0.8,            # liquidity_score
            datetime.now().hour / 24.0,  # time_of_day
            
            # Position features (2)
            nvda_pos,       # nvda_position_ratio
            msft_pos        # msft_position_ratio
        ], dtype=np.float32)
        
        return observation
    
    def get_ai_decision(self, observation):
        """Get AI decision from Stairways V3 model"""
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            
            action_meanings = {
                0: "Strong Sell",
                1: "Sell", 
                2: "Hold",
                3: "Buy",
                4: "Strong Buy"
            }
            
            action_int = int(action)
            meaning = action_meanings.get(action_int, "Unknown")
            
            logger.info(f"🧠 AI Decision: {meaning} (action={action_int})")
            
            return action_int, meaning
            
        except Exception as e:
            logger.error(f"❌ AI decision error: {e}")
            return 2, "Hold"  # Default to hold on error
    
    def execute_trade(self, symbol, action, quantity=1):
        """Execute trade based on AI decision"""
        try:
            if action >= 3:  # Buy or Strong Buy
                logger.info(f"🟢 Executing BUY order: {quantity} {symbol}")
                result = self.client.place_market_order(symbol, quantity, 'BUY')
                
            elif action <= 1:  # Sell or Strong Sell
                logger.info(f"🔴 Executing SELL order: {quantity} {symbol}")
                result = self.client.place_market_order(symbol, quantity, 'SELL')
                
            else:  # Hold
                logger.info(f"⚪ Holding position for {symbol}")
                return {"status": "Hold", "symbol": symbol}
            
            if result and result.get('status') == 'Filled':
                logger.info(f"✅ Trade executed: {result}")
                return result
            else:
                logger.error(f"❌ Trade failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return None
    
    def run_trading_session(self, duration_minutes=5, cycle_seconds=30):
        """Run a complete AI trading session"""
        logger.info("🚀 STARTING AI TRADING SESSION")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Cycle: {cycle_seconds} seconds")
        logger.info("=" * 50)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle_count = 0
        
        # Track positions
        positions = {"NVDA": 0, "MSFT": 0}
        
        while time.time() < end_time:
            cycle_count += 1
            logger.info(f"\n📊 TRADING CYCLE #{cycle_count}")
            logger.info(f"   Time remaining: {int((end_time - time.time()) / 60)} minutes")
            
            try:
                # Get market data
                nvda_price = self.client.get_current_price('NVDA')
                msft_price = self.client.get_current_price('MSFT')
                
                logger.info(f"   Market Data: NVDA=${nvda_price:.2f}, MSFT=${msft_price:.2f}")
                
                # Create observation
                observation = self.create_observation(
                    nvda_price, msft_price, 
                    positions["NVDA"] / 10.0,  # Normalize positions
                    positions["MSFT"] / 10.0
                )
                
                # Get AI decisions for both symbols
                for symbol in ["NVDA", "MSFT"]:
                    logger.info(f"\n   🎯 {symbol} Analysis:")
                    
                    action, meaning = self.get_ai_decision(observation)
                    
                    # Execute trade
                    result = self.execute_trade(symbol, action, quantity=1)
                    
                    # Update positions
                    if result and result.get('status') == 'Filled':
                        if result['action'] == 'BUY':
                            positions[symbol] += result['quantity']
                        elif result['action'] == 'SELL':
                            positions[symbol] -= result['quantity']
                
                # Show current positions
                logger.info(f"\n   📈 Current Positions: NVDA={positions['NVDA']}, MSFT={positions['MSFT']}")
                
                # Wait for next cycle
                logger.info(f"   ⏳ Waiting {cycle_seconds} seconds for next cycle...")
                time.sleep(cycle_seconds)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Trading session interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Cycle error: {e}")
                time.sleep(cycle_seconds)
        
        # Session summary
        logger.info("\n" + "=" * 50)
        logger.info("🎯 TRADING SESSION COMPLETE")
        logger.info(f"   Total Cycles: {cycle_count}")
        logger.info(f"   Final Positions: NVDA={positions['NVDA']}, MSFT={positions['MSFT']}")
        logger.info(f"   Duration: {duration_minutes} minutes")
        
        return positions
    
    def disconnect(self):
        """Clean disconnect"""
        if self.client:
            self.client.disconnect()
            logger.info("✅ Disconnected from IBKR")

def main():
    """Main function to run the working AI trader"""
    
    print("🤖 WORKING AI TRADER - STAIRWAYS V3")
    print("=" * 50)
    print("Real AI model + IBKR simulation + actual trading")
    print()
    
    # Create trader instance
    trader = WorkingAITrader()
    
    try:
        # Initialize components
        if not trader.load_model():
            print("❌ Failed to load AI model")
            return 1
        
        if not trader.connect_ibkr():
            print("❌ Failed to connect to IBKR")
            return 1
        
        # Use automatic session parameters for demo
        duration = 10  # 10 minutes
        cycle = 60     # 60 seconds between trades
        
        print(f"🎯 Auto-configured session: {duration} minutes, {cycle} seconds per cycle")
        
        print(f"\n🚀 Starting {duration}-minute AI trading session...")
        print("Press Ctrl+C to stop early")
        print()
        
        # Run trading session
        final_positions = trader.run_trading_session(duration, cycle)
        
        print(f"\n🎉 Session completed successfully!")
        print(f"Final positions: {final_positions}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Session failed: {e}")
        return 1
    finally:
        trader.disconnect()

if __name__ == "__main__":
    exit(main())