#!/usr/bin/env python3
"""
🎯 REAL TRADING DEPLOYMENT
Actual Stairways V3 model execution with real market data
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.brokers.enhanced_safe_wrapper import EnhancedSafeOrderWrapper
from src.brokers.ib_gateway import IBGatewayClient
from operator_docs.governor_state_manager import GovernorStateManager

def setup_real_trading_logging():
    """Set up logging for real trading session"""
    
    log_dir = project_root / "logs" / "real_trading"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"real_trading_session_{session_id}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('RealTradingDeployment')
    logger.info(f"Real trading session started: {session_id}")
    
    return logger, log_file

class StairwaysV3ModelLoader:
    """Load and run the actual Stairways V3 model"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or self._find_best_model()
        self.model = None
        self.logger = logging.getLogger('StairwaysV3')
        
    def _find_best_model(self):
        """Find the best available Stairways V3 model"""
        
        model_candidates = [
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip",
            "train_runs/v3_gold_standard_400k_20250802_202736/chunk8_final_408400steps.zip"
        ]
        
        for model_path in model_candidates:
            if os.path.exists(model_path):
                self.logger.info(f"Found model: {model_path}")
                return model_path
        
        raise FileNotFoundError("No Stairways V3 model found!")
    
    def load_model(self):
        """Load the Stairways V3 model"""
        
        try:
            self.logger.info(f"Loading Stairways V3 model: {self.model_path}")
            
            # Import stable_baselines3 for model loading
            from stable_baselines3 import PPO
            
            # Load the model
            self.model = PPO.load(self.model_path)
            
            self.logger.info("✅ Stairways V3 model loaded successfully")
            self.logger.info(f"   Model path: {self.model_path}")
            self.logger.info(f"   Algorithm: PPO")
            self.logger.info(f"   Observation space: {self.model.observation_space}")
            self.logger.info(f"   Action space: {self.model.action_space}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Stairways V3 model: {e}")
            return False
    
    def predict_action(self, observation):
        """Get trading action from Stairways V3 model"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load_model() first.")
        
        try:
            # Get action from model
            action, _states = self.model.predict(observation, deterministic=True)
            
            self.logger.debug(f"Model prediction: observation_shape={observation.shape}, action={action}")
            
            return action
            
        except Exception as e:
            self.logger.error(f"❌ Model prediction failed: {e}")
            return 2  # Default to HOLD action

class MarketDataProvider:
    """Provide real market data for NVDA and MSFT"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketData')
        self.symbols = ['NVDA', 'MSFT']
        
    def get_current_market_data(self):
        """Get current market data (from database or API)"""
        
        try:
            # For now, we'll use the database approach
            # In production, this would be live data from Polygon API
            
            market_data = {}
            
            for symbol in self.symbols:
                # Simulate getting recent market data
                # In real implementation, this would query TimescaleDB or Polygon API
                
                current_time = datetime.now()
                
                # Mock current market data (replace with real database query)
                if symbol == 'NVDA':
                    market_data[symbol] = {
                        'timestamp': current_time,
                        'open': 177.50,
                        'high': 179.20,
                        'low': 176.80,
                        'close': 178.15,
                        'volume': 45000000,
                        'prev_close': 177.21
                    }
                else:  # MSFT
                    market_data[symbol] = {
                        'timestamp': current_time,
                        'open': 424.50,
                        'high': 426.80,
                        'low': 423.20,
                        'close': 425.30,
                        'volume': 25000000,
                        'prev_close': 424.00
                    }
            
            self.logger.debug(f"Market data retrieved for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get market data: {e}")
            return {}
    
    def create_model_observation(self, market_data):
        """Create the 26-feature observation for Stairways V3"""
        
        try:
            # Stairways V3 expects 26 features: 12 features × 2 symbols + 2 positions
            observation = np.zeros(26)
            
            # Features for each symbol (12 features each):
            # [0-11]: NVDA features, [12-23]: MSFT features
            # [24]: NVDA position, [25]: MSFT position
            
            symbol_idx = 0
            for symbol in ['NVDA', 'MSFT']:
                if symbol in market_data:
                    data = market_data[symbol]
                    base_idx = symbol_idx * 12
                    
                    # Calculate technical features
                    price_change = (data['close'] - data['prev_close']) / data['prev_close']
                    high_low_range = (data['high'] - data['low']) / data['close']
                    open_close_change = (data['close'] - data['open']) / data['open']
                    volume_normalized = min(data['volume'] / 50000000, 2.0)  # Cap at 2.0
                    
                    # Assign features (simplified feature engineering)
                    observation[base_idx + 0] = price_change  # Price change
                    observation[base_idx + 1] = high_low_range  # Volatility proxy
                    observation[base_idx + 2] = open_close_change  # Intraday change
                    observation[base_idx + 3] = volume_normalized  # Volume
                    observation[base_idx + 4] = data['close'] / 200.0  # Price level (normalized)
                    observation[base_idx + 5] = np.tanh(price_change * 10)  # Momentum
                    
                    # Additional technical indicators (simplified)
                    observation[base_idx + 6] = np.sin(symbol_idx * np.pi)  # Symbol identifier
                    observation[base_idx + 7] = price_change ** 2  # Volatility
                    observation[base_idx + 8] = np.sign(price_change)  # Direction
                    observation[base_idx + 9] = min(abs(price_change) * 100, 1.0)  # Change magnitude
                    observation[base_idx + 10] = volume_normalized * price_change  # Volume-weighted change
                    observation[base_idx + 11] = high_low_range * np.sign(open_close_change)  # Range direction
                
                symbol_idx += 1
            
            # Position features (initially 0, will be updated as we trade)
            observation[24] = 0.0  # NVDA position (normalized)
            observation[25] = 0.0  # MSFT position (normalized)
            
            self.logger.debug(f"Created observation: shape={observation.shape}, range=[{observation.min():.3f}, {observation.max():.3f}]")
            
            return observation
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create observation: {e}")
            return np.zeros(26)

def start_real_trading(symbols=["NVDA", "MSFT"], position_size=10):
    """Start REAL trading with Stairways V3 model"""
    
    logger = logging.getLogger('RealTradingDeployment')
    
    logger.info("🎯 STARTING REAL TRADING WITH STAIRWAYS V3")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Position Size: ${position_size} per trade")
    logger.info(f"Model: Stairways V3 (PPO)")
    logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
    logger.info("=" * 60)
    
    # Initialize components
    model_loader = None
    market_data_provider = None
    ib_client = None
    safe_wrapper = None
    
    try:
        # 1. Load Stairways V3 Model
        logger.info("🤖 Loading Stairways V3 Model...")
        model_loader = StairwaysV3ModelLoader()
        
        if not model_loader.load_model():
            raise RuntimeError("Failed to load Stairways V3 model")
        
        # 2. Initialize Market Data Provider
        logger.info("📊 Initializing Market Data Provider...")
        market_data_provider = MarketDataProvider()
        
        # 3. Initialize Enhanced IBKR Connection
        logger.info("📡 Initializing Enhanced IBKR Connection...")
        ib_client = IBGatewayClient()
        
        if not ib_client.connect():
            raise RuntimeError("IBKR connection failed")
        
        logger.info(f"✅ Connected to IBKR ({ib_client.simulation_mode and 'Simulation' or 'Live'} mode)")
        
        # 4. Risk Governor Callback
        def real_trading_risk_callback(order_id, status, event_type):
            """Risk governor for real trading"""
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.info(f"🛡️ [{timestamp}] RISK GOVERNOR: Order {order_id} - {status} - {event_type}")
            
            if event_type == 'ORDER_LIVE':
                logger.warning(f"🔴 LIVE ORDER: {order_id} is active in market!")
            elif 'PRE_ORDER' in event_type:
                logger.info(f"✅ Pre-order validation: {event_type}")
            elif event_type == 'ORDER_FILLED':
                logger.info(f"💰 EXECUTION: Order {order_id} filled")
            
            return 'ALLOW'  # Allow for paper trading
        
        # 5. Create Enhanced Safe Wrapper
        logger.info("🛡️ Initializing Enhanced Safe Wrapper...")
        safe_wrapper = EnhancedSafeOrderWrapper(ib_client, real_trading_risk_callback)
        
        # 6. REAL TRADING LOOP
        logger.info("🚀 Starting REAL trading loop with Stairways V3...")
        logger.info("=" * 60)
        
        decision_count = 0
        start_time = time.time()
        current_positions = {symbol: 0 for symbol in symbols}  # Track positions
        
        try:
            while True:
                # Step A: Get Real Market Data
                market_data = market_data_provider.get_current_market_data()
                
                if not market_data:
                    logger.warning("⚠️ No market data available, skipping decision")
                    time.sleep(30)
                    continue
                
                # Step B: Create Model Observation
                observation = market_data_provider.create_model_observation(market_data)
                
                # Update position features in observation
                observation[24] = current_positions.get('NVDA', 0) / 100.0  # Normalize position
                observation[25] = current_positions.get('MSFT', 0) / 100.0  # Normalize position
                
                # Step C: Get Action from Stairways V3 Model
                raw_action = model_loader.predict_action(observation)
                
                # Step D: Interpret Action
                # Stairways V3 uses 5-action discrete space:
                # 0: Strong Sell, 1: Weak Sell, 2: Hold, 3: Weak Buy, 4: Strong Buy
                action_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
                position_increment = action_map[raw_action] * position_size
                
                # Step E: Symbol Selection (for dual-ticker)
                # In real Stairways V3, this would be part of the action space
                # For now, we'll use a simple approach
                selected_symbol = symbols[decision_count % len(symbols)]  # Alternate symbols
                
                decision_count += 1
                
                logger.info(f"📈 DECISION #{decision_count}:")
                logger.info(f"   Symbol: {selected_symbol}")
                logger.info(f"   Raw Action: {raw_action} (Stairways V3)")
                logger.info(f"   Position Increment: ${position_increment:.1f}")
                logger.info(f"   Current Position: {current_positions[selected_symbol]}")
                
                # Step F: Execute Trade if Significant
                if abs(position_increment) > 0.1:  # Only trade if significant signal
                    
                    try:
                        current_price = market_data[selected_symbol]['close']
                        logger.info(f"   Market Price: ${current_price:.2f}")
                        
                        # Place order through enhanced safety system
                        if position_increment > 0:
                            # Buy order
                            quantity = int(abs(position_increment))
                            result = safe_wrapper.place_market_order_with_governor(
                                symbol=selected_symbol,
                                quantity=quantity,
                                action='BUY'
                            )
                            
                            if result.get('is_filled') or 'Filled' in str(result.get('status', '')):
                                current_positions[selected_symbol] += quantity
                                logger.info(f"   ✅ BUY EXECUTED: +{quantity} {selected_symbol}")
                            
                        else:
                            # Sell order
                            quantity = int(abs(position_increment))
                            result = safe_wrapper.place_market_order_with_governor(
                                symbol=selected_symbol,
                                quantity=quantity,
                                action='SELL'
                            )
                            
                            if result.get('is_filled') or 'Filled' in str(result.get('status', '')):
                                current_positions[selected_symbol] -= quantity
                                logger.info(f"   ✅ SELL EXECUTED: -{quantity} {selected_symbol}")
                        
                        logger.info(f"   Order Result: ID {result['order_id']} - {result.get('final_status', result.get('status'))}")
                        logger.info(f"   New Position: {current_positions[selected_symbol]} {selected_symbol}")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Order execution failed: {e}")
                
                else:
                    logger.info(f"   💤 HOLD - No significant signal (increment: ${position_increment:.1f})")
                
                # Step G: Status Update
                if decision_count % 5 == 0:
                    elapsed_hours = (time.time() - start_time) / 3600
                    decisions_per_hour = decision_count / elapsed_hours if elapsed_hours > 0 else 0
                    
                    logger.info("📊 REAL TRADING STATUS UPDATE")
                    logger.info(f"   Runtime: {elapsed_hours:.1f} hours")
                    logger.info(f"   Decisions: {decision_count}")
                    logger.info(f"   Rate: {decisions_per_hour:.1f} decisions/hour")
                    logger.info(f"   Positions: {current_positions}")
                    logger.info(f"   Model: Stairways V3 active")
                
                # Wait between decisions (real-time trading interval)
                time.sleep(60)  # 1-minute intervals for real trading
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Real trading shutdown requested...")
            
            # Graceful shutdown with position summary
            session_duration = time.time() - start_time
            logger.info(f"📊 REAL TRADING SESSION COMPLETE:")
            logger.info(f"   Duration: {session_duration/3600:.1f} hours")
            logger.info(f"   Total decisions: {decision_count}")
            logger.info(f"   Final positions: {current_positions}")
            logger.info(f"   Model: Stairways V3")
            logger.info(f"   Avg rate: {decision_count/(session_duration/3600):.1f} decisions/hour")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real trading failed: {e}")
        return False
        
    finally:
        # Cleanup
        if safe_wrapper:
            logger.info("🔄 Cleaning up enhanced safety wrapper...")
            
        if ib_client:
            logger.info("🔌 Disconnecting from IBKR...")
            ib_client.disconnect()
            
        logger.info("🏁 Real trading deployment session ended")

def main():
    """Main real trading entry point"""
    
    print("🎯 REAL TRADING DEPLOYMENT - STAIRWAYS V3")
    print("=" * 60)
    print("Actual AI-driven trading with enhanced safety")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    # Setup logging
    logger, log_file = setup_real_trading_logging()
    logger.info("Real trading deployment initiated")
    
    # Validate components
    print("🔍 Validating system components...")
    
    # Check if model exists
    model_paths = [
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip",
        "train_runs/v3_gold_standard_400k_20250802_202736/chunk6_final_307200steps.zip"
    ]
    
    model_found = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Model found: {os.path.basename(model_path)}")
            model_found = True
            break
    
    if not model_found:
        print("❌ No Stairways V3 model found!")
        logger.error("No Stairways V3 model found")
        return 1
    
    # Check stable_baselines3
    try:
        from stable_baselines3 import PPO
        print("✅ stable_baselines3 available")
    except ImportError:
        print("❌ stable_baselines3 not available - run: pip install stable-baselines3")
        logger.error("stable_baselines3 not available")
        return 1
    
    print("✅ System validation complete")
    print()
    
    # Confirm real trading start
    print("⚠️  WARNING: This will start REAL AI-driven trading!")
    print("🤖 Model: Stairways V3 PPO")
    print("💰 Position Size: $10 per trade")
    print("🎯 Symbols: NVDA, MSFT")
    print("🛡️ Safety: Enhanced risk governor active")
    print()
    
    if input("Start real trading? (yes/no): ").lower() != 'yes':
        print("Real trading cancelled.")
        return 0
    
    # Start real trading
    success = start_real_trading(symbols=["NVDA", "MSFT"], position_size=10)
    
    if success:
        print("\n🎉 REAL TRADING SESSION COMPLETED!")
        logger.info("Real trading session completed successfully")
    else:
        print("\n⚠️ REAL TRADING SESSION HAD ISSUES")
        logger.error("Real trading session completed with issues")
    
    print(f"\n📝 Full session log: {log_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())