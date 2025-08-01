#!/usr/bin/env python3
"""
Live AI-Driven Trading System
Replaces random trading logic with AI model predictions
Connects to ai_inference_service for real-time decisions
"""

import os
import time
import logging
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from src.brokers.ib_gateway import IBGatewayClient
from src.gym_env.dual_ticker_data_adapter import DualTickerDataAdapter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading configuration
TRANSACTION_FEE = 0.65
BID_ASK_SPREAD_PCT = 0.002
AI_INFERENCE_URL = os.getenv('AI_INFERENCE_URL', 'http://localhost:8000')
TRADE_QUANTITY = int(os.getenv('TRADE_QUANTITY', '5'))

class LiveAITrader:
    """Live trading system with AI decision making"""
    
    def __init__(self):
        # IBKR connection
        self.ib_client = IBGatewayClient()
        
        # Data adapter for features
        self.data_adapter = DualTickerDataAdapter()
        
        # Trading state
        self.positions = {"NVDA": 0, "MSFT": 0}
        self.cash = 50000.0
        self.total_fees_paid = 0.0
        self.trades_count = 0
        self.portfolio_value = 100000.0
        
        # AI service connection with timeout
        self.ai_session = requests.Session()
        self.ai_session.timeout = 10  # Global timeout
        self.ai_available = self._check_ai_service()
        
        # Risk limits
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '500'))
        self.max_position_value = float(os.getenv('MAX_POSITION_VALUE', '10000'))
        self.daily_pnl = 0.0
        
        # Market data history for features
        self.price_history = {"NVDA": [], "MSFT": []}
        self.lookback_window = 50
        
    def _check_ai_service(self) -> bool:
        """Check if AI inference service is available"""
        try:
            response = self.ai_session.get(f"{AI_INFERENCE_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get("model_loaded", False):
                    logger.info("✅ AI inference service connected and model loaded")
                    return True
                else:
                    logger.warning("⚠️ AI service online but model not loaded")
                    return False
            else:
                logger.warning(f"⚠️ AI service unhealthy: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ AI service not available: {e}")
            return False
    
    def get_market_features(self, symbol: str, price: float) -> List[float]:
        """Extract market features for the model"""
        
        # Add current price to history
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback_window:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_window:]
        
        # If not enough history, pad with current price
        prices = self.price_history[symbol]
        if len(prices) < self.lookback_window:
            prices = [price] * (self.lookback_window - len(prices)) + prices
        
        # Calculate technical indicators (simplified)
        prices_array = np.array(prices)
        
        # Price-based features
        returns = np.diff(prices_array) / prices_array[:-1] if len(prices_array) > 1 else [0.0]
        
        # Simple technical indicators
        sma_5 = np.mean(prices_array[-5:]) if len(prices_array) >= 5 else price
        sma_20 = np.mean(prices_array[-20:]) if len(prices_array) >= 20 else price
        
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        momentum = (price - prices_array[0]) / prices_array[0] if len(prices_array) > 1 else 0.0
        
        # Normalize features
        features = [
            price / 1000.0,  # Normalized price
            (price - sma_5) / sma_5 if sma_5 != 0 else 0.0,  # Price vs SMA5
            (price - sma_20) / sma_20 if sma_20 != 0 else 0.0,  # Price vs SMA20
            volatility,
            momentum,
            len(returns) / self.lookback_window,  # Data completeness
            # Additional technical features
            1.0 if price > sma_5 else 0.0,  # Above SMA5
            1.0 if sma_5 > sma_20 else 0.0,  # SMA5 > SMA20
            np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,  # Recent momentum
            max(returns[-10:]) if len(returns) >= 10 else 0.0,  # Max recent return
            min(returns[-10:]) if len(returns) >= 10 else 0.0,  # Min recent return
            volatility * 100.0  # Scaled volatility
        ]
        
        return features
    
    def get_observation(self, nvda_price: float, msft_price: float) -> List[float]:
        """Create full observation for the model (26 features)"""
        
        # Get features for both symbols (12 each)
        nvda_features = self.get_market_features("NVDA", nvda_price)
        msft_features = self.get_market_features("MSFT", msft_price)
        
        # Position features (2)
        position_features = [
            self.positions["NVDA"] / 100.0,  # Normalized NVDA position
            self.positions["MSFT"] / 100.0   # Normalized MSFT position
        ]
        
        # Combine all features (12 + 12 + 2 = 26)
        observation = nvda_features + msft_features + position_features
        
        return observation
    
    def get_ai_decision(self, observation: List[float]) -> Tuple[int, str]:
        """Get trading decision from AI model"""
        
        if not self.ai_available:
            # Fallback to simple logic if AI unavailable
            logger.warning("🤖 AI not available, using fallback logic")
            return 0, "HOLD_HOLD"  # Conservative fallback
        
        try:
            payload = {
                "observation": observation,
                "timestamp": datetime.now().isoformat(),
                "symbols": ["NVDA", "MSFT"]
            }
            
            response = self.ai_session.post(
                f"{AI_INFERENCE_URL}/predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result["action"]
                action_name = result["action_name"]
                logger.info(f"🤖 AI Decision: {action_name} (action={action})")
                return action, action_name
            else:
                logger.error(f"AI prediction failed: {response.status_code}")
                return 0, "HOLD_HOLD"
                
        except Exception as e:
            logger.error(f"AI request failed: {e}")
            return 0, "HOLD_HOLD"
    
    def execute_trade_action(self, action: int, nvda_price: float, msft_price: float, market_data: Dict) -> bool:
        """Execute the AI-determined trade action"""
        
        # Action mapping:
        # 0: HOLD_HOLD, 1: HOLD_BUY, 2: HOLD_SELL, 3: BUY_HOLD, 4: BUY_BUY, 
        # 5: BUY_SELL, 6: SELL_HOLD, 7: SELL_BUY, 8: SELL_SELL
        
        executed_trades = []
        
        # NVDA action (first part)
        nvda_action = action // 3  # 0=HOLD, 1=BUY, 2=SELL
        msft_action = action % 3   # 0=HOLD, 1=BUY, 2=SELL
        
        # Execute NVDA trade
        if nvda_action == 1:  # BUY
            if self._execute_buy("NVDA", TRADE_QUANTITY, market_data["NVDA"]["ask"]):
                executed_trades.append(f"BUY {TRADE_QUANTITY} NVDA")
                
        elif nvda_action == 2:  # SELL
            sell_qty = min(TRADE_QUANTITY, self.positions["NVDA"])
            if sell_qty > 0 and self._execute_sell("NVDA", sell_qty, market_data["NVDA"]["bid"]):
                executed_trades.append(f"SELL {sell_qty} NVDA")
        
        # Execute MSFT trade
        if msft_action == 1:  # BUY
            if self._execute_buy("MSFT", TRADE_QUANTITY, market_data["MSFT"]["ask"]):
                executed_trades.append(f"BUY {TRADE_QUANTITY} MSFT")
                
        elif msft_action == 2:  # SELL
            sell_qty = min(TRADE_QUANTITY, self.positions["MSFT"])
            if sell_qty > 0 and self._execute_sell("MSFT", sell_qty, market_data["MSFT"]["bid"]):
                executed_trades.append(f"SELL {sell_qty} MSFT")
        
        if executed_trades:
            logger.info(f"🎯 Executed: {', '.join(executed_trades)}")
            return True
        
        return False
    
    def _execute_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute buy order with real-time account check"""
        # Re-query account for latest cash (safety against concurrent trades)
        account_info = self.ib_client.get_account_info()
        if account_info:
            available_funds = account_info.get('available_funds', self.cash)
            if abs(available_funds - self.cash) > 1000:  # Significant discrepancy
                logger.warning(f"Cash discrepancy detected: local=${self.cash:.2f}, IBKR=${available_funds:.2f}")
                self.cash = available_funds  # Sync with IBKR
        
        trade_cost = quantity * price + TRANSACTION_FEE
        
        if self.cash >= trade_cost:
            self.positions[symbol] += quantity
            self.cash -= trade_cost
            self.total_fees_paid += TRANSACTION_FEE
            self.trades_count += 1
            logger.info(f"🟢 BUY {quantity} {symbol} @ ${price:.2f} + ${TRANSACTION_FEE:.2f} fee")
            return True
        else:
            logger.warning(f"❌ Insufficient cash for {quantity} {symbol} (need ${trade_cost:.2f}, have ${self.cash:.2f})")
            return False
    
    def _execute_sell(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute sell order"""
        if self.positions[symbol] >= quantity:
            trade_proceeds = quantity * price - TRANSACTION_FEE
            self.positions[symbol] -= quantity
            self.cash += trade_proceeds
            self.total_fees_paid += TRANSACTION_FEE
            self.trades_count += 1
            logger.info(f"🔴 SELL {quantity} {symbol} @ ${price:.2f} - ${TRANSACTION_FEE:.2f} fee")
            return True
        else:
            logger.warning(f"❌ Insufficient {symbol} position (have {self.positions[symbol]}, need {quantity})")
            return False
    
    def update_portfolio_value(self, market_data: Dict):
        """Update total portfolio value"""
        self.portfolio_value = self.cash
        
        for symbol, qty in self.positions.items():
            if qty > 0 and symbol in market_data:
                mid_price = market_data[symbol]["price"]
                self.portfolio_value += qty * mid_price
    
    def push_metrics(self):
        """Push metrics to monitoring system with symbol labels"""
        try:
            # Enhanced metrics with per-symbol labels
            metrics_data = {
                "ai_paper_portfolio_value": self.portfolio_value,
                "ai_paper_cash": self.cash,
                "ai_paper_nvda_position": self.positions["NVDA"],
                "ai_paper_msft_position": self.positions["MSFT"],
                "ai_paper_trades_count": self.trades_count,
                "ai_paper_fees_paid": self.total_fees_paid,
                "ai_paper_daily_pnl": getattr(self, 'daily_pnl', 0.0),
                "ai_paper_timestamp": int(time.time())
            }
            
            metrics_text = "\\n".join([f"{key} {value}" for key, value in metrics_data.items()]) + "\\n"
            
            response = requests.post(
                "http://localhost:9091/metrics/job/ai_paper_trading",
                data=metrics_text,
                timeout=2.0
            )
            
            if response.status_code == 200:
                logger.info(f"📊 Metrics pushed: Portfolio=${self.portfolio_value:.2f}, Trades={self.trades_count}")
                
        except Exception as e:
            logger.debug(f"Metrics push failed: {e}")
    
    def run_trading_session(self, duration_minutes: int = 30):
        """Run AI-driven trading session"""
        
        logger.info(f"🚀 Starting AI Trading Session ({duration_minutes} minutes)")
        
        # Connect to IBKR
        if not self.ib_client.connect():
            logger.error("❌ Failed to connect to IBKR")
            return False
        
        logger.info("✅ Connected to IBKR")
        
        # Get account info
        account_info = self.ib_client.get_account_info()
        if account_info:
            self.cash = account_info.get('available_funds', self.cash)
            logger.info(f"💰 Account: {account_info['account_id']}, Cash: ${self.cash:,.2f}")
        
        # Trading loop
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle = 0
        
        try:
            while time.time() < end_time:
                cycle += 1
                logger.info(f"\\n🔄 AI Trading Cycle {cycle}")
                
                # Get market data
                market_data = {}
                symbols = ["NVDA", "MSFT"]
                
                for symbol in symbols:
                    price = self.ib_client.get_current_price(symbol)
                    if price > 0:
                        # Add realistic bid/ask spread
                        spread = price * BID_ASK_SPREAD_PCT
                        bid_price = price - (spread / 2)
                        ask_price = price + (spread / 2)
                        
                        market_data[symbol] = {
                            'price': price,
                            'bid': bid_price,
                            'ask': ask_price
                        }
                        logger.info(f"📊 {symbol}: ${price:.2f} (Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f})")
                
                if len(market_data) == 2:  # Both symbols available
                    # Create observation for AI
                    observation = self.get_observation(
                        market_data["NVDA"]["price"],
                        market_data["MSFT"]["price"]
                    )
                    
                    # Get AI decision
                    action, action_name = self.get_ai_decision(observation)
                    
                    # Execute trades
                    if action != 0:  # Not HOLD_HOLD
                        self.execute_trade_action(action, 
                                                market_data["NVDA"]["price"],
                                                market_data["MSFT"]["price"], 
                                                market_data)
                    
                    # Update portfolio value
                    self.update_portfolio_value(market_data)
                    
                    # Push metrics
                    self.push_metrics()
                    
                    # Status update
                    logger.info(f"💼 Portfolio: ${self.portfolio_value:.2f}, Cash: ${self.cash:.2f}, Fees: ${self.total_fees_paid:.2f}")
                    logger.info(f"📊 Positions: NVDA={self.positions['NVDA']}, MSFT={self.positions['MSFT']}")
                
                # Wait between cycles (configurable via env)
                cycle_delay = int(os.getenv('TRADING_CYCLE_SECONDS', '30'))
                time.sleep(cycle_delay)
                
        except KeyboardInterrupt:
            logger.info("\\n⏹️ Trading session interrupted by user")
        
        finally:
            # Final summary
            pnl = self.portfolio_value - 100000
            logger.info(f"\\n🎉 AI Trading Session Complete!")
            logger.info(f"📊 Final Portfolio: ${self.portfolio_value:.2f}")
            logger.info(f"💰 Final Cash: ${self.cash:.2f}")
            logger.info(f"📈 Total Trades: {self.trades_count}")
            logger.info(f"💸 Total Fees: ${self.total_fees_paid:.2f}")
            logger.info(f"🎯 P&L: ${pnl:.2f} ({pnl/100000*100:.2f}%)")
            
            # Disconnect
            self.ib_client.disconnect()
            logger.info("🔌 Disconnected from IBKR")

def main():
    """Main entry point"""
    trader = LiveAITrader()
    
    # Run 30-minute session by default
    duration = int(os.getenv('TRADING_SESSION_MINUTES', '30'))
    trader.run_trading_session(duration)

if __name__ == "__main__":
    main()