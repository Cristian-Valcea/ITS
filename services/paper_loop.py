#!/usr/bin/env python3
"""
📈 PAPER TRADING LOOP
Live trading loop with Redis ticks → Feature pipeline → Inference API → Risk Guard → IB Execution
"""

import argparse
import asyncio
import json
import time
import logging
import signal
import sys
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Any
import redis
import requests
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    symbols: List[str]
    model_endpoint: str
    portfolio_cap: float
    risk_daily_max_loss: float
    risk_intraday_dd: float
    trading_start: dt_time
    trading_end: dt_time
    log_file: str

@dataclass
class PortfolioState:
    timestamp: float
    total_value: float
    cash: float
    positions: Dict[str, float]
    daily_pnl: float
    trades_today: int
    drawdown: float

@dataclass
class TradingDecision:
    timestamp: float
    symbol: str
    action: str
    quantity: int
    confidence: float
    price: float
    risk_approved: bool
    risk_reason: str
    latency_ms: float

class PaperTradingLoop:
    """Main paper trading loop orchestrator"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Portfolio state
        self.portfolio = PortfolioState(
            timestamp=time.time(),
            total_value=config.portfolio_cap,
            cash=config.portfolio_cap,
            positions={symbol: 0.0 for symbol in config.symbols},
            daily_pnl=0.0,
            trades_today=0,
            drawdown=0.0
        )
        
        # Market data cache
        self.market_prices = {symbol: 0.0 for symbol in config.symbols}
        self.last_tick_time = time.time()
        
        # Feature pipeline state
        self.price_history = {symbol: [] for symbol in config.symbols}
        self.volume_history = {symbol: [] for symbol in config.symbols}
        self.feature_window = 50  # Match model training
        
        # Performance tracking
        self.decisions_log = []
        self.metrics_last_push = time.time()
        self.session_start = time.time()
        
        # Log file
        self.log_file = open(config.log_file, 'a')
        
        logger.info(f"📈 Paper Trading Loop initialized")
        logger.info(f"   Symbols: {config.symbols}")
        logger.info(f"   Portfolio Cap: ${config.portfolio_cap:,.2f}")
        logger.info(f"   Daily Max Loss: ${config.risk_daily_max_loss:,.2f}")
        logger.info(f"   Intraday DD: {config.risk_intraday_dd:.1%}")
        logger.info(f"   Trading Hours: {config.trading_start} - {config.trading_end}")
    
    def is_trading_hours(self) -> bool:
        """Check if currently in trading hours"""
        now = datetime.now().time()
        return self.config.trading_start <= now <= self.config.trading_end
    
    def should_stop_trading(self) -> bool:
        """Check if trading should stop (risk limits or time)"""
        
        # Time check
        if not self.is_trading_hours():
            return True
            
        # Daily loss limit
        if self.portfolio.daily_pnl <= -self.config.risk_daily_max_loss:
            logger.warning(f"🚫 Daily loss limit hit: ${self.portfolio.daily_pnl:.2f}")
            return True
            
        # Intraday drawdown limit
        if self.portfolio.drawdown >= self.config.risk_intraday_dd:
            logger.warning(f"🚫 Intraday drawdown limit hit: {self.portfolio.drawdown:.1%}")
            return True
            
        return False
    
    async def read_market_ticks(self) -> List[Dict[str, Any]]:
        """Read latest market ticks from Redis streams"""
        
        try:
            # Read latest ticks from polygon:ticks stream
            messages = self.redis_client.xread({'polygon:ticks': '$'}, count=10, block=100)
            
            ticks = []
            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    # Parse tick data
                    try:
                        tick = {
                            'timestamp': float(fields.get('t', time.time() * 1000)) / 1000,
                            'symbol': fields.get('sym', ''),
                            'price': float(fields.get('c', fields.get('p', 0))),
                            'volume': int(fields.get('v', 0)),
                            'open': float(fields.get('o', 0)),
                            'high': float(fields.get('h', 0)),
                            'low': float(fields.get('l', 0))
                        }
                        
                        if tick['symbol'] in self.config.symbols and tick['price'] > 0:
                            ticks.append(tick)
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing tick: {e}")
                        continue
            
            return ticks
            
        except Exception as e:
            logger.error(f"❌ Error reading market ticks: {e}")
            return []
    
    def update_feature_pipeline(self, tick: Dict[str, Any]):
        """Update feature pipeline with new tick data"""
        
        symbol = tick['symbol']
        price = tick['price']
        volume = tick['volume']
        
        # Update market prices
        self.market_prices[symbol] = price
        self.last_tick_time = tick['timestamp']
        
        # Update price history
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.feature_window:
            self.price_history[symbol] = self.price_history[symbol][-self.feature_window:]
        
        # Update volume history
        self.volume_history[symbol].append(volume)
        if len(self.volume_history[symbol]) > self.feature_window:
            self.volume_history[symbol] = self.volume_history[symbol][-self.feature_window:]
    
    def generate_features(self) -> Optional[np.ndarray]:
        """Generate features for model inference"""
        
        try:
            features = []
            
            for symbol in self.config.symbols:
                if len(self.price_history[symbol]) < 10:  # Need minimum history
                    return None
                
                prices = np.array(self.price_history[symbol][-10:])  # Last 10 prices
                volumes = np.array(self.volume_history[symbol][-10:])  # Last 10 volumes
                
                # Basic features (simplified - production would use full technical indicators)
                price_mean = np.mean(prices)
                price_std = np.std(prices) if len(prices) > 1 else 0.0
                volume_mean = np.mean(volumes)
                
                # Returns
                returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0.0]
                recent_return = returns[-1] if len(returns) > 0 else 0.0
                
                # Normalized features
                current_price = prices[-1]
                price_norm = current_price / 1000.0  # Rough normalization
                volume_norm = volumes[-1] / 10000.0 if len(volumes) > 0 else 0.0
                
                # Add features for this symbol (12 features per symbol to match training)
                symbol_features = [
                    price_norm,
                    volume_norm, 
                    recent_return,
                    price_std / price_mean if price_mean > 0 else 0.0,
                    (current_price - price_mean) / price_mean if price_mean > 0 else 0.0,
                    volume_norm,
                    0.0,  # placeholder for RSI
                    0.0,  # placeholder for MACD
                    0.0,  # placeholder for BB
                    0.0,  # placeholder for ATR
                    0.0,  # placeholder for additional feature
                    0.0   # placeholder for additional feature
                ]
                
                features.extend(symbol_features)
            
            # Add portfolio state features (2 features)
            portfolio_return = (self.portfolio.total_value - self.config.portfolio_cap) / self.config.portfolio_cap
            drawdown_norm = self.portfolio.drawdown
            
            features.extend([portfolio_return, drawdown_norm])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Error generating features: {e}")
            return None
    
    async def get_model_prediction(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get trading decision from inference API"""
        
        try:
            start_time = time.time()
            
            # Prepare request data
            market_data = {
                "timestamp": time.time(),
                "nvda_price": self.market_prices.get('NVDA', 0.0),
                "nvda_volume": 1000,  # Simplified
                "msft_price": self.market_prices.get('MSFT', 0.0),
                "msft_volume": 1000   # Simplified
            }
            
            # Call inference API
            response = requests.post(
                f"{self.config.model_endpoint}",
                json=market_data,
                timeout=1.0  # Fast timeout for live trading
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction = response.json()
                prediction['latency_ms'] = latency_ms
                return prediction
            else:
                logger.warning(f"⚠️ Model prediction failed: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Model prediction timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting model prediction: {e}")
            return None
    
    def convert_action_to_orders(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert model action to trading orders"""
        
        action = prediction.get('action', 0)
        confidence = prediction.get('confidence', 0.0)
        
        # Action mapping (same as training)
        action_map = {
            0: {"NVDA": 0, "MSFT": 0},      # HOLD_BOTH
            1: {"NVDA": 1, "MSFT": 0},      # BUY_NVDA_HOLD_MSFT
            2: {"NVDA": -1, "MSFT": 0},     # SELL_NVDA_HOLD_MSFT
            3: {"NVDA": 0, "MSFT": 1},      # HOLD_NVDA_BUY_MSFT
            4: {"NVDA": 1, "MSFT": 1},      # BUY_BOTH
            5: {"NVDA": -1, "MSFT": 1},     # SELL_NVDA_BUY_MSFT
            6: {"NVDA": 0, "MSFT": -1},     # HOLD_NVDA_SELL_MSFT
            7: {"NVDA": 1, "MSFT": -1},     # BUY_NVDA_SELL_MSFT
            8: {"NVDA": -1, "MSFT": -1}     # SELL_BOTH
        }
        
        position_changes = action_map.get(action, {"NVDA": 0, "MSFT": 0})
        
        orders = []
        for symbol, change in position_changes.items():
            if change != 0:
                # Calculate order size based on confidence and portfolio
                base_size = 10  # Base shares
                confidence_mult = max(0.5, confidence)
                order_size = int(base_size * confidence_mult)
                
                order = {
                    "symbol": symbol,
                    "action": "BUY" if change > 0 else "SELL",
                    "quantity": order_size,
                    "price": self.market_prices.get(symbol, 0.0),
                    "timestamp": time.time()
                }
                orders.append(order)
        
        return orders
    
    async def execute_orders_via_risk_guard(self, orders: List[Dict[str, Any]]) -> List[TradingDecision]:
        """Execute orders through risk guard and IB executor"""
        
        decisions = []
        
        for order in orders:
            try:
                # Create trading recommendation for risk guard
                recommendation = {
                    "timestamp": order["timestamp"],
                    "action": 1 if order["action"] == "BUY" else 2,  # Simplified action mapping
                    "action_name": f"{order['action']}_{order['symbol']}",
                    "confidence": 0.75,
                    "portfolio_value": self.portfolio.total_value,
                    "positions": json.dumps(self.portfolio.positions),
                    "risk_metrics": json.dumps({"symbol": order["symbol"]})
                }
                
                # Send to risk guard via Redis
                self.redis_client.xadd("trading:recommendations", recommendation)
                
                # For simulation, assume risk approval (real system would wait for response)
                decision = TradingDecision(
                    timestamp=order["timestamp"],
                    symbol=order["symbol"],
                    action=order["action"],
                    quantity=order["quantity"],
                    confidence=0.75,
                    price=order["price"],
                    risk_approved=True,
                    risk_reason="Paper trading simulation",
                    latency_ms=5.0
                )
                
                decisions.append(decision)
                
                # Update portfolio state (simplified)
                if decision.risk_approved:
                    symbol = decision.symbol
                    quantity_change = decision.quantity if decision.action == "BUY" else -decision.quantity
                    cost = quantity_change * decision.price
                    
                    self.portfolio.positions[symbol] += quantity_change
                    self.portfolio.cash -= cost
                    self.portfolio.trades_today += 1
                    
                    logger.info(f"📊 Executed: {decision.action} {decision.quantity} {symbol} @ ${decision.price:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error executing order: {e}")
                continue
        
        return decisions
    
    def update_portfolio_state(self):
        """Update portfolio valuation and risk metrics"""
        
        # Calculate current portfolio value
        position_value = sum(
            self.portfolio.positions[symbol] * self.market_prices.get(symbol, 0.0)
            for symbol in self.config.symbols
        )
        
        self.portfolio.total_value = self.portfolio.cash + position_value
        self.portfolio.timestamp = time.time()
        
        # Calculate daily P&L (simplified - assumes session start = day start)
        self.portfolio.daily_pnl = self.portfolio.total_value - self.config.portfolio_cap
        
        # Calculate drawdown
        max_value = max(self.config.portfolio_cap, self.portfolio.total_value)
        self.portfolio.drawdown = (max_value - self.portfolio.total_value) / max_value
    
    async def push_prometheus_metrics(self):
        """Push P&L metrics to Prometheus via Pushgateway"""
        
        try:
            metrics_data = {
                "paper_loop_portfolio_value": self.portfolio.total_value,
                "paper_loop_daily_pnl": self.portfolio.daily_pnl,
                "paper_loop_drawdown": self.portfolio.drawdown,
                "paper_loop_trades_today": self.portfolio.trades_today,
                "paper_loop_cash": self.portfolio.cash,
                "paper_loop_nvda_position": self.portfolio.positions.get("NVDA", 0),
                "paper_loop_msft_position": self.portfolio.positions.get("MSFT", 0),
                "paper_loop_session_minutes": (time.time() - self.session_start) / 60
            }
            
            # Format for Pushgateway
            metrics_text = "\n".join([f"{key} {value}" for key, value in metrics_data.items()])
            
            response = requests.post(
                "http://localhost:9091/metrics/job/paper_loop",
                data=metrics_text,
                timeout=2.0
            )
            
            if response.status_code == 200:
                logger.debug("📊 Metrics pushed to Prometheus")
            
        except Exception as e:
            logger.debug(f"Prometheus push failed: {e}")
    
    def log_decision(self, decision: TradingDecision):
        """Log trading decision to JSON Lines file"""
        
        log_entry = {
            "timestamp": decision.timestamp,
            "symbol": decision.symbol,
            "action": decision.action,
            "quantity": decision.quantity,
            "price": decision.price,
            "confidence": decision.confidence,
            "risk_approved": decision.risk_approved,
            "risk_reason": decision.risk_reason,
            "latency_ms": decision.latency_ms,
            "portfolio_value": self.portfolio.total_value,
            "daily_pnl": self.portfolio.daily_pnl,
            "drawdown": self.portfolio.drawdown
        }
        
        self.log_file.write(json.dumps(log_entry) + '\n')
        self.log_file.flush()
    
    async def trading_loop(self):
        """Main trading loop"""
        
        logger.info("🚀 Starting paper trading loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check if should stop trading
                if self.should_stop_trading():
                    if not self.is_trading_hours():
                        logger.info("🕐 Trading hours ended - stopping loop")
                    break
                
                # Read market ticks
                ticks = await self.read_market_ticks()
                
                # Process each tick
                for tick in ticks:
                    self.update_feature_pipeline(tick)
                
                # Generate trading signals (every few ticks to avoid overtrading)
                if len(ticks) > 0 and time.time() - self.metrics_last_push > 30:
                    
                    # Generate features
                    features = self.generate_features()
                    if features is not None:
                        
                        # Get model prediction
                        prediction = await self.get_model_prediction(features)
                        if prediction:
                            
                            # Convert to orders
                            orders = self.convert_action_to_orders(prediction)
                            
                            # Execute via risk guard
                            if orders:
                                decisions = await self.execute_orders_via_risk_guard(orders)
                                
                                # Log decisions
                                for decision in decisions:
                                    self.log_decision(decision)
                                    self.decisions_log.append(decision)
                    
                    # Update portfolio state
                    self.update_portfolio_state()
                    
                    # Push metrics every 30 seconds
                    await self.push_prometheus_metrics()
                    self.metrics_last_push = time.time()
                
                # Sleep briefly to avoid excessive CPU usage
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(5.0)
        
        # Flatten positions at end of day
        await self.flatten_positions()
        
        logger.info("📈 Paper trading loop stopped")
    
    async def flatten_positions(self):
        """Flatten all positions at end of trading day"""
        
        logger.info("📉 Flattening positions at end of day")
        
        flatten_orders = []
        for symbol, position in self.portfolio.positions.items():
            if abs(position) > 0.01:  # Has significant position
                order = {
                    "symbol": symbol,
                    "action": "SELL" if position > 0 else "BUY",
                    "quantity": abs(int(position)),
                    "price": self.market_prices.get(symbol, 0.0),
                    "timestamp": time.time()
                }
                flatten_orders.append(order)
        
        if flatten_orders:
            decisions = await self.execute_orders_via_risk_guard(flatten_orders)
            for decision in decisions:
                self.log_decision(decision)
                logger.info(f"📉 Flattened: {decision.symbol} {decision.quantity} shares")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📨 Received signal {signum} - shutting down")
        self.shutdown_event.set()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.log_file:
            self.log_file.close()
    
    async def run(self):
        """Run the paper trading loop"""
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        try:
            await self.trading_loop()
        finally:
            self.cleanup()

def parse_time(time_str: str) -> dt_time:
    """Parse time string to time object"""
    return datetime.strptime(time_str, "%H:%M:%S").time()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Paper Trading Loop")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "MSFT"], help="Trading symbols")
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8000/inference", help="Model inference endpoint")
    parser.add_argument("--portfolio-cap", type=float, default=100000, help="Portfolio cap")
    parser.add_argument("--risk-daily-max-loss", type=float, default=1500, help="Daily max loss")
    parser.add_argument("--risk-intraday-dd", type=float, default=0.02, help="Intraday drawdown limit")
    parser.add_argument("--log-file", required=True, help="Log file path")
    
    args = parser.parse_args()
    
    # Get trading hours from environment
    trading_start = parse_time(os.getenv("TRADING_START", "09:35:00"))
    trading_end = parse_time(os.getenv("TRADING_END", "15:55:00"))
    
    # Create config
    config = TradingConfig(
        symbols=args.symbols,
        model_endpoint=args.model_endpoint,
        portfolio_cap=args.portfolio_cap,
        risk_daily_max_loss=args.risk_daily_max_loss,
        risk_intraday_dd=args.risk_intraday_dd,
        trading_start=trading_start,
        trading_end=trading_end,
        log_file=args.log_file
    )
    
    # Run paper trading loop
    loop = PaperTradingLoop(config)
    
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        logger.info("📈 Paper trading loop stopped by user")

if __name__ == "__main__":
    import os
    main()