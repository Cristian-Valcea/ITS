#!/usr/bin/env python3
"""
Simple IBKR Paper Trading Demo
Direct connection to IBKR for market data and paper trading
"""

import time
import logging
import random
import os
from datetime import datetime
from dotenv import load_dotenv
from src.brokers.ib_gateway import IBGatewayClient
import requests

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading fees and spread configuration
TRANSACTION_FEE = 0.65  # IBKR fixed fee per trade
BID_ASK_SPREAD_PCT = 0.002  # 0.2% spread (buy higher, sell lower)
PRICE_VOLATILITY = 0.005  # 0.5% price movement range

def push_metrics_to_grafana(portfolio_value, cash, positions, trades_count, fees_paid=0):
    """Push trading metrics to Prometheus for Grafana"""
    
    metrics_data = {
        "simple_paper_portfolio_value": portfolio_value,
        "simple_paper_cash": cash,
        "simple_paper_nvda_position": positions.get("NVDA", 0),
        "simple_paper_msft_position": positions.get("MSFT", 0),
        "simple_paper_trades_count": trades_count,
        "simple_paper_fees_paid": fees_paid,
        "simple_paper_timestamp": int(time.time())
    }
    
    metrics_text = "\n".join([f"{key} {value}" for key, value in metrics_data.items()]) + "\n"
    
    try:
        response = requests.post(
            "http://localhost:9091/metrics/job/simple_paper_trading",
            data=metrics_text,
            timeout=2.0
        )
        
        if response.status_code == 200:
            logger.info(f"📊 Pushed metrics: Portfolio=${portfolio_value:.2f}, Trades={trades_count}, Fees=${fees_paid:.2f}")
        
    except Exception as e:
        logger.debug(f"Metrics push failed: {e}")

def main():
    """Simple paper trading demo"""
    
    logger.info("🚀 Starting Simple IBKR Paper Trading Demo")
    
    # Connect to IBKR
    ib_client = IBGatewayClient()
    
    if not ib_client.connect():
        logger.error("❌ Failed to connect to IBKR")
        return
    
    logger.info("✅ Connected to IBKR Paper Trading")
    
    # Get account info
    account_info = ib_client.get_account_info()
    if account_info:
        logger.info(f"💰 Account: {account_info['account_id']}")
        logger.info(f"💰 Net Liquidation: ${account_info['net_liquidation']:,.2f}")
        logger.info(f"💰 Available Funds: ${account_info['available_funds']:,.2f}")
    
    # Initialize portfolio tracking
    portfolio_value = account_info.get('net_liquidation', 100000) if account_info else 100000
    cash = account_info.get('available_funds', 50000) if account_info else 50000
    positions = {"NVDA": 0, "MSFT": 0}
    trades_count = 0
    total_fees_paid = 0.0
    
    # Push initial metrics
    push_metrics_to_grafana(portfolio_value, cash, positions, trades_count, total_fees_paid)
    
    # Get market data for NVDA and MSFT
    symbols = ["NVDA", "MSFT"]
    market_data = {}
    
    for symbol in symbols:
        logger.info(f"📊 Requesting market data for {symbol}...")
        price = ib_client.get_current_price(symbol)
        if price > 0:
            market_data[symbol] = {'price': price}
            logger.info(f"📈 {symbol}: ${price:.2f}")
    
    # Simple trading simulation
    logger.info("📈 Starting paper trading simulation...")
    
    for i in range(5):  # 5 trading cycles
        logger.info(f"\n🔄 Trading Cycle {i+1}/5")
        
        # Get updated market data with realistic price movements
        for symbol in symbols:
            base_price = ib_client.get_current_price(symbol)
            if base_price > 0:
                # Add random price movement (±0.5%)
                price_change = random.uniform(-PRICE_VOLATILITY, PRICE_VOLATILITY)
                realistic_price = base_price * (1 + price_change)
                
                # Calculate bid/ask spread
                spread = realistic_price * BID_ASK_SPREAD_PCT
                bid_price = realistic_price - (spread / 2)
                ask_price = realistic_price + (spread / 2)
                
                market_data[symbol] = {
                    'price': realistic_price,
                    'bid': bid_price,
                    'ask': ask_price
                }
                logger.info(f"📊 {symbol}: ${realistic_price:.2f} (Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f})")
        
        # Simple trading logic (buy if even cycle, sell if odd)
        if i % 2 == 0:  # Buy cycle
            symbol = "NVDA" if i % 4 == 0 else "MSFT"
            quantity = 5
            
            if market_data.get(symbol):
                # Buy at ASK price (higher)
                ask_price = market_data[symbol].get('ask', 0)
                if ask_price > 0:
                    trade_cost = quantity * ask_price + TRANSACTION_FEE
                    logger.info(f"🟢 BUY {quantity} {symbol} @ ${ask_price:.2f} + ${TRANSACTION_FEE:.2f} fee")
                    positions[symbol] += quantity
                    cash -= trade_cost
                    total_fees_paid += TRANSACTION_FEE
                    trades_count += 1
        
        else:  # Sell cycle
            symbol = "NVDA" if positions["NVDA"] > 0 else "MSFT"
            quantity = min(3, positions[symbol])
            
            if quantity > 0 and market_data.get(symbol):
                # Sell at BID price (lower)
                bid_price = market_data[symbol].get('bid', 0)
                if bid_price > 0:
                    trade_proceeds = quantity * bid_price - TRANSACTION_FEE
                    logger.info(f"🔴 SELL {quantity} {symbol} @ ${bid_price:.2f} - ${TRANSACTION_FEE:.2f} fee")
                    positions[symbol] -= quantity
                    cash += trade_proceeds
                    total_fees_paid += TRANSACTION_FEE
                    trades_count += 1
        
        # Update portfolio value using mid prices
        portfolio_value = cash
        for symbol, qty in positions.items():
            if qty > 0 and market_data.get(symbol):
                mid_price = market_data[symbol].get('price', 0)
                portfolio_value += qty * mid_price
        
        # Push metrics to Grafana
        push_metrics_to_grafana(portfolio_value, cash, positions, trades_count, total_fees_paid)
        
        logger.info(f"💼 Portfolio: ${portfolio_value:.2f}, Cash: ${cash:.2f}, Fees: ${total_fees_paid:.2f}")
        logger.info(f"📊 Positions: NVDA={positions['NVDA']}, MSFT={positions['MSFT']}")
        
        # Wait between cycles
        time.sleep(10)
    
    logger.info(f"\n🎉 Paper Trading Demo Complete!")
    logger.info(f"📊 Final Portfolio: ${portfolio_value:.2f}")
    logger.info(f"💰 Final Cash: ${cash:.2f}")
    logger.info(f"📈 Total Trades: {trades_count}")
    logger.info(f"💸 Total Fees: ${total_fees_paid:.2f}")
    logger.info(f"🎯 P&L: ${portfolio_value - 100000:.2f} (including fees)")
    
    # Disconnect
    ib_client.disconnect()
    logger.info("🔌 Disconnected from IBKR")

if __name__ == "__main__":
    main()