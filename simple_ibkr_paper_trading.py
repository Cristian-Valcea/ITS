#!/usr/bin/env python3
"""
Simple IBKR Paper Trading Demo
Direct connection to IBKR for market data and paper trading
"""

import time
import logging
from datetime import datetime
from src.brokers.ib_gateway import IBGatewayClient
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def push_metrics_to_grafana(portfolio_value, cash, positions, trades_count):
    """Push trading metrics to Prometheus for Grafana"""
    
    metrics_data = {
        "simple_paper_portfolio_value": portfolio_value,
        "simple_paper_cash": cash,
        "simple_paper_nvda_position": positions.get("NVDA", 0),
        "simple_paper_msft_position": positions.get("MSFT", 0),
        "simple_paper_trades_count": trades_count,
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
            logger.info(f"📊 Pushed metrics: Portfolio=${portfolio_value:.2f}, Trades={trades_count}")
        
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
    
    # Push initial metrics
    push_metrics_to_grafana(portfolio_value, cash, positions, trades_count)
    
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
        
        # Get updated market data
        for symbol in symbols:
            price = ib_client.get_current_price(symbol)
            if price > 0:
                market_data[symbol] = {'price': price}
                logger.info(f"📊 {symbol}: ${price:.2f}")
        
        # Simple trading logic (buy if even cycle, sell if odd)
        if i % 2 == 0:  # Buy cycle
            symbol = "NVDA" if i % 4 == 0 else "MSFT"
            quantity = 5
            
            if market_data.get(symbol):
                price = market_data[symbol].get('price', 0)
                if price > 0:
                    logger.info(f"🟢 BUY {quantity} {symbol} @ ${price}")
                    positions[symbol] += quantity
                    cash -= quantity * price
                    trades_count += 1
        
        else:  # Sell cycle
            symbol = "NVDA" if positions["NVDA"] > 0 else "MSFT"
            quantity = min(3, positions[symbol])
            
            if quantity > 0 and market_data.get(symbol):
                price = market_data[symbol].get('price', 0)
                if price > 0:
                    logger.info(f"🔴 SELL {quantity} {symbol} @ ${price}")
                    positions[symbol] -= quantity
                    cash += quantity * price
                    trades_count += 1
        
        # Update portfolio value
        portfolio_value = cash
        for symbol, qty in positions.items():
            if qty > 0 and market_data.get(symbol):
                portfolio_value += qty * market_data[symbol].get('price', 0)
        
        # Push metrics to Grafana
        push_metrics_to_grafana(portfolio_value, cash, positions, trades_count)
        
        logger.info(f"💼 Portfolio: ${portfolio_value:.2f}, Cash: ${cash:.2f}")
        logger.info(f"📊 Positions: NVDA={positions['NVDA']}, MSFT={positions['MSFT']}")
        
        # Wait between cycles
        time.sleep(10)
    
    logger.info(f"\n🎉 Paper Trading Demo Complete!")
    logger.info(f"📊 Final Portfolio: ${portfolio_value:.2f}")
    logger.info(f"💰 Final Cash: ${cash:.2f}")
    logger.info(f"📈 Total Trades: {trades_count}")
    logger.info(f"🎯 P&L: ${portfolio_value - 100000:.2f}")
    
    # Disconnect
    ib_client.disconnect()
    logger.info("🔌 Disconnected from IBKR")

if __name__ == "__main__":
    main()