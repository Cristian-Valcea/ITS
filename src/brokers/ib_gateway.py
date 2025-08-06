#!/usr/bin/env python3
"""
Interactive Brokers Gateway Connection for Paper Trading
Implements basic IB connection, authentication, and order management
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Load .env from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    load_dotenv(env_file)
    
except ImportError:
    pass  # dotenv not available, use system environment variables

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("ib_insync not available - IB Gateway will run in simulation mode")

logger = logging.getLogger(__name__)

class IBGatewayClient:
    """Interactive Brokers Gateway client for paper trading"""
    
    def __init__(self, host: str = None, port: int = None, client_id: int = None):
        # Debug environment variables
        logger.info(f"Environment check: IBKR_HOST_IP={os.getenv('IBKR_HOST_IP')}")
        
        # Use environment variables with fallbacks
        self.host = host or os.getenv('IBKR_HOST_IP', '127.0.0.1')
        self.port = port or int(os.getenv('IBKR_PORT', '7497'))
        self.client_id = client_id or int(os.getenv('IBKR_CLIENT_ID', '1'))
        
        # Debug: Log the configuration being used
        logger.info(f"IBKR Configuration: host={self.host}, port={self.port}, client_id={self.client_id}")
        self.ib = None
        self.connected = False
        self.simulation_mode = not IB_AVAILABLE
        
        # Paper trading credentials (optional - not needed for TWS connection)
        self.username = os.getenv('IB_USERNAME')
        self.password = os.getenv('IB_PASSWORD')
        
        # Don't force simulation mode just because credentials are missing
        # IBKR Paper Trading Workstation doesn't require username/password for API connection
        
        # Supported symbols for dual-ticker system
        self.supported_symbols = ['NVDA', 'MSFT']
        self.contracts = {}
        
        # Simulation data for testing
        self.simulation_positions = {'NVDA': 0, 'MSFT': 0}
        self.simulation_orders = []
        
    def connect(self) -> bool:
        """Connect to IB Gateway"""
        if self.simulation_mode:
            logger.info("🎭 IB Gateway running in SIMULATION MODE")
            self.connected = True
            return True
        
        try:
            self.ib = IB()
            logger.info(f"🔌 Attempting connection to {self.host}:{self.port} (Client ID: {self.client_id})")
            
            # Connect with extended timeout for initial handshake
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=60)
            
            # Wait a moment for connection to stabilize
            self.ib.sleep(1)
            
            if not self.ib.isConnected():
                raise ConnectionError("Connection established but not confirmed")
            
            self.connected = True
            logger.info(f"✅ Connected to IB Gateway at {self.host}:{self.port}")
            logger.info(f"📊 Server version: {self.ib.client.serverVersion()}")
            
            # Test basic functionality
            try:
                accounts = self.ib.managedAccounts()
                logger.info(f"👤 Managed accounts: {accounts}")
            except Exception as e:
                logger.warning(f"⚠️  Could not get accounts: {e}")
            
            # Initialize contracts for supported symbols
            self._initialize_contracts()
            
            return True
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"❌ Failed to connect to IB Gateway: {error_type}: {e}")
            
            # Provide specific guidance based on error type
            if "TimeoutError" in error_type:
                logger.error("💡 Timeout suggests IBKR API is not properly configured")
                logger.error("   Check: API enabled, trusted IPs, port 7497")
            elif "ConnectionRefusedError" in error_type:
                logger.error("💡 Connection refused suggests IBKR is not running")
                logger.error("   Check: IBKR Workstation running, correct port")
            
            logger.info("🎭 Switching to simulation mode")
            self.simulation_mode = True
            self.connected = True
            return True
    
    def test_real_connection(self) -> bool:
        """Test real IBKR connection without falling back to simulation"""
        try:
            test_ib = IB()
            logger.info(f"🧪 Testing real connection to {self.host}:{self.port}")
            
            # Try connection with timeout
            test_ib.connect(self.host, self.port, clientId=self.client_id + 100, timeout=15)
            
            if test_ib.isConnected():
                logger.info("✅ Real IBKR connection successful!")
                
                # Get basic info
                try:
                    accounts = test_ib.managedAccounts()
                    logger.info(f"👤 Accounts: {accounts}")
                    
                    server_version = test_ib.client.serverVersion()
                    logger.info(f"📊 Server version: {server_version}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Info retrieval failed: {e}")
                
                test_ib.disconnect()
                return True
            else:
                logger.error("❌ Connection not confirmed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Real connection test failed: {type(e).__name__}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.ib and not self.simulation_mode:
            try:
                self.ib.disconnect()
                logger.info("Disconnected from IB Gateway")
            except Exception as e:
                logger.error(f"Error disconnecting from IB Gateway: {e}")
        
        self.connected = False
    
    def _initialize_contracts(self):
        """Initialize stock contracts for supported symbols"""
        if self.simulation_mode:
            # Simulation contracts
            for symbol in self.supported_symbols:
                self.contracts[symbol] = {
                    'symbol': symbol,
                    'exchange': 'NASDAQ',
                    'currency': 'USD',
                    'type': 'simulation'
                }
            return
        
        for symbol in self.supported_symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                self.contracts[symbol] = contract
                logger.info(f"✅ Qualified contract for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to qualify contract for {symbol}: {e}")
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if self.simulation_mode:
            return {
                'account_id': 'SIMULATION',
                'net_liquidation': 100000.0,
                'available_funds': 50000.0,
                'buying_power': 200000.0,
                'currency': 'USD',
                'timestamp': datetime.now().isoformat(),
                'mode': 'simulation'
            }
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        try:
            account_values = self.ib.accountValues()
            account_summary = {}
            
            for value in account_values:
                if value.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower']:
                    account_summary[value.tag] = float(value.value)
            
            return {
                'account_id': account_values[0].account if account_values else 'UNKNOWN',
                'net_liquidation': account_summary.get('NetLiquidation', 0.0),
                'available_funds': account_summary.get('AvailableFunds', 0.0),
                'buying_power': account_summary.get('BuyingPower', 0.0),
                'currency': 'USD',
                'timestamp': datetime.now().isoformat(),
                'mode': 'live'
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions for supported symbols"""
        if self.simulation_mode:
            positions = {}
            for symbol in self.supported_symbols:
                positions[symbol] = {
                    'symbol': symbol,
                    'position': self.simulation_positions[symbol],
                    'market_price': 150.0 + hash(symbol) % 100,  # Fake price
                    'market_value': self.simulation_positions[symbol] * (150.0 + hash(symbol) % 100),
                    'avg_cost': 145.0 + hash(symbol) % 100,
                    'unrealized_pnl': self.simulation_positions[symbol] * 5.0,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'simulation'
                }
            return positions
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        try:
            positions = {}
            ib_positions = self.ib.positions()
            
            # Initialize all supported symbols with zero positions
            for symbol in self.supported_symbols:
                positions[symbol] = {
                    'symbol': symbol,
                    'position': 0,
                    'market_price': 0.0,
                    'market_value': 0.0,
                    'avg_cost': 0.0,
                    'unrealized_pnl': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'live'
                }
            
            # Update with actual positions
            for pos in ib_positions:
                symbol = pos.contract.symbol
                if symbol in self.supported_symbols:
                    positions[symbol].update({
                        'position': pos.position,
                        'market_price': pos.marketPrice,
                        'market_value': pos.marketValue,
                        'avg_cost': pos.avgCost,
                        'unrealized_pnl': pos.unrealizedPNL
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
    
    def place_market_order(self, symbol: str, quantity: int, action: str = 'BUY') -> Dict:
        """Place market order"""
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} not supported. Use: {self.supported_symbols}")
        
        if action not in ['BUY', 'SELL']:
            raise ValueError(f"Action must be BUY or SELL, got: {action}")
        
        if self.simulation_mode:
            order_id = len(self.simulation_orders) + 1
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'MKT',
                'status': 'Filled',
                'fill_price': 150.0 + hash(symbol) % 100,  # Fake fill price
                'timestamp': datetime.now().isoformat(),
                'mode': 'simulation'
            }
            
            # Update simulation positions
            position_change = quantity if action == 'BUY' else -quantity
            self.simulation_positions[symbol] += position_change
            
            self.simulation_orders.append(order)
            logger.info(f"🎭 Simulation order: {action} {quantity} {symbol} @ ${order['fill_price']:.2f}")
            return order
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        if symbol not in self.contracts:
            raise ValueError(f"Contract not available for {symbol}")
        
        try:
            contract = self.contracts[symbol]
            order = MarketOrder(action, quantity)
            
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be acknowledged
            self.ib.sleep(1)
            
            return {
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'MKT',
                'status': trade.orderStatus.status,
                'fill_price': trade.orderStatus.avgFillPrice or 0.0,
                'timestamp': datetime.now().isoformat(),
                'mode': 'live'
            }
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise
    
    def place_limit_order(self, symbol: str, quantity: int, price: float, action: str = 'BUY') -> Dict:
        """Place limit order"""
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} not supported. Use: {self.supported_symbols}")
        
        if action not in ['BUY', 'SELL']:
            raise ValueError(f"Action must be BUY or SELL, got: {action}")
        
        if self.simulation_mode:
            order_id = len(self.simulation_orders) + 1
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'LMT',
                'limit_price': price,
                'status': 'Submitted',
                'fill_price': 0.0,
                'timestamp': datetime.now().isoformat(),
                'mode': 'simulation'
            }
            
            self.simulation_orders.append(order)
            logger.info(f"🎭 Simulation limit order: {action} {quantity} {symbol} @ ${price:.2f}")
            return order
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        if symbol not in self.contracts:
            raise ValueError(f"Contract not available for {symbol}")
        
        try:
            contract = self.contracts[symbol]
            order = LimitOrder(action, quantity, price)
            
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be acknowledged
            self.ib.sleep(1)
            
            return {
                'order_id': trade.order.orderId,
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'LMT',
                'limit_price': price,
                'status': trade.orderStatus.status,
                'fill_price': trade.orderStatus.avgFillPrice or 0.0,
                'timestamp': datetime.now().isoformat(),
                'mode': 'live'
            }
            
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise
    
    def get_open_orders(self) -> List[Dict]:
        """Get open orders"""
        if self.simulation_mode:
            return [order for order in self.simulation_orders if order['status'] in ['Submitted', 'PreSubmitted']]
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        try:
            trades = self.ib.openTrades()
            orders = []
            
            for trade in trades:
                if trade.contract.symbol in self.supported_symbols:
                    orders.append({
                        'order_id': trade.order.orderId,
                        'symbol': trade.contract.symbol,
                        'quantity': trade.order.totalQuantity,
                        'action': trade.order.action,
                        'order_type': trade.order.orderType,
                        'limit_price': getattr(trade.order, 'lmtPrice', 0.0),
                        'status': trade.orderStatus.status,
                        'fill_price': trade.orderStatus.avgFillPrice or 0.0,
                        'timestamp': datetime.now().isoformat(),
                        'mode': 'live'
                    })
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel order by ID"""
        if self.simulation_mode:
            for order in self.simulation_orders:
                if order['order_id'] == order_id and order['status'] in ['Submitted', 'PreSubmitted']:
                    order['status'] = 'Cancelled'
                    logger.info(f"🎭 Simulation order {order_id} cancelled")
                    return True
            return False
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        try:
            trades = self.ib.openTrades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True
            
            logger.warning(f"Order {order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    def health_check(self) -> Dict:
        """Check IB Gateway connection health"""
        try:
            if self.simulation_mode:
                return {
                    'status': 'healthy',
                    'mode': 'simulation',
                    'connected': True,
                    'supported_symbols': self.supported_symbols,
                    'timestamp': datetime.now().isoformat()
                }
            
            if not self.connected or not self.ib:
                return {
                    'status': 'unhealthy',
                    'mode': 'live',
                    'connected': False,
                    'error': 'Not connected to IB Gateway',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Test connection with account info request
            account_info = self.get_account_info()
            
            return {
                'status': 'healthy',
                'mode': 'live',
                'connected': True,
                'account_id': account_info.get('account_id', 'UNKNOWN'),
                'supported_symbols': self.supported_symbols,
                'contracts_qualified': len(self.contracts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'mode': 'live' if not self.simulation_mode else 'simulation',
                'connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} not supported. Use: {self.supported_symbols}")
        
        if self.simulation_mode:
            # Return simulated prices based on symbol
            simulation_prices = {
                'NVDA': 485.50 + (hash(datetime.now().isoformat()[:13]) % 100) / 10,  # ±$10 range
                'MSFT': 412.25 + (hash(datetime.now().isoformat()[:13]) % 80) / 10    # ±$8 range
            }
            return simulation_prices.get(symbol, 150.0)
        
        if not self.connected or not self.ib:
            raise ValueError("Not connected to IB Gateway")
        
        if symbol not in self.contracts:
            raise ValueError(f"Contract not available for {symbol}")
        
        try:
            contract = self.contracts[symbol]
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Wait for market data
            
            if ticker.last and ticker.last > 0:
                return float(ticker.last)
            elif ticker.close and ticker.close > 0:
                return float(ticker.close)
            else:
                logger.warning(f"No market data available for {symbol}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise

def main():
    """CLI interface for testing IB Gateway client"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IB Gateway Client')
    parser.add_argument('--host', default=None, help='IB Gateway host (uses IBKR_HOST_IP env var if not specified)')
    parser.add_argument('--port', type=int, default=None, help='IB Gateway port (uses IBKR_PORT env var if not specified)')
    parser.add_argument('--client-id', type=int, default=None, help='Client ID (uses IBKR_CLIENT_ID env var if not specified)')
    parser.add_argument('--test', choices=['connect', 'real-test', 'account', 'positions', 'order'], 
                       default='connect', help='Test to run')
    parser.add_argument('--symbol', choices=['NVDA', 'MSFT'], default='NVDA',
                       help='Symbol for order test')
    parser.add_argument('--quantity', type=int, default=10, help='Quantity for order test')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        client = IBGatewayClient(args.host, args.port, args.client_id)
        
        # Connect
        if not client.connect():
            print("❌ Failed to connect to IB Gateway")
            return 1
        
        print(f"✅ Connected to IB Gateway (mode: {'simulation' if client.simulation_mode else 'live'})")
        
        if args.test == 'connect':
            health = client.health_check()
            print(json.dumps(health, indent=2))
        
        elif args.test == 'real-test':
            print("🧪 Testing real IBKR connection (no simulation fallback)...")
            real_success = client.test_real_connection()
            if real_success:
                print("🎉 Real IBKR connection is working!")
            else:
                print("❌ Real IBKR connection failed - check configuration")
                return 1
        
        elif args.test == 'account':
            account_info = client.get_account_info()
            print(json.dumps(account_info, indent=2))
        
        elif args.test == 'positions':
            positions = client.get_positions()
            print(json.dumps(positions, indent=2))
        
        elif args.test == 'order':
            print(f"Placing test market order: BUY {args.quantity} {args.symbol}")
            order = client.place_market_order(args.symbol, args.quantity, 'BUY')
            print(json.dumps(order, indent=2))
        
        client.disconnect()
        print("✅ IB Gateway client test successful")
        return 0
        
    except Exception as e:
        print(f"❌ IB Gateway client test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())