#!/usr/bin/env python3
"""
💰 PNL TRACKER SERVICE
Tracks portfolio performance, P&L, and generates executive dashboard metrics
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import redis
import requests
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor

# Local imports
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    position_values: Dict[str, float]  # symbol -> market value
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_return: float
    drawdown: float
    trades_today: int

@dataclass
class ExecutionRecord:
    timestamp: datetime
    order_id: str
    action: str
    symbol: str
    quantity: float
    price: float
    value: float
    commission: float
    pnl_impact: float

class PnLTrackerService:
    """Service to track portfolio performance and P&L"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.db_connection = None
        
        # Portfolio state
        self.initial_balance = 100000.0
        self.current_portfolio = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=100000.0,
            cash=100000.0,
            positions={"NVDA": 0.0, "MSFT": 0.0},
            position_values={"NVDA": 0.0, "MSFT": 0.0},
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            daily_pnl=0.0,
            total_return=0.0,
            drawdown=0.0,
            trades_today=0
        )
        
        # Performance tracking
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.execution_history: List[ExecutionRecord] = []
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Market data cache
        self.market_prices = {"NVDA": 0.0, "MSFT": 0.0}
        self.last_price_update = datetime.now()
        
        # Performance metrics
        self.performance_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_trades": 0,
            "days_active": 1
        }
    
    async def initialize_database(self):
        """Initialize database connection"""
        
        try:
            db_url = SecretsHelper.get_database_url()
            self.db_connection = psycopg2.connect(db_url)
            
            # Create paper_pnl table if not exists
            await self.create_pnl_table()
            
            logger.info("✅ Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def create_pnl_table(self):
        """Create paper trading P&L table"""
        
        create_sql = """
        CREATE TABLE IF NOT EXISTS paper_pnl (
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            total_value DOUBLE PRECISION NOT NULL,
            cash DOUBLE PRECISION NOT NULL,
            positions JSONB NOT NULL,
            position_values JSONB NOT NULL,
            unrealized_pnl DOUBLE PRECISION NOT NULL,
            realized_pnl DOUBLE PRECISION NOT NULL,
            daily_pnl DOUBLE PRECISION NOT NULL,
            total_return DOUBLE PRECISION NOT NULL,
            drawdown DOUBLE PRECISION NOT NULL,
            trades_today INTEGER NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create hypertable for time-series optimization
        SELECT create_hypertable('paper_pnl', 'timestamp', if_not_exists => TRUE);
        
        -- Create index for performance
        CREATE INDEX IF NOT EXISTS idx_paper_pnl_timestamp ON paper_pnl (timestamp DESC);
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(create_sql)
                self.db_connection.commit()
            logger.info("✅ paper_pnl table ready")
            
        except Exception as e:
            logger.error(f"❌ Error creating paper_pnl table: {e}")
    
    async def update_market_prices(self):
        """Update market prices from Redis streams"""
        
        try:
            # Get latest prices from polygon stream
            for symbol in ["NVDA", "MSFT"]:
                # Get most recent price from Redis
                latest_messages = self.redis_client.xrevrange('polygon:ticks', count=10)
                
                for message_id, fields in latest_messages:
                    if fields.get('sym') == symbol:
                        price = float(fields.get('p', 0))
                        if price > 0:
                            self.market_prices[symbol] = price
                            break
            
            self.last_price_update = datetime.now()
            
            logger.debug(f"💰 Prices: NVDA=${self.market_prices['NVDA']:.2f}, MSFT=${self.market_prices['MSFT']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating market prices: {e}")
    
    def calculate_position_values(self) -> Dict[str, float]:
        """Calculate current market value of positions"""
        
        position_values = {}
        
        for symbol, quantity in self.current_portfolio.positions.items():
            price = self.market_prices.get(symbol, 0.0)
            position_values[symbol] = quantity * price
        
        return position_values
    
    def calculate_portfolio_metrics(self):
        """Calculate current portfolio metrics"""
        
        # Update position values
        self.current_portfolio.position_values = self.calculate_position_values()
        
        # Calculate total portfolio value
        total_position_value = sum(self.current_portfolio.position_values.values())
        self.current_portfolio.total_value = self.current_portfolio.cash + total_position_value
        
        # Calculate returns
        self.current_portfolio.total_return = (self.current_portfolio.total_value - self.initial_balance) / self.initial_balance
        
        # Calculate drawdown
        if self.portfolio_history:
            peak_value = max(snapshot.total_value for snapshot in self.portfolio_history[-100:])  # Last 100 snapshots
            peak_value = max(peak_value, self.initial_balance)
        else:
            peak_value = self.initial_balance
        
        self.current_portfolio.drawdown = (peak_value - self.current_portfolio.total_value) / peak_value
        
        # Calculate daily P&L
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_snapshots = [s for s in self.portfolio_history if s.timestamp >= today_start]
        
        if daily_snapshots:
            start_of_day_value = daily_snapshots[0].total_value
            self.current_portfolio.daily_pnl = self.current_portfolio.total_value - start_of_day_value
        else:
            self.current_portfolio.daily_pnl = self.current_portfolio.total_value - self.initial_balance
        
        # Update timestamp
        self.current_portfolio.timestamp = datetime.now()
    
    async def process_execution_update(self, execution_data: Dict[str, str]):
        """Process execution update from IB executor"""
        
        try:
            # Parse execution data
            order_id = execution_data.get('order_id', '')
            status = execution_data.get('status', '')
            positions_json = execution_data.get('positions', '{}')
            
            if status == 'filled':
                # Update positions from execution result
                new_positions = json.loads(positions_json)
                
                # Update portfolio positions (excluding cash for now)
                for symbol in ["NVDA", "MSFT"]:
                    if symbol in new_positions:
                        old_quantity = self.current_portfolio.positions[symbol]
                        new_quantity = float(new_positions[symbol])
                        quantity_change = new_quantity - old_quantity
                        
                        if abs(quantity_change) > 0.01:  # Significant change
                            # Update position
                            self.current_portfolio.positions[symbol] = new_quantity
                            
                            # Estimate cash impact (simplified)
                            price = self.market_prices.get(symbol, 100.0)
                            cash_impact = -quantity_change * price  # Negative for purchases
                            self.current_portfolio.cash += cash_impact
                            
                            # Count trade
                            self.current_portfolio.trades_today += 1
                            
                            logger.info(f"💰 Position update: {symbol} {old_quantity} → {new_quantity}")
            
            # Recalculate metrics
            self.calculate_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"❌ Error processing execution update: {e}")
    
    async def save_portfolio_snapshot(self):
        """Save current portfolio snapshot to database"""
        
        try:
            if not self.db_connection:
                return
            
            insert_sql = """
            INSERT INTO paper_pnl (
                timestamp, total_value, cash, positions, position_values,
                unrealized_pnl, realized_pnl, daily_pnl, total_return,
                drawdown, trades_today
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(insert_sql, (
                    self.current_portfolio.timestamp,
                    self.current_portfolio.total_value,
                    self.current_portfolio.cash,
                    json.dumps(self.current_portfolio.positions),
                    json.dumps(self.current_portfolio.position_values),
                    self.current_portfolio.unrealized_pnl,
                    self.current_portfolio.realized_pnl,
                    self.current_portfolio.daily_pnl,
                    self.current_portfolio.total_return,
                    self.current_portfolio.drawdown,
                    self.current_portfolio.trades_today
                ))
                self.db_connection.commit()
            
            logger.debug("💰 Portfolio snapshot saved to database")
            
        except Exception as e:
            logger.error(f"❌ Error saving portfolio snapshot: {e}")
    
    def push_monitoring_metrics(self):
        """Push P&L metrics to monitoring systems"""
        
        try:
            # Metrics for Pushgateway
            metrics_data = {
                "portfolio_total_value": self.current_portfolio.total_value,
                "portfolio_daily_pnl": self.current_portfolio.daily_pnl,
                "portfolio_total_return": self.current_portfolio.total_return,
                "portfolio_drawdown": self.current_portfolio.drawdown,
                "portfolio_cash": self.current_portfolio.cash,
                "portfolio_trades_today": self.current_portfolio.trades_today,
                "portfolio_nvda_position": self.current_portfolio.positions["NVDA"],
                "portfolio_msft_position": self.current_portfolio.positions["MSFT"],
                "portfolio_nvda_value": self.current_portfolio.position_values["NVDA"],
                "portfolio_msft_value": self.current_portfolio.position_values["MSFT"]
            }
            
            # Store in Redis for real-time access
            self.redis_client.hset("portfolio_metrics", mapping=metrics_data)
            
            # Push to Pushgateway (if running)
            try:
                pushgateway_url = "http://localhost:9091/metrics/job/pnl_tracker"
                
                metrics_text = "\n".join([
                    f"{key} {value}" for key, value in metrics_data.items()
                ])
                
                response = requests.post(pushgateway_url, data=metrics_text, timeout=5)
                if response.status_code == 200:
                    logger.debug("💰 Metrics pushed to Pushgateway")
                
            except Exception as e:
                logger.debug(f"Pushgateway not available: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error pushing monitoring metrics: {e}")
    
    async def generate_executive_dashboard(self):
        """Generate executive dashboard data"""
        
        try:
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "portfolio": {
                    "total_value": self.current_portfolio.total_value,
                    "daily_pnl": self.current_portfolio.daily_pnl,
                    "total_return": self.current_portfolio.total_return * 100,  # As percentage
                    "drawdown": self.current_portfolio.drawdown * 100,
                    "trades_today": self.current_portfolio.trades_today
                },
                "positions": {
                    "cash": self.current_portfolio.cash,
                    "NVDA": {
                        "quantity": self.current_portfolio.positions["NVDA"],
                        "value": self.current_portfolio.position_values["NVDA"],
                        "price": self.market_prices["NVDA"]
                    },
                    "MSFT": {
                        "quantity": self.current_portfolio.positions["MSFT"],
                        "value": self.current_portfolio.position_values["MSFT"],
                        "price": self.market_prices["MSFT"]
                    }
                },
                "performance": self.performance_metrics,
                "status": "active" if abs(self.current_portfolio.daily_pnl) > 1 else "stable"
            }
            
            # Save to file for executive access
            dashboard_file = f"reports/executive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            # Store in Redis for API access
            self.redis_client.set("executive_dashboard", json.dumps(dashboard_data))
            
            logger.debug("💰 Executive dashboard updated")
            
        except Exception as e:
            logger.error(f"❌ Error generating executive dashboard: {e}")
    
    async def track_portfolio(self):
        """Main loop to track portfolio performance"""
        
        logger.info("💰 P&L Tracker service started")
        
        # Initialize database
        db_success = await self.initialize_database()
        if not db_success:
            logger.warning("⚠️ Database not available - metrics will be memory-only")
        
        logger.info("✅ P&L Tracker ready")
        
        while True:
            try:
                # Update market prices
                await self.update_market_prices()
                
                # Check for execution updates
                messages = self.redis_client.xread({'trading:executions': '$'}, count=5, block=1000)
                
                # Process execution updates
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self.process_execution_update(fields)
                
                # Calculate current metrics
                self.calculate_portfolio_metrics()
                
                # Save snapshot every minute
                current_time = datetime.now()
                if len(self.portfolio_history) == 0 or (current_time - self.portfolio_history[-1].timestamp).seconds >= 60:
                    
                    # Add to history
                    self.portfolio_history.append(PortfolioSnapshot(**asdict(self.current_portfolio)))
                    
                    # Keep last 1440 snapshots (24 hours at 1-minute intervals)
                    if len(self.portfolio_history) > 1440:
                        self.portfolio_history = self.portfolio_history[-1440:]
                    
                    # Save to database
                    await self.save_portfolio_snapshot()
                    
                    # Push metrics
                    self.push_monitoring_metrics()
                    
                    # Update executive dashboard every 5 minutes
                    if current_time.minute % 5 == 0:
                        await self.generate_executive_dashboard()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in P&L tracking loop: {e}")
                await asyncio.sleep(30)
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        
        return {
            "portfolio": asdict(self.current_portfolio),
            "market_prices": self.market_prices,
            "performance": self.performance_metrics,
            "history_length": len(self.portfolio_history),
            "last_updated": datetime.now().isoformat()
        }

async def main():
    """Main function to run P&L tracker service"""
    
    tracker = PnLTrackerService()
    
    try:
        await tracker.track_portfolio()
    except KeyboardInterrupt:
        logger.info("💰 P&L Tracker service stopped")
    except Exception as e:
        logger.error(f"❌ P&L Tracker service error: {e}")

if __name__ == "__main__":
    asyncio.run(main())