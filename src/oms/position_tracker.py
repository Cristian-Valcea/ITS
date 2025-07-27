"""
Order Management System - Position Tracker
Minimal ORM model for current positions tracking
"""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor
import os

@dataclass
class Position:
    """Current position model"""
    id: int
    symbol: str
    qty: Decimal
    avg_price: Optional[Decimal]
    market_value: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal('0')
    last_updated: datetime = None

class PositionTracker:
    """Minimal position tracking for paper trading"""
    
    def __init__(self, db_host: str = "localhost", db_port: int = 5432):
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'database': 'intradayjules',
            'user': 'postgres',
            'password': os.environ.get('DB_PASSWORD', 'testpass')
        }
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM current_positions WHERE symbol = %s",
                    (symbol,)
                )
                row = cur.fetchone()
                if row:
                    return Position(**dict(row))
                return None
    
    def get_all_positions(self) -> List[Position]:
        """Get all current positions"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM current_positions ORDER BY symbol")
                rows = cur.fetchall()
                return [Position(**dict(row)) for row in rows]
    
    def update_position(self, symbol: str, qty_change: Decimal, price: Decimal):
        """Update position after trade execution"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get current position
                cur.execute(
                    "SELECT qty, avg_price FROM current_positions WHERE symbol = %s",
                    (symbol,)
                )
                current = cur.fetchone()
                
                if current:
                    current_qty, current_avg = current
                    new_qty = current_qty + qty_change
                    
                    # Calculate new average price
                    if new_qty != 0:
                        if current_avg is None:
                            new_avg = price
                        else:
                            total_cost = (current_qty * current_avg) + (qty_change * price)
                            new_avg = total_cost / new_qty
                    else:
                        new_avg = None
                    
                    # Update position
                    cur.execute("""
                        UPDATE current_positions 
                        SET qty = %s, avg_price = %s, last_updated = NOW()
                        WHERE symbol = %s
                    """, (new_qty, new_avg, symbol))
                    
                    conn.commit()
                    return True
                return False
    
    def calculate_unrealized_pnl(self, symbol: str, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L for position"""
        position = self.get_position(symbol)
        if position and position.qty != 0 and position.avg_price:
            return position.qty * (current_price - position.avg_price)
        return Decimal('0')
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary for CLI logging"""
        positions = self.get_all_positions()
        summary = {
            'positions': [],
            'total_value': Decimal('0'),
            'total_pnl': Decimal('0')
        }
        
        for pos in positions:
            if pos.qty != 0:
                pos_data = {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_price': float(pos.avg_price) if pos.avg_price else None,
                    'market_value': float(pos.market_value) if pos.market_value else None,
                    'unrealized_pnl': float(pos.unrealized_pnl)
                }
                summary['positions'].append(pos_data)
                
                if pos.market_value:
                    summary['total_value'] += pos.market_value
                summary['total_pnl'] += pos.unrealized_pnl
        
        return summary

# CLI utility for logging what the bot "owns"
def log_portfolio_status():
    """CLI utility to show current positions"""
    tracker = PositionTracker()
    summary = tracker.get_portfolio_summary()
    
    print("🏦 CURRENT PORTFOLIO STATUS")
    print("=" * 40)
    
    if not summary['positions']:
        print("📊 No active positions")
    else:
        for pos in summary['positions']:
            print(f"📈 {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_price']:.2f}")
            if pos['unrealized_pnl'] != 0:
                pnl_sign = "📈" if pos['unrealized_pnl'] > 0 else "📉"
                print(f"   {pnl_sign} P&L: ${pos['unrealized_pnl']:.2f}")
    
    print(f"💰 Total Portfolio Value: ${float(summary['total_value']):.2f}")
    print(f"📊 Total Unrealized P&L: ${float(summary['total_pnl']):.2f}")

if __name__ == "__main__":
    log_portfolio_status()