#!/usr/bin/env python3
"""
Prometheus Data Generator for AI Trading Dashboard
Generates realistic dummy metrics for Grafana dashboard testing

Creates time-series data for:
- Portfolio performance metrics
- Trading activity indicators  
- AI model performance metrics
"""

import time
import random
import math
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import threading
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingMetricsGenerator:
    """
    Generates realistic trading metrics for dashboard demonstration
    Simulates a live AI trading system with realistic market behavior
    """
    
    def __init__(self, prometheus_gateway_url: str = "http://localhost:9091"):
        self.prometheus_gateway_url = prometheus_gateway_url
        self.job_name = "ai_trading_system"
        
        # Initialize trading state
        self.portfolio_value = 100000.0  # Start with $100K
        self.cash_balance = 50000.0      # $50K cash
        self.nvda_position = 50          # 50 shares NVDA
        self.msft_position = 120         # 120 shares MSFT
        self.total_trades = 0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = self.portfolio_value
        
        # AI model state
        self.model_confidence = 0.75
        self.prediction_accuracy = 0.68
        self.episode_reward = 2.4
        
        # Market simulation parameters
        self.nvda_price = 485.0
        self.msft_price = 415.0
        self.market_volatility = 0.02
        
        logger.info("🤖 Trading Metrics Generator initialized")
        logger.info(f"   Prometheus Gateway: {prometheus_gateway_url}")
        logger.info(f"   Initial Portfolio: ${self.portfolio_value:,.2f}")
    
    def simulate_market_tick(self) -> None:
        """Simulate one market tick with realistic price movements"""
        
        # Market hours volatility (higher during open/close)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Market hours
            vol_multiplier = 1.5 if current_hour in [9, 15, 16] else 1.0
        else:
            vol_multiplier = 0.3  # After hours
        
        # Price movements with mean reversion
        nvda_return = np.random.normal(0, self.market_volatility * vol_multiplier)
        msft_return = np.random.normal(0, self.market_volatility * vol_multiplier * 0.8)  # MSFT less volatile
        
        # Apply correlation between stocks
        correlation = 0.6
        msft_return = correlation * nvda_return + (1 - correlation) * msft_return
        
        # Update prices
        self.nvda_price *= (1 + nvda_return)
        self.msft_price *= (1 + msft_return)
        
        # Ensure prices stay reasonable
        self.nvda_price = max(400, min(600, self.nvda_price))
        self.msft_price = max(350, min(500, self.msft_price))
        
        # Update portfolio value
        new_portfolio_value = (
            self.cash_balance + 
            self.nvda_position * self.nvda_price + 
            self.msft_position * self.msft_price
        )
        
        # Calculate daily P&L
        self.daily_pnl = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Update drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def simulate_ai_model_update(self) -> None:
        """Simulate AI model performance metrics"""
        
        # Model confidence varies with market conditions
        market_stress = abs(self.daily_pnl) / 1000  # Normalize by $1000
        stress_factor = max(0.5, 1.0 - market_stress * 0.1)
        
        # Update confidence with some persistence
        confidence_change = np.random.normal(0, 0.05) * stress_factor
        self.model_confidence = np.clip(
            self.model_confidence + confidence_change, 
            0.4, 0.95
        )
        
        # Prediction accuracy correlates with confidence
        accuracy_base = 0.5 + 0.3 * self.model_confidence
        accuracy_noise = np.random.normal(0, 0.08)
        self.prediction_accuracy = np.clip(
            accuracy_base + accuracy_noise,
            0.45, 0.85
        )
        
        # Episode reward correlates with recent performance
        recent_return = self.daily_pnl / 1000  # Normalize
        reward_change = np.random.normal(recent_return * 0.1, 0.2)
        self.episode_reward = np.clip(
            self.episode_reward + reward_change,
            -1.0, 6.0
        )
    
    def simulate_trading_activity(self) -> None:
        """Simulate trading decisions and position changes"""
        
        # Trading probability based on model confidence and market volatility
        trade_probability = self.model_confidence * 0.1  # Max 10% chance per tick
        
        if random.random() < trade_probability:
            # Execute a trade
            trade_type = random.choice(['buy_nvda', 'sell_nvda', 'buy_msft', 'sell_msft'])
            
            if trade_type == 'buy_nvda' and self.cash_balance > self.nvda_price * 10:
                shares = random.randint(1, 10)
                cost = shares * self.nvda_price
                self.nvda_position += shares
                self.cash_balance -= cost
                self.total_trades += 1
                logger.info(f"🔵 BUY {shares} NVDA @ ${self.nvda_price:.2f}")
                
            elif trade_type == 'sell_nvda' and self.nvda_position > 10:
                shares = random.randint(1, min(10, self.nvda_position // 2))
                proceeds = shares * self.nvda_price
                self.nvda_position -= shares
                self.cash_balance += proceeds
                self.total_trades += 1
                logger.info(f"🔴 SELL {shares} NVDA @ ${self.nvda_price:.2f}")
                
            elif trade_type == 'buy_msft' and self.cash_balance > self.msft_price * 10:
                shares = random.randint(1, 15)
                cost = shares * self.msft_price
                self.msft_position += shares
                self.cash_balance -= cost
                self.total_trades += 1
                logger.info(f"🔵 BUY {shares} MSFT @ ${self.msft_price:.2f}")
                
            elif trade_type == 'sell_msft' and self.msft_position > 15:
                shares = random.randint(1, min(15, self.msft_position // 2))
                proceeds = shares * self.msft_price
                self.msft_position -= shares
                self.cash_balance += proceeds
                self.total_trades += 1
                logger.info(f"🔴 SELL {shares} MSFT @ ${self.msft_price:.2f}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics for Prometheus"""
        
        return {
            # Portfolio metrics
            'trading_portfolio_value_usd': self.portfolio_value,
            'trading_cash_balance_usd': self.cash_balance,
            'trading_daily_pnl_usd': self.daily_pnl,
            'trading_max_drawdown_pct': self.max_drawdown,
            
            # Position metrics
            'trading_nvda_position_shares': self.nvda_position,
            'trading_msft_position_shares': self.msft_position,
            'trading_nvda_price_usd': self.nvda_price,
            'trading_msft_price_usd': self.msft_price,
            
            # Trading activity
            'trading_total_trades_count': self.total_trades,
            
            # AI model metrics
            'ai_model_confidence_score': self.model_confidence,
            'ai_prediction_accuracy_1h': self.prediction_accuracy,
            'ai_episode_reward_mean': self.episode_reward,
        }
    
    def push_metrics_to_prometheus(self) -> bool:
        """Push current metrics to Prometheus Pushgateway"""
        
        try:
            metrics = self.get_current_metrics()
            
            # Format metrics for Prometheus
            prometheus_data = []
            for metric_name, value in metrics.items():
                prometheus_data.append(f"{metric_name} {value}")
            
            # Add timestamp
            timestamp = int(time.time() * 1000)  # Milliseconds
            prometheus_payload = "\n".join(prometheus_data)
            
            # Push to Prometheus Pushgateway
            url = f"{self.prometheus_gateway_url}/metrics/job/{self.job_name}"
            
            headers = {'Content-Type': 'text/plain'}
            response = requests.post(url, data=prometheus_payload, headers=headers, timeout=5)
            
            if response.status_code == 200:
                logger.debug(f"✅ Metrics pushed to Prometheus")
                return True
            else:
                logger.warning(f"⚠️ Failed to push metrics: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pushing metrics to Prometheus: {e}")
            return False


class DashboardDataGenerator:
    """
    Main class to generate continuous data for Grafana dashboard
    Runs simulation loop and updates metrics in real-time
    """
    
    def __init__(self, 
                 prometheus_gateway_url: str = "http://localhost:9091",
                 update_interval_seconds: int = 30):
        
        self.generator = TradingMetricsGenerator(prometheus_gateway_url)
        self.update_interval = update_interval_seconds
        self.running = False
        
        logger.info("📊 Dashboard Data Generator initialized")
        logger.info(f"   Update interval: {update_interval_seconds} seconds")
    
    def start_continuous_generation(self) -> None:
        """Start continuous metric generation"""
        
        logger.info("🚀 Starting continuous metric generation...")
        self.running = True
        
        while self.running:
            try:
                # Simulate market tick
                self.generator.simulate_market_tick()
                
                # Update AI model metrics
                self.generator.simulate_ai_model_update()
                
                # Simulate trading activity
                self.generator.simulate_trading_activity()
                
                # Push metrics to Prometheus
                success = self.generator.push_metrics_to_prometheus()
                
                if success:
                    metrics = self.generator.get_current_metrics()
                    logger.info(f"📊 Portfolio: ${metrics['trading_portfolio_value_usd']:,.2f}, "
                              f"P&L: ${metrics['trading_daily_pnl_usd']:+.2f}, "
                              f"Confidence: {metrics['ai_model_confidence_score']:.2%}")
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping metric generation...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in generation loop: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def generate_historical_data(self, hours: int = 24) -> None:
        """Generate historical data for dashboard backfill"""
        
        logger.info(f"📈 Generating {hours} hours of historical data...")
        
        # Calculate number of data points (every 30 seconds)
        total_points = hours * 60 * 2  # 2 points per minute
        
        for i in range(total_points):
            # Simulate passage of time
            self.generator.simulate_market_tick()
            self.generator.simulate_ai_model_update()
            
            # Occasional trades
            if random.random() < 0.05:  # 5% chance per tick
                self.generator.simulate_trading_activity()
            
            # Push every 10th point to avoid overwhelming
            if i % 10 == 0:
                self.generator.push_metrics_to_prometheus()
                
                if i % 100 == 0:
                    progress = (i / total_points) * 100
                    logger.info(f"   Progress: {progress:.1f}% ({i}/{total_points})")
        
        logger.info("✅ Historical data generation complete")
    
    def save_sample_data(self, filename: str = "monitoring/sample_metrics.json") -> None:
        """Save sample metrics data for testing"""
        
        # Generate sample data points
        sample_data = []
        
        for _ in range(100):  # 100 sample points
            self.generator.simulate_market_tick()
            self.generator.simulate_ai_model_update()
            
            metrics = self.generator.get_current_metrics()
            sample_data.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
        
        # Save to file
        output_path = Path(filename)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"📄 Sample data saved to: {output_path}")


def setup_prometheus_config():
    """Generate Prometheus configuration for dashboard metrics"""
    
    config = {
        'global': {
            'scrape_interval': '30s',
            'evaluation_interval': '30s'
        },
        'scrape_configs': [
            {
                'job_name': 'ai_trading_system',
                'static_configs': [
                    {
                        'targets': ['localhost:9091']
                    }
                ],
                'scrape_interval': '30s',
                'metrics_path': '/metrics'
            }
        ]
    }
    
    config_path = Path("monitoring/prometheus.yml")
    config_path.parent.mkdir(exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"📝 Prometheus config saved to: {config_path}")
    return str(config_path)


def main():
    """Main function to run dashboard data generation"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate data for AI Trading Dashboard")
    parser.add_argument("--mode", choices=['continuous', 'historical', 'sample'], 
                       default='sample', help="Generation mode")
    parser.add_argument("--hours", type=int, default=4, 
                       help="Hours of historical data to generate")
    parser.add_argument("--interval", type=int, default=30,
                       help="Update interval in seconds")
    parser.add_argument("--prometheus-url", default="http://localhost:9091",
                       help="Prometheus Pushgateway URL")
    
    args = parser.parse_args()
    
    print("📊 AI TRADING DASHBOARD DATA GENERATOR")
    print("=" * 50)
    
    # Setup Prometheus configuration
    prometheus_config = setup_prometheus_config()
    print(f"📝 Prometheus config: {prometheus_config}")
    
    # Initialize generator
    generator = DashboardDataGenerator(
        prometheus_gateway_url=args.prometheus_url,
        update_interval_seconds=args.interval
    )
    
    if args.mode == 'continuous':
        print("🔄 Starting continuous data generation...")
        print("   Press Ctrl+C to stop")
        generator.start_continuous_generation()
        
    elif args.mode == 'historical':
        print(f"📈 Generating {args.hours} hours of historical data...")
        generator.generate_historical_data(hours=args.hours)
        
    elif args.mode == 'sample':
        print("📄 Generating sample data file...")
        generator.save_sample_data()
        
        # Also generate a small batch for immediate testing
        print("📊 Generating test data batch...")
        for i in range(10):
            generator.generator.simulate_market_tick()
            generator.generator.simulate_ai_model_update()
            success = generator.generator.push_metrics_to_prometheus()
            
            if success:
                print(f"   ✅ Batch {i+1}/10 pushed to Prometheus")
            else:
                print(f"   ⚠️ Batch {i+1}/10 failed (Prometheus may not be running)")
            
            time.sleep(2)
    
    print("\n🎯 Dashboard ready!")
    print("📊 Grafana Dashboard: http://localhost:3000")
    print("📈 Prometheus: http://localhost:9090")
    print("🔧 Pushgateway: http://localhost:9091")


if __name__ == "__main__":
    main()