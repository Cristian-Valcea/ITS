#!/usr/bin/env python3
"""
Polygon WebSocket Router
Real-time tick data from Polygon.io to Redis Streams
"""

import os
import json
import asyncio
import websockets
import redis
import time
import logging
from datetime import datetime
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonWebSocketRouter:
    def __init__(self):
        # Get API key from vault
        self.api_key = SecretsHelper.get_polygon_api_key()
        # Use delayed WebSocket for Starter plan (15-minute delay)
        self.ws_url = f"wss://delayed.polygon.io/stocks"
        
        # Redis connection
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # Metrics tracking
        self.message_count = 0
        self.last_metric_time = time.time()
        
    async def connect_and_stream(self):
        """Main WebSocket connection and streaming loop"""
        try:
            async with websockets.connect(
                self.ws_url, 
                ping_interval=30,
                ping_timeout=10
            ) as websocket:
                
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "params": self.api_key
                }
                await websocket.send(json.dumps(auth_msg))
                
                # Wait for auth response
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                logger.info(f"🔐 Auth response: {auth_data}")
                
                # Check for successful connection (Polygon returns "Connected Successfully")
                if not any(msg.get('message') == 'Connected Successfully' for msg in auth_data):
                    raise Exception(f"Authentication failed: {auth_data}")
                
                # Subscribe to NVDA and MSFT ticks and aggregates
                subscribe_msg = {
                    "action": "subscribe",
                    "params": "T.NVDA,T.MSFT,A.NVDA,A.MSFT"
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for subscription response
                sub_response = await websocket.recv()
                sub_data = json.loads(sub_response)
                logger.info(f"📊 Subscription response: {sub_data}")
                
                logger.info("✅ Connected to Polygon WebSocket, subscribed to NVDA/MSFT")
                
                # Message processing loop
                async for message in websocket:
                    await self.process_message(message)
                    
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
            raise
    
    async def process_message(self, raw_message):
        """Process incoming WebSocket message"""
        try:
            ts_start = time.time()
            
            # Parse message
            messages = json.loads(raw_message)
            if not isinstance(messages, list):
                messages = [messages]
            
            # Process each message in the batch
            for msg in messages:
                msg_type = msg.get('ev')
                
                if msg_type == 'T':  # Trade tick
                    await self.handle_trade_tick(msg)
                elif msg_type == 'A':  # Aggregate (minute bar)
                    await self.handle_aggregate(msg)
                elif msg_type == 'status':
                    logger.info(f"📊 Status: {msg}")
            
            # Update metrics
            latency_ms = (time.time() - ts_start) * 1000
            self.update_metrics(latency_ms, len(messages))
            
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
    
    async def handle_trade_tick(self, tick):
        """Handle individual trade tick"""
        try:
            # Validate required fields
            if not all(tick.get(field) is not None for field in ['sym', 'p', 's', 't']):
                logger.debug(f"⚠️ Incomplete tick data: {tick}")
                return
            
            # Add to Redis stream with TTL
            stream_data = {
                'symbol': str(tick.get('sym')),
                'price': str(tick.get('p')),
                'size': str(tick.get('s')),
                'timestamp': str(tick.get('t')),
                'exchange': str(tick.get('x', '')),
                'conditions': json.dumps(tick.get('c', [])),
                'type': 'trade'
            }
            
            # Add to ticks stream (high frequency, short retention)
            self.redis_client.xadd(
                'ticks', 
                stream_data, 
                maxlen=500000  # ~1 trading day
            )
            
            logger.debug(f"✅ Stored tick: {tick.get('sym')} @ {tick.get('p')}")
            
        except Exception as e:
            logger.error(f"❌ Trade tick error: {e}")
    
    async def handle_aggregate(self, agg):
        """Handle minute aggregate"""
        try:
            # Validate required fields
            required_fields = ['sym', 'o', 'h', 'l', 'c', 'v', 't']
            if not all(agg.get(field) is not None for field in required_fields):
                logger.debug(f"⚠️ Incomplete aggregate data: {agg}")
                return
            
            # Add to aggregates stream (lower frequency, longer retention)
            stream_data = {
                'symbol': str(agg.get('sym')),
                'open': str(agg.get('o')),
                'high': str(agg.get('h')),
                'low': str(agg.get('l')),
                'close': str(agg.get('c')),
                'volume': str(agg.get('v')),
                'timestamp': str(agg.get('t')),
                'type': 'aggregate'
            }
            
            # Add to aggregates stream
            self.redis_client.xadd(
                'agg_minute', 
                stream_data, 
                maxlen=100000  # ~2 months of minute bars
            )
            
            logger.debug(f"✅ Stored aggregate: {agg.get('sym')} @ {agg.get('c')}")
            
        except Exception as e:
            logger.error(f"❌ Aggregate error: {e}")
    
    def update_metrics(self, latency_ms, message_count):
        """Update performance metrics"""
        try:
            self.message_count += message_count
            current_time = time.time()
            
            # Update Redis metrics
            metrics = {
                'router.latency_ms': latency_ms,
                'router.messages_total': self.message_count,
                'router.last_update': current_time
            }
            
            self.redis_client.hset('metrics', mapping=metrics)
            
            # Log periodic stats
            if current_time - self.last_metric_time > 60:  # Every minute
                msg_per_sec = self.message_count / (current_time - self.last_metric_time)
                logger.info(f"📊 Router: {msg_per_sec:.1f} msg/s, {latency_ms:.1f}ms latency")
                self.last_metric_time = current_time
                self.message_count = 0
                
        except Exception as e:
            logger.error(f"❌ Metrics error: {e}")

async def main():
    """Main entry point"""
    router = PolygonWebSocketRouter()
    
    while True:
        try:
            await router.connect_and_stream()
        except Exception as e:
            logger.error(f"❌ Router crashed: {e}")
            logger.info("🔄 Restarting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())