#!/usr/bin/env python3
"""
Redis to TimescaleDB Loader
Batch processing from Redis streams to TimescaleDB
"""

import os
import sys
import json
import time
import redis
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisToTimescaleLoader:
    def __init__(self):
        # Redis connection
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        
        # TimescaleDB connection
        db_url = SecretsHelper.get_database_url()
        self.pg_conn = psycopg2.connect(db_url)
        self.pg_conn.autocommit = False
        
        # Processing state
        self.last_processed = {'agg_minute': '0-0'}
        
    def process_aggregates_batch(self):
        """Process batch of minute aggregates from Redis to TimescaleDB"""
        try:
            # Read from Redis stream
            streams = self.redis_client.xread(
                {'agg_minute': self.last_processed['agg_minute']}, 
                count=500, 
                block=5000  # 5 second timeout
            )
            
            if not streams:
                return 0
            
            # Process messages
            rows = []
            last_id = None
            
            for stream_name, messages in streams:
                for message_id, fields in messages:
                    last_id = message_id
                    
                    # Convert Redis message to database row
                    try:
                        row = (
                            datetime.fromtimestamp(int(fields['timestamp']) / 1000),  # timestamp
                            fields['symbol'],  # symbol
                            float(fields['open']),  # open
                            float(fields['high']),  # high
                            float(fields['low']),  # low
                            float(fields['close']),  # close
                            int(fields['volume']),  # volume
                            'polygon_websocket',  # source
                            datetime.now()  # created_at
                        )
                        rows.append(row)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"⚠️ Skipping invalid message {message_id}: {e}")
                        continue
            
            if rows:
                # Batch insert to TimescaleDB
                with self.pg_conn.cursor() as cursor:
                    psycopg2.extras.execute_values(
                        cursor,
                        """
                        INSERT INTO market_data 
                        (timestamp, symbol, open, high, low, close, volume, source, created_at)
                        VALUES %s
                        ON CONFLICT (timestamp, symbol) DO NOTHING
                        """,
                        rows,
                        template=None,
                        page_size=100
                    )
                
                self.pg_conn.commit()
                self.last_processed['agg_minute'] = last_id
                
                logger.info(f"✅ Loaded {len(rows)} aggregates to TimescaleDB")
                
                # Update metrics
                self.redis_client.hset('metrics', mapping={
                    'loader.last_batch_size': len(rows),
                    'loader.last_update': time.time(),
                    'loader.total_processed': self.redis_client.hget('metrics', 'loader.total_processed') or 0 + len(rows)
                })
            
            return len(rows)
            
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            self.pg_conn.rollback()
            return 0
    
    def run_continuous(self):
        """Run continuous processing loop"""
        logger.info("🔄 Starting continuous Redis → TimescaleDB loader")
        
        while True:
            try:
                processed = self.process_aggregates_batch()
                
                if processed == 0:
                    time.sleep(1)  # Brief pause if no data
                    
            except KeyboardInterrupt:
                logger.info("👋 Shutting down loader")
                break
            except Exception as e:
                logger.error(f"❌ Loader error: {e}")
                time.sleep(5)  # Pause on error
        
        self.pg_conn.close()

if __name__ == "__main__":
    loader = RedisToTimescaleLoader()
    loader.run_continuous()