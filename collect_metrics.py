#!/usr/bin/env python3
"""
System Metrics Collector
Collects metrics from Redis and stores in TimescaleDB
"""

import time
import redis
import psycopg2
import logging
from datetime import datetime
from secrets_helper import SecretsHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        db_url = SecretsHelper.get_database_url()
        self.pg_conn = psycopg2.connect(db_url)
        
    def collect_and_store(self):
        """Collect metrics from Redis and store in TimescaleDB"""
        try:
            # Get all metrics from Redis
            metrics = self.redis_client.hgetall('metrics')
            
            if not metrics:
                return
            
            # Prepare batch insert
            rows = []
            timestamp = datetime.now()
            
            for key, value in metrics.items():
                if key.startswith(('router.', 'loader.')):
                    service = key.split('.')[0]
                    metric_name = key.split('.', 1)[1]
                    
                    try:
                        metric_value = float(value)
                        rows.append((timestamp, service, metric_name, metric_value, '{}'))
                    except ValueError:
                        continue
            
            if rows:
                with self.pg_conn.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO sys_metrics 
                        (timestamp, service, metric_name, metric_value, tags)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        rows
                    )
                self.pg_conn.commit()
                logger.info(f"✅ Stored {len(rows)} metrics")
                
        except Exception as e:
            logger.error(f"❌ Metrics collection error: {e}")
    
    def run_continuous(self):
        """Run continuous metrics collection"""
        logger.info("📊 Starting continuous metrics collection")
        while True:
            try:
                self.collect_and_store()
                time.sleep(30)  # Collect every 30 seconds
            except KeyboardInterrupt:
                logger.info("👋 Shutting down metrics collector")
                break
            except Exception as e:
                logger.error(f"❌ Collector error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    collector = MetricsCollector()
    collector.run_continuous()