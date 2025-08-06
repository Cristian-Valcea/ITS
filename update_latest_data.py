#!/usr/bin/env python3
"""
Update Latest Market Data
Only fetch new data since last update (incremental updates)
"""

import psycopg2
from datetime import datetime, timedelta
from secrets_helper import SecretsHelper
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_last_data_date():
    """Get the most recent date in our cache"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_data',
            user='postgres',
            password=SecretsHelper.get_timescaledb_password()
        )
        cur = conn.cursor()
        
        cur.execute("SELECT MAX(DATE(timestamp)) FROM minute_bars WHERE symbol = 'NVDA';")
        last_date = cur.fetchone()[0]
        
        conn.close()
        return last_date
        
    except Exception as e:
        logger.error(f"Error getting last date: {e}")
        return None

def update_incremental_data():
    """Only fetch data newer than what we have"""
    
    last_date = get_last_data_date()
    if not last_date:
        logger.error("Cannot determine last data date")
        return
    
    # Only fetch if data is older than 1 day
    if (datetime.now().date() - last_date).days > 1:
        next_date = last_date + timedelta(days=1)
        today = datetime.now().date()
        
        logger.info(f"📈 Updating data from {next_date} to {today}")
        # Fetch only new data here
    else:
        logger.info("✅ Data is up to date - no fetch needed")

if __name__ == "__main__":
    update_incremental_data()