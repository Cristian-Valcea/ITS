"""
Monitoring API Routes
FastAPI + Prometheus metrics for system health monitoring
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime
from typing import Dict, Any
import psycopg2
import os

# Prometheus metrics
data_ingestion_counter = Counter('data_bars_ingested_total', 'Total bars ingested', ['symbol'])
data_quality_failures = Counter('data_quality_failures_total', 'Data quality failures', ['reason'])
database_query_duration = Histogram('database_query_duration_seconds', 'Database query duration')
active_positions_gauge = Gauge('active_positions_total', 'Number of active positions')
portfolio_value_gauge = Gauge('portfolio_value_usd', 'Total portfolio value in USD')

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint - JSON format for browser viewing
    Returns system component status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check database connection
    try:
        # Use TimescaleDB service hostname for container network
        db_host = os.environ.get('DB_HOST', 'timescaledb' if os.environ.get('DOCKER_ENV') else 'localhost')
        db_config = {
            'host': db_host,
            'port': int(os.environ.get('DB_PORT', '5432')),
            'database': 'intradayjules',
            'user': 'postgres',
            'password': os.environ.get('DB_PASSWORD', 'testpass')
        }
        
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Connection successful"
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Connection failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check data freshness (if data exists)
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol, COUNT(*) as bar_count, MAX(timestamp) as latest_bar
                    FROM dual_ticker_bars 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                    GROUP BY symbol
                """)
                recent_data = cur.fetchall()
                
                if recent_data:
                    health_status["components"]["data_freshness"] = {
                        "status": "healthy",
                        "recent_bars": {row[0]: {"count": row[1], "latest": row[2].isoformat()} for row in recent_data}
                    }
                else:
                    health_status["components"]["data_freshness"] = {
                        "status": "no_data",
                        "message": "No recent data found (expected during development)"
                    }
    except Exception as e:
        health_status["components"]["data_freshness"] = {
            "status": "unknown",
            "message": f"Could not check data freshness: {str(e)}"
        }
    
    # Check positions
    try:
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM current_positions WHERE qty != 0")
                active_positions = cur.fetchone()[0]
                
                health_status["components"]["positions"] = {
                    "status": "healthy",
                    "active_positions": active_positions
                }
                
                # Update Prometheus gauge
                active_positions_gauge.set(active_positions)
                
    except Exception as e:
        health_status["components"]["positions"] = {
            "status": "unknown",
            "message": f"Could not check positions: {str(e)}"
        }
    
    return health_status

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format for scraping
    """
    try:
        # Update some real-time metrics before serving
        db_config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': int(os.environ.get('DB_PORT', '5432')),
            'database': 'intradayjules',
            'user': 'postgres',
            'password': os.environ.get('DB_PASSWORD', 'testpass')
        }
        
        with database_query_duration.time():
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    # Update data ingestion counters
                    cur.execute("""
                        SELECT symbol, COUNT(*) 
                        FROM dual_ticker_bars 
                        WHERE created_at > NOW() - INTERVAL '1 hour'
                        GROUP BY symbol
                    """)
                    for symbol, count in cur.fetchall():
                        data_ingestion_counter.labels(symbol=symbol)._value._value = count
                    
                    # Update portfolio value
                    cur.execute("""
                        SELECT COALESCE(SUM(market_value), 0) 
                        FROM current_positions 
                        WHERE market_value IS NOT NULL
                    """)
                    portfolio_value = cur.fetchone()[0]
                    portfolio_value_gauge.set(float(portfolio_value))
        
        return generate_latest()
        
    except Exception as e:
        # Return basic metrics even if database is unavailable
        data_quality_failures.labels(reason="metrics_collection_error").inc()
        return generate_latest()

@router.get("/status")
async def system_status():
    """
    Simple status endpoint for quick CLI checks
    curl http://localhost:8000/monitoring/status
    """
    return {
        "system": "IntradayJules Dual-Ticker Trading System",
        "status": "operational",
        "symbols": ["NVDA", "MSFT"],
        "timestamp": datetime.utcnow().isoformat()
    }