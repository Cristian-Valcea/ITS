#!/usr/bin/env python3
"""
Live Monitoring Endpoints Connected to Actual Running Services
Implements real health checks, metrics collection, and service status monitoring
"""

import logging
import psycopg2
import os
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json

# Import our service clients
try:
    from ..data.alpha_vantage_client import AlphaVantageClient
    from ..data.quality_validator import DataQualityValidator
    from ..brokers.ib_gateway import IBGatewayClient
    CLIENTS_AVAILABLE = True
except ImportError:
    CLIENTS_AVAILABLE = False
    logging.warning("Service clients not available - monitoring will show limited data")

logger = logging.getLogger(__name__)

# Prometheus metrics
health_check_counter = Counter('health_check_total', 'Total health checks performed', ['service', 'status'])
service_response_time = Histogram('service_response_seconds', 'Service response time', ['service'])
data_ingestion_counter = Counter('data_ingestion_total', 'Total data points ingested', ['symbol', 'source'])
active_connections = Gauge('active_connections', 'Active service connections', ['service'])
last_data_timestamp = Gauge('last_data_timestamp', 'Timestamp of last data point', ['symbol'])

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

class LiveMonitoringService:
    """Live monitoring service that connects to actual running services"""
    
    def __init__(self):
        self.services = {}
        self.last_health_check = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize connections to monitored services"""
        if not CLIENTS_AVAILABLE:
            logger.warning("Service clients not available - using mock services")
            return
        
        try:
            # Initialize Alpha Vantage client
            if os.getenv('ALPHA_VANTAGE_KEY'):
                self.services['alpha_vantage'] = AlphaVantageClient()
                logger.info("✅ Alpha Vantage client initialized")
            else:
                logger.warning("❌ ALPHA_VANTAGE_KEY not found")
            
            # Initialize IB Gateway client
            self.services['ib_gateway'] = IBGatewayClient()
            logger.info("✅ IB Gateway client initialized")
            
            # Initialize Data Quality Validator
            self.services['quality_validator'] = DataQualityValidator()
            logger.info("✅ Data Quality Validator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
    
    def check_database_health(self) -> Dict:
        """Check TimescaleDB connection and basic functionality"""
        start_time = datetime.now()
        
        try:
            # Database connection parameters
            db_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'trading'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
            
            # Test connection
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Check if TimescaleDB extension is available
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb';")
            timescale_available = cursor.fetchone() is not None
            
            # Check dual_ticker_bars table
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'dual_ticker_bars';
            """)
            table_exists = cursor.fetchone()[0] > 0
            
            # Get recent data count
            recent_data_count = 0
            if table_exists:
                cursor.execute("""
                    SELECT COUNT(*) FROM dual_ticker_bars 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour';
                """)
                recent_data_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'healthy',
                'version': version,
                'timescale_available': timescale_available,
                'table_exists': table_exists,
                'recent_data_count': recent_data_count,
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='database', status='success').inc()
            service_response_time.labels(service='database').observe(response_time)
            active_connections.labels(service='database').set(1)
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='database', status='error').inc()
            service_response_time.labels(service='database').observe(response_time)
            active_connections.labels(service='database').set(0)
            
            return result
    
    def check_alpha_vantage_health(self) -> Dict:
        """Check Alpha Vantage API connectivity and data freshness"""
        start_time = datetime.now()
        
        if 'alpha_vantage' not in self.services:
            return {
                'status': 'unavailable',
                'error': 'Alpha Vantage client not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            client = self.services['alpha_vantage']
            
            # Test with a simple quote request for NVDA
            quote = client.get_quote('NVDA')
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'healthy',
                'last_price': quote['price'],
                'last_update': quote['timestamp'],
                'response_time_seconds': response_time,
                'rate_limit_delay': client.rate_limit_delay,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='alpha_vantage', status='success').inc()
            service_response_time.labels(service='alpha_vantage').observe(response_time)
            active_connections.labels(service='alpha_vantage').set(1)
            data_ingestion_counter.labels(symbol='NVDA', source='alpha_vantage').inc()
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='alpha_vantage', status='error').inc()
            service_response_time.labels(service='alpha_vantage').observe(response_time)
            active_connections.labels(service='alpha_vantage').set(0)
            
            return result
    
    def check_ib_gateway_health(self) -> Dict:
        """Check IB Gateway connection and account status"""
        start_time = datetime.now()
        
        if 'ib_gateway' not in self.services:
            return {
                'status': 'unavailable',
                'error': 'IB Gateway client not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            client = self.services['ib_gateway']
            
            # Connect if not already connected
            if not client.connected:
                client.connect()
            
            # Get health check from client
            health = client.health_check()
            
            response_time = (datetime.now() - start_time).total_seconds()
            health['response_time_seconds'] = response_time
            
            status = 'success' if health['status'] == 'healthy' else 'error'
            health_check_counter.labels(service='ib_gateway', status=status).inc()
            service_response_time.labels(service='ib_gateway').observe(response_time)
            active_connections.labels(service='ib_gateway').set(1 if health['connected'] else 0)
            
            return health
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='ib_gateway', status='error').inc()
            service_response_time.labels(service='ib_gateway').observe(response_time)
            active_connections.labels(service='ib_gateway').set(0)
            
            return result
    
    def check_data_quality_health(self) -> Dict:
        """Check data quality validation service"""
        start_time = datetime.now()
        
        if 'quality_validator' not in self.services:
            return {
                'status': 'unavailable',
                'error': 'Quality validator not initialized',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            validator = self.services['quality_validator']
            
            # Get recent data from database for validation
            db_health = self.check_database_health()
            
            if db_health['status'] != 'healthy':
                return {
                    'status': 'unhealthy',
                    'error': 'Database not available for quality validation',
                    'timestamp': datetime.now().isoformat()
                }
            
            # For now, just check that validator is configured properly
            config_status = 'healthy' if validator.config else 'unhealthy'
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': config_status,
                'config_loaded': bool(validator.config),
                'missing_data_threshold_ci': validator.config.get('missing_data', {}).get('ci_threshold', 0.05),
                'missing_data_threshold_prod': validator.config.get('missing_data', {}).get('prod_threshold', 0.01),
                'ohlc_validation_enabled': validator.config.get('ohlc_validation', {}).get('enabled', True),
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='quality_validator', status='success').inc()
            service_response_time.labels(service='quality_validator').observe(response_time)
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_seconds': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            health_check_counter.labels(service='quality_validator', status='error').inc()
            service_response_time.labels(service='quality_validator').observe(response_time)
            
            return result
    
    def get_overall_health(self) -> Dict:
        """Get overall system health status"""
        services_health = {
            'database': self.check_database_health(),
            'alpha_vantage': self.check_alpha_vantage_health(),
            'ib_gateway': self.check_ib_gateway_health(),
            'quality_validator': self.check_data_quality_health()
        }
        
        # Determine overall status
        healthy_services = sum(1 for health in services_health.values() if health['status'] == 'healthy')
        total_services = len(services_health)
        
        if healthy_services == total_services:
            overall_status = 'healthy'
        elif healthy_services >= total_services * 0.5:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'healthy_services': healthy_services,
            'total_services': total_services,
            'services': services_health,
            'timestamp': datetime.now().isoformat()
        }

# Initialize monitoring service
monitoring_service = LiveMonitoringService()

@router.get("/health")
async def health_check():
    """Overall system health check"""
    try:
        health = monitoring_service.get_overall_health()
        
        if health['status'] == 'unhealthy':
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/health/{service}")
async def service_health_check(service: str):
    """Individual service health check"""
    service_checks = {
        'database': monitoring_service.check_database_health,
        'alpha_vantage': monitoring_service.check_alpha_vantage_health,
        'ib_gateway': monitoring_service.check_ib_gateway_health,
        'quality_validator': monitoring_service.check_data_quality_health
    }
    
    if service not in service_checks:
        raise HTTPException(status_code=404, detail=f"Service '{service}' not found")
    
    try:
        health = service_checks[service]()
        
        if health['status'] == 'unhealthy':
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service {service} health check failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        # Update some real-time metrics
        monitoring_service.get_overall_health()  # This updates the metrics
        
        # Generate Prometheus format
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/status")
async def system_status():
    """Human-readable system status"""
    try:
        health = monitoring_service.get_overall_health()
        
        # Format for CLI/human consumption
        status_lines = [
            f"🎯 SYSTEM STATUS: {health['status'].upper()}",
            f"📊 Services: {health['healthy_services']}/{health['total_services']} healthy",
            f"⏰ Last Check: {health['timestamp']}",
            "",
            "📋 SERVICE DETAILS:"
        ]
        
        for service_name, service_health in health['services'].items():
            status_emoji = "✅" if service_health['status'] == 'healthy' else "❌" if service_health['status'] == 'unhealthy' else "⚠️"
            status_lines.append(f"{status_emoji} {service_name}: {service_health['status']}")
            
            if 'error' in service_health:
                status_lines.append(f"   Error: {service_health['error']}")
            
            if 'response_time_seconds' in service_health:
                status_lines.append(f"   Response: {service_health['response_time_seconds']:.3f}s")
        
        return {"status_text": "\n".join(status_lines), "health_data": health}
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@router.get("/data-ingestion")
async def data_ingestion_status():
    """Data ingestion pipeline status"""
    try:
        # Check recent data ingestion
        db_health = monitoring_service.check_database_health()
        av_health = monitoring_service.check_alpha_vantage_health()
        
        return {
            'database_status': db_health['status'],
            'recent_data_count': db_health.get('recent_data_count', 0),
            'alpha_vantage_status': av_health['status'],
            'last_api_call': av_health.get('last_update', 'unknown'),
            'ingestion_active': db_health['status'] == 'healthy' and av_health['status'] == 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data ingestion status check failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

def main():
    """CLI interface for testing live monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Monitoring Service')
    parser.add_argument('--check', choices=['health', 'database', 'alpha_vantage', 'ib_gateway', 'quality'], 
                       default='health', help='Check to perform')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        service = LiveMonitoringService()
        
        if args.check == 'health':
            result = service.get_overall_health()
        elif args.check == 'database':
            result = service.check_database_health()
        elif args.check == 'alpha_vantage':
            result = service.check_alpha_vantage_health()
        elif args.check == 'ib_gateway':
            result = service.check_ib_gateway_health()
        elif args.check == 'quality':
            result = service.check_data_quality_health()
        
        print(json.dumps(result, indent=2))
        
        if result['status'] == 'healthy':
            print("✅ Live monitoring test successful")
            return 0
        else:
            print(f"❌ Service status: {result['status']}")
            return 1
        
    except Exception as e:
        print(f"❌ Live monitoring test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())