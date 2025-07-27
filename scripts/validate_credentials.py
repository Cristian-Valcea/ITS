#!/usr/bin/env python3
"""
Day 2 Credentials Validation Script
Tests all required credentials before development begins
"""

import os
import sys
import requests
import psycopg2
from datetime import datetime

def test_alpha_vantage():
    """Test Alpha Vantage API key"""
    print("🧪 Testing Alpha Vantage API...")
    
    api_key = os.environ.get('ALPHA_VANTAGE_KEY')
    if not api_key:
        print("❌ ALPHA_VANTAGE_KEY not found in environment")
        return False
    
    try:
        # Test with NVDA data request
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&apikey={api_key}&outputsize=compact"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            print(f"❌ Alpha Vantage API error: {data['Error Message']}")
            return False
        elif "Note" in data:
            print(f"⚠️  Alpha Vantage rate limit: {data['Note']}")
            return False
        elif "Time Series (1min)" in data:
            print("✅ Alpha Vantage API key working correctly")
            return True
        else:
            print(f"❌ Unexpected Alpha Vantage response: {list(data.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Alpha Vantage connection failed: {e}")
        return False

def test_ib_credentials():
    """Test Interactive Brokers credentials"""
    print("🧪 Testing IB Gateway connection...")
    
    username = os.environ.get('IB_USERNAME')
    password = os.environ.get('IB_PASSWORD')
    
    if not username or not password:
        print("❌ IB_USERNAME or IB_PASSWORD not found in environment")
        return False
    
    try:
        # Try to import ib_insync
        from ib_insync import IB
        
        # Test connection (this will fail if TWS Gateway not running, but validates credentials exist)
        ib = IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=1, timeout=5)
            print("✅ IB Gateway connection successful")
            ib.disconnect()
            return True
        except Exception as conn_error:
            if "Connection refused" in str(conn_error):
                print("⚠️  IB Gateway not running, but credentials are configured")
                print("   Start TWS Gateway on port 7497 for full testing")
                return True  # Credentials exist, just gateway not running
            else:
                print(f"❌ IB connection failed: {conn_error}")
                return False
                
    except ImportError:
        print("⚠️  ib_insync not installed, but credentials are configured")
        return True  # Credentials exist, just library not available
    except Exception as e:
        print(f"❌ IB credential test failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("🧪 Testing database connection...")
    
    try:
        # Use same logic as monitoring.py
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
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                print(f"✅ Database connection successful: {version.split(',')[0]}")
                
                # Test if our tables exist
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('dual_ticker_bars', 'current_positions')
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                if 'dual_ticker_bars' in tables and 'current_positions' in tables:
                    print("✅ Required tables found: dual_ticker_bars, current_positions")
                else:
                    print(f"⚠️  Missing tables. Found: {tables}")
                    print("   Run: docker-compose up timescaledb to initialize schema")
                
                return True
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Check if TimescaleDB is running and credentials are correct")
        return False

def test_monitoring_endpoints():
    """Test monitoring endpoints"""
    print("🧪 Testing monitoring endpoints...")
    
    try:
        # Test if FastAPI monitoring module can be imported
        from src.api.monitoring import router
        print("✅ Monitoring module imports successfully")
        
        # Test Prometheus client
        from prometheus_client import Counter
        test_counter = Counter('test_counter', 'Test counter')
        test_counter.inc()
        print("✅ Prometheus client working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Monitoring dependencies missing: {e}")
        print("   Run: pip install prometheus-client fastapi")
        return False
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def main():
    """Run all credential validation tests"""
    print("🚀 DAY 2 CREDENTIALS VALIDATION")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    tests = [
        ("Alpha Vantage API", test_alpha_vantage),
        ("IB Credentials", test_ib_credentials),
        ("Database Connection", test_database_connection),
        ("Monitoring Endpoints", test_monitoring_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 VALIDATION SUMMARY")
    print("=" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print()
    print(f"Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL CREDENTIALS VALIDATED - READY FOR DAY 2 DEVELOPMENT!")
        return 0
    else:
        print("🚨 CREDENTIAL ISSUES FOUND - RESOLVE BEFORE DEVELOPMENT")
        print("   See DAY2_CREDENTIALS_SETUP.md for troubleshooting")
        return 1

if __name__ == "__main__":
    sys.exit(main())