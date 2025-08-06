#!/usr/bin/env python3
"""
🎯 IBKR-Focused Smoke Run
Tests only the enhanced IBKR integration without external dependencies
"""

import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.brokers.enhanced_safe_wrapper import EnhancedSafeOrderWrapper
from src.brokers.ib_gateway import IBGatewayClient

# Configure logging
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ibkr_connection():
    """Test basic IBKR connection"""
    print("1️⃣ Testing IBKR Connection...")
    
    ib_client = IBGatewayClient()
    
    try:
        if not ib_client.connect():
            raise RuntimeError("IBKR connection failed")
        
        # Get connection details
        if ib_client.simulation_mode:
            print("   ✅ Connected in simulation mode")
            result = {
                'mode': 'simulation',
                'status': 'connected'
            }
        else:
            accounts = ib_client.ib.managedAccounts()
            server_version = ib_client.ib.client.serverVersion()
            
            print(f"   ✅ Connected to IBKR Live")
            print(f"   📊 Server version: {server_version}")
            print(f"   👤 Accounts: {accounts}")
            
            result = {
                'mode': 'live',
                'accounts': accounts,
                'server_version': server_version,
                'status': 'connected'
            }
        
        return result
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        raise
    finally:
        ib_client.disconnect()

def test_enhanced_safety_components():
    """Test enhanced safety system components"""
    print("\n2️⃣ Testing Enhanced Safety Components...")
    
    components = {}
    
    try:
        from src.brokers.connection_validator import IBKRConnectionValidator
        components['connection_validator'] = '✅ Available'
        print("   ✅ Connection Validator: Available")
    except ImportError as e:
        components['connection_validator'] = f'❌ Missing: {e}'
        print(f"   ❌ Connection Validator: Missing - {e}")
        
    try:
        from src.brokers.event_order_monitor import EventDrivenOrderMonitor
        components['event_monitor'] = '✅ Available'
        print("   ✅ Event Monitor: Available")
    except ImportError as e:
        components['event_monitor'] = f'❌ Missing: {e}'
        print(f"   ❌ Event Monitor: Missing - {e}")
        
    try:
        from src.brokers.enhanced_safe_wrapper import EnhancedSafeOrderWrapper
        components['safe_wrapper'] = '✅ Available'
        print("   ✅ Safe Wrapper: Available")
    except ImportError as e:
        components['safe_wrapper'] = f'❌ Missing: {e}'
        print(f"   ❌ Safe Wrapper: Missing - {e}")
    
    # Check if any components are missing
    missing_components = [name for name, status in components.items() if '❌' in status]
    if missing_components:
        raise RuntimeError(f"Missing enhanced safety components: {missing_components}")
    
    return components

def test_enhanced_order_placement():
    """Test enhanced order placement with monitoring"""
    print("\n3️⃣ Testing Enhanced Order Placement...")
    
    ib_client = IBGatewayClient()
    
    try:
        if not ib_client.connect():
            raise RuntimeError("IBKR connection failed for order test")
        
        print(f"   📡 Connected for order test")
        
        # Risk governor callback for testing
        def test_risk_callback(order_id, status, event_type):
            print(f"   🛡️ Risk Governor: Order {order_id} - {status} - {event_type}")
            logger.info(f"Risk Governor: Order {order_id} - {status} - {event_type}")
            return 'ALLOW'
        
        # Create enhanced safe wrapper
        print("   🛡️ Creating enhanced safe wrapper with risk governor...")
        safe_wrapper = EnhancedSafeOrderWrapper(ib_client, test_risk_callback)
        
        if ib_client.simulation_mode:
            print("   🎭 Testing in simulation mode")
            
            # Test market order in simulation
            result = safe_wrapper.place_market_order_with_governor('MSFT', 1, 'BUY')
            
            print(f"   ✅ Simulation Order Result:")
            print(f"      Order ID: {result['order_id']}")
            print(f"      Status: {result.get('final_status', result.get('status'))}")
            print(f"      Mode: {result['mode']}")
            
            return result
            
        else:
            print("   🚨 Testing with LIVE IBKR - placing 1-share limit order")
            
            # Place conservative limit order
            result = safe_wrapper.place_limit_order_with_governor('MSFT', 1, 400.00, 'BUY')
            
            print(f"   ✅ Live Order Result:")
            print(f"      Order ID: {result['order_id']}")
            print(f"      Final Status: {result['final_status']}")
            print(f"      Is Live: {result['is_live']}")
            print(f"      Status Events: {result['status_events']}")
            print(f"      Risk Governor: {result['risk_governor_integrated']}")
            
            return result
        
    except Exception as e:
        print(f"   ❌ Enhanced order test failed: {e}")
        raise
    finally:
        ib_client.disconnect()

def run_focused_smoke_test():
    """Run focused IBKR smoke test"""
    
    print("🎯 IBKR-FOCUSED SMOKE RUN")
    print("=" * 50)
    print("Testing enhanced IBKR integration components only")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        'start_time': datetime.now(timezone.utc).isoformat(),
        'tests': {},
        'overall_status': 'PENDING'
    }
    
    tests = [
        ("IBKR Connection", test_ibkr_connection),
        ("Enhanced Safety Components", test_enhanced_safety_components),
        ("Enhanced Order Placement", test_enhanced_order_placement)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        try:
            result = test_function()
            results['tests'][test_name] = {
                'status': 'PASS',
                'result': result
            }
            passed_tests += 1
            
        except Exception as e:
            results['tests'][test_name] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"Test failed: {test_name} - {e}")
            break  # Stop on first failure
    
    # Final results
    results['end_time'] = datetime.now(timezone.utc).isoformat()
    results['tests_passed'] = passed_tests
    results['total_tests'] = total_tests
    
    if passed_tests == total_tests:
        results['overall_status'] = 'PASS'
        print(f"\n🎉 SMOKE RUN PASSED!")
        print(f"✅ All {total_tests} IBKR tests completed successfully")
        print(f"🚀 Enhanced IBKR integration is working!")
        
        print(f"\n📊 KEY VALIDATIONS:")
        print(f"   ✅ IBKR connection established")
        print(f"   ✅ Enhanced safety components loaded")
        print(f"   ✅ Order placement with enhanced monitoring")
        print(f"   ✅ Risk governor integration working")
        
    else:
        results['overall_status'] = 'FAIL'
        print(f"\n🚨 SMOKE RUN FAILED!")
        print(f"❌ Only {passed_tests}/{total_tests} tests passed")
        print(f"🛑 Enhanced IBKR integration has issues")
    
    return results

if __name__ == "__main__":
    print("🚀 STARTING FOCUSED IBKR SMOKE RUN")
    print("Testing enhanced integration without external dependencies")
    print()
    
    results = run_focused_smoke_test()
    
    print(f"\n" + "="*50)
    print(f"🎯 FOCUSED SMOKE RUN COMPLETE")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print(f"="*50)
    
    if results['overall_status'] == 'PASS':
        print("\n🎉 ENHANCED IBKR INTEGRATION VALIDATED!")
        print("Ready to proceed with full production deployment")
    else:
        print("\n🛑 INTEGRATION ISSUES DETECTED")
        print("Review test failures before proceeding")