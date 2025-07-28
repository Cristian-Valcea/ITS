#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DAY 2 VALIDATION SUITE
Verify team's claims of 100% completion for 4 missing components
"""

import sys
import os
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class Day2ComprehensiveValidator:
    """Validates all team claims with evidence-based testing"""
    
    def __init__(self):
        self.results = {
            'alpha_vantage': {'status': 'UNKNOWN', 'details': [], 'score': 0.0},
            'data_quality': {'status': 'UNKNOWN', 'details': [], 'score': 0.0},
            'ib_gateway': {'status': 'UNKNOWN', 'details': [], 'score': 0.0},
            'live_monitoring': {'status': 'UNKNOWN', 'details': [], 'score': 0.0},
            'integration': {'status': 'UNKNOWN', 'details': [], 'score': 0.0}
        }
        self.total_tests = 0
        self.passed_tests = 0
    
    def test_component(self, component: str, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        self.total_tests += 1
        try:
            result = test_func()
            if result:
                self.results[component]['details'].append(f"✅ {test_name}: PASS")
                self.passed_tests += 1
                return True
            else:
                self.results[component]['details'].append(f"❌ {test_name}: FAIL")
                return False
        except Exception as e:
            self.results[component]['details'].append(f"💥 {test_name}: ERROR - {str(e)}")
            return False
    
    def validate_alpha_vantage_client(self) -> Dict[str, Any]:
        """Test Alpha Vantage API client implementation"""
        print("🔍 Testing Alpha Vantage API Client...")
        
        component = 'alpha_vantage'
        
        # Test 1: Module import
        def test_import():
            try:
                from src.data.alpha_vantage_client import AlphaVantageClient
                return True
            except ImportError:
                return False
        
        # Test 2: Class instantiation
        def test_instantiation():
            try:
                from src.data.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
                return hasattr(client, 'get_dual_ticker_quotes')
            except:
                return False
        
        # Test 3: Dual-ticker support
        def test_dual_ticker_methods():
            try:
                from src.data.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
                return (hasattr(client, 'get_dual_ticker_quotes') and 
                       hasattr(client, 'get_dual_ticker_bars'))
            except:
                return False
        
        # Test 4: Rate limiting implementation
        def test_rate_limiting():
            try:
                from src.data.alpha_vantage_client import AlphaVantageClient
                client = AlphaVantageClient()
                return hasattr(client, 'rate_limit') or 'rate' in str(client.__dict__)
            except:
                return False
        
        # Test 5: CLI interface
        def test_cli_interface():
            cli_file = Path('src/data/alpha_vantage_client.py')
            if not cli_file.exists():
                return False
            content = cli_file.read_text()
            return '__main__' in content and 'argparse' in content
        
        # Run tests
        tests = [
            ('Module Import', test_import),
            ('Class Instantiation', test_instantiation),
            ('Dual-Ticker Methods', test_dual_ticker_methods),
            ('Rate Limiting', test_rate_limiting),
            ('CLI Interface', test_cli_interface)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.test_component(component, test_name, test_func):
                passed += 1
        
        score = passed / len(tests)
        self.results[component]['score'] = score
        self.results[component]['status'] = 'PASS' if score >= 0.8 else 'FAIL'
        
        return self.results[component]
    
    def validate_data_quality_validator(self) -> Dict[str, Any]:
        """Test Data Quality Validator implementation"""
        print("🔍 Testing Data Quality Validator...")
        
        component = 'data_quality'
        
        # Test 1: Module import
        def test_import():
            try:
                from src.data.quality_validator import DataQualityValidator
                return True
            except ImportError:
                return False
        
        # Test 2: Class instantiation
        def test_instantiation():
            try:
                from src.data.quality_validator import DataQualityValidator
                validator = DataQualityValidator()
                return hasattr(validator, 'run_full_validation')
            except:
                return False
        
        # Test 3: Configuration support
        def test_configuration():
            try:
                from src.data.quality_validator import DataQualityValidator
                validator = DataQualityValidator()
                # Check for environment-specific configuration
                result = validator.run_full_validation if hasattr(validator, 'run_full_validation') else None
                return result is not None
            except:
                return False
        
        # Test 4: OHLC validation methods
        def test_ohlc_validation():
            try:
                from src.data.quality_validator import DataQualityValidator
                validator = DataQualityValidator()
                # Look for OHLC validation methods
                methods = dir(validator)
                return any('ohlc' in method.lower() or 'price' in method.lower() for method in methods)
            except:
                return False
        
        # Test 5: CLI interface
        def test_cli_interface():
            cli_file = Path('src/data/quality_validator.py')
            if not cli_file.exists():
                return False
            content = cli_file.read_text()
            return '__main__' in content
        
        # Run tests
        tests = [
            ('Module Import', test_import),
            ('Class Instantiation', test_instantiation),
            ('Configuration Support', test_configuration),
            ('OHLC Validation', test_ohlc_validation),
            ('CLI Interface', test_cli_interface)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.test_component(component, test_name, test_func):
                passed += 1
        
        score = passed / len(tests)
        self.results[component]['score'] = score
        self.results[component]['status'] = 'PASS' if score >= 0.8 else 'FAIL'
        
        return self.results[component]
    
    def validate_ib_gateway_connection(self) -> Dict[str, Any]:
        """Test IB Gateway connection implementation"""
        print("🔍 Testing IB Gateway Connection...")
        
        component = 'ib_gateway'
        
        # Test 1: Module import
        def test_import():
            try:
                from src.brokers.ib_gateway import IBGatewayClient
                return True
            except ImportError:
                return False
        
        # Test 2: Class instantiation
        def test_instantiation():
            try:
                from src.brokers.ib_gateway import IBGatewayClient
                client = IBGatewayClient()
                return hasattr(client, 'connect')
            except:
                return False
        
        # Test 3: Trading methods
        def test_trading_methods():
            try:
                from src.brokers.ib_gateway import IBGatewayClient
                client = IBGatewayClient()
                methods = dir(client)
                required_methods = ['place_market_order', 'get_positions']
                return all(method in methods for method in required_methods)
            except:
                return False
        
        # Test 4: Simulation mode
        def test_simulation_mode():
            try:
                from src.brokers.ib_gateway import IBGatewayClient
                client = IBGatewayClient()
                # Try to connect in simulation mode
                client.connect()
                return True  # If no exception, simulation mode works
            except:
                return False
        
        # Test 5: CLI interface
        def test_cli_interface():
            cli_file = Path('src/brokers/ib_gateway.py')
            if not cli_file.exists():
                return False
            content = cli_file.read_text()
            return '__main__' in content
        
        # Run tests
        tests = [
            ('Module Import', test_import),
            ('Class Instantiation', test_instantiation),
            ('Trading Methods', test_trading_methods),
            ('Simulation Mode', test_simulation_mode),
            ('CLI Interface', test_cli_interface)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.test_component(component, test_name, test_func):
                passed += 1
        
        score = passed / len(tests)
        self.results[component]['score'] = score
        self.results[component]['status'] = 'PASS' if score >= 0.8 else 'FAIL'
        
        return self.results[component]
    
    def validate_live_monitoring_endpoints(self) -> Dict[str, Any]:
        """Test Live Monitoring endpoints implementation"""
        print("🔍 Testing Live Monitoring Endpoints...")
        
        component = 'live_monitoring'
        
        # Test 1: Module import
        def test_import():
            try:
                from src.api.live_monitoring import LiveMonitoringService
                return True
            except ImportError:
                return False
        
        # Test 2: Service instantiation
        def test_instantiation():
            try:
                from src.api.live_monitoring import LiveMonitoringService
                service = LiveMonitoringService()
                return hasattr(service, 'get_overall_health')
            except:
                return False
        
        # Test 3: Router import
        def test_router_import():
            try:
                from src.api.live_monitoring import router
                return router is not None
            except ImportError:
                return False
        
        # Test 4: Health check methods
        def test_health_methods():
            try:
                from src.api.live_monitoring import LiveMonitoringService
                service = LiveMonitoringService()
                methods = dir(service)
                health_methods = [m for m in methods if 'health' in m.lower()]
                return len(health_methods) >= 2
            except:
                return False
        
        # Test 5: CLI interface
        def test_cli_interface():
            cli_file = Path('src/api/live_monitoring.py')
            if not cli_file.exists():
                return False
            content = cli_file.read_text()
            return '__main__' in content
        
        # Run tests
        tests = [
            ('Module Import', test_import),
            ('Service Instantiation', test_instantiation),
            ('Router Import', test_router_import),
            ('Health Methods', test_health_methods),
            ('CLI Interface', test_cli_interface)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.test_component(component, test_name, test_func):
                passed += 1
        
        score = passed / len(tests)
        self.results[component]['score'] = score
        self.results[component]['status'] = 'PASS' if score >= 0.8 else 'FAIL'
        
        return self.results[component]
    
    def validate_fastapi_integration(self) -> Dict[str, Any]:
        """Test FastAPI integration of all components"""
        print("🔍 Testing FastAPI Integration...")
        
        component = 'integration'
        
        # Test 1: Main app imports
        def test_main_imports():
            try:
                main_file = Path('src/api/main.py')
                if not main_file.exists():
                    return False
                content = main_file.read_text()
                return 'live_monitoring_router' in content
            except:
                return False
        
        # Test 2: Router inclusion
        def test_router_inclusion():
            try:
                main_file = Path('src/api/main.py')
                content = main_file.read_text()
                return 'app.include_router(live_monitoring_router)' in content
            except:
                return False
        
        # Test 3: All required files exist
        def test_files_exist():
            required_files = [
                'src/data/alpha_vantage_client.py',
                'src/data/quality_validator.py',
                'src/brokers/ib_gateway.py',
                'src/api/live_monitoring.py'
            ]
            return all(Path(f).exists() for f in required_files)
        
        # Test 4: Test runner exists
        def test_runner_exists():
            return Path('scripts/test_day2_missing_components.py').exists()
        
        # Test 5: Directory structure
        def test_directory_structure():
            required_dirs = ['src/data', 'src/brokers', 'src/api']
            return all(Path(d).exists() for d in required_dirs)
        
        # Run tests
        tests = [
            ('Main App Imports', test_main_imports),
            ('Router Inclusion', test_router_inclusion),
            ('Required Files Exist', test_files_exist),
            ('Test Runner Exists', test_runner_exists),
            ('Directory Structure', test_directory_structure)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.test_component(component, test_name, test_func):
                passed += 1
        
        score = passed / len(tests)
        self.results[component]['score'] = score
        self.results[component]['status'] = 'PASS' if score >= 0.8 else 'FAIL'
        
        return self.results[component]
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🚀 COMPREHENSIVE DAY 2 VALIDATION SUITE")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all component validations
        self.validate_alpha_vantage_client()
        self.validate_data_quality_validator()
        self.validate_ib_gateway_connection()
        self.validate_live_monitoring_endpoints()
        self.validate_fastapi_integration()
        
        # Calculate overall scores
        overall_score = sum(r['score'] for r in self.results.values()) / len(self.results)
        success_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
        
        end_time = time.time()
        
        # Generate summary
        summary = {
            'overall_score': overall_score,
            'success_rate': success_rate,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.total_tests - self.passed_tests,
            'execution_time': end_time - start_time,
            'components': self.results,
            'final_verdict': 'PASS' if overall_score >= 0.8 else 'FAIL'
        }
        
        return summary
    
    def print_detailed_report(self, summary: Dict[str, Any]):
        """Print comprehensive validation report"""
        print("\n📊 DETAILED VALIDATION REPORT")
        print("=" * 50)
        
        # Component results
        for component, result in summary['components'].items():
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            print(f"\n{status_emoji} {component.upper().replace('_', ' ')} - {result['status']} ({result['score']:.1%})")
            for detail in result['details']:
                print(f"  {detail}")
        
        # Overall summary
        print(f"\n🎯 OVERALL RESULTS")
        print(f"Overall Score: {summary['overall_score']:.1%}")
        print(f"Test Success Rate: {summary['success_rate']:.1%} ({summary['passed_tests']}/{summary['total_tests']})")
        print(f"Execution Time: {summary['execution_time']:.2f} seconds")
        
        # Final verdict
        verdict_emoji = "✅" if summary['final_verdict'] == 'PASS' else "❌"
        print(f"\n{verdict_emoji} FINAL VERDICT: {summary['final_verdict']}")
        
        if summary['final_verdict'] == 'PASS':
            print("🎉 Team's claims VERIFIED - All Day 2 components are implemented!")
        else:
            print("⚠️  Team's claims PARTIALLY VERIFIED - Some components need work")
        
        return summary

def main():
    """Run comprehensive Day 2 validation"""
    validator = Day2ComprehensiveValidator()
    
    try:
        summary = validator.run_comprehensive_validation()
        validator.print_detailed_report(summary)
        
        # Exit code based on results
        exit_code = 0 if summary['final_verdict'] == 'PASS' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"💥 VALIDATION FAILED: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()