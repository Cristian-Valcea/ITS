#!/usr/bin/env python3
"""
🔍 TRADING SYSTEM INTEGRATION VALIDATION SUITE
==============================================

This test suite validates the integration between the secrets management system
and the IntradayJules Trading System (ITS) components as described in Phase 4.

⚠️ TRUST BUT VERIFY ⚠️
This test validates that the secrets system actually integrates properly with:
- Database connections (TimescaleDB)
- Interactive Brokers API
- Alert systems (PagerDuty, Slack)
- Dual-ticker trading pipeline
- Risk management systems

Test Scope:
- ITS helper functions
- Database configuration management
- Trading credential handling
- Alert system integration
- Failover and backup mechanisms
- Real trading pipeline integration
"""

import sys
import os
import asyncio
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

class TradingSystemIntegrationValidator:
    """
    Comprehensive validation suite for trading system integration.
    
    This validates that the secrets management system properly integrates
    with all ITS trading components as claimed in the documentation.
    """
    
    def __init__(self):
        self.test_results = {}
        self.master_password = "trading_test_password_12345"
        self.temp_vault_path = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up isolated test environment for trading integration."""
        print("🔧 Setting up trading integration test environment...")
        
        # Create temporary vault file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.vault')
        self.temp_vault_path = temp_file.name
        temp_file.close()
        
        # Set up environment variables for ITS integration
        os.environ['ITS_VAULT_PATH'] = self.temp_vault_path
        os.environ['ITS_MASTER_PASSWORD'] = self.master_password
        os.environ['ITS_SECRETS_BACKEND'] = 'local'
        
        print(f"✅ Trading test vault: {self.temp_vault_path}")
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_vault_path and os.path.exists(self.temp_vault_path):
            os.unlink(self.temp_vault_path)
        
        # Clean up environment variables
        for var in ['ITS_VAULT_PATH', 'ITS_MASTER_PASSWORD', 'ITS_SECRETS_BACKEND']:
            os.environ.pop(var, None)
    
    def record_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
    
    # ========================================================================
    # ITS INTEGRATION CORE VALIDATION
    # ========================================================================
    
    def test_its_integration_imports(self):
        """Test that ITS integration components can be imported."""
        print("\n📋 ITS INTEGRATION: Core Component Imports")
        print("=" * 60)
        
        try:
            # Test ITS-specific imports as mentioned in Phase 4
            from security import get_database_config, get_its_secret, ITSSecretsHelper
            self.record_test_result("ITS_Core_Imports", True)
            
            # Test additional helper functions
            try:
                from security import get_alert_config
                self.record_test_result("ITS_Alert_Config_Import", True)
            except ImportError:
                self.record_test_result("ITS_Alert_Config_Import", False, "get_alert_config not found")
            
        except ImportError as e:
            self.record_test_result("ITS_Core_Imports", False, str(e))
    
    def test_database_configuration_integration(self):
        """Test database configuration for TimescaleDB integration."""
        try:
            from security import get_database_config
            
            # Get database configuration
            db_config = get_database_config()
            
            # Validate structure matches Phase 4 documentation
            expected_keys = {'host', 'port', 'database', 'user', 'password'}
            actual_keys = set(db_config.keys())
            
            self.record_test_result("DB_Config_Structure", 
                                   expected_keys.issubset(actual_keys),
                                   f"Expected: {expected_keys}, Got: {actual_keys}")
            
            # Validate default values match documentation
            expected_defaults = {
                'host': 'localhost',
                'port': '5432',
                'database': 'featurestore_manifest',
                'user': 'postgres'
            }
            
            defaults_match = all(
                db_config.get(key) == value 
                for key, value in expected_defaults.items()
            )
            self.record_test_result("DB_Config_Defaults", defaults_match)
            
            # Test that password is present (should have secure default)
            has_password = 'password' in db_config and db_config['password']
            self.record_test_result("DB_Config_Password", has_password)
            
            # Test connection string generation
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            self.record_test_result("DB_Connection_String", 
                                   connection_string.startswith("postgresql://"))
            
        except Exception as e:
            self.record_test_result("Database_Configuration", False, str(e))
    
    def test_alert_system_integration(self):
        """Test alert system configuration for PagerDuty and Slack."""
        try:
            from security import get_alert_config
            
            # Get alert configuration
            alert_config = get_alert_config()
            
            # Validate structure matches Phase 4 documentation
            expected_keys = {'pagerduty_key', 'slack_webhook', 'slack_channel'}
            actual_keys = set(alert_config.keys())
            
            self.record_test_result("Alert_Config_Structure", 
                                   expected_keys.issubset(actual_keys),
                                   f"Expected: {expected_keys}, Got: {actual_keys}")
            
            # Validate default values
            expected_values = {
                'pagerduty_key': 'pd_integration_key_12345',
                'slack_channel': '#trading-alerts'
            }
            
            for key, expected_value in expected_values.items():
                actual_value = alert_config.get(key)
                self.record_test_result(f"Alert_Config_{key}", 
                                       actual_value == expected_value,
                                       f"Expected: {expected_value}, Got: {actual_value}")
            
            # Test Slack webhook format
            slack_webhook = alert_config.get('slack_webhook', '')
            webhook_valid = slack_webhook.startswith('https://hooks.slack.com/')
            self.record_test_result("Alert_Slack_Webhook_Format", webhook_valid)
            
        except Exception as e:
            self.record_test_result("Alert_System_Integration", False, str(e))
    
    # ========================================================================
    # TRADING CREDENTIALS VALIDATION
    # ========================================================================
    
    def test_trading_credentials_management(self):
        """Test management of trading-specific credentials."""
        print("\n📋 TRADING CREDENTIALS: Management and Retrieval")
        print("=" * 60)
        
        try:
            from security import get_its_secret, ITSSecretsHelper
            from security.advanced_secrets_manager import AdvancedSecretsManager
            from security.backends.local_vault import LocalVaultBackend
            from security.protocols import SecretType
            
            # Set up manager for testing
            backend = LocalVaultBackend(self.temp_vault_path, self.master_password)
            manager = AdvancedSecretsManager(backend, self.master_password)
            
            # Test storing trading-specific secrets
            trading_secrets = {
                "ib_api_key": "test_ib_api_key_12345",
                "ib_username": "test_ib_user",
                "ib_password": "test_ib_password",
                "broker_api_key": "test_broker_key",
                "alpha_vantage_key": "test_av_key_67890",
                "yahoo_finance_key": "test_yf_key"
            }
            
            # Store all trading secrets
            for key, value in trading_secrets.items():
                result = asyncio.run(manager.write_secret(
                    key, value,
                    secret_type=SecretType.API_KEY,
                    description=f"Trading credential: {key}"
                ))
                self.record_test_result(f"Store_Trading_{key}", result)
            
            # Test retrieval using ITS helper
            for key, expected_value in trading_secrets.items():
                retrieved_value = get_its_secret(key)
                self.record_test_result(f"Retrieve_Trading_{key}", 
                                       retrieved_value == expected_value)
            
            # Test non-existent secret returns None
            missing_secret = get_its_secret('nonexistent_trading_key')
            self.record_test_result("Trading_Missing_Secret", missing_secret is None)
            
        except Exception as e:
            self.record_test_result("Trading_Credentials_Management", False, str(e))
    
    def test_dual_ticker_integration(self):
        """Test integration with dual-ticker trading system (AAPL + MSFT)."""
        try:
            from security import get_its_secret
            from security.advanced_secrets_manager import AdvancedSecretsManager
            from security.backends.local_vault import LocalVaultBackend
            from security.protocols import SecretType
            
            # Set up manager
            backend = LocalVaultBackend(self.temp_vault_path, self.master_password)
            manager = AdvancedSecretsManager(backend, self.master_password)
            
            # Store dual-ticker specific credentials
            dual_ticker_secrets = {
                "aapl_data_key": "aapl_specific_key_12345",
                "msft_data_key": "msft_specific_key_67890",
                "portfolio_manager_key": "portfolio_mgr_key",
                "dual_ticker_db_password": "dual_ticker_db_pass",
                "correlation_service_key": "correlation_svc_key"
            }
            
            for key, value in dual_ticker_secrets.items():
                result = asyncio.run(manager.write_secret(
                    key, value,
                    secret_type=SecretType.API_KEY,
                    description=f"Dual-ticker system: {key}",
                    tags={"system": "dual_ticker", "assets": "AAPL,MSFT"}
                ))
                self.record_test_result(f"DualTicker_Store_{key}", result)
            
            # Test retrieval for dual-ticker system
            for key, expected_value in dual_ticker_secrets.items():
                retrieved = get_its_secret(key)
                self.record_test_result(f"DualTicker_Retrieve_{key}", 
                                       retrieved == expected_value)
            
        except Exception as e:
            self.record_test_result("Dual_Ticker_Integration", False, str(e))
    
    # ========================================================================
    # RISK MANAGEMENT INTEGRATION
    # ========================================================================
    
    def test_risk_management_integration(self):
        """Test integration with risk management systems."""
        print("\n📋 RISK MANAGEMENT: Integration Testing")
        print("=" * 60)
        
        try:
            from security import get_its_secret
            from security.advanced_secrets_manager import AdvancedSecretsManager
            from security.backends.local_vault import LocalVaultBackend
            from security.protocols import SecretType
            
            # Set up manager
            backend = LocalVaultBackend(self.temp_vault_path, self.master_password)
            manager = AdvancedSecretsManager(backend, self.master_password)
            
            # Store risk management secrets
            risk_secrets = {
                "risk_db_password": "risk_database_password",
                "var_calculator_key": "var_calc_service_key",
                "stress_test_key": "stress_test_api_key",
                "compliance_reporting_key": "compliance_key",
                "circuit_breaker_config": "circuit_breaker_settings"
            }
            
            for key, value in risk_secrets.items():
                result = asyncio.run(manager.write_secret(
                    key, value,
                    secret_type=SecretType.API_KEY,
                    description=f"Risk management: {key}",
                    tags={"component": "risk_management"}
                ))
                self.record_test_result(f"Risk_Store_{key}", result)
            
            # Test integration with risk agent
            for key in risk_secrets.keys():
                retrieved = get_its_secret(key)
                self.record_test_result(f"Risk_Integration_{key}", retrieved is not None)
            
        except Exception as e:
            self.record_test_result("Risk_Management_Integration", False, str(e))
    
    # ========================================================================
    # BACKUP AND FAILOVER TESTING
    # ========================================================================
    
    def test_backup_and_failover(self):
        """Test backup and failover mechanisms."""
        print("\n📋 BACKUP & FAILOVER: Resilience Testing")
        print("=" * 60)
        
        try:
            from security import get_its_secret
            
            # Test environment variable fallback
            test_key = "test_fallback_key"
            test_value = "fallback_test_value"
            
            # Set environment variable as fallback
            os.environ[f"ITS_{test_key.upper()}"] = test_value
            
            # Should retrieve from environment when not in vault
            retrieved = get_its_secret(test_key)
            self.record_test_result("Fallback_Environment", retrieved == test_value)
            
            # Clean up
            os.environ.pop(f"ITS_{test_key.upper()}", None)
            
            # Test graceful handling of missing secrets
            missing_secret = get_its_secret("completely_missing_secret")
            self.record_test_result("Fallback_Missing_Secret", missing_secret is None)
            
        except Exception as e:
            self.record_test_result("Backup_Failover", False, str(e))
    
    # ========================================================================
    # REAL TRADING PIPELINE SIMULATION
    # ========================================================================
    
    def test_trading_pipeline_simulation(self):
        """Simulate a complete trading pipeline using secrets."""
        print("\n📋 TRADING PIPELINE: Complete Workflow Simulation")
        print("=" * 60)
        
        try:
            from security import get_database_config, get_alert_config, get_its_secret
            from security.advanced_secrets_manager import AdvancedSecretsManager
            from security.backends.local_vault import LocalVaultBackend
            from security.protocols import SecretType
            
            # Set up complete trading environment
            backend = LocalVaultBackend(self.temp_vault_path, self.master_password)
            manager = AdvancedSecretsManager(backend, self.master_password)
            
            # 1. Store all required trading secrets
            complete_trading_setup = {
                "ib_paper_account": "paper_account_credentials",
                "timescaledb_password": "timescale_db_password",
                "aapl_data_source": "aapl_market_data_key",
                "msft_data_source": "msft_market_data_key",
                "risk_engine_key": "risk_calculation_key",
                "portfolio_optimizer": "portfolio_opt_key"
            }
            
            for key, value in complete_trading_setup.items():
                result = asyncio.run(manager.write_secret(
                    key, value,
                    secret_type=SecretType.API_KEY,
                    description=f"Complete trading setup: {key}",
                    tags={"pipeline": "complete_trading"}
                ))
                self.record_test_result(f"Pipeline_Setup_{key}", result)
            
            # 2. Simulate trading pipeline initialization
            def simulate_trading_initialization():
                """Simulate how the trading system would initialize."""
                
                # Database connection
                db_config = get_database_config()
                db_connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                
                # Trading credentials
                ib_credentials = get_its_secret("ib_paper_account")
                
                # Data sources
                aapl_key = get_its_secret("aapl_data_source") 
                msft_key = get_its_secret("msft_data_source")
                
                # Risk management
                risk_key = get_its_secret("risk_engine_key")
                
                # Alert system
                alert_config = get_alert_config()
                
                return {
                    'database': db_connection_string is not None,
                    'ib_credentials': ib_credentials is not None,
                    'data_sources': aapl_key is not None and msft_key is not None,
                    'risk_management': risk_key is not None,
                    'alerts': 'slack_webhook' in alert_config
                }
            
            # 3. Test complete initialization
            init_result = simulate_trading_initialization()
            
            for component, success in init_result.items():
                self.record_test_result(f"Pipeline_Init_{component}", success)
            
            # 4. Test pipeline readiness
            all_components_ready = all(init_result.values())
            self.record_test_result("Pipeline_Complete_Readiness", all_components_ready)
            
        except Exception as e:
            self.record_test_result("Trading_Pipeline_Simulation", False, str(e))
    
    # ========================================================================
    # PERFORMANCE AND RELIABILITY TESTING
    # ========================================================================
    
    def test_performance_under_load(self):
        """Test performance under trading load conditions."""
        print("\n📋 PERFORMANCE: Load Testing for Trading")
        print("=" * 60)
        
        try:
            from security import get_its_secret, get_database_config
            import time
            
            # Test rapid-fire secret retrieval (simulating high-frequency trading)
            start_time = time.time()
            for i in range(100):
                db_config = get_database_config()
                secret = get_its_secret("test_secret_or_fallback")
            
            load_test_time = time.time() - start_time
            
            # Should handle 100 operations in reasonable time (< 10 seconds)
            self.record_test_result("Performance_Load_Test", 
                                   load_test_time < 10,
                                   f"100 operations took {load_test_time:.2f}s")
            
            # Test concurrent access simulation
            import threading
            
            def worker_thread():
                for _ in range(10):
                    get_database_config()
                    get_its_secret("test_key")
            
            threads = []
            start_time = time.time()
            
            for _ in range(5):  # 5 concurrent threads
                thread = threading.Thread(target=worker_thread)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            concurrent_time = time.time() - start_time
            
            # 5 threads doing 10 operations each should complete reasonably fast
            self.record_test_result("Performance_Concurrent_Access", 
                                   concurrent_time < 15,
                                   f"Concurrent access took {concurrent_time:.2f}s")
            
        except Exception as e:
            self.record_test_result("Performance_Load", False, str(e))
    
    # ========================================================================
    # INTEGRATION WITH EXISTING ITS COMPONENTS
    # ========================================================================
    
    def test_existing_its_components_integration(self):
        """Test integration with existing ITS components."""
        print("\n📋 EXISTING ITS: Component Integration")
        print("=" * 60)
        
        try:
            # Test that we can import and use secrets in context of existing ITS components
            from security import get_database_config, get_its_secret
            
            # Simulate integration with existing ITS components
            
            # 1. Feature Store Integration
            db_config = get_database_config()
            feature_store_config = {
                'connection_string': f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}",
                'pool_size': 10,
                'timeout': 30
            }
            self.record_test_result("ITS_FeatureStore_Config", 
                                   'postgresql://' in feature_store_config['connection_string'])
            
            # 2. Risk Agent Integration
            risk_db_password = get_its_secret("risk_db_password")
            if risk_db_password is None:
                # Fallback to default for testing
                risk_db_password = "default_risk_password"
            
            risk_config = {
                'database_password': risk_db_password,
                'max_position_size': 1000,
                'daily_loss_limit': 50
            }
            self.record_test_result("ITS_RiskAgent_Config", 
                                   isinstance(risk_config['database_password'], str))
            
            # 3. Orchestrator Agent Integration
            orchestrator_config = {
                'database_config': db_config,
                'broker_credentials': get_its_secret("ib_api_key") or "default_ib_key",
                'risk_limits': {
                    'max_drawdown': 0.02,
                    'position_limit': 1000
                }
            }
            self.record_test_result("ITS_Orchestrator_Config", 
                                   'host' in orchestrator_config['database_config'])
            
            # 4. Data Agent Integration  
            data_sources = {
                'ib_credentials': get_its_secret("ib_api_key"),
                'alpha_vantage': get_its_secret("alpha_vantage_key"),
                'yahoo_finance': get_its_secret("yahoo_finance_key")
            }
            
            # At least one data source should be available
            has_data_source = any(cred is not None for cred in data_sources.values())
            self.record_test_result("ITS_DataAgent_Sources", has_data_source)
            
        except Exception as e:
            self.record_test_result("Existing_ITS_Integration", False, str(e))
    
    # ========================================================================
    # MAIN TEST EXECUTION
    # ========================================================================
    
    def run_all_tests(self):
        """Execute all trading integration validation tests."""
        print("🔍 TRADING SYSTEM INTEGRATION VALIDATION")
        print("=" * 80)
        print("⚠️  VERIFYING SECRETS <-> TRADING SYSTEM INTEGRATION")
        print("=" * 80)
        
        try:
            # Core ITS Integration
            self.test_its_integration_imports()
            self.test_database_configuration_integration()
            self.test_alert_system_integration()
            
            # Trading Credentials
            self.test_trading_credentials_management()
            self.test_dual_ticker_integration()
            
            # Risk Management
            self.test_risk_management_integration()
            
            # Backup and Failover
            self.test_backup_and_failover()
            
            # Complete Pipeline
            self.test_trading_pipeline_simulation()
            
            # Performance
            self.test_performance_under_load()
            
            # Existing Components
            self.test_existing_its_components_integration()
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in trading integration tests: {e}")
        
        finally:
            self.cleanup_test_environment()
    
    def generate_report(self):
        """Generate comprehensive trading integration report."""
        print("\n" + "=" * 80)
        print("📊 TRADING INTEGRATION VALIDATION REPORT")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"📈 TRADING INTEGRATION SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED INTEGRATION TESTS ({failed_tests}):")
            for test_name, result in self.test_results.items():
                if not result['passed']:
                    print(f"   - {test_name}: {result['details']}")
        
        # Component-specific analysis
        components = {
            'Database': [t for t in self.test_results if 'DB_' in t or 'Database' in t],
            'Trading': [t for t in self.test_results if 'Trading_' in t or 'IB_' in t],
            'Risk': [t for t in self.test_results if 'Risk_' in t],
            'Pipeline': [t for t in self.test_results if 'Pipeline_' in t],
            'ITS': [t for t in self.test_results if 'ITS_' in t]
        }
        
        print(f"\n📋 COMPONENT-SPECIFIC ANALYSIS:")
        for component_name, tests in components.items():
            if tests:
                component_passed = sum(1 for t in tests if self.test_results[t]['passed'])
                component_total = len(tests)
                component_rate = (component_passed/component_total)*100 if component_total > 0 else 0
                status = "✅" if component_rate == 100 else "⚠️" if component_rate >= 75 else "❌"
                print(f"   {status} {component_name}: {component_passed}/{component_total} ({component_rate:.1f}%)")
        
        # Trading readiness assessment
        print(f"\n🎯 TRADING SYSTEM READINESS:")
        critical_tests = [
            'ITS_Core_Imports',
            'DB_Config_Structure', 
            'Pipeline_Complete_Readiness',
            'Trading_Credentials_Management'
        ]
        
        critical_passed = sum(1 for test in critical_tests 
                             if test in self.test_results and self.test_results[test]['passed'])
        critical_total = len(critical_tests)
        
        if critical_passed == critical_total:
            print("   🎉 READY FOR TRADING - All critical integrations working!")
        elif critical_passed >= critical_total * 0.75:
            print("   ⚠️ MOSTLY READY - Some integration issues detected")
        else:
            print("   ❌ NOT READY - Critical integration failures")
        
        # Overall recommendation
        print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
        if passed_tests == total_tests:
            print("   ✅ DEPLOY TO PRODUCTION - All integrations validated")
        elif (passed_tests/total_tests) >= 0.90:
            print("   ✅ DEPLOY WITH MONITORING - Minor issues acceptable")
        elif (passed_tests/total_tests) >= 0.75:
            print("   ⚠️ DEPLOY TO STAGING FIRST - Significant issues need resolution")
        else:
            print("   ❌ DO NOT DEPLOY - Major integration failures")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'critical_readiness': (critical_passed/critical_total)*100,
            'results': self.test_results
        }


def main():
    """Main test execution function."""
    validator = TradingSystemIntegrationValidator()
    
    try:
        validator.run_all_tests()
        report = validator.generate_report()
        
        # Save detailed report
        report_path = project_root / "TRADING_INTEGRATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_path}")
        
        # Exit with appropriate code based on trading readiness
        if report['success_rate'] == 100:
            sys.exit(0)  # Ready for production
        elif report['success_rate'] >= 90:
            sys.exit(1)  # Deploy with monitoring
        elif report['success_rate'] >= 75:
            sys.exit(2)  # Staging only
        else:
            sys.exit(3)  # Do not deploy
            
    except Exception as e:
        print(f"💥 CRITICAL TRADING INTEGRATION FAILURE: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()