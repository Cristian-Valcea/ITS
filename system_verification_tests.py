#!/usr/bin/env python3
"""
System Verification Tests
Comprehensive testing to verify what's actually working vs. claimed functionality
"""

import os
import sys
import subprocess
import requests
import time
import json
from pathlib import Path
import importlib.util

def test_file_existence():
    """Test if claimed files actually exist"""
    print("🔍 FILE EXISTENCE TEST")
    print("=" * 40)
    
    files_to_check = [
        "minimal_launcher.py",
        "real_trading_deployment.py", 
        "TRADING_SYSTEM_GUIDE.md",
        "models/chunk7_final_358400steps.zip",
        "src/brokers/ib_gateway.py",
        "src/agents/trainer_agent.py",
    ]
    
    results = {}
    for file_path in files_to_check:
        full_path = Path(file_path)
        exists = full_path.exists()
        size = full_path.stat().st_size if exists else 0
        results[file_path] = {"exists": exists, "size": size}
        
        status = "✅" if exists else "❌"
        size_info = f"({size} bytes)" if exists else ""
        print(f"   {status} {file_path} {size_info}")
    
    return results

def test_python_imports():
    """Test if claimed Python modules can be imported"""
    print("\n🐍 PYTHON IMPORT TEST")
    print("=" * 40)
    
    modules_to_test = [
        ("ib_insync", "IBKR integration"),
        ("stable_baselines3", "AI model framework"),
        ("gymnasium", "RL environment"),
        ("flask", "Web interface"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data processing"),
    ]
    
    results = {}
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            results[module_name] = True
            print(f"   ✅ {module_name} - {description}")
        except ImportError as e:
            results[module_name] = False
            print(f"   ❌ {module_name} - {description} (Error: {e})")
    
    return results

def test_web_interface():
    """Test if web interfaces are actually running"""
    print("\n🌐 WEB INTERFACE TEST")
    print("=" * 40)
    
    ports_to_test = [
        (5000, "Flask app (claimed)"),
        (9000, "Minimal launcher (claimed)"),
        (8000, "FastAPI (existing)"),
        (3000, "Grafana (monitoring)"),
    ]
    
    results = {}
    for port, description in ports_to_test:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=3)
            results[port] = {
                "accessible": True,
                "status_code": response.status_code,
                "content_length": len(response.text)
            }
            print(f"   ✅ Port {port} - {description} (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            results[port] = {"accessible": False, "error": str(e)}
            print(f"   ❌ Port {port} - {description} (Error: {type(e).__name__})")
    
    return results

def test_ai_model_loading():
    """Test if AI models can actually be loaded"""
    print("\n🤖 AI MODEL LOADING TEST")
    print("=" * 40)
    
    model_paths = [
        "models/chunk7_final_358400steps.zip",
        "models/stairways_v3.zip",
        "models/latest_model.zip",
    ]
    
    results = {}
    for model_path in model_paths:
        try:
            if Path(model_path).exists():
                # Try to load with stable_baselines3
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(model_path)
                    results[model_path] = {
                        "exists": True,
                        "loadable": True,
                        "type": str(type(model))
                    }
                    print(f"   ✅ {model_path} - Loadable PPO model")
                except Exception as e:
                    results[model_path] = {
                        "exists": True,
                        "loadable": False,
                        "error": str(e)
                    }
                    print(f"   ⚠️  {model_path} - Exists but not loadable ({e})")
            else:
                results[model_path] = {"exists": False}
                print(f"   ❌ {model_path} - File not found")
        except Exception as e:
            results[model_path] = {"error": str(e)}
            print(f"   ❌ {model_path} - Error: {e}")
    
    return results

def test_ibkr_connection():
    """Test IBKR connection capabilities"""
    print("\n🔌 IBKR CONNECTION TEST")
    print("=" * 40)
    
    try:
        # Try to import and test IBKR gateway
        sys.path.append('src')
        from brokers.ib_gateway import IBGatewayClient
        
        client = IBGatewayClient()
        connected = client.connect()
        
        if connected:
            mode = "simulation" if client.simulation_mode else "live"
            print(f"   ✅ IBKR Connection successful (mode: {mode})")
            
            # Test basic functionality
            try:
                price = client.get_current_price('NVDA')
                print(f"   ✅ Market data: NVDA ${price}")
            except Exception as e:
                print(f"   ⚠️  Market data error: {e}")
            
            try:
                account_info = client.get_account_info()
                print(f"   ✅ Account info: {account_info.get('mode', 'unknown')}")
            except Exception as e:
                print(f"   ⚠️  Account info error: {e}")
            
            client.disconnect()
            return {"connected": True, "mode": mode}
        else:
            print("   ❌ IBKR Connection failed")
            return {"connected": False}
            
    except Exception as e:
        print(f"   ❌ IBKR Test error: {e}")
        return {"error": str(e)}

def test_claimed_scripts():
    """Test if claimed scripts actually work"""
    print("\n📜 SCRIPT EXECUTION TEST")
    print("=" * 40)
    
    scripts_to_test = [
        ("minimal_launcher.py", "Web interface launcher"),
        ("real_trading_deployment.py", "AI trading deployment"),
        ("src/brokers/ib_gateway.py --test connect", "IBKR gateway test"),
    ]
    
    results = {}
    for script, description in scripts_to_test:
        try:
            print(f"   🔄 Testing {script}...")
            
            # Run script with timeout
            result = subprocess.run(
                f"python {script}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            results[script] = {
                "returncode": result.returncode,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"   ✅ {script} - {description} (Success)")
            else:
                print(f"   ❌ {script} - {description} (Exit code: {result.returncode})")
                if result.stderr:
                    print(f"      Error: {result.stderr[:100]}...")
                    
        except subprocess.TimeoutExpired:
            results[script] = {"timeout": True}
            print(f"   ⏰ {script} - {description} (Timeout - may be running)")
        except Exception as e:
            results[script] = {"error": str(e)}
            print(f"   ❌ {script} - {description} (Error: {e})")
    
    return results

def test_environment_config():
    """Test environment configuration"""
    print("\n🔧 ENVIRONMENT CONFIG TEST")
    print("=" * 40)
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file exists")
        with open(env_file) as f:
            env_content = f.read()
            
        # Check key variables
        key_vars = [
            "IBKR_HOST_IP",
            "IBKR_PORT", 
            "TRADING_CYCLE_SECONDS",
            "MAX_DAILY_LOSS",
            "MAX_POSITION_VALUE"
        ]
        
        for var in key_vars:
            if var in env_content:
                print(f"   ✅ {var} configured")
            else:
                print(f"   ❌ {var} missing")
    else:
        print("   ❌ .env file not found")
    
    # Check virtual environment
    if os.environ.get('VIRTUAL_ENV'):
        print(f"   ✅ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
    else:
        print("   ⚠️  Virtual environment not detected")

def generate_verification_report(all_results):
    """Generate comprehensive verification report"""
    print("\n" + "=" * 60)
    print("📊 SYSTEM VERIFICATION REPORT")
    print("=" * 60)
    
    # Count successes and failures
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n📋 {category.upper()}:")
        
        if isinstance(results, dict):
            for item, result in results.items():
                total_tests += 1
                if isinstance(result, dict):
                    success = result.get('exists', result.get('connected', result.get('success', False)))
                elif isinstance(result, bool):
                    success = result
                else:
                    success = False
                
                if success:
                    passed_tests += 1
                    print(f"   ✅ {item}")
                else:
                    print(f"   ❌ {item}")
    
    # Overall score
    score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 OVERALL SYSTEM HEALTH: {score:.1f}% ({passed_tests}/{total_tests})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if score >= 80:
        print("   ✅ System appears to be mostly functional")
        print("   🎯 Focus on fixing remaining issues")
    elif score >= 50:
        print("   ⚠️  System has significant issues")
        print("   🔧 Major fixes needed before production use")
    else:
        print("   ❌ System has critical problems")
        print("   🚨 Extensive debugging required")
    
    return score

def main():
    """Run comprehensive system verification"""
    print("🔍 COMPREHENSIVE SYSTEM VERIFICATION")
    print("=" * 60)
    print("Testing claimed functionality vs. actual implementation")
    print()
    
    # Run all tests
    all_results = {}
    
    all_results["File Existence"] = test_file_existence()
    all_results["Python Imports"] = test_python_imports()
    all_results["Web Interfaces"] = test_web_interface()
    all_results["AI Models"] = test_ai_model_loading()
    all_results["IBKR Connection"] = test_ibkr_connection()
    all_results["Script Execution"] = test_claimed_scripts()
    
    test_environment_config()
    
    # Generate report
    score = generate_verification_report(all_results)
    
    # Save results to file
    with open("system_verification_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: system_verification_results.json")
    
    return score >= 70  # Return True if system is mostly functional

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)