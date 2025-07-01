#!/usr/bin/env python3
"""
Test Phase 3 UI functionality
"""

import requests
import time

API_BASE_URL = "http://127.0.0.1:8000"

def test_ui_endpoints():
    """Test the Phase 3 UI endpoints"""
    print("🌐 Testing Phase 3 UI Endpoints...")
    
    endpoints = [
        ("/ui/dashboard", "Dashboard"),
        ("/ui/config/main_config", "Main Config Editor"),
        ("/ui/config/model_params", "Model Params Editor"),
        ("/ui/config/risk_limits", "Risk Limits Editor"),
        ("/ui/train", "Training Form"),
        ("/docs", "API Documentation"),
        ("/api/v1/status", "API Status")
    ]
    
    passed = 0
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {name}: OK ({len(response.content)} bytes)")
                passed += 1
            else:
                print(f"❌ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    return passed, len(endpoints)

def main():
    print("🚀 Phase 3 UI Testing")
    print(f"API Base URL: {API_BASE_URL}")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    passed, total = test_ui_endpoints()
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 All Phase 3 UI endpoints are working!")
        print("\n📖 Available UI Pages:")
        print("   • Dashboard: http://127.0.0.1:8000/ui/dashboard")
        print("   • Config Editor: http://127.0.0.1:8000/ui/config/main_config")
        print("   • Training Form: http://127.0.0.1:8000/ui/train")
        print("   • API Docs: http://127.0.0.1:8000/docs")
    else:
        print("⚠️  Some UI endpoints failed. Check the server logs.")

if __name__ == "__main__":
    main()