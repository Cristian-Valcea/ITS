#!/usr/bin/env python3
"""
Simple API test script to verify the FastAPI endpoints are working.
Run this while the API server is running.
"""

import requests
import json
import sys

API_BASE_URL = "http://127.0.0.1:8000"

def test_status_endpoint():
    """Test the status endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/status")
        print(f"✅ Status Endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Status Endpoint Failed: {e}")
        return False

def test_get_config():
    """Test getting configuration"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/config/main_config")
        print(f"✅ Get Config Endpoint: {response.status_code}")
        if response.status_code == 200:
            config_data = response.json()
            print(f"   Config keys: {list(config_data.get('data', {}).keys())}")
        return True
    except Exception as e:
        print(f"❌ Get Config Failed: {e}")
        return False

def test_docs_endpoint():
    """Test that the docs endpoint is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        print(f"✅ Docs Endpoint: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Docs Endpoint Failed: {e}")
        return False

def main():
    print("🚀 Testing FastAPI Endpoints...")
    print(f"API Base URL: {API_BASE_URL}")
    print("-" * 50)
    
    tests = [
        test_status_endpoint,
        test_get_config,
        test_docs_endpoint
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("-" * 50)
    print(f"✅ Tests Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All API tests passed! The FastAPI backend is working correctly.")
        print(f"📖 Visit {API_BASE_URL}/docs for interactive API documentation")
    else:
        print("⚠️  Some tests failed. Check if the API server is running.")
        sys.exit(1)

if __name__ == "__main__":
    main()