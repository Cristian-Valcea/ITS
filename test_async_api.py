#!/usr/bin/env python3
"""
Test the Phase 2 async training functionality
"""

import requests
import json
import time

API_BASE_URL = "http://127.0.0.1:8000"

def test_async_training():
    """Test async training endpoint"""
    print("🚀 Testing Async Training...")
    
    # Training request
    request_data = {
        "symbol": "AAPL",
        "start_date": "2023-01-01", 
        "end_date": "2023-01-05",  # Short period for quick test
        "interval": "1min",
        "use_cached_data": True,
        "run_evaluation_after_train": False
    }
    
    try:
        # 1. Start async training
        response = requests.post(f"{API_BASE_URL}/api/v1/pipelines/train", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training started!")
            print(f"Task ID: {result.get('task_id')}")
            
            # 2. Check task status
            task_id = result.get('task_id')
            if task_id:
                status_response = requests.get(f"{API_BASE_URL}/api/v1/pipelines/train/status/{task_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"✅ Status check working!")
                    print(f"Current status: {status.get('status')}")
                    return True
        else:
            print(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    print("Testing Phase 2 Async API...")
    if test_async_training():
        print("🎉 Phase 2 async functionality working!")
    else:
        print("❌ Phase 2 test failed")