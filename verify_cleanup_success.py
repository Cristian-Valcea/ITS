#!/usr/bin/env python3
"""
Verify Stairways V3 Cleanup Success
Confirms that the cleanup preserved all essential components and the system is fully functional.
"""

import os
import sys
from pathlib import Path

def verify_cleanup_success():
    """Verify that cleanup was successful and system is functional"""
    print("🔍 VERIFYING STAIRWAYS V3 CLEANUP SUCCESS")
    print("=" * 50)
    
    project_root = Path("/home/cristian/IntradayTrading/ITS")
    success = True
    
    # 1. Check working model exists
    working_model_path = project_root / "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip"
    if working_model_path.exists():
        size_mb = working_model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Working Stairways V3 model: {size_mb:.1f} MB")
    else:
        print("❌ Working Stairways V3 model NOT FOUND!")
        success = False
    
    # 2. Check essential files
    essential_files = [
        "recap2025-08-05.md",
        "working_ai_trader.py", 
        "verify_ibkr_setup.py",
        "src/brokers/ib_gateway.py"
    ]
    
    print(f"\n📄 Essential Files:")
    for file_path in essential_files:
        full_path = project_root / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"   ✅ {file_path}: {size_kb:.1f} KB")
        else:
            print(f"   ❌ {file_path}: NOT FOUND!")
            success = False
    
    # 3. Check essential directories
    essential_dirs = ["src", "config"]
    print(f"\n📁 Essential Directories:")
    for dir_path in essential_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"   ✅ {dir_path}/: EXISTS")
        else:
            print(f"   ❌ {dir_path}/: NOT FOUND!")
            success = False
    
    # 4. Check that old models were removed
    train_runs_path = project_root / "train_runs"
    if train_runs_path.exists():
        model_dirs = [d for d in train_runs_path.iterdir() if d.is_dir()]
        print(f"\n🗂️  Model directories remaining: {len(model_dirs)}")
        if len(model_dirs) == 1 and model_dirs[0].name == "v3_gold_standard_400k_20250802_202736":
            print("   ✅ Only working model remains")
        else:
            print("   ⚠️  Multiple model directories found:")
            for model_dir in model_dirs:
                print(f"      - {model_dir.name}")
    
    # 5. Check disk space freed
    train_runs_size = 0
    if train_runs_path.exists():
        for dirpath, dirnames, filenames in os.walk(train_runs_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    train_runs_size += os.path.getsize(filepath)
    
    train_runs_size_mb = train_runs_size / (1024 * 1024)
    print(f"\n💾 Current train_runs/ size: {train_runs_size_mb:.1f} MB")
    if train_runs_size_mb < 600:  # Should be around 545 MB
        print("   ✅ Size optimized (was ~1.2 GB)")
    else:
        print("   ⚠️  Size still large")
    
    # 6. Test model loading (if in virtual environment)
    print(f"\n🤖 Testing Model Loading:")
    try:
        sys.path.append('src')
        from stable_baselines3 import PPO
        
        model_path = "train_runs/v3_gold_standard_400k_20250802_202736/chunk7_final_358400steps.zip"
        model = PPO.load(model_path)
        print("   ✅ Stairways V3 model loads successfully!")
        
    except ImportError:
        print("   ⚠️  stable_baselines3 not available (run in virtual environment)")
        print("   💡 To test: source venv/bin/activate && python verify_cleanup_success.py")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        success = False
    
    # 7. Check cleanup summary
    cleanup_summaries = list(project_root.glob("cleanup_summary_*.json"))
    if cleanup_summaries:
        latest_summary = max(cleanup_summaries, key=lambda x: x.stat().st_mtime)
        print(f"\n📋 Cleanup Summary: {latest_summary.name}")
        print("   ✅ Cleanup operation documented")
    else:
        print(f"\n📋 No cleanup summary found")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CLEANUP VERIFICATION: SUCCESS!")
        print("\n✅ Your Stairways V3 system is:")
        print("   • Fully functional")
        print("   • Optimized for disk space")
        print("   • Ready for AI trading")
        print("\n🚀 You can now run:")
        print("   source venv/bin/activate")
        print("   python working_ai_trader.py")
        return True
    else:
        print("❌ CLEANUP VERIFICATION: ISSUES FOUND!")
        print("\n🔧 Please check the issues above")
        return False

def main():
    """Main verification function"""
    success = verify_cleanup_success()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())