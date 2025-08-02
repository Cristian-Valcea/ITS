#!/usr/bin/env python3
"""
🎯 V3 TUNING EXECUTION SCRIPT
Quick execution script for V3 warm-start tuning

USAGE:
    python run_v3_tuning.py

PROCESS:
1. Load V3 gold standard model (409K steps)
2. Retrain 50K steps with tuned weights
3. Compare trading behavior
4. Generate analysis report
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Execute V3 tuning process"""
    
    logger.info("🎯 Starting V3 Tuning Process...")
    
    # Check if base model exists
    base_model_path = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
    
    if not os.path.exists(base_model_path):
        logger.error(f"❌ Base model not found: {base_model_path}")
        logger.info("Please ensure the V3 gold standard model is available")
        return
    
    logger.info(f"✅ Base model found: {base_model_path}")
    
    # Check if configuration exists
    config_path = "config/v3_tuned_warmstart.yml"
    
    if not os.path.exists(config_path):
        logger.error(f"❌ Configuration not found: {config_path}")
        logger.info("Please ensure the tuning configuration is available")
        return
    
    logger.info(f"✅ Configuration found: {config_path}")
    
    try:
        # Step 1: Run warm-start tuning
        logger.info("🚀 Step 1: Running warm-start tuning...")
        
        from scripts.v3_warmstart_tuning import main as run_tuning
        run_tuning()
        
        logger.info("✅ Step 1 complete: Tuning finished")
        
        # Step 2: Compare models (optional, requires manual path update)
        logger.info("📊 Step 2: Model comparison available")
        logger.info("To compare models, run: python scripts/compare_v3_tuning.py")
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 V3 TUNING PROCESS COMPLETE!")
        print("="*60)
        print("\n🎯 WHAT WAS DONE:")
        print("✅ Loaded V3 gold standard model (409K steps)")
        print("✅ Applied tuned weights:")
        print("   - Hold bonus: 0.01 → 0.0005 (20x reduction)")
        print("   - Ticket cost: $0.50 → $0.20 (60% reduction)")
        print("✅ Retrained 50K steps with warm-start")
        print("✅ Preserved existing 409K steps of learning")
        
        print("\n🔍 NEXT STEPS:")
        print("1. Check output in train_runs/v3_tuned_warmstart_50k_*/")
        print("2. Run comparison: python scripts/compare_v3_tuning.py")
        print("3. Analyze trading frequency changes")
        print("4. Deploy to paper trading if satisfactory")
        
        print("\n📊 EXPECTED OUTCOMES:")
        print("- Increased trading frequency (target: 12 → 25 trades/episode)")
        print("- Reduced holding percentage (target: 80% → 60%)")
        print("- Maintained profitability (acceptable Sharpe degradation)")
        print("- Better alpha signal utilization")
        
    except Exception as e:
        logger.error(f"❌ Tuning process failed: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure CUDA is available for GPU training")
        print("2. Check that all dependencies are installed")
        print("3. Verify data files are accessible")
        print("4. Check disk space for model checkpoints")
        raise

if __name__ == "__main__":
    main()