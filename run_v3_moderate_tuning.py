#!/usr/bin/env python3
"""
🎯 V3 MODERATE TUNING RUNNER
Conservative parameter adjustments to fix over-aggressive trading behavior
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    logger.info("🎯 Starting V3 Moderate Tuning Process...")
    
    # Check for required files
    base_model = "train_runs/v3_gold_standard_400k_20250802_202736/v3_gold_standard_final_409600steps.zip"
    config_file = "config/v3_moderate_tuning.yml"
    
    if not os.path.exists(base_model):
        logger.error(f"❌ Base model not found: {base_model}")
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("1. Ensure the original V3 gold standard model exists")
        logger.info("2. Check the path is correct")
        return
    
    if not os.path.exists(config_file):
        logger.error(f"❌ Configuration not found: {config_file}")
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("1. Ensure the moderate tuning config exists")
        logger.info("2. Check the config file was created correctly")
        return
    
    logger.info("✅ Base model found: " + base_model)
    logger.info("✅ Configuration found: " + config_file)
    
    # Step 1: Run moderate tuning
    logger.info("🚀 Step 1: Running moderate tuning...")
    logger.info("📊 Parameters:")
    logger.info("   Hold bonus: 0.01 → 0.001 (10x reduction)")
    logger.info("   Ticket cost: $0.50 → $0.50 (unchanged)")
    logger.info("   Training: 409K → 459K steps (+50K)")
    
    try:
        # Import and run the moderate tuning
        from scripts.v3_moderate_tuning import main as run_moderate_tuning
        
        logger.info("🎯 Starting moderate tuning with conservative parameters...")
        run_moderate_tuning()
        
        logger.info("✅ Step 1 completed: Moderate tuning finished")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("1. Ensure the moderate tuning script exists")
        logger.info("2. Check all dependencies are installed")
        return
        
    except Exception as e:
        logger.error(f"❌ Moderate tuning failed: {str(e)}")
        logger.info("🔧 TROUBLESHOOTING:")
        logger.info("1. Check GPU memory availability")
        logger.info("2. Verify data files are accessible")
        logger.info("3. Check disk space for model checkpoints")
        return
    
    # Step 2: Quick validation
    logger.info("🚀 Step 2: Quick validation...")
    
    try:
        # Check if output was generated
        from glob import glob
        moderate_runs = glob("train_runs/v3_moderate_tuning_50k_*")
        
        if moderate_runs:
            latest_run = sorted(moderate_runs)[-1]
            logger.info(f"✅ Moderate tuning output found: {latest_run}")
            
            # Check for key files
            best_model = f"{latest_run}/best_model.zip"
            if os.path.exists(best_model):
                logger.info("✅ Best model saved successfully")
            else:
                logger.warning("⚠️ Best model not found")
            
            # Quick episode analysis
            train_monitor = f"{latest_run}/train_monitor.csv"
            if os.path.exists(train_monitor):
                import pandas as pd
                df = pd.read_csv(train_monitor, comment='#')
                avg_length = df['l'].mean()
                early_term_rate = (df['l'] < 100).sum() / len(df) * 100
                
                logger.info(f"📊 Quick validation results:")
                logger.info(f"   Average episode length: {avg_length:.1f}")
                logger.info(f"   Early termination rate: {early_term_rate:.1f}%")
                
                if avg_length > 200 and early_term_rate < 50:
                    logger.info("✅ IMPROVEMENT: Episodes are longer and more stable!")
                elif avg_length > 100:
                    logger.info("🔄 PARTIAL IMPROVEMENT: Some progress made")
                else:
                    logger.warning("⚠️ LIMITED IMPROVEMENT: May need further adjustment")
            
        else:
            logger.warning("⚠️ No moderate tuning output found")
            
    except Exception as e:
        logger.warning(f"⚠️ Validation failed: {str(e)}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("🎯 V3 MODERATE TUNING COMPLETED")
    logger.info("=" * 60)
    logger.info("📊 CHANGES APPLIED:")
    logger.info("   Hold bonus: 0.01 → 0.001 (moderate 10x reduction)")
    logger.info("   Ticket cost: $0.50 → $0.50 (unchanged - maintains friction)")
    logger.info("   Training: +50K steps from proven 409K base")
    logger.info("")
    logger.info("🔍 NEXT STEPS:")
    logger.info("   1. Run evaluation: python analyze_moderate_tuning.py")
    logger.info("   2. Compare with original: python compare_moderate_vs_original.py")
    logger.info("   3. If successful, proceed to paper trading validation")
    logger.info("")
    logger.info("🎉 Moderate tuning process completed!")

if __name__ == "__main__":
    main()