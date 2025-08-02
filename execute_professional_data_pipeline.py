#!/usr/bin/env python3
"""
Execute Professional Data Pipeline - Complete Implementation
Based on institutional standards for intraday RL trading systems

This script implements the complete data methodology grid:
✅ 36-month data horizon (3 market regimes)
✅ 70/15/15 time-ordered splits
✅ Earnings/FOMC/holiday filtering
✅ Volume/price guardrails
✅ Professional validation gates
✅ Rate-limited Polygon API integration

Usage:
    python execute_professional_data_pipeline.py --full-pipeline
    python execute_professional_data_pipeline.py --validate-only
    python execute_professional_data_pipeline.py --dry-run
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from data.professional_data_pipeline import ProfessionalDataPipeline
from validation.data_quality_gates import DataQualityGates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/professional_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create necessary directory structure"""
    directories = [
        'data/raw/parquet',
        'data/processed', 
        'logs',
        'models/professional',
        'reports/validation'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")


def validate_prerequisites():
    """Validate system prerequisites"""
    logger.info("🔍 Validating prerequisites...")
    
    issues = []
    
    # Check config file exists
    config_path = Path("config/data_methodology.yaml")
    if not config_path.exists():
        issues.append(f"Config file missing: {config_path}")
    
    # Check secrets access
    try:
        from secrets_helper import SecretsHelper
        api_key = SecretsHelper.get_polygon_api_key()
        if not api_key:
            issues.append("Polygon API key not configured")
    except Exception as e:
        issues.append(f"Secrets access failed: {e}")
    
    # Check Python environment
    try:
        import pandas as pd
        import numpy as np
        import requests
        import yaml
        logger.info("✅ Required packages available")
    except ImportError as e:
        issues.append(f"Missing required package: {e}")
    
    if issues:
        logger.error("❌ Prerequisites validation failed:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    
    logger.info("✅ Prerequisites validation passed")
    return True


def execute_data_pipeline(dry_run: bool = False):
    """Execute the complete professional data pipeline"""
    
    logger.info("🚀 EXECUTING PROFESSIONAL DATA PIPELINE")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = ProfessionalDataPipeline("config/data_methodology.yaml")
    
    if dry_run:
        logger.info("🧪 DRY RUN MODE - No data will be downloaded")
        
        # Calculate and display splits
        train_split, val_split, test_split = pipeline.calculate_data_splits()
        
        logger.info("📊 Data split calculation complete:")
        logger.info(f"   🎯 Train: {train_split.start_date.date()} → {train_split.end_date.date()}")
        logger.info(f"   🎯 Val:   {val_split.start_date.date()} → {val_split.end_date.date()}")
        logger.info(f"   🎯 Test:  {test_split.start_date.date()} → {test_split.end_date.date()}")
        
        return {"status": "dry_run_complete", "splits_calculated": True}
    
    # Execute full pipeline
    results = pipeline.execute_full_pipeline()
    
    # Calculate total bars processed
    total_bars = sum(
        sum(split_data['bars_count'] for split_data in symbol_data.values())
        for symbol_data in results['splits'].values()
    )
    
    logger.info("✅ PROFESSIONAL DATA PIPELINE COMPLETE")
    logger.info(f"📊 Total bars processed: {total_bars:,}")
    logger.info(f"📁 Results saved to: data/processed/")
    
    return results


def execute_validation_gates(model_path: str = None):
    """Execute professional validation gates"""
    
    logger.info("🔍 EXECUTING VALIDATION GATES")
    logger.info("=" * 40)
    
    # Initialize validator
    validator = DataQualityGates("config/data_methodology.yaml")
    
    # Run complete validation suite
    suite = validator.run_complete_validation_suite(model_path)
    
    # Save validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/validation/validation_report_{timestamp}.json"
    validator.save_validation_report(suite, report_path)
    
    # Send notifications
    validator.send_slack_notification(suite)
    
    return suite


def generate_implementation_summary():
    """Generate summary of what has been implemented"""
    
    logger.info("📋 IMPLEMENTATION SUMMARY")
    logger.info("=" * 50)
    
    implemented_features = [
        "✅ 36-month data horizon with regime coverage (2022-08 to present)",
        "✅ 70/15/15 time-ordered train/val/test splits (leak-proof)",
        "✅ Earnings exclusion (day-1, day0, day+1) via Polygon calendar",
        "✅ FOMC announcement filtering (14:00-15:00 ET)",
        "✅ Holiday and half-day exclusions (configurable YAML)",
        "✅ Extreme volatility filtering (5% max bar-to-bar returns)",
        "✅ Volume guardrails (20k shares median 1-min volume)",
        "✅ Price guardrails ($5 minimum price threshold)",
        "✅ Rate-limited Polygon API with token bucket (5 req/min)",
        "✅ Professional validation gates (CI/CD ready)",
        "✅ OHLC data integrity validation",
        "✅ Parquet storage with compression",
        "✅ Cross-asset validation framework",
        "✅ Comprehensive logging and error handling"
    ]
    
    for feature in implemented_features:
        logger.info(f"   {feature}")
    
    logger.info("")
    logger.info("🎯 NEXT STEPS:")
    next_steps = [
        "1. Run: python execute_professional_data_pipeline.py --full-pipeline",
        "2. Validate data quality with CI/CD gates",
        "3. Implement rolling-window cross-validation training",
        "4. Set up nightly training cadence (25k steps)",
        "5. Configure weekly full retraining (200k steps)",
        "6. Deploy model with professional risk limits"
    ]
    
    for step in next_steps:
        logger.info(f"   {step}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Professional Data Pipeline for Intraday RL Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline execution
    python execute_professional_data_pipeline.py --full-pipeline
    
    # Dry run (calculate splits only)
    python execute_professional_data_pipeline.py --dry-run
    
    # Validation only
    python execute_professional_data_pipeline.py --validate-only
    
    # Show implementation summary
    python execute_professional_data_pipeline.py --summary
        """
    )
    
    parser.add_argument("--full-pipeline", action="store_true", 
                       help="Execute complete data pipeline with real data download")
    parser.add_argument("--dry-run", action="store_true",
                       help="Calculate data splits only, no data download")  
    parser.add_argument("--validate-only", action="store_true",
                       help="Run validation gates only")
    parser.add_argument("--summary", action="store_true",
                       help="Show implementation summary")
    parser.add_argument("--model-path", help="Path to model for validation")
    parser.add_argument("--config", default="config/data_methodology.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Show summary if requested
    if args.summary:
        generate_implementation_summary()
        return
    
    # Validate prerequisites
    if not validate_prerequisites():
        logger.error("❌ Prerequisites validation failed, aborting")
        return 1
    
    try:
        # Execute based on mode
        if args.validate_only:
            # Run validation gates only
            suite = execute_validation_gates(args.model_path)
            
            if not suite.overall_passed:
                logger.error("❌ Validation gates failed")
                return 1
                
        elif args.dry_run:
            # Dry run mode
            results = execute_data_pipeline(dry_run=True)
            
        elif args.full_pipeline:
            # Full pipeline execution
            logger.info("🚀 STARTING FULL PROFESSIONAL PIPELINE")
            
            # Step 1: Execute data pipeline
            pipeline_results = execute_data_pipeline(dry_run=False)
            
            # Step 2: Run validation gates
            validation_suite = execute_validation_gates(args.model_path)
            
            if not validation_suite.overall_passed:
                logger.error("❌ Post-pipeline validation failed")
                return 1
            
            logger.info("🎉 PROFESSIONAL PIPELINE COMPLETE & VALIDATED")
            logger.info("📋 Ready for model training on real market data")
            
        else:
            # Show help if no mode specified
            parser.print_help()
            logger.info("")
            logger.info("💡 Tip: Start with --dry-run to validate configuration")
            logger.info("💡 Then use --full-pipeline for complete execution")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())