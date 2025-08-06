#!/usr/bin/env python3
"""
🔬 ALL CHECKPOINTS VALIDATION SCRIPT
Phase 1A: Management Enhanced Validation

Validates that ALL checkpoints pass the tightened criteria:
- ALL three checkpoints must score ≥0.5 reward
- NONE can fall below 0.3 reward floor
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_all_checkpoints(results: List[float], min_reward: float, min_floor: float) -> Dict[str, Any]:
    """
    Validate all checkpoints against tightened criteria
    
    Args:
        results: List of mean rewards from each checkpoint
        min_reward: Minimum reward threshold (0.5)
        min_floor: Minimum floor threshold (0.3)
    
    Returns:
        Validation results dictionary
    """
    
    logger.info(f"🔍 Validating {len(results)} checkpoints...")
    logger.info(f"   Minimum reward threshold: {min_reward}")
    logger.info(f"   Minimum floor threshold: {min_floor}")
    
    # Convert to numpy array for easier processing
    results_array = np.array(results)
    
    # Check individual criteria
    all_pass_threshold = np.all(results_array >= min_reward)
    none_below_floor = np.all(results_array >= min_floor)
    
    # Calculate statistics
    mean_reward = np.mean(results_array)
    std_reward = np.std(results_array)
    min_result = np.min(results_array)
    max_result = np.max(results_array)
    
    # Count passes and failures
    pass_threshold_count = np.sum(results_array >= min_reward)
    below_floor_count = np.sum(results_array < min_floor)
    
    # Overall validation result
    validation_passed = all_pass_threshold and none_below_floor
    
    validation_results = {
        'validation_passed': validation_passed,
        'all_pass_threshold': all_pass_threshold,
        'none_below_floor': none_below_floor,
        
        # Statistics
        'num_checkpoints': len(results),
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward_observed': min_result,
        'max_reward_observed': max_result,
        
        # Counts
        'pass_threshold_count': int(pass_threshold_count),
        'below_floor_count': int(below_floor_count),
        
        # Individual results
        'checkpoint_results': results,
        
        # Criteria
        'min_reward_threshold': min_reward,
        'min_floor_threshold': min_floor
    }
    
    # Log results
    logger.info(f"📊 Validation Results:")
    logger.info(f"   Overall validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    logger.info(f"   All pass ≥{min_reward}: {'✅' if all_pass_threshold else '❌'} ({pass_threshold_count}/{len(results)})")
    logger.info(f"   None below {min_floor}: {'✅' if none_below_floor else '❌'} ({below_floor_count} below floor)")
    logger.info(f"   Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    logger.info(f"   Range: [{min_result:.3f}, {max_result:.3f}]")
    
    # Detailed checkpoint analysis
    for i, result in enumerate(results):
        checkpoint_name = f"checkpoint_{[5, 10, 15][i]}k" if i < 3 else f"checkpoint_{i+1}"
        status = "✅" if result >= min_reward else ("⚠️" if result >= min_floor else "❌")
        logger.info(f"   {checkpoint_name}: {result:.3f} {status}")
    
    return validation_results

def save_validation_results(results: Dict[str, Any], output_path: str):
    """Save validation results to CSV file"""
    
    # Create summary for CSV
    summary_data = {
        'validation_passed': [results['validation_passed']],
        'all_pass_threshold': [results['all_pass_threshold']],
        'none_below_floor': [results['none_below_floor']],
        'num_checkpoints': [results['num_checkpoints']],
        'mean_reward': [results['mean_reward']],
        'std_reward': [results['std_reward']],
        'min_reward_observed': [results['min_reward_observed']],
        'max_reward_observed': [results['max_reward_observed']],
        'pass_threshold_count': [results['pass_threshold_count']],
        'below_floor_count': [results['below_floor_count']],
        'min_reward_threshold': [results['min_reward_threshold']],
        'min_floor_threshold': [results['min_floor_threshold']]
    }
    
    # Add individual checkpoint results
    for i, result in enumerate(results['checkpoint_results']):
        checkpoint_name = f"checkpoint_{[5, 10, 15][i]}k_reward" if i < 3 else f"checkpoint_{i+1}_reward"
        summary_data[checkpoint_name] = [result]
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Validation results saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Validate All Checkpoints - Management Enhanced')
    parser.add_argument('--results', nargs='+', type=float, required=True, 
                       help='Mean reward results from each checkpoint')
    parser.add_argument('--min_reward', type=float, default=0.5,
                       help='Minimum reward threshold (default: 0.5)')
    parser.add_argument('--min_floor', type=float, default=0.3,
                       help='Minimum floor threshold (default: 0.3)')
    parser.add_argument('--output', required=True,
                       help='Path to save validation results CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.results) == 0:
        logger.error("❌ No checkpoint results provided")
        sys.exit(1)
    
    if args.min_reward <= 0 or args.min_floor <= 0:
        logger.error("❌ Invalid threshold values")
        sys.exit(1)
    
    if args.min_floor > args.min_reward:
        logger.error("❌ Floor threshold cannot be higher than reward threshold")
        sys.exit(1)
    
    logger.info(f"🔬 Starting Phase 1A checkpoint validation...")
    logger.info(f"   Checkpoint results: {args.results}")
    logger.info(f"   Criteria: ALL ≥{args.min_reward}, NONE <{args.min_floor}")
    
    try:
        # Run validation
        validation_results = validate_all_checkpoints(
            args.results, 
            args.min_reward, 
            args.min_floor
        )
        
        # Save results
        save_validation_results(validation_results, args.output)
        
        # Exit with appropriate code
        if validation_results['validation_passed']:
            logger.info("🎉 Phase 1A validation PASSED - all checkpoints meet criteria!")
            sys.exit(0)
        else:
            logger.error("💥 Phase 1A validation FAILED - criteria not met")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()