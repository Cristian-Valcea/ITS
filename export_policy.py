#!/usr/bin/env python3
"""
🎯 EXPORT POLICY WEIGHTS
Export successful single-ticker policy for dual-ticker warm-start
"""

import sys
from pathlib import Path
import argparse
import json
import zipfile
import logging
from datetime import datetime

import torch
import numpy as np
from sb3_contrib import RecurrentPPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_policy_weights(model_path: str, output_path: str, config_info: dict = None):
    """Export policy weights and metadata for dual-ticker warm-start"""
    
    logger.info(f"🎯 EXPORTING POLICY WEIGHTS")
    logger.info(f"   Source: {model_path}")
    logger.info(f"   Output: {output_path}")
    
    try:
        # Load the successful model
        model = RecurrentPPO.load(model_path)
        logger.info(f"✅ Model loaded successfully")
        
        # Extract policy network weights
        policy_state_dict = model.policy.state_dict()
        
        # Try to extract VecNormalize stats if available (important for real-data transition)
        vecnormalize_stats = None
        try:
            # These would come from the training environment
            # This is a placeholder - actual implementation would extract from training
            vecnormalize_stats = {
                'obs_rms_mean': None,  # Would be extracted from VecNormalize
                'obs_rms_var': None,   # Would be extracted from VecNormalize
                'ret_rms_mean': None,  # Would be extracted from VecNormalize
                'ret_rms_var': None,   # Would be extracted from VecNormalize
                'epsilon': 1e-8,
                'gamma': 0.999,
                'note': 'VecNormalize stats crucial for real-data transition'
            }
            logger.info("💡 VecNormalize stats placeholder added - extract from training environment")
        except Exception as e:
            logger.warning(f"Could not extract VecNormalize stats: {e}")
        
        # Separate shared weights from action head weights
        shared_weights = {}
        action_head_weights = {}
        
        for key, tensor in policy_state_dict.items():
            if 'action_net' in key or 'output' in key:
                # These are action-specific weights (3 actions → 9 actions)
                action_head_weights[key] = tensor
            else:
                # These are shared feature extraction weights (reusable)
                shared_weights[key] = tensor
        
        logger.info(f"   Shared weights: {len(shared_weights)} tensors")
        logger.info(f"   Action head weights: {len(action_head_weights)} tensors")
        
        # Prepare export package
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'RecurrentPPO',
            'source_model_path': model_path,
            'shared_weights': shared_weights,
            'action_head_weights': action_head_weights,
            'vecnormalize_stats': vecnormalize_stats,  # Added VecNormalize stats
            'config_info': config_info or {},
            'export_metadata': {
                'source_action_space': 3,  # Single-ticker: SELL, HOLD, BUY
                'target_action_space': 9,  # Dual-ticker: 3x3 matrix
                'obs_space_source': 13,    # Single-ticker + alpha
                'obs_space_target': 28,    # Dual-ticker + positions + alpha
                'warm_start_strategy': 'freeze_shared_retrain_action_head',
                'vecnormalize_note': 'Stats crucial for real-data transition'
            }
        }
        
        # Save as compressed archive
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save main export data
            import pickle
            zf.writestr('export_data.pkl', pickle.dumps(export_data))
            
            # Save metadata as JSON for easy inspection
            metadata = {k: v for k, v in export_data.items() if k not in ['shared_weights', 'action_head_weights']}
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # Save readable summary
            summary = f"""
🎯 POLICY EXPORT SUMMARY
========================

Export Date: {export_data['timestamp']}
Source Model: {model_path}
Model Type: {export_data['model_type']}

Architecture:
- Source Action Space: {export_data['export_metadata']['source_action_space']} (single-ticker)
- Target Action Space: {export_data['export_metadata']['target_action_space']} (dual-ticker)
- Source Obs Space: {export_data['export_metadata']['obs_space_source']}
- Target Obs Space: {export_data['export_metadata']['obs_space_target']}

Warm-Start Strategy:
{export_data['export_metadata']['warm_start_strategy']}

Weights Exported:
- Shared weights: {len(shared_weights)} tensors (feature extraction, LSTM)
- Action head weights: {len(action_head_weights)} tensors (3→9 action mapping)

Config Info:
{json.dumps(config_info or {}, indent=2)}
"""
            zf.writestr('README.txt', summary)
        
        logger.info(f"✅ Export completed successfully")
        logger.info(f"   Archive: {output_path}")
        logger.info(f"   Components: export_data.pkl, metadata.json, README.txt")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return False

def load_winning_config():
    """Load winning HPO configuration if available"""
    
    config_file = Path('winning_hpo_config.json')
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None

def find_model_path_from_config(config_info):
    """Extract model path from winning config"""
    
    if not config_info:
        return None
    
    # Try to construct model path from config info
    # This is a placeholder - the actual HPO script should save model path
    config_params = config_info.get('config', {})
    
    # Construct directory name from parameters
    n_steps = config_params.get('n_steps', 2048)
    lr = config_params.get('learning_rate', 7e-5)
    ent_coef = config_params.get('ent_coef', 0.002)
    
    # Format scientific notation properly
    lr_str = f"{lr:.0e}".replace('+', '').replace('-0', '-')
    ent_str = f"{ent_coef:.3f}".rstrip('0').rstrip('.')
    
    dir_name = f"singletick_ns{n_steps}_lr{lr_str}_ent{ent_str}_seed42"
    
    # Look for model files in logs directory
    possible_paths = [
        f"logs/{dir_name}/best_model.zip",
        f"logs/{dir_name}/final_model.zip", 
        f"logs/{dir_name}/model.zip",
        f"{dir_name}.zip",
        "latest_model.zip"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"✅ Found model at: {path}")
            return path
    
    logger.warning(f"⚠️ No model found for config, tried: {possible_paths}")
    return None

def main():
    """Main export function"""
    
    parser = argparse.ArgumentParser(description='Export policy weights for dual-ticker warm-start')
    parser.add_argument('--model_path', type=str, 
                       help='Path to trained model file (.zip) - auto-detected if not provided')
    parser.add_argument('--output', type=str, default='models/singleticker_gatepass.zip',
                       help='Output path for exported weights')
    parser.add_argument('--config_file', type=str, 
                       help='Optional config file with training details')
    
    args = parser.parse_args()
    
    # Load config info if available
    config_info = None
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_info = json.load(f)
    else:
        # Try to load winning config automatically
        config_info = load_winning_config()
    
    # Auto-detect model path if not provided
    model_path = args.model_path
    if not model_path:
        model_path = find_model_path_from_config(config_info)
        if not model_path:
            logger.error("❌ No model path provided and auto-detection failed")
            logger.error("   Use --model_path argument or ensure winning_hpo_config.json exists")
            sys.exit(1)
        logger.info(f"🔍 Auto-detected model path: {model_path}")
    
    # Verify model path exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export policy weights
    success = export_policy_weights(model_path, str(output_path), config_info)
    
    if success:
        print(f"✅ Policy exported successfully to {output_path}")
        print(f"Ready for dual-ticker warm-start!")
    else:
        print(f"❌ Export failed")
        sys.exit(1)

if __name__ == "__main__":
    main()