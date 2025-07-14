#!/usr/bin/env python3
"""
Test script for the fixed reward audit system.

Tests the fixes for:
1. VecEnv support (multiple parallel environments)
2. Memory pressure handling (disk dumping)
3. NaN correlation handling (degenerate cases)
"""

import sys
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reward_pnl_audit import RewardPnLAudit


def test_vecenv_support():
    """Test that VecEnv with multiple parallel environments is handled correctly."""
    print("🧪 Testing VecEnv support...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit = RewardPnLAudit(
            output_dir=temp_dir,
            memory_dump_threshold=100,  # Low threshold for testing
            verbose=False
        )
        
        # Mock training environment with 3 parallel environments
        audit.locals = {
            "rewards": [0.1, 0.2, -0.1],  # 3 environments
            "infos": [
                {"realized_pnl_step": 0.05, "total_pnl_step": 0.05},
                {"realized_pnl_step": 0.10, "total_pnl_step": 0.10},
                {"realized_pnl_step": -0.05, "total_pnl_step": -0.05}
            ],
            "actions": [1, 2, 0]
        }
        
        # Process step
        audit._on_step()
        
        # Check that all 3 environments were processed
        assert len(audit.current_episode_data) == 3, f"Expected 3 envs, got {len(audit.current_episode_data)}"
        
        # Check data for each environment
        for env_idx in range(3):
            env_data = audit.current_episode_data[env_idx]
            assert len(env_data['rewards']) == 1, f"Env {env_idx} should have 1 reward"
            assert len(env_data['realized_pnl']) == 1, f"Env {env_idx} should have 1 P&L"
        
        print("✅ VecEnv support test passed")


def test_memory_pressure_handling():
    """Test that memory pressure triggers disk dumping."""
    print("🧪 Testing memory pressure handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit = RewardPnLAudit(
            output_dir=temp_dir,
            memory_dump_threshold=5,  # Very low threshold for testing
            verbose=False
        )
        
        # Simulate many steps to trigger memory pressure
        for step in range(10):
            audit.locals = {
                "rewards": [0.1],
                "infos": [{"realized_pnl_step": 0.05, "total_pnl_step": 0.05}],
                "actions": [1]
            }
            audit._on_step()
        
        # Check that dump files were created
        dump_files = list(Path(temp_dir).glob("episode_dump_*.pkl"))
        assert len(dump_files) > 0, "Expected dump files to be created"
        
        # Check that dumped episodes are tracked
        assert len(audit.dumped_episodes) > 0, "Expected dumped episodes to be tracked"
        
        print("✅ Memory pressure handling test passed")


def test_nan_correlation_handling():
    """Test that NaN correlations (degenerate cases) are handled properly."""
    print("🧪 Testing NaN correlation handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit = RewardPnLAudit(
            output_dir=temp_dir,
            min_correlation_threshold=0.5,
            alert_episodes=2,
            verbose=False
        )
        
        # Create degenerate case: zero variance in rewards
        episode_data = {
            'rewards': [0.0, 0.0, 0.0, 0.0],  # Zero variance
            'realized_pnl': [0.1, 0.2, -0.1, 0.05],  # Non-zero variance
            'unrealized_pnl': [0.0, 0.0, 0.0, 0.0],
            'total_pnl': [0.1, 0.2, -0.1, 0.05],
            'fees': [0.01, 0.01, 0.01, 0.01],
            'net_pnl': [0.09, 0.19, -0.11, 0.04],
            'raw_rewards': [0.0, 0.0, 0.0, 0.0],
            'scaled_rewards': [0.0, 0.0, 0.0, 0.0],
            'actions': [1, 2, 0, 1],
            'portfolio_values': [1000, 1010, 1005, 1008],
            'timestamps': [None, None, None, None]
        }
        
        # Calculate metrics for degenerate case
        metrics = audit._calculate_episode_metrics(episode_data)
        
        # Check that correlation is NaN (not 0.0)
        assert np.isnan(metrics['step_correlation']), f"Expected NaN, got {metrics['step_correlation']}"
        assert np.isnan(metrics['total_pnl_correlation']), f"Expected NaN, got {metrics['total_pnl_correlation']}"
        
        # Test alert handling with NaN
        try:
            audit._check_correlation_alerts(metrics)
            # Should not raise exception, just log warning
        except Exception as e:
            assert False, f"NaN correlation handling should not raise exception: {e}"
        
        print("✅ NaN correlation handling test passed")


def test_safe_correlation_function():
    """Test the safe_correlation function specifically."""
    print("🧪 Testing safe_correlation function...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit = RewardPnLAudit(output_dir=temp_dir, verbose=False)
        
        # Test cases for safe_correlation
        test_cases = [
            # (x, y, expected_result_type)
            ([1, 2, 3], [1, 2, 3], "valid"),  # Perfect correlation
            ([1, 1, 1], [1, 2, 3], "nan"),    # Zero variance in x
            ([1, 2, 3], [1, 1, 1], "nan"),    # Zero variance in y
            ([1], [1], "nan"),                # Not enough data points
            ([], [], "nan"),                  # Empty arrays
        ]
        
        for x, y, expected_type in test_cases:
            episode_data = audit._create_empty_episode_data()
            episode_data['rewards'] = x
            episode_data['realized_pnl'] = y
            
            if len(x) > 0:
                # Fill other required fields
                for key in episode_data:
                    if key not in ['rewards', 'realized_pnl']:
                        episode_data[key] = [0] * len(x)
            
            try:
                metrics = audit._calculate_episode_metrics(episode_data)
                corr = metrics['step_correlation']
                
                if expected_type == "nan":
                    assert np.isnan(corr), f"Expected NaN for {x}, {y}, got {corr}"
                elif expected_type == "valid":
                    assert not np.isnan(corr), f"Expected valid correlation for {x}, {y}, got {corr}"
                    
            except Exception as e:
                if expected_type != "error":
                    assert False, f"Unexpected error for {x}, {y}: {e}"
        
        print("✅ safe_correlation function test passed")


def test_integration():
    """Test full integration with all fixes."""
    print("🧪 Testing full integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audit = RewardPnLAudit(
            output_dir=temp_dir,
            memory_dump_threshold=20,
            min_correlation_threshold=0.5,
            alert_episodes=3,
            verbose=False
        )
        
        # Simulate training with VecEnv, memory pressure, and degenerate cases
        episode_count = 0
        
        for step in range(30):
            if step < 10:
                # Normal case
                audit.locals = {
                    "rewards": [0.1, 0.2],
                    "infos": [
                        {"realized_pnl_step": 0.05, "total_pnl_step": 0.05},
                        {"realized_pnl_step": 0.10, "total_pnl_step": 0.10}
                    ],
                    "actions": [1, 2]
                }
            elif step < 20:
                # Degenerate case (zero variance)
                audit.locals = {
                    "rewards": [0.0, 0.0],
                    "infos": [
                        {"realized_pnl_step": 0.05, "total_pnl_step": 0.05},
                        {"realized_pnl_step": 0.10, "total_pnl_step": 0.10}
                    ],
                    "actions": [1, 2]
                }
            else:
                # Poor correlation case
                audit.locals = {
                    "rewards": [0.1, 0.2],
                    "infos": [
                        {"realized_pnl_step": -0.05, "total_pnl_step": -0.05},
                        {"realized_pnl_step": -0.10, "total_pnl_step": -0.10}
                    ],
                    "actions": [1, 2]
                }
            
            audit._on_step()
            
            # Simulate episode end every 5 steps
            if (step + 1) % 5 == 0:
                audit._on_rollout_end()
                episode_count += 1
        
        # Final report
        audit._on_training_end()
        
        # Check that files were created
        csv_file = Path(temp_dir) / "reward_pnl_audit.csv"
        summary_file = Path(temp_dir) / "audit_summary.txt"
        
        assert csv_file.exists(), "CSV file should be created"
        assert summary_file.exists(), "Summary file should be created"
        
        # Check that summary handles NaN values
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "NaN" in content or "degenerate" in content, "Summary should mention NaN/degenerate cases"
        
        print("✅ Full integration test passed")


def main():
    """Run all tests."""
    print("🎯 Testing Reward Audit System Fixes...")
    
    try:
        test_vecenv_support()
        test_memory_pressure_handling()
        test_nan_correlation_handling()
        test_safe_correlation_function()
        test_integration()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Fixes Verified:")
        print("✅ VecEnv support - Multiple parallel environments handled correctly")
        print("✅ Memory pressure - Automatic disk dumping prevents RAM issues")
        print("✅ NaN correlation - Degenerate cases made visible, not hidden")
        print("✅ Integration - All fixes work together seamlessly")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())