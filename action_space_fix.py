#!/usr/bin/env python3
"""
🔧 ACTION SPACE FIX
Fix the action space corruption and implement the 5-action system
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def analyze_action_space_requirements():
    """Analyze what the correct action space should be."""
    
    print("🔍 ACTION SPACE ANALYSIS")
    print("=" * 40)
    
    print("Current system (9 actions):")
    print("   0: SELL_NVDA + SELL_MSFT")
    print("   1: SELL_NVDA + HOLD_MSFT") 
    print("   2: SELL_NVDA + BUY_MSFT")
    print("   3: HOLD_NVDA + SELL_MSFT")
    print("   4: HOLD_NVDA + HOLD_MSFT")  # This is the HOLD action
    print("   5: HOLD_NVDA + BUY_MSFT")
    print("   6: BUY_NVDA + SELL_MSFT")
    print("   7: BUY_NVDA + HOLD_MSFT")
    print("   8: BUY_NVDA + BUY_MSFT")
    
    print("\nUser's expected system (5 actions):")
    print("   0: Buy A (NVDA)")
    print("   1: Sell A (NVDA)")
    print("   2: Buy B (MSFT)")
    print("   3: Sell B (MSFT)")
    print("   4: Hold (both)")
    
    print("\n🤔 ANALYSIS:")
    print("The user expects a simplified 5-action system where:")
    print("- Actions 0-3 are individual ticker actions")
    print("- Action 4 is hold both")
    print("- This is different from the current 9-action dual system")
    
    print("\n🚀 SOLUTION OPTIONS:")
    print("1. Create a new 5-action environment")
    print("2. Map the 9-action system to 5 actions")
    print("3. Use action 4 (HOLD_BOTH) as the primary hold action")
    
    return True

def create_5_action_environment():
    """Create a 5-action version of the environment."""
    
    print("\n🔧 CREATING 5-ACTION ENVIRONMENT")
    print("=" * 40)
    
    # Read the current environment
    env_path = Path("src/gym_env/dual_ticker_trading_env_v3_enhanced.py")
    
    if not env_path.exists():
        print(f"❌ Environment file not found: {env_path}")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Create the 5-action version
    new_content = content.replace(
        "self.action_space = spaces.Discrete(9)",
        "self.action_space = spaces.Discrete(5)"
    )
    
    # Update the decode action method
    old_decode = '''    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action (IDENTICAL TO V3)"""
        nvda_action = action // 3
        msft_action = action % 3
        return nvda_action, msft_action'''
    
    new_decode = '''    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode 5-action system: 0=Buy A, 1=Sell A, 2=Buy B, 3=Sell B, 4=Hold Both"""
        if action == 0:  # Buy A (NVDA)
            return 2, 1  # Buy NVDA, Hold MSFT
        elif action == 1:  # Sell A (NVDA)
            return 0, 1  # Sell NVDA, Hold MSFT
        elif action == 2:  # Buy B (MSFT)
            return 1, 2  # Hold NVDA, Buy MSFT
        elif action == 3:  # Sell B (MSFT)
            return 1, 0  # Hold NVDA, Sell MSFT
        elif action == 4:  # Hold Both
            return 1, 1  # Hold NVDA, Hold MSFT
        else:
            # Fallback to hold both for invalid actions
            return 1, 1'''
    
    new_content = new_content.replace(old_decode, new_decode)
    
    # Save the fixed environment
    fixed_env_path = Path("src/gym_env/dual_ticker_trading_env_v3_enhanced_5action.py")
    
    with open(fixed_env_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Created 5-action environment: {fixed_env_path.name}")
    
    return True

def main():
    """Main function."""
    
    print("🚀 ACTION SPACE FIX UTILITY")
    print("=" * 50)
    
    # Analyze the requirements
    analyze_action_space_requirements()
    
    # Create the 5-action environment
    success = create_5_action_environment()
    
    if success:
        print("\n✅ ACTION SPACE FIX COMPLETE")
        print("🔧 Next steps:")
        print("1. Update training scripts to use 5-action environment")
        print("2. Increase base_hold_bonus to 0.020")
        print("3. Add action space validation")
        print("4. Retry Cycle 7 with fixed configuration")
    else:
        print("\n❌ ACTION SPACE FIX FAILED")
        print("🔧 Manual intervention required")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)