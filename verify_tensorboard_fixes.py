#!/usr/bin/env python3
"""
Verify TensorBoard optimization and episode summary enhancements
"""

import sys
sys.path.append('src')
import os

def verify_episode_summary_enhancements():
    """Verify that episode summary code includes Sharpe and Sortino ratios"""
    
    print('🧪 VERIFYING EPISODE SUMMARY ENHANCEMENTS')
    print('=' * 60)
    
    env_file = 'src/gym_env/intraday_trading_env.py'
    
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for the new risk-adjusted metrics
        checks = [
            ('_calculate_risk_adjusted_returns', 'Risk-adjusted calculation method'),
            ('sharpe_ratio', 'Sharpe ratio in episode summary'),
            ('sortino_ratio', 'Sortino ratio in episode summary'),
            ('volatility', 'Volatility in episode summary'),
            ('Risk-Adjusted: Sharpe=', 'Risk metrics in logging')
        ]
        
        all_passed = True
        
        for check_text, description in checks:
            if check_text in content:
                print(f'✅ {description}: Found')
            else:
                print(f'❌ {description}: Missing')
                all_passed = False
        
        if all_passed:
            print('✅ Episode summary enhancements: SUCCESS')
            return True
        else:
            print('❌ Episode summary enhancements: INCOMPLETE')
            return False
            
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return False

def verify_curriculum_callback_frequency():
    """Verify that curriculum callback has frequency control"""
    
    print('\n🧪 VERIFYING CURRICULUM CALLBACK FREQUENCY CONTROL')
    print('=' * 60)
    
    callback_file = 'src/training/core/curriculum_callback.py'
    
    try:
        with open(callback_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for frequency control
        if 'current_episode % 10 == 0' in content:
            print('✅ Frequency control found: "current_episode % 10 == 0"')
            
            # Check context
            if 'Log to tensorboard if available (with frequency control)' in content:
                print('✅ Proper context: TensorBoard logging with frequency control')
                print('✅ Curriculum callback frequency control: SUCCESS')
                return True
            else:
                print('❌ Frequency control found but context unclear')
                return False
        else:
            print('❌ No frequency control found')
            return False
            
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return False

def verify_risk_callback_frequency():
    """Verify that risk callbacks already have frequency control"""
    
    print('\n🧪 VERIFYING RISK CALLBACK FREQUENCY CONTROL')
    print('=' * 60)
    
    files_to_check = [
        'src/training/trainer_agent.py',
        'src/training/core/risk_callbacks.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for existing frequency control
            if 'self.num_timesteps % self.log_freq == 0' in content:
                print(f'✅ {os.path.basename(file_path)}: Has proper frequency control')
            elif 'num_timesteps % log_freq' in content:
                print(f'✅ {os.path.basename(file_path)}: Has frequency control (variant)')
            else:
                print(f'❌ {os.path.basename(file_path)}: No frequency control found')
                all_good = False
                
        except Exception as e:
            print(f'❌ Error reading {file_path}: {e}')
            all_good = False
    
    if all_good:
        print('✅ Risk callback frequency control: SUCCESS')
        return True
    else:
        print('❌ Risk callback frequency control: ISSUES FOUND')
        return False

def verify_tensorboard_monitoring_frequency():
    """Verify that TensorBoard monitoring has frequency control"""
    
    print('\n🧪 VERIFYING TENSORBOARD MONITORING FREQUENCY')
    print('=' * 60)
    
    monitoring_file = 'src/training/core/tensorboard_monitoring.py'
    
    try:
        with open(monitoring_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for audit frequency control
        if 'audit_frequency' in content and 'last_audit_step' in content:
            print('✅ TensorBoard monitoring has audit frequency control')
            print('✅ TensorBoard monitoring frequency: SUCCESS')
            return True
        else:
            print('❌ TensorBoard monitoring frequency control not found')
            return False
            
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return False

def create_summary_report():
    """Create a summary of all optimizations"""
    
    print('\n📊 OPTIMIZATION SUMMARY REPORT')
    print('=' * 60)
    
    optimizations = [
        {
            'name': 'Episode Summary Sharpe & Sortino',
            'description': 'Added risk-adjusted metrics to episode summaries',
            'benefit': 'Better offline hyperparameter sweeps',
            'files_modified': ['src/gym_env/intraday_trading_env.py']
        },
        {
            'name': 'Curriculum Callback Frequency Control',
            'description': 'Reduced TensorBoard logging from every episode to every 10 episodes',
            'benefit': '10x reduction in TensorBoard writes',
            'files_modified': ['src/training/core/curriculum_callback.py']
        },
        {
            'name': 'Risk Callback Frequency Control',
            'description': 'Existing frequency control verified (log_freq parameter)',
            'benefit': 'Controlled TensorBoard logging frequency',
            'files_modified': ['Already implemented']
        },
        {
            'name': 'TensorBoard Monitoring Frequency',
            'description': 'Existing audit frequency control verified',
            'benefit': 'Controlled monitoring overhead',
            'files_modified': ['Already implemented']
        }
    ]
    
    for i, opt in enumerate(optimizations, 1):
        print(f'{i}. {opt["name"]}')
        print(f'   Description: {opt["description"]}')
        print(f'   Benefit: {opt["benefit"]}')
        print(f'   Files: {", ".join(opt["files_modified"])}')
        print()
    
    print('🎯 EXPECTED IMPACT:')
    print('   📊 Episode summaries now include Sharpe & Sortino ratios')
    print('   ⚡ TensorBoard logging reduced by ~90% (curriculum callback)')
    print('   🎯 Risk callbacks already properly controlled')
    print('   📈 Better data for offline hyperparameter optimization')

if __name__ == '__main__':
    print('🎯 TENSORBOARD OPTIMIZATION VERIFICATION')
    print('=' * 80)
    
    # Verify all optimizations
    test1_passed = verify_episode_summary_enhancements()
    test2_passed = verify_curriculum_callback_frequency()
    test3_passed = verify_risk_callback_frequency()
    test4_passed = verify_tensorboard_monitoring_frequency()
    
    print('\n🎯 FINAL VERIFICATION RESULTS:')
    print('=' * 50)
    
    if test1_passed and test2_passed and test3_passed and test4_passed:
        print('✅ ALL OPTIMIZATIONS VERIFIED!')
        create_summary_report()
    else:
        print('❌ SOME OPTIMIZATIONS FAILED VERIFICATION:')
        if not test1_passed:
            print('   ❌ Episode summary enhancements')
        if not test2_passed:
            print('   ❌ Curriculum callback frequency control')
        if not test3_passed:
            print('   ❌ Risk callback frequency control')
        if not test4_passed:
            print('   ❌ TensorBoard monitoring frequency')