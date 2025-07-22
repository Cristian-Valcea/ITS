@echo off
echo 🚀 SAFE UPGRADE TO 50K PRODUCTION TRAINING
echo =========================================
echo.
echo This script will safely upgrade from 15K to 50K configuration
echo.
echo ⚠️  WARNING: Only run this AFTER 15K validation succeeds!
echo.
set /p confirm="Are you sure you want to upgrade to 50K? (y/N): "
if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b
)

echo.
echo 🔧 Safely upgrading configuration to 50K...

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

:: Create backup with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
copy phase1_fast_recovery_training.py "phase1_fast_recovery_training_15k_backup_%TIMESTAMP%.py" >nul

echo ✅ Backup created: phase1_fast_recovery_training_15k_backup_%TIMESTAMP%.py

:: Use Python to safely replace text (avoids encoding issues)
python -c "
import re

# Read the file
with open('phase1_fast_recovery_training.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Safe replacements
content = re.sub(r'total_timesteps.*15000.*', 'total_timesteps\': 50000,  # 🚀 PRODUCTION: Full 50K training run', content)
content = re.sub(r'lambda_schedule_steps.*15000.*', 'lambda_schedule_steps\'] = 50000  # 🚀 PRODUCTION: Linear increase over full 50k steps', content)
content = re.sub(r'tensorboard_phase1_15k', 'tensorboard_phase1_50k', content)
content = re.sub(r'15K ENGINEERING VALIDATION', '50K PRODUCTION TRAINING', content)
content = re.sub(r'15,000 timesteps', '50,000 timesteps', content)
content = re.sub(r'smart incremental approach', 'full-scale deployment', content)
content = re.sub(r'VALIDATION TRAINING CRITERIA', 'PRODUCTION TRAINING CRITERIA', content)
content = re.sub(r'Engineering validation training \(15000 steps\)', 'Full production training (50000 steps)', content)
content = re.sub(r'phase1_15k', 'phase1_50k', content)

# Write back
with open('phase1_fast_recovery_training.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Configuration safely upgraded to 50K!')
"

if errorlevel 1 (
    echo ❌ Upgrade failed! Restoring backup...
    copy "phase1_fast_recovery_training_15k_backup_%TIMESTAMP%.py" phase1_fast_recovery_training.py >nul
    echo ✅ Backup restored
    pause
    exit /b 1
)

echo ✅ Configuration upgraded to 50K!
echo ✅ Backup saved with timestamp
echo.
echo 🚀 You can now run: launch_50k_training.bat
echo.
pause