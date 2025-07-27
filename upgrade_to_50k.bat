@echo off
echo 🚀 UPGRADE TO 50K PRODUCTION TRAINING
echo ====================================
echo.
echo This script will upgrade the configuration from 15K to 50K
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
echo 🔧 Upgrading configuration to 50K...

cd /d "c:\Projects\IntradayJules"

:: Create backup
copy phase1_fast_recovery_training.py phase1_fast_recovery_training_15k_backup.py >nul

:: Upgrade timesteps
powershell -Command "(Get-Content phase1_fast_recovery_training.py) -replace 'total_timesteps.*15000.*', 'total_timesteps'': 50000,  # 🚀 PRODUCTION: Full 50K training run' | Set-Content phase1_fast_recovery_training.py"

:: Upgrade lambda schedule
powershell -Command "(Get-Content phase1_fast_recovery_training.py) -replace 'lambda_schedule_steps.*15000.*', 'lambda_schedule_steps'' = 50000  # 🚀 PRODUCTION: Linear increase over full 50k steps' | Set-Content phase1_fast_recovery_training.py"

:: Upgrade tensorboard directory
powershell -Command "(Get-Content phase1_fast_recovery_training.py) -replace 'tensorboard_phase1_15k', 'tensorboard_phase1_50k' | Set-Content phase1_fast_recovery_training.py"

:: Upgrade log messages
powershell -Command "(Get-Content phase1_fast_recovery_training.py) -replace '15K ENGINEERING VALIDATION', '50K PRODUCTION TRAINING' | Set-Content phase1_fast_recovery_training.py"
powershell -Command "(Get-Content phase1_fast_recovery_training.py) -replace '15,000 timesteps', '50,000 timesteps' | Set-Content phase1_fast_recovery_training.py"

echo ✅ Configuration upgraded to 50K!
echo ✅ Backup saved as: phase1_fast_recovery_training_15k_backup.py
echo.
echo 🚀 You can now run: launch_50k_training.bat
echo.
pause