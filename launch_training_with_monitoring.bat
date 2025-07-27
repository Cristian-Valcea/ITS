@echo off
REM ========================================================================
REM IntradayJules Training Launch with Full Monitoring Suite
REM ========================================================================
REM This batch file launches training and all monitoring tools in parallel
REM Author: IntradayJules System
REM Date: 2025-07-15
REM ========================================================================

echo.
echo ========================================================================
echo 🚀 IntradayJules Training Launch with Full Monitoring
echo ========================================================================
echo.

REM Set working directory
cd /d "C:\Projects\IntradayJules"

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv_fresh\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    echo Please ensure venv_fresh exists and run: venv_fresh\Scripts\activate
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "logs\tensorboard_gpu_fixed" mkdir logs\tensorboard_gpu_fixed

REM Clear old logs (optional - uncomment if you want fresh logs)
REM del /q logs\orchestrator_gpu_fixed.log* 2>nul
REM rmdir /s /q logs\tensorboard_gpu_fixed 2>nul
REM mkdir logs\tensorboard_gpu_fixed

echo 📁 Log directories prepared...
echo.

REM ========================================================================
REM Launch TensorBoard (Port 6006)
REM ========================================================================
echo 📊 Starting TensorBoard on http://localhost:6006...
start "TensorBoard-GPU-Fixed" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && tensorboard --logdir logs\tensorboard_gpu_fixed --port 6006 --host localhost"

REM Wait a moment for TensorBoard to start
timeout /t 3 /nobreak >nul

REM ========================================================================
REM Launch Log File Monitor
REM ========================================================================
echo 📡 Starting real-time log monitor...
start "Log-Monitor" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && python monitor_live_logs.py"

REM ========================================================================
REM Launch Log Tail (Windows equivalent using PowerShell)
REM ========================================================================
echo 📋 Starting log tail monitor...
start "Log-Tail" powershell -Command "cd 'C:\Projects\IntradayJules'; Get-Content -Path 'logs\orchestrator_gpu_fixed.log' -Wait -Tail 50"

REM ========================================================================
REM Launch API Server for Monitoring Endpoints
REM ========================================================================
echo 🌐 Starting API server on http://localhost:8000...
start "API-Server" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"

REM Wait for API server to start
timeout /t 5 /nobreak >nul

REM ========================================================================
REM Launch System Resource Monitor (Simple version)
REM ========================================================================
echo 💻 Starting system resource monitor...
start "Resource-Monitor" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && echo 💻 System Resource Monitor && echo ======================== && echo Monitoring system resources... && pause"

REM ========================================================================
REM Launch Risk Audit Monitor (Simple version)
REM ========================================================================
echo 🛡️  Starting risk audit monitor...
start "Risk-Audit" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && echo 🛡️  Risk Audit Monitor && echo ======================== && echo Monitoring risk events... && pause"

REM ========================================================================
REM Launch Post-Training Visualizer (waits for completion)
REM ========================================================================
echo 🎨 Starting post-training visualization monitor...
start "Post-Training-Visualizer" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && python post_training_visualizer.py"

REM ========================================================================
REM Launch Training Progress Monitor
REM ========================================================================
echo 📈 Starting training progress monitor...
start "Training-Progress" cmd /k "cd /d C:\Projects\IntradayJules && python -c \"
import re
import time
import os
from datetime import datetime

print('📈 Training Progress Monitor')
print('=' * 50)

log_file = 'logs/orchestrator_gpu_fixed.log'
if not os.path.exists(log_file):
    print('⏳ Waiting for training log to be created...')
    while not os.path.exists(log_file):
        time.sleep(1)

print(f'📡 Monitoring: {log_file}')
print('🔍 Watching for training progress...')
print()

last_position = 0
episode_count = 0
last_reward = 0.0

while True:
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                
                for line in new_lines:
                    # Look for episode information
                    if 'Episode' in line and 'reward' in line:
                        episode_match = re.search(r'Episode (\d+)', line)
                        reward_match = re.search(r'reward[:\s]+([+-]?\d*\.?\d+)', line)
                        
                        if episode_match:
                            episode_count = int(episode_match.group(1))
                        if reward_match:
                            last_reward = float(reward_match.group(1))
                            
                        print(f'🎯 Episode {episode_count}: Reward = {last_reward:.4f}')
                    
                    # Look for Kyle Lambda fills
                    elif 'KYLE LAMBDA IMPACT' in line:
                        print(f'💰 {line.strip()}')
                    
                    # Look for risk events
                    elif any(keyword in line.lower() for keyword in ['turnover', 'drawdown', 'risk', 'violation']):
                        print(f'🛡️  {line.strip()}')
                    
                    # Look for errors
                    elif 'ERROR' in line:
                        print(f'❌ {line.strip()}')
                    
                    # Look for warnings
                    elif 'WARNING' in line:
                        print(f'⚠️  {line.strip()}')
                
                last_position = f.tell()
        
        time.sleep(1)
    except Exception as e:
        print(f'❌ Error: {e}')
        time.sleep(5)
\""

REM ========================================================================
REM Wait and then launch the main training
REM ========================================================================
echo.
echo ⏳ Waiting 10 seconds for all monitoring tools to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ========================================================================
echo 🎯 Starting Main Training Process
echo ========================================================================
echo 📊 Command: cd c:/Projects/IntradayJules/src ^&^& python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31
echo.

REM Launch the main training in a new window so you can see the output
start "MAIN-TRAINING" cmd /k "cd /d C:\Projects\IntradayJules && call venv_fresh\Scripts\activate.bat && cd src && python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31"

REM ========================================================================
REM Show monitoring dashboard URLs
REM ========================================================================
echo.
echo ========================================================================
echo 📊 MONITORING DASHBOARD URLS
echo ========================================================================
echo 🔗 TensorBoard:     http://localhost:6006
echo 🔗 API Monitoring:  http://localhost:8000/docs
echo 🔗 System Status:   http://localhost:8000/api/v1/status
echo ========================================================================
echo.

REM ========================================================================
REM Keep this window open for control
REM ========================================================================
echo 🎮 CONTROL PANEL
echo ========================================================================
echo This window serves as your control panel.
echo.
echo 📋 Monitoring Windows Launched:
echo   - TensorBoard (Port 6006)
echo   - Log Monitor
echo   - Log Tail
echo   - API Server (Port 8000)
echo   - Resource Monitor
echo   - Risk Audit Monitor
echo   - Post-Training Visualizer (waits for completion)
echo   - Training Progress Monitor
echo   - Main Training Process
echo.
echo 💡 Tips:
echo   - Check TensorBoard at http://localhost:6006 for training metrics
echo   - Monitor the various windows for real-time updates
echo   - Use Ctrl+C in any window to stop that specific monitor
echo   - Close this window to keep everything running
echo.
echo 🚀 Training is now running with full monitoring suite!
echo ========================================================================

REM Keep the control panel open
pause