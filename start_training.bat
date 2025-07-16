@echo off
REM ========================================================================
REM IntradayJules Training Launch with Full Monitoring
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
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    echo Please ensure venv exists and run: venv\Scripts\activate
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"
if not exist "logs\tensorboard_gpu_fixed" mkdir "logs\tensorboard_gpu_fixed"

echo 📁 Log directories prepared...
echo.

REM ========================================================================
REM Launch TensorBoard (Port 6006)
REM ========================================================================
echo 📊 Starting TensorBoard on http://localhost:6006...
start "TensorBoard-GPU-Fixed" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && tensorboard --logdir logs\tensorboard_gpu_fixed --port 6006 --host 0.0.0.0"

REM Wait a moment for TensorBoard to start
timeout /t 3 /nobreak >nul

REM ========================================================================
REM Launch Log File Monitor
REM ========================================================================
echo 📡 Starting real-time log monitor...
start "Log-Monitor" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && python monitor_live_logs.py"

REM ========================================================================
REM Launch Log Tail (Windows equivalent using PowerShell)
REM ========================================================================
echo 📋 Starting log tail monitor...
start "Log-Tail" powershell -Command "& {Set-Location 'C:\Projects\IntradayJules'; Write-Host '📋 Log Tail Monitor - logs\orchestrator_gpu_fixed.log' -ForegroundColor Cyan; Write-Host ('=' * 60) -ForegroundColor Gray; if (Test-Path 'logs\orchestrator_gpu_fixed.log') {Get-Content -Path 'logs\orchestrator_gpu_fixed.log' -Wait -Tail 50} else {Write-Host '⏳ Waiting for log file...' -ForegroundColor Yellow; while (!(Test-Path 'logs\orchestrator_gpu_fixed.log')) {Start-Sleep -Seconds 1}; Get-Content -Path 'logs\orchestrator_gpu_fixed.log' -Wait -Tail 50}}"

REM ========================================================================
REM Launch API Server for Monitoring Endpoints
REM ========================================================================
echo 🌐 Starting API server on http://localhost:8000...
start "API-Server" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"

REM Wait for API server to start
timeout /t 5 /nobreak >nul

REM ========================================================================
REM Launch System Resource Monitor (Simple version)
REM ========================================================================
echo 💻 Starting system resource monitor...
start "Resource-Monitor" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && echo 💻 System Resource Monitor && echo ======================== && echo Use Task Manager for detailed system monitoring && pause"

REM ========================================================================
REM Launch Risk Audit Monitor (Simple version)
REM ========================================================================
echo 🛡️ Starting risk audit monitor...
start "Risk-Audit" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && echo 🛡️ Risk Audit Monitor && echo ======================== && echo Monitoring logs\risk_audit.jsonl && echo Check this file for risk events && pause"

REM ========================================================================
REM Launch Post-Training Visualizer (waits for completion)
REM ========================================================================
echo 🎨 Starting post-training visualization monitor...
start "Post-Training-Visualizer" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && python post_training_visualizer.py"

REM ========================================================================
REM Launch Training Progress Monitor (Simple version)
REM ========================================================================
echo 📈 Starting training progress monitor...
start "Training-Progress" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && echo 📈 Training Progress Monitor && echo ======================== && echo Monitoring logs\orchestrator_gpu_fixed.log && echo Watch for Episode rewards and training progress && pause"

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
start "MAIN-TRAINING" cmd /k "cd /d C:\Projects\IntradayJules && call venv\Scripts\activate.bat && cd src && python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31"

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
echo 🎯 All monitoring tools are now running!
echo 📊 Check the URLs above for web-based monitoring
echo 🖼️ Performance plots will auto-open when training completes
echo.
echo ========================================================================
echo 💡 TIPS:
echo   - Keep this window open to monitor the system
echo   - Close any monitoring window if not needed
echo   - Training will continue even if you close monitoring windows
echo   - Performance visualizations will appear automatically when done
echo ========================================================================
echo.

pause