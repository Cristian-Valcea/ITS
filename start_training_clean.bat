@echo off
REM ========================================================================
REM IntradayJules Training Launch - CLEAN VERSION (No Duplicates)
REM ========================================================================

echo.
echo ========================================================================
echo IntradayJules Training Launch - CLEAN VERSION
echo ========================================================================
echo.

REM Set working directory
cd /d "C:\Projects\IntradayJules"

REM ========================================================================
REM PRE-FLIGHT CHECKS - Clean up potential conflicts
REM ========================================================================

echo [PRE-FLIGHT] Running system cleanup checks...
echo.

echo [1/6] Checking for conflicting Python processes...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    if not "%%~i"=="" (
        echo   - Terminating Python process %%~i
        taskkill /PID %%~i /F >nul 2>&1
    )
)
REM Also check for python processes without .exe extension
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python" /FO CSV /NH 2^>nul') do (
    if not "%%~i"=="" (
        echo   - Terminating Python process %%~i
        taskkill /PID %%~i /F >nul 2>&1
    )
)

echo [2/6] Checking for TensorBoard processes...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq tensorboard.exe" /FO CSV /NH 2^>nul') do (
    if not "%%~i"=="" (
        echo   - Terminating TensorBoard process %%~i
        taskkill /PID %%~i /F >nul 2>&1
    )
)

echo [3/6] Checking for processes using port 6006 (TensorBoard)...
for /f "tokens=5" %%i in ('netstat -ano ^| findstr ":6006" 2^>nul') do (
    if not "%%i"=="" (
        echo   - Terminating process using port 6006: %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)

echo [4/6] Checking for processes using port 8000 (API Server)...
for /f "tokens=5" %%i in ('netstat -ano ^| findstr ":8000" 2^>nul') do (
    if not "%%i"=="" (
        echo   - Terminating process using port 8000: %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)

echo [5/6] Cleaning DuckDB lock files...
REM Force kill any remaining Python processes that might be holding DuckDB
wmic process where "name='python.exe'" get ProcessId /format:value 2>nul | findstr "ProcessId" > temp_pids.txt
for /f "tokens=2 delims==" %%i in (temp_pids.txt) do (
    if not "%%i"=="" (
        echo   - Force terminating Python process %%i
        taskkill /PID %%i /F >nul 2>&1
    )
)
del temp_pids.txt >nul 2>&1

REM Also use PowerShell to kill any remaining Python processes
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" >nul 2>&1

if exist "%USERPROFILE%\.feature_cache\manifest.duckdb" (
    echo   - Removing locked DuckDB manifest file...
    del /F /Q "%USERPROFILE%\.feature_cache\manifest.duckdb" >nul 2>&1
    if exist "%USERPROFILE%\.feature_cache\manifest.duckdb" (
        echo   - File still locked, waiting and retrying...
        timeout /t 3 /nobreak >nul
        powershell -Command "Remove-Item '%USERPROFILE%\.feature_cache\manifest.duckdb' -Force -ErrorAction SilentlyContinue" >nul 2>&1
    )
)
if exist "%USERPROFILE%\.feature_cache\*.wal" (
    echo   - Removing DuckDB WAL files...
    del /F /Q "%USERPROFILE%\.feature_cache\*.wal" >nul 2>&1
)
if exist "%USERPROFILE%\.feature_cache\*.tmp" (
    echo   - Removing DuckDB temp files...
    del /F /Q "%USERPROFILE%\.feature_cache\*.tmp" >nul 2>&1
)
if exist "%USERPROFILE%\.feature_cache\*.lock" (
    echo   - Removing DuckDB lock files...
    del /F /Q "%USERPROFILE%\.feature_cache\*.lock" >nul 2>&1
)

echo [6/6] Cleaning old log files...
if exist "logs\orchestrator_gpu_fixed_rainbow_qrdqn.log" (
    echo   - Archiving previous log file...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    move "logs\orchestrator_gpu_fixed_rainbow_qrdqn.log" "logs\orchestrator_gpu_fixed_rainbow_qrdqn_!mydate!_!mytime!.log" >nul 2>&1
)

echo.
echo [PRE-FLIGHT] All checks completed successfully!
echo.
echo ========================================================================
echo Starting Training System Components
echo ========================================================================
echo.

REM Activate virtual environment ONCE
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated successfully

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"
if not exist "logs\tensorboard_gpu_recurrent_ppo_microstructural" mkdir "logs\tensorboard_gpu_recurrent_ppo_microstructural"

REM Create required data directories
if not exist "data" mkdir "data"
if not exist "data\raw_orch_gpu_rainbow_qrdqn" mkdir "data\raw_orch_gpu_rainbow_qrdqn"
if not exist "data\processed_orch_gpu_rainbow_qrdqn" mkdir "data\processed_orch_gpu_rainbow_qrdqn"
if not exist "models" mkdir "models"
if not exist "models\orch_gpu_rainbow_qrdqn" mkdir "models\orch_gpu_rainbow_qrdqn"
if not exist "reports" mkdir "reports"
if not exist "reports\orch_gpu_rainbow_qrdqn" mkdir "reports\orch_gpu_rainbow_qrdqn"

REM Create placeholder log file for tail monitoring
echo [%date% %time%] Log monitoring started... > "logs\orchestrator_gpu_fixed_rainbow_qrdqn.log"
echo Log directories prepared...
echo.

REM ========================================================================
REM Launch monitoring tools (each in separate window, inheriting venv)
REM ========================================================================

echo Starting TensorBoard on http://localhost:6006...
start "TensorBoard" cmd /k "tensorboard --logdir logs\tensorboard_gpu_recurrent_ppo_microstructural --port 6006 --host localhost"
timeout /t 2 /nobreak >nul

echo Starting log monitor...
start "Log-Monitor" cmd /k "python monitor_live_logs.py"
timeout /t 1 /nobreak >nul

echo Starting API server on http://localhost:8000...
start "API-Server" cmd /k "set DISABLE_FEATURESTORE_MONITORING=true && python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul

echo Starting post-training visualizer...
start "Visualizer" cmd /k "chcp 65001 >nul && python post_training_visualizer.py"
timeout /t 1 /nobreak >nul

REM ========================================================================
REM Launch main training
REM ========================================================================
echo.
echo Waiting 10 seconds for monitoring tools to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ========================================================================
echo Starting Main Training Process with Rolling Window Backtest
echo ========================================================================
echo.
echo 🔄 ROLLING WINDOW BACKTEST ENABLED:
echo    - 3-month training windows
echo    - 1-month evaluation periods  
echo    - Comprehensive robustness validation
echo    - Automated deployment recommendations
echo.

REM Final system readiness check
echo [FINAL CHECK] Verifying system readiness...
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)
if not exist "src\main.py" (
    echo ERROR: Main training script not found!
    pause
    exit /b 1
)
if not exist "config\main_config_orchestrator_gpu_fixed.yaml" (
    echo ERROR: Training configuration not found!
    pause
    exit /b 1
)
if not exist "config\model_params.yaml" (
    echo ERROR: Model parameters configuration not found!
    pause
    exit /b 1
)
if not exist "config\risk_limits.yaml" (
    echo ERROR: Risk limits configuration not found!
    pause
    exit /b 1
)
echo [FINAL CHECK] System ready - launching training...
echo.

REM Change to src directory and run training
cd src
start "MAIN-TRAINING" cmd /k "python main.py train --main_config ../config/main_config_orchestrator_gpu_fixed.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-01-31"

REM Return to root directory
cd ..

REM Wait for training to start creating logs, then start log tail
timeout /t 20 /nobreak >nul
echo Starting log tail now that training has begun...
start "Log-Tail" cmd /k "powershell -Command \"Get-Content -Path 'logs\orchestrator_gpu_fixed_rainbow_qrdqn.log' -Wait -Tail 20 -ErrorAction SilentlyContinue\""

REM ========================================================================
REM Show status
REM ========================================================================
echo.
echo ========================================================================
echo 📊 MONITORING DASHBOARD - Enhanced with Rolling Window Backtest
echo ========================================================================
echo 🔗 TensorBoard:     http://localhost:6006
echo 🔗 API Monitoring:  http://localhost:8000/docs
echo 🔗 API Status:      http://localhost:8000/api/v1/status
echo ========================================================================
echo.
echo LAUNCHED WINDOWS:
echo   1. TensorBoard      - Training metrics and visualizations
echo   2. Log Monitor      - Real-time log analysis
echo   3. API Server       - REST API for monitoring and control
echo   4. Visualizer       - Post-training analysis tools
echo   5. MAIN TRAINING    - Core training process
echo   6. Log Tail         - Live log streaming (started after training begins)
echo.
echo 🔄 NEW FEATURES ENABLED:
echo   ✅ 3-Month Rolling Window Walk-Forward Backtest
echo   ✅ Automated Robustness Validation
echo   ✅ Market Regime Analysis
echo   ✅ Deployment Recommendations
echo   ✅ Enhanced Risk Management
echo.
echo ✅ All tools launched! Training with robustness validation is starting...
echo 💡 Keep this window open as control panel
echo 💡 Rolling window backtest will run automatically after training
echo.

pause