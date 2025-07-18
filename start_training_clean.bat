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
if exist "logs\turnover_penalty_orchestrator_gpu.log" (
    echo   - Archiving previous turnover penalty log file...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    move "logs\turnover_penalty_orchestrator_gpu.log" "logs\turnover_penalty_orchestrator_gpu_!mydate!_!mytime!.log" >nul 2>&1
)
if exist "logs\emergency_fix_orchestrator_gpu.log" (
    echo   - Archiving previous emergency fix log file...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    move "logs\emergency_fix_orchestrator_gpu.log" "logs\emergency_fix_orchestrator_gpu_!mydate!_!mytime!.log" >nul 2>&1
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
if not exist "logs\tensorboard_emergency_fix" mkdir "logs\tensorboard_emergency_fix"

REM Create required data directories
if not exist "data" mkdir "data"
if not exist "data\raw_emergency_fix" mkdir "data\raw_emergency_fix"
if not exist "data\processed_emergency_fix" mkdir "data\processed_emergency_fix"
if not exist "models" mkdir "models"
if not exist "models\emergency_fix" mkdir "models\emergency_fix"
if not exist "reports" mkdir "reports"
if not exist "reports\emergency_fix" mkdir "reports\emergency_fix"

REM Create placeholder log file for tail monitoring
if "%CONFIG_FILE%"=="turnover_penalty_orchestrator_gpu.yaml" (
    echo [%date% %time%] Turnover Penalty Training Log Started... > "logs\turnover_penalty_orchestrator_gpu.log"
    set LOG_FILE=turnover_penalty_orchestrator_gpu.log
) else (
    echo [%date% %time%] Emergency Fix Training Log Started... > "logs\emergency_fix_orchestrator_gpu.log"
    set LOG_FILE=emergency_fix_orchestrator_gpu.log
)
echo Log directories prepared...
echo.

REM ========================================================================
REM Launch monitoring tools (each in separate window, inheriting venv)
REM ========================================================================

echo Starting TensorBoard on http://localhost:6006...
if "%CONFIG_FILE%"=="turnover_penalty_orchestrator_gpu.yaml" (
    echo   📊 Using turnover penalty TensorBoard logs
    start "TensorBoard" cmd /k "tensorboard --logdir logs\tensorboard_turnover_penalty --port 6006 --host localhost --reload_interval 30"
) else (
    echo   📊 Using emergency fix TensorBoard logs
    start "TensorBoard" cmd /k "tensorboard --logdir logs\tensorboard_emergency_fix --port 6006 --host localhost"
)
timeout /t 2 /nobreak >nul

echo Starting enhanced TensorBoard launcher...
start "TensorBoard-Enhanced" cmd /k "python launch_tensorboard.py --port 6007 --logdir runs"
timeout /t 2 /nobreak >nul

echo Starting log monitor...
start "Log-Monitor" cmd /k "python monitor_live_logs.py logs\%LOG_FILE%"
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
if not exist "config\turnover_penalty_orchestrator_gpu.yaml" (
    echo ERROR: Turnover penalty configuration not found!
    echo Falling back to emergency fix configuration...
    if not exist "config\emergency_fix_orchestrator_gpu.yaml" (
        echo ERROR: No valid training configuration found!
        pause
        exit /b 1
    )
    set CONFIG_FILE=emergency_fix_orchestrator_gpu.yaml
) else (
    set CONFIG_FILE=turnover_penalty_orchestrator_gpu.yaml
    echo ✅ Using enhanced turnover penalty configuration
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
REM Test TensorBoard and turnover penalty integration
echo Testing system components...
python validate_training_setup.py

echo [FINAL CHECK] System ready - launching training...
echo.

REM Change to src directory and run training
cd src
if "%CONFIG_FILE%"=="turnover_penalty_orchestrator_gpu.yaml" (
    echo 🎯 Launching training with TURNOVER PENALTY SYSTEM
    start "MAIN-TRAINING-TURNOVER-PENALTY" cmd /k "python main.py train --main_config ../config/turnover_penalty_orchestrator_gpu.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-03-31"
) else (
    echo 🔧 Launching training with emergency fix configuration
    start "MAIN-TRAINING-EMERGENCY-FIX" cmd /k "python main.py train --main_config ../config/emergency_fix_orchestrator_gpu.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-03-31"
)

REM Return to root directory
cd ..

REM Wait for training to start creating logs, then start log tail
timeout /t 20 /nobreak >nul
echo Starting log tail now that training has begun...
start "Log-Tail" cmd /k "powershell -Command \"Get-Content -Path 'logs\%LOG_FILE%' -Wait -Tail 20 -ErrorAction SilentlyContinue\""

REM ========================================================================
REM Show status
REM ========================================================================
echo.
echo ========================================================================
echo 📊 MONITORING DASHBOARD - Enhanced with TensorBoard Integration
echo ========================================================================
echo 🔗 Primary TensorBoard:    http://localhost:6006
echo 🔗 Enhanced TensorBoard:   http://localhost:6007
echo 🔗 API Monitoring:         http://localhost:8000/docs
echo 🔗 API Status:             http://localhost:8000/api/v1/status
echo ========================================================================
echo.
echo LAUNCHED WINDOWS:
echo   1. TensorBoard (Primary)     - Standard training metrics
echo   2. TensorBoard (Enhanced)    - Comprehensive turnover penalty metrics
echo   3. Log Monitor              - Real-time log analysis
echo   4. API Server               - REST API for monitoring and control
echo   5. Visualizer               - Post-training analysis tools
echo   6. MAIN TRAINING            - Core training process
echo   7. Log Tail                 - Live log streaming (started after training begins)
echo.
if "%CONFIG_FILE%"=="turnover_penalty_orchestrator_gpu.yaml" (
    echo 🎯 TURNOVER PENALTY FEATURES ENABLED:
    echo   ✅ Normalized turnover penalty system
    echo   ✅ Dynamic portfolio value tracking
    echo   ✅ Smooth sigmoid penalty curves
    echo   ✅ Real-time TensorBoard monitoring
    echo   ✅ Comprehensive performance analytics
    echo   ✅ Target: 2%% normalized turnover
    echo.
    echo 📊 KEY TENSORBOARD METRICS TO MONITOR:
    echo   🎯 turnover/penalty_current - Current penalty value
    echo   📊 turnover/normalized_current - Current turnover ratio
    echo   🎯 turnover/target - Target turnover (2%%)
    echo   📈 performance/win_rate - Success percentage
    echo   📉 performance/sharpe_ratio - Risk-adjusted returns
    echo   ⚠️  performance/max_drawdown - Maximum drawdown
) else (
    echo 🔄 EMERGENCY FIX FEATURES ENABLED:
    echo   ✅ Emergency reward system fixes
    echo   ✅ Enhanced risk management
    echo   ✅ Standard TensorBoard monitoring
)
echo.
echo ✅ All tools launched! Training with robustness validation is starting...
echo 💡 Keep this window open as control panel
echo 💡 Rolling window backtest will run automatically after training
echo.

pause