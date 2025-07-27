@echo off
REM ========================================================================
REM IntradayJules Training Launch - TURNOVER PENALTY SYSTEM
REM ========================================================================

echo.
echo ========================================================================
echo IntradayJules Training Launch - TURNOVER PENALTY SYSTEM
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
    echo   - Archiving previous log file...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    move "logs\turnover_penalty_orchestrator_gpu.log" "logs\turnover_penalty_orchestrator_gpu_!mydate!_!mytime!.log" >nul 2>&1
)

echo.
echo [PRE-FLIGHT] All checks completed successfully!
echo.

REM ========================================================================
REM DIRECTORY SETUP - Create required directories
REM ========================================================================

echo [SETUP] Creating required directories...

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated successfully

REM Create logs directory structure
if not exist "logs" mkdir "logs"
if not exist "logs\tensorboard_turnover_penalty" mkdir "logs\tensorboard_turnover_penalty"

REM Create data directories for turnover penalty system
if not exist "data" mkdir "data"
if not exist "data\raw_turnover_penalty" mkdir "data\raw_turnover_penalty"
if not exist "data\processed_turnover_penalty" mkdir "data\processed_turnover_penalty"

REM Create model directories
if not exist "models" mkdir "models"
if not exist "models\turnover_penalty" mkdir "models\turnover_penalty"

REM Create reports directories
if not exist "reports" mkdir "reports"
if not exist "reports\turnover_penalty" mkdir "reports\turnover_penalty"

REM Create runs directory for our new TensorBoard exporter
if not exist "runs" mkdir "runs"

REM Create placeholder log file for monitoring
echo [%date% %time%] Turnover Penalty Training Log Started... > "logs\turnover_penalty_orchestrator_gpu.log"
echo Directory structure prepared...
echo.

REM ========================================================================
REM CONFIGURATION VALIDATION
REM ========================================================================

echo [CONFIG] Validating configuration files...

if not exist "config\turnover_penalty_orchestrator_gpu.yaml" (
    echo ERROR: Turnover penalty configuration not found!
    echo Expected: config\turnover_penalty_orchestrator_gpu.yaml
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

echo ✅ Configuration files validated
echo.

REM ========================================================================
REM LAUNCH MONITORING TOOLS
REM ========================================================================

echo ========================================================================
echo Starting Enhanced Monitoring Tools
echo ========================================================================
echo.

echo [1/5] Starting TensorBoard with turnover penalty logs...
echo   📊 TensorBoard URL: http://localhost:6006
echo   📁 Log Directory: logs\tensorboard_turnover_penalty
start "TensorBoard-TurnoverPenalty" cmd /k "tensorboard --logdir logs\tensorboard_turnover_penalty --port 6006 --host localhost --reload_interval 30"
timeout /t 3 /nobreak >nul

echo [2/5] Starting enhanced TensorBoard launcher...
echo   🚀 Enhanced TensorBoard: http://localhost:6007
start "TensorBoard-Enhanced" cmd /k "python launch_tensorboard.py --port 6007 --logdir runs"
timeout /t 2 /nobreak >nul

echo [3/5] Starting log monitor...
start "Log-Monitor" cmd /k "python monitor_live_logs.py logs\turnover_penalty_orchestrator_gpu.log"
timeout /t 1 /nobreak >nul

echo [4/5] Starting API server...
echo   🔗 API Server: http://localhost:8000
echo   📚 API Docs: http://localhost:8000/docs
start "API-Server" cmd /k "set DISABLE_FEATURESTORE_MONITORING=true && python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul

echo [5/5] Starting post-training visualizer...
start "Visualizer" cmd /k "chcp 65001 >nul && python post_training_visualizer.py"
timeout /t 1 /nobreak >nul

echo.
echo ✅ All monitoring tools launched successfully!
echo.

REM ========================================================================
REM SYSTEM READINESS CHECK
REM ========================================================================

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

REM Test TensorBoard integration
echo Testing TensorBoard integration...
python -c "
import sys
sys.path.append('src')
try:
    from training.core.tensorboard_exporter import TensorBoardExporter
    print('✅ TensorBoard exporter available')
except ImportError as e:
    print(f'❌ TensorBoard exporter error: {e}')
    sys.exit(1)

try:
    from gym_env.components.turnover_penalty import TurnoverPenaltyCalculator
    print('✅ Turnover penalty calculator available')
except ImportError as e:
    print(f'❌ Turnover penalty calculator error: {e}')
    sys.exit(1)

print('✅ All components ready for training')
"

if %errorlevel% neq 0 (
    echo ERROR: Component validation failed!
    pause
    exit /b 1
)

echo [FINAL CHECK] System ready - launching training...
echo.

REM ========================================================================
REM LAUNCH MAIN TRAINING
REM ========================================================================

echo ========================================================================
echo Starting Main Training Process - TURNOVER PENALTY SYSTEM
echo ========================================================================
echo.
echo 🎯 TURNOVER PENALTY FEATURES ENABLED:
echo    ✅ Normalized turnover penalty system
echo    ✅ Dynamic portfolio value tracking
echo    ✅ Smooth sigmoid penalty curves
echo    ✅ Real-time TensorBoard monitoring
echo    ✅ Comprehensive performance analytics
echo    ✅ Target: 2%% normalized turnover
echo    ✅ Algorithm: RecurrentPPO with LSTM
echo.

echo Waiting 10 seconds for monitoring tools to initialize...
timeout /t 10 /nobreak >nul

echo.
echo 🚀 LAUNCHING TRAINING...
echo.

REM Change to src directory and run training with turnover penalty config
cd src
start "MAIN-TRAINING-TURNOVER-PENALTY" cmd /k "python main.py train --main_config ../config/turnover_penalty_orchestrator_gpu.yaml --symbol NVDA --start_date 2024-01-01 --end_date 2024-03-31"

REM Return to root directory
cd ..

REM Wait for training to start creating logs, then start log tail
timeout /t 20 /nobreak >nul
echo Starting enhanced log tail...
start "Log-Tail-Enhanced" cmd /k "powershell -Command \"Get-Content -Path 'logs\turnover_penalty_orchestrator_gpu.log' -Wait -Tail 30 -ErrorAction SilentlyContinue\""

REM ========================================================================
REM MONITORING DASHBOARD
REM ========================================================================
echo.
echo ========================================================================
echo 📊 TURNOVER PENALTY MONITORING DASHBOARD
echo ========================================================================
echo 🔗 Primary TensorBoard:    http://localhost:6006
echo 🔗 Enhanced TensorBoard:   http://localhost:6007
echo 🔗 API Monitoring:         http://localhost:8000/docs
echo 🔗 API Status:             http://localhost:8000/api/v1/status
echo ========================================================================
echo.
echo 📊 TENSORBOARD METRICS TO MONITOR:
echo   🎯 Turnover Penalty Evolution:
echo      • turnover/penalty_current - Current penalty value
echo      • turnover/normalized_current - Current turnover ratio
echo      • turnover/target - Target turnover (2%%)
echo      • turnover/excess_current - Excess over target
echo.
echo   📈 Performance Metrics:
echo      • episode/total_reward - Episode rewards
echo      • episode/portfolio_value - Portfolio evolution
echo      • performance/win_rate - Success percentage
echo      • performance/sharpe_ratio - Risk-adjusted returns
echo      • performance/max_drawdown - Maximum drawdown
echo.
echo   🧠 Training Metrics:
echo      • training/loss - Training loss convergence
echo      • training/q_value_variance - Q-value spread
echo      • training/learning_rate - Learning rate schedule
echo.
echo   ⚠️  Risk Metrics:
echo      • risk/volatility - Portfolio volatility
echo      • monitoring/drawdown_pct_mean - Average drawdown
echo.
echo LAUNCHED WINDOWS:
echo   1. TensorBoard (Primary)     - Standard training metrics
echo   2. TensorBoard (Enhanced)    - Comprehensive turnover penalty metrics
echo   3. Log Monitor              - Real-time log analysis
echo   4. API Server               - REST API for monitoring and control
echo   5. Visualizer               - Post-training analysis tools
echo   6. MAIN TRAINING            - Core training with turnover penalty
echo   7. Log Tail (Enhanced)      - Live log streaming with more context
echo.
echo 🎯 EXPECTED TRAINING BEHAVIOR:
echo   📊 Early Episodes: High turnover penalty, learning phase
echo   📈 Mid Episodes: Decreasing penalty as agent learns control
echo   🎯 Late Episodes: Stable penalty near 2%% target, improved performance
echo   ✅ Convergence: Consistent performance with controlled turnover
echo.
echo ✅ Turnover Penalty Training System Launched!
echo 💡 Keep this window open as control panel
echo 💡 Monitor TensorBoard for real-time turnover penalty evolution
echo 💡 Training will automatically save models and generate reports
echo.

pause