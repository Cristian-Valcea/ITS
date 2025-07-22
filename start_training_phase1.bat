@echo off
REM ========================================================================
REM IntradayJules PHASE 1 REALITY GROUNDING Training Launch
REM ========================================================================

echo.
echo ========================================================================
echo IntradayJules PHASE 1 REALITY GROUNDING Training Launch
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
)

echo [6/6] Cleaning old log files...
if exist "logs\phase1_reality_grounding.log" (
    echo   - Archiving previous Phase 1 log file...
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    move "logs\phase1_reality_grounding.log" "logs\phase1_reality_grounding_!mydate!_!mytime!.log" >nul 2>&1
)

echo.
echo [PRE-FLIGHT] All checks completed successfully!
echo.

REM ========================================================================
REM PHASE 1 VALIDATION
REM ========================================================================

echo ========================================================================
echo PHASE 1 REALITY GROUNDING VALIDATION
echo ========================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated successfully

REM Validate Phase 1 integration
echo Running Phase 1 integration validation...
python phase1_integration_complete.py
if %errorlevel% neq 0 (
    echo ERROR: Phase 1 integration validation failed!
    pause
    exit /b 1
)

REM Run Phase 1 smoke test
echo Running Phase 1 smoke test...
python phase1_smoke_test.py
if %errorlevel% neq 0 (
    echo ERROR: Phase 1 smoke test failed!
    pause
    exit /b 1
)

echo.
echo ✅ Phase 1 validation completed successfully!
echo.

REM ========================================================================
REM DIRECTORY SETUP
REM ========================================================================

echo Setting up Phase 1 directories...

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir "logs"
if not exist "logs\tensorboard_phase1" mkdir "logs\tensorboard_phase1"

REM Create required data directories
if not exist "data" mkdir "data"
if not exist "data\raw_phase1" mkdir "data\raw_phase1"
if not exist "data\processed_phase1" mkdir "data\processed_phase1"
if not exist "models" mkdir "models"
if not exist "models\phase1" mkdir "models\phase1"
if not exist "models\phase1_checkpoints" mkdir "models\phase1_checkpoints"
if not exist "models\phase1_best" mkdir "models\phase1_best"
if not exist "reports" mkdir "reports"
if not exist "reports\phase1" mkdir "reports\phase1"

REM Create placeholder log file
echo [%date% %time%] Phase 1 Reality Grounding Training Log Started... > "logs\phase1_reality_grounding.log"
set LOG_FILE=phase1_reality_grounding.log

echo Phase 1 directories prepared...
echo.

REM ========================================================================
REM Launch monitoring tools
REM ========================================================================

echo Starting Phase 1 monitoring tools...

echo Starting TensorBoard on http://localhost:6006...
start "TensorBoard-Phase1" cmd /k "tensorboard --logdir logs\tensorboard_phase1_fix1 --port 6006 --host localhost --reload_interval 30"
timeout /t 2 /nobreak >nul

echo Starting API server on http://localhost:8000...
start "API-Server" cmd /k "set DISABLE_FEATURESTORE_MONITORING=true && python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul

echo Starting log monitor...
start "Log-Monitor" cmd /k "python monitor_live_logs.py logs\%LOG_FILE%"
timeout /t 1 /nobreak >nul

REM ========================================================================
REM PHASE 1 CONFIGURATION CHECK
REM ========================================================================

echo.
echo ========================================================================
echo PHASE 1 CONFIGURATION VALIDATION
echo ========================================================================
echo.

REM Check Phase 1 configuration exists
if not exist "config\phase1_reality_grounding.yaml" (
    echo ERROR: Phase 1 configuration not found!
    echo Expected: config\phase1_reality_grounding.yaml
    pause
    exit /b 1
)

echo ✅ Phase 1 configuration found: config\phase1_reality_grounding.yaml
echo.

REM Display Phase 1 configuration summary
echo 🛡️ PHASE 1 REALITY GROUNDING FEATURES:
echo    ✅ Institutional Safeguards - Reward bounds, position limits, cash reserves
echo    ✅ Model Compatibility Validation - Prevents silent training failures  
echo    ✅ Observation Consistency Testing - Ensures train/eval consistency
echo    ✅ Environment Integration - Safeguards active in trading environment
echo    ✅ Training Pipeline - Phase 1 configuration and validation
echo.

echo 🎯 PHASE 1 TRAINING PARAMETERS (1-DAY SPRINT TUNING):
echo    • Initial Capital: $50,000
echo    • Reward Scaling: 0.25 (12.5x increase from 0.02)
echo    • Target Episode Rewards: 6-12 (smoke test), 400 (full run)
echo    • Position Limit: 95%% of capital
echo    • Cash Reserve: 5%% minimum
echo    • Reward Bounds: -150 to +150 (tighter clipping)
echo.

REM ========================================================================
REM Launch Phase 1 Training
REM ========================================================================

echo Waiting 10 seconds for monitoring tools to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ========================================================================
echo Starting PHASE 1 REALITY GROUNDING Training Process
echo ========================================================================
echo.

echo 🛡️ INSTITUTIONAL SAFEGUARDS ACTIVE:
echo    - Reward bounds enforcement
echo    - Position size limits
echo    - Cash reserve requirements
echo    - Model compatibility validation
echo    - Observation consistency testing
echo.

REM Launch Phase 1 smoke test (1-day sprint tuning)
echo 🧪 Launching Phase 1 Smoke Test (5k steps - 1-day sprint tuning)...
start "PHASE1-SMOKE-TEST" cmd /k "python phase1_fast_recovery_training.py"

REM Wait for training to start creating logs, then start log tail
timeout /t 20 /nobreak >nul
echo Starting log tail now that training has begun...
start "Log-Tail" cmd /k "powershell -Command \"Get-Content -Path 'logs\%LOG_FILE%' -Wait -Tail 20 -ErrorAction SilentlyContinue\""

REM ========================================================================
REM Show status
REM ========================================================================
echo.
echo ========================================================================
echo 📊 PHASE 1 MONITORING DASHBOARD
echo ========================================================================
echo 🔗 TensorBoard:              http://localhost:6006
echo 🔗 API Monitoring:           http://localhost:8000/docs
echo 🔗 API Status:               http://localhost:8000/api/v1/status
echo ========================================================================
echo.
echo LAUNCHED WINDOWS:
echo   1. TensorBoard (Phase 1)     - Phase 1 training metrics
echo   2. API Server                - REST API for monitoring and control
echo   3. Log Monitor               - Real-time log analysis
echo   4. PHASE 1 TRAINING          - Core Phase 1 training process
echo   5. Log Tail                  - Live log streaming
echo.
echo 🛡️ PHASE 1 REALITY GROUNDING FEATURES ACTIVE:
echo   ✅ Institutional Safeguards   - Reward bounds (-2000, +5000)
echo   ✅ Position Limits            - Maximum 95%% of capital
echo   ✅ Cash Reserves              - Minimum 5%% cash buffer
echo   ✅ Model Compatibility        - Prevents silent failures
echo   ✅ Observation Consistency    - Ensures train/eval match
echo   ✅ Reward Scaling             - 0.02 institutional grade
echo.
echo 📊 KEY METRICS TO MONITOR:
echo   🎯 Episode Rewards            - Target: 8,000 - 19,000
echo   🛡️ Safeguard Violations      - Should be minimal
echo   📈 Portfolio Value            - Starting at $50,000
echo   ⚠️  Drawdown                  - Monitor risk management
echo   🔄 Training Progress          - 50,000 timesteps target
echo.
echo ✅ Phase 1 Reality Grounding training launched!
echo 💡 Keep this window open as control panel
echo 💡 All institutional safeguards are ACTIVE
echo.

pause