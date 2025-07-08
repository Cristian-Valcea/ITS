@echo off
REM scripts/feature_store_gc.bat
REM Windows batch script for IntradayJules Feature Store Garbage Collection
REM
REM Schedule with Windows Task Scheduler to run daily at 2 AM:
REM schtasks /create /tn "FeatureStoreGC" /tr "C:\path\to\IntradayJules\scripts\feature_store_gc.bat" /sc daily /st 02:00

setlocal enabledelayedexpansion

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR:~0,-1%
for %%i in ("%PROJECT_ROOT%") do set PROJECT_ROOT=%%~dpi
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

set PYTHON_ENV=%PROJECT_ROOT%\venv\Scripts\python.exe
set GC_SCRIPT=%PROJECT_ROOT%\src\shared\disk_gc_service.py

REM Default settings (can be overridden by environment variables)
if not defined FEATURE_STORE_GC_RETENTION_WEEKS set FEATURE_STORE_GC_RETENTION_WEEKS=4
if not defined FEATURE_STORE_PATH set FEATURE_STORE_PATH=%USERPROFILE%\.feature_cache
if not defined FEATURE_STORE_LOG_DIR set FEATURE_STORE_LOG_DIR=%PROJECT_ROOT%\logs
if not defined FEATURE_STORE_GC_DRY_RUN set FEATURE_STORE_GC_DRY_RUN=false

set RETENTION_WEEKS=%FEATURE_STORE_GC_RETENTION_WEEKS%
set CACHE_ROOT=%FEATURE_STORE_PATH%
set LOG_DIR=%FEATURE_STORE_LOG_DIR%
set DRY_RUN=%FEATURE_STORE_GC_DRY_RUN%

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Log file with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

set LOG_FILE=%LOG_DIR%\feature_store_gc_%timestamp%.log
set JSON_OUTPUT=%LOG_DIR%\feature_store_gc_%timestamp%.json

REM Function to log with timestamp
call :log "Starting Feature Store Garbage Collection"
call :log "Cache root: %CACHE_ROOT%"
call :log "Retention: %RETENTION_WEEKS% weeks"
call :log "Dry run: %DRY_RUN%"
call :log "Log file: %LOG_FILE%"

REM Check if Python environment exists
if not exist "%PYTHON_ENV%" (
    call :log "ERROR: Python environment not found at %PYTHON_ENV%"
    exit /b 1
)

REM Check if GC script exists
if not exist "%GC_SCRIPT%" (
    call :log "ERROR: GC script not found at %GC_SCRIPT%"
    exit /b 1
)

REM Check if cache directory exists
if not exist "%CACHE_ROOT%" (
    call :log "WARNING: Cache directory does not exist: %CACHE_ROOT%"
    call :log "Creating cache directory..."
    mkdir "%CACHE_ROOT%"
)

REM Build command arguments
set GC_ARGS=--cache-root "%CACHE_ROOT%" --retention-weeks %RETENTION_WEEKS% --output-json "%JSON_OUTPUT%" --verbose

if "%DRY_RUN%"=="true" (
    set GC_ARGS=%GC_ARGS% --dry-run
)

REM Run garbage collection
call :log "Executing: %PYTHON_ENV% %GC_SCRIPT% %GC_ARGS%"

"%PYTHON_ENV%" "%GC_SCRIPT%" %GC_ARGS% >> "%LOG_FILE%" 2>&1

if %errorlevel% equ 0 (
    call :log "Garbage collection completed successfully"
    
    REM Check if JSON output exists
    if exist "%JSON_OUTPUT%" (
        call :log "JSON output saved to: %JSON_OUTPUT%"
    )
    
    set exit_code=0
) else (
    call :log "ERROR: Garbage collection failed with exit code %errorlevel%"
    set exit_code=1
)

REM Cleanup old log files (keep last 30 days)
forfiles /p "%LOG_DIR%" /m feature_store_gc_*.log /d -30 /c "cmd /c del @path" 2>nul
forfiles /p "%LOG_DIR%" /m feature_store_gc_*.json /d -30 /c "cmd /c del @path" 2>nul

call :log "Feature Store GC completed with exit code: %exit_code%"
exit /b %exit_code%

:log
echo [%date% %time%] %~1 >> "%LOG_FILE%"
echo [%date% %time%] %~1
goto :eof