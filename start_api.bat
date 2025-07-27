@echo off
echo Starting IntradayJules API Server...
echo Web Interface will be available at: http://127.0.0.1:8000/ui/dashboard
echo NVDA DQN Training at: http://127.0.0.1:8000/ui/nvda-dqn
echo API Documentation at: http://127.0.0.1:8000/docs
echo.

REM Activate virtual environment if it exists
if exist ".\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .\venv\Scripts\activate.bat
)

REM Start the API server
echo Starting FastAPI server...
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

echo.
echo API server stopped!
pause