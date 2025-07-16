@echo off
echo ========================================================================
echo 🧪 Testing IntradayJules Environment
echo ========================================================================

cd /d "C:\Projects\IntradayJules"

echo 🔧 Activating virtual environment...
call venv_fresh\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    echo Please check if venv_fresh directory exists
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo.

echo 🐍 Python version:
python --version
echo.

echo 📦 Key packages:
python -c "import stable_baselines3; print('✅ stable-baselines3:', stable_baselines3.__version__)"
python -c "import gymnasium; print('✅ gymnasium:', gymnasium.__version__)"
python -c "import torch; print('✅ torch:', torch.__version__)"
python -c "import tensorboard; print('✅ tensorboard: available')"
echo.

echo 🔍 Testing system validation:
python test_system_status.py
echo.

echo ========================================================================
echo 🎯 Environment Test Complete
echo ========================================================================
pause