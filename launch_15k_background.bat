@echo off
echo 🎯 INTRADAYJULES 15K ENGINEERING VALIDATION - BACKGROUND MODE
echo =============================================================
echo.

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo ✅ Starting 15K validation training in background...
echo ✅ Output will be logged to: logs\15k_training_%TIMESTAMP%.log
echo ✅ TensorBoard available at: logs\tensorboard_phase1_15k
echo.
echo 🔍 Monitor progress with:
echo    tensorboard --logdir=logs\tensorboard_phase1_15k --port=6006
echo.
echo ⏱️ Expected duration: 2-4 hours
echo 📊 Training started at: %date% %time%
echo.

start /min cmd /c "python phase1_fast_recovery_training.py > logs\15k_training_%TIMESTAMP%.log 2>&1"

echo ✅ 15K validation training launched in background!
echo ✅ Check Task Manager for "python.exe" process
echo ✅ Monitor logs\15k_training_%TIMESTAMP%.log for progress
echo.
echo 🎯 This is the smart engineering approach - validate before 50K!
echo.
pause