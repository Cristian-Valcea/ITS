@echo off
echo 🚀 INTRADAYJULES 50K PRODUCTION TRAINING LAUNCHER
echo ================================================
echo.
echo Starting 50K production training run...
echo Expected duration: 8-12 hours
echo TensorBoard: logs/tensorboard_phase1_50k
echo.

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo ✅ Starting training...
echo.

python phase1_fast_recovery_training.py

echo.
echo 🎉 Training completed!
echo Check logs/ directory for results
echo.
pause