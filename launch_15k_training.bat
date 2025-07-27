@echo off
echo 🎯 INTRADAYJULES 15K ENGINEERING VALIDATION LAUNCHER
echo ==================================================
echo.
echo 🧠 SMART ENGINEERING APPROACH:
echo    - 15K timesteps (3x longer than 5K smoke test)
echo    - Expected duration: 2-4 hours (manageable)
echo    - Perfect for validation before full 50K commitment
echo    - All precision calibrations active
echo.
echo Starting 15K validation training run...
echo TensorBoard: logs/tensorboard_phase1_15k
echo.

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo ✅ Starting 15K validation training...
echo.

python phase1_fast_recovery_training.py

echo.
echo 🎉 15K Training completed!
echo 📊 Check logs/ directory for results
echo 📈 Review TensorBoard metrics for validation
echo.
echo 🚀 If results look good, proceed to 50K production run!
echo.
pause