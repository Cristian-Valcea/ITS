@echo off
echo 🔍 15K VALIDATION RESULTS ANALYSIS
echo ==================================
echo.

cd /d "c:\Projects\IntradayJules"

echo 📊 CHECKING 15K VALIDATION RESULTS...
echo.

echo 📁 Log Files:
dir /b logs\*15k* 2>nul
if errorlevel 1 echo    No 15K log files found yet
echo.

echo 📁 TensorBoard Data:
dir /b logs\tensorboard_phase1_15k 2>nul
if errorlevel 1 echo    No TensorBoard data found yet
echo.

echo 📁 Model Files:
dir /b models\*phase1* 2>nul
if errorlevel 1 echo    No model files found yet
echo.

echo 🎯 VALIDATION DECISION CRITERIA:
echo ================================
echo.
echo ✅ PROCEED TO 50K IF:
echo    - ep_rew_mean stable in 4-6 range
echo    - entropy_loss healthy (-0.7 to -1.0)
echo    - explained_variance improving (≥0.1)
echo    - No prolonged drawdown periods
echo    - Penalty frequency reasonable (≤20%%)
echo.
echo ❌ NEED MORE TUNING IF:
echo    - ep_rew_mean outside 4-6 range
echo    - entropy_loss too negative (≤-1.5)
echo    - explained_variance not improving
echo    - Excessive penalty frequency (≥30%%)
echo    - Unstable reward patterns
echo.
echo 🚀 If validation passes, run: launch_50k_training.bat
echo 🔧 If validation fails, analyze and retune parameters
echo.
pause