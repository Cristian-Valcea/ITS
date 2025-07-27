@echo off
echo 📊 TENSORBOARD MONITORING FOR 50K TRAINING
echo ==========================================
echo.

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

echo ✅ Starting TensorBoard...
echo ✅ Monitoring: logs\tensorboard_phase1_50k
echo ✅ Web interface: http://localhost:6006
echo.
echo 🔍 Key metrics to watch:
echo    - ep_rew_mean: Should stay in 4-6 range
echo    - entropy_loss: Should be around -0.7 to -1.0
echo    - explained_variance: Should improve over time
echo.
echo Press Ctrl+C to stop TensorBoard
echo.

tensorboard --logdir=logs\tensorboard_phase1_50k --port=6006