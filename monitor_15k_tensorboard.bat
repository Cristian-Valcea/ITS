@echo off
echo 📊 TENSORBOARD MONITORING FOR 15K VALIDATION TRAINING
echo =====================================================
echo.

cd /d "c:\Projects\IntradayJules"
call .\venv\Scripts\activate.bat

echo ✅ Starting TensorBoard for 15K validation...
echo ✅ Monitoring: logs\tensorboard_phase1_15k
echo ✅ Web interface: http://localhost:6006
echo.
echo 🎯 15K VALIDATION SUCCESS CRITERIA:
echo    - ep_rew_mean: Should stay in 4-6 range (CRITICAL)
echo    - entropy_loss: Should be around -0.7 to -1.0 (exploration)
echo    - explained_variance: Should improve over time (critic learning)
echo    - policy_gradient_loss: Should converge (stability)
echo    - value_loss: Should decrease (value function learning)
echo.
echo ⏱️ Expected training duration: 2-4 hours
echo 🚀 If metrics look good, proceed to 50K production!
echo.
echo Press Ctrl+C to stop TensorBoard
echo.

tensorboard --logdir=logs\tensorboard_phase1_15k --port=6006