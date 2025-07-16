@echo off
REM ========================================================================
REM Manual Performance Visualization Creator
REM Run this after training to create performance plots manually
REM ========================================================================

echo.
echo ========================================================================
echo 🎨 IntradayJules Performance Visualization Creator
echo ========================================================================
echo.

cd /d "C:\Projects\IntradayJules"

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

echo.
echo 🎨 Creating performance visualizations...
echo.

REM Run the visualization creator directly
python -c "
import sys
import os
sys.path.append('src')

from evaluation.performance_visualizer import PerformanceVisualizer
from datetime import datetime
import yaml

print('🎨 Loading configuration...')
with open('config/main_config_orchestrator_gpu_fixed.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('🎨 Initializing visualizer...')
visualizer = PerformanceVisualizer(config['orchestrator'])

# Demo metrics (replace with actual metrics when available)
metrics = {
    'total_return': 0.15,
    'sharpe_ratio': 1.8,
    'max_drawdown': -0.08,
    'avg_turnover': 2.1,
    'num_trades': 150,
    'win_rate': 0.58,
    'sortino_ratio': 2.2,
    'calmar_ratio': 1.5,
    'volatility': 0.12
}

model_name = f'NVDA_GPU_Fixed_{datetime.now().strftime(\"%Y%m%d\")}'

print('🎨 Creating performance plots...')
plot_files = visualizer.create_performance_plots(metrics, None, model_name)

if plot_files:
    print('✅ Performance plots created successfully!')
    print('📁 Generated files:')
    for plot_file in plot_files:
        print(f'   📊 {plot_file}')
    
    print('🖼️ Opening plots...')
    import subprocess
    for plot_file in plot_files:
        try:
            os.startfile(plot_file)
        except:
            print(f'Could not open {plot_file}')
else:
    print('❌ No plots were created')
"

echo.
echo ========================================================================
echo 🎉 Performance visualization creation completed!
echo ========================================================================
pause