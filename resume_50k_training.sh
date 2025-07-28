#!/bin/bash
# Resume 50K Dual-Ticker Training from Checkpoint
# IntradayJules Trading System - Training Continuation
# Created: July 28, 2025 Evening

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/home/cristian/IntradayTrading/ITS"
VENV_PATH="$PROJECT_ROOT/venv"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/dual_ticker_50k_10000_steps.zip"
LOG_DIR="$PROJECT_ROOT/logs"

echo -e "${BLUE}🚀 IntradayJules 50K Training Resumption${NC}"
echo -e "${BLUE}======================================${NC}"
echo "Project Root: $PROJECT_ROOT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"
echo -e "${YELLOW}📁 Changed to project directory: $(pwd)${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at: $VENV_PATH${NC}"
    echo -e "${RED}Please create virtual environment first${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Virtual environment activated: $VIRTUAL_ENV${NC}"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${BLUE}🐍 Python version: $PYTHON_VERSION${NC}"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}❌ Checkpoint not found: $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}Available checkpoints:${NC}"
    ls -la "$PROJECT_ROOT/checkpoints/" || echo "No checkpoints directory found"
    exit 1
fi

# Check checkpoint details
CHECKPOINT_SIZE=$(stat -c%s "$CHECKPOINT_PATH")
echo -e "${GREEN}✅ Checkpoint found: $(basename $CHECKPOINT_PATH)${NC}"
echo -e "${BLUE}   Size: $CHECKPOINT_SIZE bytes${NC}"
echo -e "${BLUE}   Modified: $(stat -c%y "$CHECKPOINT_PATH")${NC}"

# Verify training script exists
TRAINING_SCRIPT="$PROJECT_ROOT/train_50k_dual_ticker_resume.py"
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo -e "${RED}❌ Enhanced training script not found: $TRAINING_SCRIPT${NC}"
    echo -e "${YELLOW}Falling back to original script...${NC}"
    TRAINING_SCRIPT="$PROJECT_ROOT/train_50k_dual_ticker.py"
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        echo -e "${RED}❌ No training script found${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✅ Training script found: $(basename $TRAINING_SCRIPT)${NC}"

# Check GPU availability
echo -e "${YELLOW}🔍 Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✅ GPU detected: $GPU_INFO${NC}"
    
    # Check CUDA availability in Python
    CUDA_AVAILABLE=$(python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>/dev/null || echo "CUDA check failed")
    echo -e "${BLUE}   $CUDA_AVAILABLE${NC}"
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found - GPU may not be available${NC}"
fi

# Check required dependencies
echo -e "${YELLOW}🔍 Checking key dependencies...${NC}"
python -c "
import sys
required_packages = [
    'torch', 'stable_baselines3', 'sb3_contrib', 
    'numpy', 'pandas', 'psycopg2', 'tensorboard'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages available')
" || {
    echo -e "${RED}❌ Missing required dependencies${NC}"
    exit 1
}

# Create timestamped log directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SESSION_LOG_DIR="$LOG_DIR/resume_50k_$TIMESTAMP"
mkdir -p "$SESSION_LOG_DIR"
echo -e "${GREEN}✅ Created log directory: $SESSION_LOG_DIR${NC}"

# Check TimescaleDB connection
echo -e "${YELLOW}🔍 Testing TimescaleDB connection...${NC}"
python -c "
import psycopg2
import os
try:
    # Try to connect to TimescaleDB
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='trading_data',
        user='postgres',
        password=os.environ.get('TIMESCALEDB_PASSWORD', 'your_password')
    )
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM trading.market_data;')
    count = cursor.fetchone()[0]
    print(f'✅ TimescaleDB connected - {count} market data records')
    conn.close()
except Exception as e:
    print(f'⚠️  TimescaleDB connection issue: {e}')
    print('Training will use fallback data if needed')
"

# Display training parameters
echo -e "${BLUE}📊 Training Configuration:${NC}"
echo "   Model: RecurrentPPO with LSTM"
echo "   Environment: DualTickerTradingEnv (NVDA + MSFT)"
echo "   Total Steps: 50,000"
echo "   Resume From: 10,000 steps (29% complete)"
echo "   Remaining: 40,000 steps (~5.5 minutes at 121 steps/sec)"
echo "   Checkpoints: Every 10,000 steps"
echo "   Log Directory: $SESSION_LOG_DIR"

# Confirmation prompt
echo ""
echo -e "${YELLOW}🚨 Ready to resume 50K training from checkpoint${NC}"
echo -e "${YELLOW}   This will continue training for ~40,000 more steps${NC}"
echo -e "${YELLOW}   Estimated completion time: ~5-6 minutes${NC}"
echo ""
read -p "Continue with training resumption? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training resumption cancelled by user${NC}"
    exit 0
fi

# Start training with comprehensive logging
echo -e "${GREEN}🚀 Starting 50K training resumption...${NC}"
echo -e "${BLUE}   Checkpoint: $(basename $CHECKPOINT_PATH)${NC}"
echo -e "${BLUE}   Logs: $SESSION_LOG_DIR${NC}"
echo -e "${BLUE}   Start Time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

# Create training command with proper logging
TRAINING_LOG="$SESSION_LOG_DIR/training_output.log"
ERROR_LOG="$SESSION_LOG_DIR/training_errors.log"

# Run training with both stdout and stderr capture
python "$(basename $TRAINING_SCRIPT)" \
    --resume-from-checkpoint "$CHECKPOINT_PATH" \
    --log-dir "$SESSION_LOG_DIR" \
    --tensorboard-log "$SESSION_LOG_DIR/tensorboard" \
    --total-steps 50000 \
    2> >(tee "$ERROR_LOG" >&2) \
    | tee "$TRAINING_LOG"

TRAINING_EXIT_CODE=$?

echo ""
echo -e "${BLUE}📊 Training Session Complete${NC}"
echo -e "${BLUE}=========================${NC}"
echo "Exit Code: $TRAINING_EXIT_CODE"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs Saved: $SESSION_LOG_DIR"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    
    # Check for new checkpoints
    echo -e "${YELLOW}🔍 Checking for new checkpoints...${NC}"
    find "$PROJECT_ROOT/checkpoints" -name "dual_ticker_50k_*_steps.zip" -newer "$CHECKPOINT_PATH" | while read checkpoint; do
        echo -e "${GREEN}   New checkpoint: $(basename $checkpoint)${NC}"
    done
    
    # Display final model location
    FINAL_MODEL="$PROJECT_ROOT/models/dual_ticker_50k_final.zip"
    if [ -f "$FINAL_MODEL" ]; then
        echo -e "${GREEN}✅ Final model saved: $(basename $FINAL_MODEL)${NC}"
        echo -e "${BLUE}   Size: $(stat -c%s "$FINAL_MODEL") bytes${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 50K Training Successfully Completed!${NC}"
    echo -e "${BLUE}   Ready for 200K production training${NC}"
    
else
    echo -e "${RED}❌ Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
    echo -e "${YELLOW}   Check logs for details: $SESSION_LOG_DIR${NC}"
    echo -e "${YELLOW}   Error log: $ERROR_LOG${NC}"
    echo -e "${YELLOW}   Training log: $TRAINING_LOG${NC}"
fi

echo ""
echo -e "${BLUE}📁 Session files saved in: $SESSION_LOG_DIR${NC}"
echo -e "${BLUE}🔧 Virtual environment remains active${NC}"
echo -e "${BLUE}📊 Use 'tensorboard --logdir $SESSION_LOG_DIR/tensorboard' to view metrics${NC}"