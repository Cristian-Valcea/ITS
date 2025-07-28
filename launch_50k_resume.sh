#!/bin/bash
# Quick launcher for 50K training resumption
# IntradayJules Trading System

echo "🚀 IntradayJules 50K Training Resumption"
echo "========================================"
echo ""

# Change to project directory
cd /home/cristian/IntradayTrading/ITS

# Check if checkpoint exists
if [ -f "checkpoints/dual_ticker_50k_10000_steps.zip" ]; then
    echo "✅ Checkpoint found: dual_ticker_50k_10000_steps.zip"
    echo "📊 Ready to resume from 10K steps (29% complete)"
    echo ""
    
    # Launch the comprehensive resume script
    ./resume_50k_training.sh
else
    echo "❌ Checkpoint not found: checkpoints/dual_ticker_50k_10000_steps.zip"
    echo ""
    echo "Available checkpoints:"
    ls -la checkpoints/ 2>/dev/null || echo "No checkpoints directory found"
    echo ""
    echo "To start new training, use: python train_50k_dual_ticker.py"
fi