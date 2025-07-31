#!/bin/bash
# 🚀 Launch 200K Training in tmux for background execution

echo "🚀 Launching 200K Dual-Ticker Training in tmux"
echo "📊 Estimated duration: 8-10 hours on RTX 3060"
echo ""

# Check if tmux session already exists
if tmux has-session -t training200k 2>/dev/null; then
    echo "⚠️  tmux session 'training200k' already exists"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t training200k"
    echo "  2. Kill existing session: tmux kill-session -t training200k"
    echo ""
    read -p "Kill existing session and start new? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t training200k
        echo "✅ Killed existing session"
    else
        echo "Exiting. Use: tmux attach -t training200k"
        exit 0
    fi
fi

# Activate virtual environment and launch training
echo "🔄 Creating new tmux session 'training200k'"
tmux new-session -d -s training200k -c "$(pwd)"

# Send commands to tmux session
tmux send-keys -t training200k 'source venv/bin/activate' Enter
tmux send-keys -t training200k 'echo "🚀 Starting 200K Dual-Ticker Training"' Enter
tmux send-keys -t training200k 'echo "💡 Monitor progress: tail -f logs/200k_training_*.log"' Enter
tmux send-keys -t training200k 'echo "📊 TensorBoard: tensorboard --logdir runs/"' Enter
tmux send-keys -t training200k 'echo ""' Enter
tmux send-keys -t training200k 'python launch_200k_dual_ticker_training.py' Enter

echo "✅ Training launched in tmux session 'training200k'"
echo ""
echo "📋 Useful commands:"
echo "  • Attach to session:    tmux attach -t training200k"
echo "  • Detach from session:  Ctrl+B, then D"
echo "  • Kill session:         tmux kill-session -t training200k"
echo "  • Monitor logs:         tail -f logs/200k_training_*.log"
echo "  • TensorBoard:          tensorboard --logdir runs/"
echo ""
echo "🎯 Training will run for ~8-10 hours. Check progress periodically!"