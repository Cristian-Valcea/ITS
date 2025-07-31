#!/bin/bash
# 📊 Training Monitoring Script - Multiple monitoring options

echo "📊 200K Training Monitoring Options"
echo "=================================="

# Check if training session exists
if ! tmux has-session -t training200k 2>/dev/null; then
    echo "❌ Training session 'training200k' not found"
    echo "   Start training first: ./launch_200k_tmux.sh"
    exit 1
fi

echo "✅ Training session 'training200k' is active"
echo ""

# Show monitoring options
echo "📋 MONITORING OPTIONS:"
echo "  1. Live dashboard (refreshes every 30s)"
echo "  2. Attach to training session"
echo "  3. Tail training logs"
echo "  4. Show current status"
echo "  5. TensorBoard (web interface)"
echo "  6. Control training (pause/resume/stop)"
echo ""

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo "🎛️ Starting live dashboard..."
        python training_control_dashboard.py --watch
        ;;
    2)
        echo "🔗 Attaching to training session..."
        echo "💡 Press Ctrl+B then D to detach"
        tmux attach -t training200k
        ;;
    3)
        echo "📋 Tailing training logs..."
        echo "💡 Press Ctrl+C to exit"
        # Find latest log file
        latest_log=$(ls -t logs/200k_training_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            tail -f "$latest_log"
        else
            echo "❌ No training logs found"
        fi
        ;;
    4)
        echo "📊 Current training status:"
        python training_control_dashboard.py --action status
        ;;
    5)
        echo "🌐 Starting TensorBoard..."
        echo "💡 Open browser to: http://localhost:6006"
        echo "💡 Press Ctrl+C to stop TensorBoard"
        tensorboard --logdir runs/ --port 6006
        ;;
    6)
        echo "🎛️ Training Control Options:"
        echo "  pause  - Pause training (Ctrl+Z)"
        echo "  resume - Resume training (fg)"
        echo "  stop   - Stop training gracefully (Ctrl+C)"
        echo "  kill   - Kill training session"
        echo ""
        read -p "Enter control action: " action
        python training_control_dashboard.py --action "$action"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac