#!/bin/bash
# Auto-restart script for Research Compass
# Monitors the application and restarts on crashes or changes

PORT="${1:-7860}"
SHARE="${2:-false}"

echo "============================================================"
echo "🧭 Research Compass - Auto-Restart Monitor"
echo "============================================================"
echo ""
echo "  Port: $PORT"
echo "  Share: $SHARE"
echo "  Press Ctrl+C to stop"
echo ""
echo "============================================================"
echo ""

while true; do
    echo "🚀 Starting application..."
    
    if [ "$SHARE" = "true" ]; then
        python3 gradio_launcher.py --port "$PORT" --share
    else
        python3 gradio_launcher.py --port "$PORT"
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Application stopped normally"
        break
    else
        echo ""
        echo "⚠️  Application crashed (exit code: $EXIT_CODE)"
        echo "♻️  Restarting in 3 seconds..."
        sleep 3
    fi
done

echo ""
echo "👋 Auto-restart monitor stopped"
