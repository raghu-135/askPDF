#!/bin/bash
set -e

# Start s6-init (from base image) to run the browser in parallel
# This starts the Brave browser with the display and VNC
/init &
S6_PID=$!

echo "Starting browser (s6-init PID: $S6_PID)..."

# Adaptive wait for CDP with early exit
MAX_RETRIES=30
RETRY_COUNT=0
echo "Waiting for browser CDP to be ready..."

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:9222/json/version > /dev/null 2>&1; then
        echo "Browser CDP is ready after ${RETRY_COUNT} retries (~$((RETRY_COUNT * 2))s)"
        break
    fi
    # Exponential backoff: start fast, slow down after 10 attempts
    if [ $RETRY_COUNT -lt 10 ]; then
        sleep 0.5
    else
        sleep 1
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: CDP not available after ~60s, starting API anyway..."
fi

# Start the capture API (already in venv from Dockerfile)
echo "Starting capture API on port 8080..."
/opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080 --log-level info --app-dir /app &
API_PID=$!

# Handle shutdown gracefully
cleanup() {
    echo "Shutting down..."
    kill $API_PID 2>/dev/null || true
    kill $S6_PID 2>/dev/null || true
    wait
}
trap cleanup SIGTERM SIGINT

# Keep container running
wait
