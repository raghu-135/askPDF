#!/bin/bash
set -e

# Start s6-init (from base image) to run the browser
# This starts the Brave browser with the display and VNC
/init &
S6_PID=$!

# Wait for browser to start and CDP to be available
echo "Waiting for browser to start..."
sleep 8

# Verify CDP is accessible
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:9222/json/version > /dev/null 2>&1; then
        echo "Browser CDP is ready!"
        break
    fi
    echo "Waiting for CDP... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: CDP not available after $MAX_RETRIES retries, starting API anyway..."
fi

# Start the capture API
echo "Starting capture API on port 8080..."
python3 /app/main.py &
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
