#!/bin/bash
# Startup script for ASL Recognition WebSocket Server with ngrok

echo "=========================================="
echo "  ASL Sign Language Recognition Server"
echo "=========================================="

# Check if we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f "../.env" ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' ../.env | xargs)
fi

# Check for ngrok auth token
if [ -z "$ngrok" ]; then
    echo "Warning: No ngrok auth token found in .env file"
    echo "Please set 'ngrok=YOUR_AUTH_TOKEN' in ../.env"
fi

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
pip install -q -r requirements_fastapi.txt

# Configure ngrok authentication if token is provided
if [ ! -z "$ngrok" ]; then
    echo "Configuring ngrok authentication..."
    ngrok config add-authtoken "$ngrok" 2>/dev/null || true
fi

# Start the FastAPI server
echo ""
echo "Starting FastAPI WebSocket server on port 8000..."
echo "Press Ctrl+C to stop"
echo ""

python fastapi_websocket.py
