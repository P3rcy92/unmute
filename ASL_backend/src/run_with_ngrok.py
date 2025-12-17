"""
Startup script that runs the FastAPI server and exposes it via ngrok.
Reads ngrok auth token from .env file.
"""

import os
import sys
import subprocess
import threading
import time
import signal

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Get ngrok auth token from environment
NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTHTOKEN')
NGROK_DOMAIN = "noncryptic-suboptical-marcene.ngrok-free.dev"

def run_server():
    """Run the FastAPI server"""
    import uvicorn
    from fastapi_websocket import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

def run_ngrok():
    """Start ngrok tunnel"""
    from pyngrok import ngrok, conf
    
    # Configure ngrok with auth token
    if NGROK_AUTH_TOKEN:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        print(f"✓ ngrok authenticated")
    else:
        print("⚠ Warning: No ngrok auth token found. Tunnel may have limited functionality.")
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    # Open tunnel with static domain
    try:
        # Kill any existing tunnels first
        ngrok.kill()
        time.sleep(1)
        
        # Start tunnel with the static domain
        public_url = ngrok.connect(8000, hostname=NGROK_DOMAIN)
        print("\n" + "="*60)
        print("🌐 NGROK TUNNEL ACTIVE")
        print("="*60)
        print(f"📡 Public URL: https://{NGROK_DOMAIN}")
        print(f"🔌 WebSocket URL: wss://{NGROK_DOMAIN}/ws")
        print("\n📋 For frontend, set:")
        print(f"   NEXT_PUBLIC_WS_URL=wss://{NGROK_DOMAIN}/ws")
        print("="*60 + "\n")
        
        return public_url
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("  🤟 ASL Sign Language Recognition Server")
    print("="*60)
    print(f"🖥️  Device: {'CUDA' if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else 'CPU'}")
    print(f"🔑 ngrok token: {'✓ Found' if NGROK_AUTH_TOKEN else '✗ Not found'}")
    print("="*60 + "\n")
    
    # Start ngrok in a separate thread
    ngrok_thread = threading.Thread(target=run_ngrok, daemon=True)
    ngrok_thread.start()
    
    # Run the FastAPI server (this blocks)
    run_server()


if __name__ == "__main__":
    main()
