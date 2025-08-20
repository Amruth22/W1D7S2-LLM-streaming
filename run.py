#!/usr/bin/env python3
"""
Startup script for Smart Study Buddy System
Runs FastAPI backend (8080) and Streamlit frontend (8081)
"""

import os
import sys
import subprocess
import time
import threading
import config

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import streamlit
        import google.genai
        import faiss
        import numpy
        import requests
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["data", "data/faiss_index"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def start_fastapi():
    """Start FastAPI backend server"""
    print(f"Starting FastAPI backend on port {config.FASTAPI_PORT}...")
    try:
        subprocess.run([
            sys.executable, "fastapi_server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("FastAPI server stopped")
    except Exception as e:
        print(f"FastAPI error: {e}")

def start_streamlit():
    """Start Streamlit frontend"""
    print(f"Starting Streamlit frontend on port {config.STREAMLIT_PORT}...")
    try:
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.port", str(config.STREAMLIT_PORT),
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("Streamlit app stopped")
    except Exception as e:
        print(f"Streamlit error: {e}")

def main():
    """Main startup function"""
    print("Smart Study Buddy System Startup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check config
    if not os.path.exists("config.py"):
        print("✗ config.py not found!")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("Starting servers...")
    print(f"FastAPI Backend: http://localhost:{config.FASTAPI_PORT}")
    print(f"Streamlit Frontend: http://localhost:{config.STREAMLIT_PORT}")
    print("Press Ctrl+C to stop both servers")
    print("=" * 40)
    
    try:
        # Start FastAPI in background thread
        fastapi_thread = threading.Thread(target=start_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(3)
        
        # Start Streamlit in main thread
        start_streamlit()
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()