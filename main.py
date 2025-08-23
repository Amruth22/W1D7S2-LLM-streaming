#!/usr/bin/env python3
"""
Startup script for Smart Study Buddy API System
Runs FastAPI backend server on port 8080
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
    """Start FastAPI API server"""
    print(f"Starting FastAPI API server on port {config.FASTAPI_PORT}...")
    try:
        subprocess.run([
            sys.executable, "fastapi_server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("API server stopped")
    except Exception as e:
        print(f"API server error: {e}")

def main():
    """Main startup function"""
    print("Smart Study Buddy API Server Startup")
    print("=" * 45)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check config
    if not os.path.exists("config.py"):
        print("✗ config.py not found!")
        sys.exit(1)
    
    print("\n" + "=" * 45)
    print("Starting API server...")
    print(f"API Server: http://localhost:{config.FASTAPI_PORT}")
    print(f"API Docs: http://localhost:{config.FASTAPI_PORT}/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 45)
    
    try:
        # Start FastAPI server
        start_fastapi()
        
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
