#!/usr/bin/env python3
"""
Startup script for the Vector Search API
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import numpy
        import pymongo
        import openai
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_env_variables():
    """Check if required environment variables are set."""
    required_vars = [
        "MONGODB_CONNECTION_STRING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    
    print("✅ Environment variables are configured")
    return True

def main():
    """Main startup function."""
    print("🚀 Starting AP Gov QA Vector Search API")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment variables
    if not check_env_variables():
        sys.exit(1)
    
    # Start the API server
    print("\n🌐 Starting API server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 API Root endpoint: http://localhost:8000")
    print("❤️  Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Change to the API directory
        api_path = Path(__file__).parent / "src" / "app" / "searcher_api.py"
        subprocess.run([sys.executable, str(api_path)], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()