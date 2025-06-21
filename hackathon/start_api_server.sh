#!/bin/bash

# GridPilot-GT FastAPI Server Startup Script

echo "🚀 Starting GridPilot-GT FastAPI Backend Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📥 Installing/updating dependencies..."
pip install -q fastapi uvicorn websockets pydantic

# Install GridPilot-GT dependencies if not already installed
pip install -q pandas numpy prophet scikit-learn httpx toml

echo "✅ Dependencies installed"

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Documentation available at: http://localhost:8000/docs"
echo "🔄 WebSocket endpoint: ws://localhost:8000/ws/live-data"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

# Run with auto-reload for development
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload --log-level info 