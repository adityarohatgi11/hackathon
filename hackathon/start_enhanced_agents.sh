#!/bin/bash

# Enhanced Agent System Startup Script
echo "🚀 Starting Enhanced Agent System..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not available"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "deploy_enhanced_agents.py" ]; then
    echo "❌ deploy_enhanced_agents.py not found. Please run from the hackathon directory."
    exit 1
fi

# Set Python path
export PYTHONPATH="$PWD:$PYTHONPATH"

# Start the enhanced agent system
echo "🚀 Launching Enhanced Agent System..."
python3 deploy_enhanced_agents.py

echo "✅ Enhanced Agent System deployment complete." 