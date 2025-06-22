#!/bin/bash

# MARA Energy Management Dashboard Deployment Script
# This script deploys the GridPilot-GT dashboard for frontend access

echo "🚀 Deploying MARA Energy Management Dashboard..."

# Kill any existing Streamlit processes
echo "📋 Stopping existing dashboard instances..."
pkill -f "streamlit run ui/dashboard.py" 2>/dev/null || true
sleep 2

# Check if required dependencies are available
echo "🔍 Checking dependencies..."
python -c "import streamlit, pandas, plotly" 2>/dev/null || {
    echo "❌ Missing required dependencies. Installing..."
    pip install streamlit pandas plotly numpy
}

# Test MARA API connection
echo "🌐 Testing MARA API connection..."
python -c "
import sys
sys.path.append('.')
from api_client.client import test_mara_api_connection
result = test_mara_api_connection()
if result.get('overall_status') == 'operational':
    print('✅ MARA API connection successful')
else:
    print('⚠️ MARA API limited functionality - will use fallback data')
" || echo "⚠️ API test failed - dashboard will use fallback data"

# Find available port
PORT=8507
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo "🎯 Starting dashboard on port $PORT..."

# Start the dashboard
echo "📊 Launching MARA Energy Management Dashboard..."
echo "🌍 Dashboard will be available at:"
echo "   Local:   http://localhost:$PORT"
echo "   Network: http://0.0.0.0:$PORT"
echo ""
echo "💡 Features available:"
echo "   • Real-time energy data visualization"
echo "   • AI-powered analysis and insights"
echo "   • Q-Learning optimization"
echo "   • Stochastic forecasting models"
echo "   • Advanced game theory auctions"
echo "   • Professional DeepMind-inspired UI"
echo ""
echo "🔄 Starting in 3 seconds..."
sleep 3

# Launch Streamlit
streamlit run ui/dashboard.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false 