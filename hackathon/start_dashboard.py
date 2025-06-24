#!/usr/bin/env python3
"""
Enhanced Agent Dashboard Startup Script
Launches the beautiful web interface for the enhanced agent system.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available."""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False

def main():
    print("🚀 Enhanced Agent System Dashboard Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    dashboard_path = Path("ui/enhanced_agent_dashboard.py")
    if not dashboard_path.exists():
        print("❌ Dashboard file not found. Please run from the hackathon directory.")
        return 1
    
    # Check Streamlit installation
    if not check_streamlit():
        print("📦 Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Please install Streamlit manually: pip install streamlit plotly")
            return 1
    
    print("✅ All dependencies ready!")
    print("🌐 Starting Enhanced Agent Dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    print("✅ Dashboard shutdown complete")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 