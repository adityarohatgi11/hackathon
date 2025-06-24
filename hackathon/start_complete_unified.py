#!/usr/bin/env python3
"""
MARA Complete Unified Platform Launcher
Starts the complete unified dashboard with ALL functionality in one interface.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 80)
    print("🚀 MARA COMPLETE UNIFIED PLATFORM")
    print("=" * 80)
    print("🎯 ALL-IN-ONE: Energy Management + AI Agents + Advanced Analytics")
    print("⚡ Real-time Trading • 🤖 Intelligent Agents • 🧠 AI Insights • 📈 Analytics")
    print("🔬 Machine Learning • 🧪 Live Demos • 🎮 Q-Learning • 📊 Forecasting")
    print("=" * 80)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'), 
        ('pandas', 'pandas'),
        ('numpy', 'numpy')
    ]
    
    missing = []
    for package, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package} - Ready")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - Missing")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", *missing
            ])
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"   pip install {' '.join(missing)}")
            return False
    
    return True

def check_port(port, timeout=5):
    """Check if a port is responding."""
    import socket
    
    for _ in range(timeout):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except:
            pass
        time.sleep(1)
    return False

def start_api_server():
    """Start the API server in background."""
    print("🚀 Starting API Server...")
    try:
        api_process = subprocess.Popen(
            [sys.executable, "api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Give it time to start
        time.sleep(3)
        
        if api_process.poll() is None:
            print("✅ API Server started on port 8000")
            return api_process
        else:
            print("⚠️  API Server failed to start (continuing without it)")
            return None
    except Exception as e:
        print(f"⚠️  API Server not available: {e}")
        return None

def open_dashboard():
    """Open the dashboard in browser."""
    time.sleep(3)  # Give dashboard time to start
    try:
        webbrowser.open('http://localhost:8503')
        print("🌐 Opening complete unified dashboard in your browser...")
    except:
        print("⚠️  Please manually open http://localhost:8503 in your browser")

def main():
    """Main launcher function."""
    print_banner()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("❌ Please resolve dependency issues before continuing.")
        return 1
    
    # Check if we're in the right directory
    required_files = [
        'ui/complete_dashboard.py',
        'api_server.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please run from the hackathon directory.")
        return 1
    
    print("\n🚀 Starting MARA Complete Unified Platform...")
    
    # Start API server in background (optional)
    api_process = start_api_server()
    
    # Start the complete unified dashboard
    print("🚀 Starting Complete Unified Dashboard...")
    print("🎯 This includes ALL functionality:")
    print("   • Energy Management (from port 8502)")
    print("   • AI Agent System (from port 8501)")  
    print("   • Live Demonstrations")
    print("   • AI Insights & Analysis")
    print("   • Advanced Analytics")
    print("   • Machine Learning")
    print("   • System Status")
    print("")
    
    # Open browser in background
    import threading
    browser_thread = threading.Thread(target=open_dashboard)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start the complete dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "ui/complete_dashboard.py",
            "--server.port", "8503",
            "--server.address", "localhost", 
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🌐 Complete Dashboard will be available at: http://localhost:8503")
        print("🛑 Press Ctrl+C to stop all services")
        print("=" * 80)
        
        # Run streamlit dashboard
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Stop API server if running
        if api_process and api_process.poll() is None:
            api_process.terminate()
            api_process.wait()
            print("✅ API Server stopped")
        
        print("✅ Complete Unified Platform stopped successfully!")
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 