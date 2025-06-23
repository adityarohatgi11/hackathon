#!/usr/bin/env python3
"""
MARA Unified Platform Launcher
Starts all dashboard components and provides unified access.
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path
import webbrowser

def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 80)
    print("🚀 MARA UNIFIED PLATFORM LAUNCHER")
    print("=" * 80)
    print("🎯 Advanced Energy Management & AI Agent Platform")
    print("⚡ Real-time Trading • 🤖 Intelligent Agents • 🧠 AI Insights")
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

def start_service(name, command, port, cwd=None):
    """Start a service in the background."""
    print(f"🚀 Starting {name} on port {port}...")
    
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ {name} started successfully!")
            return process
        else:
            print(f"❌ {name} failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting {name}: {e}")
        return None

def check_port(port, timeout=10):
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

def display_dashboard_info():
    """Display information about available dashboards."""
    print("\n" + "=" * 80)
    print("🌐 DASHBOARD ACCESS INFORMATION")
    print("=" * 80)
    
    dashboards = [
        {
            'name': '🎯 Unified Platform Dashboard',
            'url': 'http://localhost:8503',
            'port': 8503,
            'description': 'Main hub with navigation to all components'
        },
        {
            'name': '⚡ Energy Management Dashboard', 
            'url': 'http://localhost:8502',
            'port': 8502,
            'description': 'Advanced energy trading & optimization'
        },
        {
            'name': '🤖 Enhanced Agent Dashboard',
            'url': 'http://localhost:8501', 
            'port': 8501,
            'description': 'AI agent system monitoring & control'
        },
        {
            'name': '📡 API Server & Documentation',
            'url': 'http://localhost:8000',
            'port': 8000,
            'description': 'REST API endpoints & interactive docs'
        }
    ]
    
    for dashboard in dashboards:
        status = "🟢 ONLINE" if check_port(dashboard['port'], timeout=3) else "🔴 OFFLINE"
        print(f"\n{dashboard['name']}")
        print(f"   URL: {dashboard['url']}")
        print(f"   Status: {status}")
        print(f"   Description: {dashboard['description']}")
    
    print("\n" + "=" * 80)

def open_main_dashboard():
    """Open the main unified dashboard in browser."""
    time.sleep(5)  # Give services time to start
    try:
        webbrowser.open('http://localhost:8503')
        print("🌐 Opening unified dashboard in your browser...")
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
        'ui/unified_dashboard.py',
        'ui/dashboard.py', 
        'ui/enhanced_agent_dashboard.py',
        'api_server.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please run from the hackathon directory.")
        return 1
    
    print("\n🚀 Starting MARA Unified Platform services...")
    
    services = []
    
    # Start API Server
    api_process = start_service(
        "API Server",
        [sys.executable, "api_server.py"],
        8000
    )
    if api_process:
        services.append(("API Server", api_process))
    
    # Start Enhanced Agent Dashboard
    agent_process = start_service(
        "Enhanced Agent Dashboard",
        [
            sys.executable, "-m", "streamlit", "run", 
            "ui/enhanced_agent_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ],
        8501
    )
    if agent_process:
        services.append(("Enhanced Agent Dashboard", agent_process))
    
    # Start Energy Management Dashboard
    energy_process = start_service(
        "Energy Management Dashboard",
        [
            sys.executable, "-m", "streamlit", "run",
            "ui/dashboard.py", 
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ],
        8502
    )
    if energy_process:
        services.append(("Energy Management Dashboard", energy_process))
    
    # Start Unified Platform Dashboard
    unified_process = start_service(
        "Unified Platform Dashboard",
        [
            sys.executable, "-m", "streamlit", "run",
            "ui/unified_dashboard.py",
            "--server.port", "8503", 
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ],
        8503
    )
    if unified_process:
        services.append(("Unified Platform Dashboard", unified_process))
    
    # Display dashboard information
    display_dashboard_info()
    
    # Open main dashboard in browser
    threading.Thread(target=open_main_dashboard, daemon=True).start()
    
    print("\n🎉 MARA Unified Platform is now running!")
    print("💡 TIP: Start with the Unified Platform Dashboard for easy navigation")
    print("🛑 Press Ctrl+C to stop all services")
    print("\n⏳ Monitoring services... (Press Ctrl+C to exit)")
    
    try:
        # Monitor services
        while True:
            time.sleep(30)
            
            # Check if any service has died
            active_services = []
            for name, process in services:
                if process.poll() is None:
                    active_services.append((name, process))
                else:
                    print(f"⚠️  {name} stopped unexpectedly")
            
            services = active_services
            
            if not services:
                print("❌ All services have stopped")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MARA Unified Platform...")
        
        # Terminate all services
        for name, process in services:
            try:
                process.terminate()
                print(f"✅ {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"⚠️  {name} force stopped")
                except:
                    print(f"❌ Could not stop {name}")
        
        print("✅ MARA Unified Platform shutdown complete")
        return 0
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 