#!/usr/bin/env python3
"""
Simple Enhanced Agent System Deployment
Starts the enhanced agent system directly without complex orchestration.
"""

import sys
import time
import signal
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 Enhanced Agent System - Simple Deployment")
print("=" * 60)

try:
    from agents.enhanced_system_manager import EnhancedSystemManager, SystemConfig
    print("✅ Enhanced system manager imported successfully")
    
    # Create configuration
    config = SystemConfig(
        data_fetch_interval=60,
        enable_monitoring=True,
        restart_on_failure=True,
        health_check_interval=30
    )
    print("✅ System configuration created")
    
    # Create system manager
    manager = EnhancedSystemManager(config)
    print("✅ System manager initialized")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}. Shutting down...")
        try:
            manager.shutdown_system()
            print("✅ System shutdown complete")
        except:
            print("⚠️  Forced shutdown")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n🎉 Starting Enhanced Agent System...")
    print("📊 The system will start data and strategy agents")
    print("🔄 Health monitoring enabled with 30-second intervals")
    print("🛑 Press Ctrl+C to stop the system")
    print("=" * 60)
    
    # Start the system (this will run indefinitely)
    manager.start_system(agents=["data", "strategy"])
    
except KeyboardInterrupt:
    print("\n🛑 System interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    print("✅ Enhanced Agent System deployment complete") 