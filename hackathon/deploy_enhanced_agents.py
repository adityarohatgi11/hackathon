#!/usr/bin/env python3
"""
Enhanced Agent System Deployment Script
Deploys and manages the enhanced agent system with health monitoring.
"""

import os
import sys
import time
import signal
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.enhanced_system_manager import EnhancedSystemManager, SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedAgentDeployment:
    """Enhanced agent system deployment manager."""
    
    def __init__(self):
        self.system_manager = None
        self.cache_dir = tempfile.mkdtemp(prefix="enhanced_agents_")
        self.running = False
        self.start_time = None
        
    def setup_environment(self):
        """Setup deployment environment."""
        logger.info("🚀 Setting up Enhanced Agent System deployment environment...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        
        # Check Python dependencies
        try:
            import pandas
            import numpy
            logger.info("✅ Core dependencies verified")
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            return False
            
        logger.info(f"✅ Cache directory: {self.cache_dir}")
        return True
    
    def deploy_system(self):
        """Deploy the enhanced agent system."""
        logger.info("🚀 Deploying Enhanced Agent System...")
        
        try:
            # Configure system
            config = SystemConfig(
                data_fetch_interval=60,  # 1 minute data fetch
                enable_monitoring=True,
                restart_on_failure=True,
                health_check_interval=30  # 30 second health checks
            )
            
            # Initialize system manager
            self.system_manager = EnhancedSystemManager(config)
            logger.info("✅ System Manager initialized")
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("✅ Signal handlers configured")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def start_system(self):
        """Start the enhanced agent system."""
        if not self.system_manager:
            logger.error("❌ System not deployed. Run deploy_system() first.")
            return False
            
        try:
            logger.info("🚀 Starting Enhanced Agent System...")
            self.start_time = datetime.now()
            self.running = True
            
            # Start the system in a separate process
            logger.info("✅ Starting system manager...")
            self.system_manager.start_system()
            
            logger.info("🎉 Enhanced Agent System successfully started!")
            logger.info("=" * 60)
            logger.info("SYSTEM STATUS:")
            logger.info(f"✅ Start Time: {self.start_time}")
            logger.info(f"✅ Cache Directory: {self.cache_dir}")
            logger.info(f"✅ Data Fetch Interval: 60 seconds")
            logger.info(f"✅ Health Check Interval: 30 seconds")
            logger.info(f"✅ Auto-restart: Enabled")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            self.running = False
            return False
    
    def monitor_system(self):
        """Monitor the running system."""
        logger.info("📊 Starting system monitoring...")
        
        try:
            while self.running and self.system_manager._running:
                # Display system status
                self._display_status()
                
                # Wait before next status check
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
    
    def _display_status(self):
        """Display current system status."""
        try:
            if self.system_manager:
                status = self.system_manager.get_system_status()
                uptime = datetime.now() - self.start_time if self.start_time else "Unknown"
                
                logger.info("📊 SYSTEM STATUS UPDATE:")
                logger.info(f"   ⏱️  Uptime: {uptime}")
                logger.info(f"   🤖 Agents Running: {status.get('agents_running', 0)}")
                logger.info(f"   💭 Message Bus: {status.get('message_bus_status', 'Unknown')}")
                logger.info(f"   💾 Memory Usage: {status.get('memory_usage_mb', 'Unknown')} MB")
                
                # Agent status
                agent_status = status.get('agent_status', {})
                for agent_name, agent_info in agent_status.items():
                    state = agent_info.get('state', 'Unknown')
                    health = agent_info.get('health', {})
                    processed = health.get('messages_processed', 0)
                    logger.info(f"   🔧 {agent_name}: {state} ({processed} messages)")
                
                logger.info("   " + "-" * 50)
                
        except Exception as e:
            logger.error(f"❌ Status display error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.stop_system()
    
    def stop_system(self):
        """Stop the enhanced agent system."""
        logger.info("🛑 Stopping Enhanced Agent System...")
        self.running = False
        
        try:
            if self.system_manager:
                self.system_manager.shutdown_system()
                logger.info("✅ System manager shutdown complete")
            
            # Cleanup cache directory
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                logger.info(f"✅ Cache directory cleaned: {self.cache_dir}")
            
            logger.info("🎉 Enhanced Agent System shutdown complete!")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


def main():
    """Main deployment function."""
    print("🚀 Enhanced Agent System Deployment")
    print("=" * 60)
    
    deployment = EnhancedAgentDeployment()
    
    try:
        # Setup environment
        if not deployment.setup_environment():
            print("❌ Environment setup failed")
            return 1
        
        # Deploy system
        if not deployment.deploy_system():
            print("❌ System deployment failed")
            return 1
        
        # Start system
        if not deployment.start_system():
            print("❌ System startup failed")
            return 1
        
        print("\n🎉 Enhanced Agent System is now running!")
        print("📊 Monitoring system status... (Press Ctrl+C to stop)")
        print("=" * 60)
        
        # Monitor system
        deployment.monitor_system()
        
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        logger.exception("Deployment failed")
        return 1
    finally:
        deployment.stop_system()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 