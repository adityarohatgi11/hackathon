#!/usr/bin/env python3
"""Startup script for the complete GridPilot-GT agent system."""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import List

# Import all agent classes
from agents.data_agent import DataAgent
from agents.forecaster_agent import ForecasterAgent
from agents.strategy_agent import StrategyAgent
from agents.local_llm_agent import LocalLLMAgent
from agents.vector_store_agent import VectorStoreAgent
from agents.message_bus import MessageBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/agent_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Global list to track agent processes
AGENT_PROCESSES: List[multiprocessing.Process] = []


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down agent system...")
    shutdown_agents()
    sys.exit(0)


def shutdown_agents():
    """Shutdown all agent processes."""
    logger.info("Shutting down all agents...")
    
    for process in AGENT_PROCESSES:
        if process.is_alive():
            logger.info(f"Terminating {process.name}")
            process.terminate()
    
    # Wait for processes to terminate
    for process in AGENT_PROCESSES:
        process.join(timeout=5)
        if process.is_alive():
            logger.warning(f"Force killing {process.name}")
            process.kill()
    
    logger.info("All agents shutdown complete")


def start_agent(agent_class, name: str, *args, **kwargs):
    """Start an agent in a separate process."""
    def run_agent():
        try:
            logger.info(f"Starting {name}")
            agent = agent_class(*args, **kwargs)
            agent.start()
        except KeyboardInterrupt:
            logger.info(f"{name} received interrupt signal")
        except Exception as exc:
            logger.exception(f"{name} crashed: {exc}")
            
    process = multiprocessing.Process(target=run_agent, name=name)
    process.start()
    AGENT_PROCESSES.append(process)
    return process


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import redis
        # Try to connect to Redis
        try:
            import redis as redis_lib
            r = redis_lib.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("✅ Redis connection successful")
        except Exception:
            logger.warning("⚠️  Redis not available - will use in-memory message bus")
    except ImportError:
        missing_deps.append("redis")
    
    try:
        import chromadb
        logger.info("✅ ChromaDB available")
    except ImportError:
        logger.warning("⚠️  ChromaDB not available - will use in-memory storage")
    
    try:
        import ray
        logger.info("✅ Ray RLlib available")
    except ImportError:
        logger.warning("⚠️  Ray RLlib not available - will use heuristic strategy")
    
    try:
        from llama_cpp import Llama
        logger.info("✅ Llama-cpp-python available")
    except ImportError:
        logger.warning("⚠️  Llama-cpp-python not available - will use rule-based analysis")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {missing_deps}")
        logger.info("System will run with fallback implementations")
    
    return len(missing_deps) == 0


def setup_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data/vectorstore", 
        "models",
        "data/cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")


def test_message_bus():
    """Test message bus connectivity."""
    try:
        bus = MessageBus()
        
        # Test publish/consume
        test_message = {"test": "startup_check", "timestamp": "2025-06-23T10:00:00"}
        bus.publish("startup-test", test_message)
        
        # Try to consume
        consumer = bus.consume("startup-test", block_ms=1000)
        try:
            received = next(consumer)
            if received.get("test") == "startup_check":
                logger.info("✅ Message bus test successful")
                return True
        except StopIteration:
            pass
        
        logger.warning("⚠️  Message bus test failed - continuing anyway")
        return False
        
    except Exception as exc:
        logger.warning(f"⚠️  Message bus test error: {exc}")
        return False


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="GridPilot-GT Agent System")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["data", "forecaster", "strategy", "llm", "vectorstore"],
        choices=["data", "forecaster", "strategy", "llm", "vectorstore"],
        help="Agents to start (default: all)"
    )
    parser.add_argument(
        "--data-interval",
        type=int,
        default=60,
        help="Data agent fetch interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Path to local LLM model file"
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default="data/vectorstore",
        help="Vector store persistence directory"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting GridPilot-GT Agent System")
    
    # Setup
    setup_directories()
    
    # Check dependencies
    deps_ok = check_dependencies()
    if args.check_only:
        sys.exit(0 if deps_ok else 1)
    
    # Test message bus
    test_message_bus()
    
    # Start agents based on arguments
    logger.info(f"Starting agents: {args.agents}")
    
    if "vectorstore" in args.agents:
        start_agent(
            VectorStoreAgent,
            "VectorStoreAgent",
            persist_directory=args.vectorstore_dir
        )
        time.sleep(2)  # Let vector store start first
    
    if "data" in args.agents:
        start_agent(
            DataAgent,
            "DataAgent", 
            fetch_interval=args.data_interval
        )
        time.sleep(2)
    
    if "forecaster" in args.agents:
        start_agent(ForecasterAgent, "ForecasterAgent")
        time.sleep(1)
    
    if "strategy" in args.agents:
        start_agent(StrategyAgent, "StrategyAgent")
        time.sleep(1)
    
    if "llm" in args.agents:
        start_agent(
            LocalLLMAgent,
            "LocalLLMAgent",
            model_path=args.llm_model
        )
    
    logger.info(f"✅ Started {len(AGENT_PROCESSES)} agents")
    
    # Monitor agent health
    try:
        while True:
            time.sleep(10)
            
            # Check if any agents have died
            for process in AGENT_PROCESSES:
                if not process.is_alive():
                    logger.error(f"❌ Agent {process.name} has died!")
                    # Could implement restart logic here
            
            # Log system status
            alive_count = sum(1 for p in AGENT_PROCESSES if p.is_alive())
            logger.info(f"System status: {alive_count}/{len(AGENT_PROCESSES)} agents running")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        shutdown_agents()


if __name__ == "__main__":
    main() 