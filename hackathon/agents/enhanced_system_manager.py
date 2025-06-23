"""Enhanced agent system manager with robust orchestration and monitoring."""

from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import os

from .enhanced_base_agent import EnhancedBaseAgent, AgentConfig, AgentState
from .enhanced_data_agent import EnhancedDataAgent
from .enhanced_strategy_agent import EnhancedStrategyAgent
from .message_bus import MessageBus

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for the enhanced agent system."""
    data_fetch_interval: int = 60
    cache_dir: str = "data/cache"
    logs_dir: str = "logs"
    enable_persistence: bool = True
    enable_monitoring: bool = True
    health_check_interval: float = 30.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3


class EnhancedSystemManager:
    """Comprehensive system manager for enhanced agents."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.bus = MessageBus()
        
        # System state
        self._running = False
        self._agents: Dict[str, multiprocessing.Process] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._restart_counts: Dict[str, int] = {}
        
        # Monitoring
        self._system_metrics: Dict[str, Any] = {}
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._last_health_check = 0.0
        self._start_time = time.time()
        
        # Setup directories
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        logger.info("[SystemManager] Enhanced agent system manager initialized")

    def start_system(self, agents: Optional[List[str]] = None) -> None:
        """Start the complete enhanced agent system."""
        logger.info("[SystemManager] Starting enhanced agent system")
        
        # Default agents to start
        if agents is None:
            agents = ["data", "strategy"]
        
        self._running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        
        # Start health monitoring
        if self.config.enable_monitoring:
            self._start_health_monitoring()
        
        # Start requested agents
        for agent_name in agents:
            self._start_agent(agent_name)
        
        # Monitor system health
        self._main_monitoring_loop()

    def _main_monitoring_loop(self) -> None:
        """Main system monitoring loop."""
        logger.info("[SystemManager] Starting system monitoring loop")
        
        while self._running:
            try:
                # Check agent health
                self._check_agent_health()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Handle failed agents
                self._handle_failed_agents()
                
                # Sleep between checks
                time.sleep(5.0)
                
            except Exception as exc:
                logger.exception(f"[SystemManager] Error in monitoring loop: {exc}")
                time.sleep(10.0)

    def _start_agent(self, agent_name: str) -> bool:
        """Start a specific agent."""
        try:
            logger.info(f"[SystemManager] Starting {agent_name} agent")
            
            # Create agent configuration
            if agent_name == "data":
                agent_config = AgentConfig(
                    cache_size=5000,
                    cache_ttl=300.0,
                    enable_caching=True,
                    enable_metrics=True,
                    max_retries=5,
                    retry_delay=2.0,
                    health_check_interval=30.0
                )
                process = multiprocessing.Process(
                    target=self._run_data_agent,
                    args=(agent_config,),
                    name=f"Enhanced{agent_name.capitalize()}Agent"
                )
            elif agent_name == "strategy":
                agent_config = AgentConfig(
                    cache_size=2000,
                    cache_ttl=180.0,
                    enable_caching=True,
                    enable_metrics=True,
                    max_retries=3,
                    retry_delay=1.0,
                    health_check_interval=30.0
                )
                process = multiprocessing.Process(
                    target=self._run_strategy_agent,
                    args=(agent_config,),
                    name=f"Enhanced{agent_name.capitalize()}Agent"
                )
            else:
                logger.error(f"[SystemManager] Unknown agent type: {agent_name}")
                return False
            
            # Start the process
            process.start()
            
            # Track the agent
            self._agents[agent_name] = process
            self._agent_configs[agent_name] = agent_config
            self._restart_counts[agent_name] = 0
            
            logger.info(f"[SystemManager] {agent_name} agent started with PID {process.pid}")
            return True
            
        except Exception as exc:
            logger.exception(f"[SystemManager] Failed to start {agent_name} agent: {exc}")
            return False

    def _run_data_agent(self, config: AgentConfig) -> None:
        """Run the enhanced data agent in a separate process."""
        try:
            agent = EnhancedDataAgent(
                fetch_interval=self.config.data_fetch_interval,
                cache_dir=self.config.cache_dir
            )
            agent.config = config
            agent.start()
        except Exception as exc:
            logger.exception(f"[DataAgent] Agent crashed: {exc}")

    def _run_strategy_agent(self, config: AgentConfig) -> None:
        """Run the enhanced strategy agent in a separate process."""
        try:
            agent = EnhancedStrategyAgent(cache_dir=self.config.cache_dir)
            agent.config = config
            agent.start()
        except Exception as exc:
            logger.exception(f"[StrategyAgent] Agent crashed: {exc}")

    def _check_agent_health(self) -> None:
        """Check health of all running agents."""
        current_time = time.time()
        
        for agent_name, process in list(self._agents.items()):
            try:
                # Check if process is alive
                if not process.is_alive():
                    logger.warning(f"[SystemManager] {agent_name} agent process is dead")
                    self._handle_dead_agent(agent_name)
                    continue
                
                # Check process status
                if hasattr(process, 'exitcode') and process.exitcode is not None:
                    logger.warning(f"[SystemManager] {agent_name} agent exited with code {process.exitcode}")
                    self._handle_dead_agent(agent_name)
                    continue
                
            except Exception as exc:
                logger.warning(f"[SystemManager] Error checking {agent_name} agent health: {exc}")

    def _handle_dead_agent(self, agent_name: str) -> None:
        """Handle a dead agent."""
        logger.warning(f"[SystemManager] Handling dead {agent_name} agent")
        
        # Clean up the dead process
        if agent_name in self._agents:
            process = self._agents[agent_name]
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                except:
                    pass
            
            del self._agents[agent_name]
        
        # Attempt restart if enabled
        if self.config.restart_on_failure:
            restart_count = self._restart_counts.get(agent_name, 0)
            if restart_count < self.config.max_restart_attempts:
                logger.info(f"[SystemManager] Attempting to restart {agent_name} agent (attempt {restart_count + 1})")
                if self._start_agent(agent_name):
                    self._restart_counts[agent_name] = restart_count + 1
                else:
                    logger.error(f"[SystemManager] Failed to restart {agent_name} agent")
            else:
                logger.error(f"[SystemManager] Max restart attempts reached for {agent_name} agent")

    def _handle_failed_agents(self) -> None:
        """Check for and handle any failed agents."""
        pass

    def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics."""
        try:
            metrics = {
                "timestamp": time.time(),
                "system_uptime": time.time() - self._start_time,
                "agents_running": len(self._agents),
                "agent_status": {},
                "restart_counts": self._restart_counts.copy(),
                "memory_usage": self._get_system_memory_usage(),
                "message_bus_status": self._check_message_bus_health()
            }
            
            # Agent-specific metrics
            for agent_name, process in self._agents.items():
                try:
                    metrics["agent_status"][agent_name] = {
                        "pid": process.pid,
                        "alive": process.is_alive(),
                        "exitcode": getattr(process, 'exitcode', None)
                    }
                except:
                    metrics["agent_status"][agent_name] = {"status": "error"}
            
            self._system_metrics = metrics
            
            # Optionally persist metrics
            if self.config.enable_persistence:
                self._persist_metrics(metrics)
                
        except Exception as exc:
            logger.warning(f"[SystemManager] Failed to collect system metrics: {exc}")

    def _get_system_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024,
                "percent_used": memory.percent
            }
        except ImportError:
            return {"status": "psutil_not_available"}

    def _check_message_bus_health(self) -> Dict[str, Any]:
        """Check message bus health."""
        try:
            test_message = {"test": "health_check", "timestamp": time.time()}
            self.bus.publish("system-health-test", test_message)
            
            return {"status": "healthy", "type": "in_memory" if self.bus._use_memory else "redis"}
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    def _persist_metrics(self, metrics: Dict[str, Any]) -> None:
        """Persist system metrics to disk."""
        try:
            metrics_file = os.path.join(self.config.logs_dir, "system_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as exc:
            logger.warning(f"[SystemManager] Failed to persist metrics: {exc}")

    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="SystemHealthMonitor",
            daemon=True
        )
        self._health_monitor_thread.start()

    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        logger.info("[SystemManager] Starting health monitoring")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Perform periodic health checks
                if current_time - self._last_health_check >= self.config.health_check_interval:
                    self._perform_comprehensive_health_check()
                    self._last_health_check = current_time
                
                # Publish system health metrics
                self._publish_system_health()
                
                time.sleep(10.0)
                
            except Exception as exc:
                logger.exception(f"[SystemManager] Error in health monitoring: {exc}")
                time.sleep(30.0)

    def _perform_comprehensive_health_check(self) -> None:
        """Perform comprehensive system health check."""
        logger.info("[SystemManager] Performing comprehensive health check")
        
        health_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "issues": [],
            "agent_health": {},
            "system_resources": self._get_system_memory_usage(),
            "message_bus": self._check_message_bus_health()
        }
        
        # Check individual agents
        for agent_name in self._agents:
            if agent_name in self._agents and self._agents[agent_name].is_alive():
                health_report["agent_health"][agent_name] = "healthy"
            else:
                health_report["agent_health"][agent_name] = "unhealthy"
                health_report["issues"].append(f"{agent_name}_agent_down")
        
        # Check system resources
        memory_info = health_report["system_resources"]
        if isinstance(memory_info, dict) and memory_info.get("percent_used", 0) > 90:
            health_report["issues"].append("high_memory_usage")
            health_report["overall_status"] = "degraded"
        
        # Check message bus
        if health_report["message_bus"].get("status") != "healthy":
            health_report["issues"].append("message_bus_unhealthy")
            health_report["overall_status"] = "degraded"
        
        # Set overall status
        if health_report["issues"]:
            if any("down" in issue for issue in health_report["issues"]):
                health_report["overall_status"] = "unhealthy"
            else:
                health_report["overall_status"] = "degraded"
        
        # Log health status
        if health_report["overall_status"] == "healthy":
            logger.info("[SystemManager] System health check: HEALTHY")
        else:
            logger.warning(f"[SystemManager] System health check: {health_report['overall_status'].upper()} - Issues: {health_report['issues']}")
        
        # Persist health report
        if self.config.enable_persistence:
            try:
                health_file = os.path.join(self.config.logs_dir, "system_health.json")
                with open(health_file, 'w') as f:
                    json.dump(health_report, f, indent=2)
            except Exception as exc:
                logger.warning(f"[SystemManager] Failed to persist health report: {exc}")

    def _publish_system_health(self) -> None:
        """Publish system health metrics to message bus."""
        try:
            health_metrics = {
                "timestamp": time.time(),
                "source": "SystemManager",
                "agents_running": len(self._agents),
                "system_metrics": self._system_metrics,
                "uptime": time.time() - self._start_time
            }
            
            self.bus.publish("system-health", health_metrics)
            
        except Exception as exc:
            logger.warning(f"[SystemManager] Failed to publish system health: {exc}")

    def _graceful_shutdown(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown signal."""
        logger.info(f"[SystemManager] Received signal {signum}, initiating graceful shutdown")
        self.shutdown_system()

    def shutdown_system(self) -> None:
        """Shutdown the entire agent system gracefully."""
        logger.info("[SystemManager] Shutting down enhanced agent system")
        self._running = False
        
        # Stop health monitoring
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5)
        
        # Shutdown all agents
        for agent_name, process in list(self._agents.items()):
            logger.info(f"[SystemManager] Shutting down {agent_name} agent")
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    
                    if process.is_alive():
                        logger.warning(f"[SystemManager] Force killing {agent_name} agent")
                        process.kill()
                        process.join(timeout=5)
                        
                logger.info(f"[SystemManager] {agent_name} agent shutdown complete")
                
            except Exception as exc:
                logger.error(f"[SystemManager] Error shutting down {agent_name} agent: {exc}")
        
        # Clear agent tracking
        self._agents.clear()
        
        logger.info("[SystemManager] System shutdown complete")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "running": self._running,
            "agents": {name: proc.is_alive() for name, proc in self._agents.items()},
            "restart_counts": self._restart_counts.copy(),
            "metrics": self._system_metrics,
            "config": {
                "data_fetch_interval": self.config.data_fetch_interval,
                "cache_dir": self.config.cache_dir,
                "monitoring_enabled": self.config.enable_monitoring,
                "restart_on_failure": self.config.restart_on_failure
            }
        }

    def restart_agent(self, agent_name: str) -> bool:
        """Manually restart a specific agent."""
        logger.info(f"[SystemManager] Manual restart requested for {agent_name} agent")
        
        # Stop the agent if running
        if agent_name in self._agents:
            self._handle_dead_agent(agent_name)
        
        # Start the agent
        return self._start_agent(agent_name)
