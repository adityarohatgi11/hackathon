"""Enhanced base agent with robust features for scalable operation."""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import deque
import uuid

from .message_bus import MessageBus

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class HealthMetrics:
    """Agent health metrics tracking."""
    messages_processed: int = 0
    messages_failed: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.messages_processed + self.messages_failed
        return self.messages_processed / total if total > 0 else 0.0
    
    @property
    def avg_processing_time(self) -> float:
        """Calculate average processing time."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0


@dataclass
class AgentConfig:
    """Configuration for enhanced agents."""
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0
    cache_size: int = 1000
    cache_ttl: float = 300.0  # 5 minutes
    max_queue_size: int = 10000
    processing_timeout: float = 60.0
    enable_metrics: bool = True
    enable_caching: bool = True


class EnhancedBaseAgent(ABC):
    """Enhanced base class for robust, scalable agents."""

    subscribe_topics: List[str] = []
    publish_topic: str | None = None

    def __init__(self, name: str, config: Optional[AgentConfig] = None, bus: MessageBus | None = None):
        self.name = name
        self.config = config or AgentConfig()
        self.bus = bus or MessageBus()
        
        # State management
        self.state = AgentState.INITIALIZING
        self.health = HealthMetrics()
        self.agent_id = str(uuid.uuid4())[:8]
        
        # Threading and control
        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Caching system
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Circuit breaker
        self._circuit_open = False
        self._circuit_failure_count = 0
        self._circuit_last_failure = 0.0
        self._circuit_reset_timeout = 60.0
        
        logger.info(f"[{self.name}] Enhanced agent initialized with ID: {self.agent_id}")

    def start(self) -> None:
        """Start the enhanced agent with health monitoring."""
        logger.info(f"[{self.name}] Starting enhanced agent")
        self.state = AgentState.HEALTHY
        self._running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)
        
        # Start health monitoring
        if self.config.enable_metrics:
            self._start_health_monitoring()
        
        # Initialize agent-specific resources
        try:
            self._initialize()
            logger.info(f"[{self.name}] Agent initialization complete")
        except Exception as exc:
            logger.error(f"[{self.name}] Initialization failed: {exc}")
            self.state = AgentState.UNHEALTHY
            return
        
        # Main processing loop
        self._main_loop()

    @abstractmethod
    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Process an incoming message and optionally return a response."""
        raise NotImplementedError

    def _initialize(self) -> None:
        """Initialize agent-specific resources. Override in subclasses."""
        pass

    def _health_check(self) -> bool:
        """Perform agent-specific health check. Override in subclasses."""
        return True

    def _main_loop(self) -> None:
        """Main processing loop with enhanced error handling."""
        while self._running:
            try:
                # Check circuit breaker
                if self._check_circuit_breaker():
                    time.sleep(1)
                    continue
                
                # Process messages
                self._process_single_messages()
                    
                # Clean up cache
                self._cleanup_cache()
                
                # Brief pause to prevent tight looping
                time.sleep(0.01)
                
            except Exception as exc:
                logger.exception(f"[{self.name}] Error in main loop: {exc}")
                self._handle_failure(exc)
                time.sleep(self.config.retry_delay)

    def _process_single_messages(self) -> None:
        """Process messages one by one."""
        for topic in self.subscribe_topics:
            for message in self.bus.consume(topic):
                if not self._running:
                    break
                    
                start_time = time.time()
                try:
                    result = self._process_message_with_retries(message)
                    if result is not None and self.publish_topic:
                        self.bus.publish(self.publish_topic, result)
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self._record_success(processing_time)
                    
                except Exception as exc:
                    self._record_failure(exc)
                    logger.warning(f"[{self.name}] Message processing failed: {exc}")

    def _process_message_with_retries(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Process message with retry logic and caching."""
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._get_cache_key(message)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = self.handle_message(message)
                
                # Cache the result
                if self.config.enable_caching and result is not None:
                    self._set_cache(cache_key, result)
                
                return result
                
            except Exception as exc:
                last_exception = exc
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"[{self.name}] Retry {attempt + 1}/{self.config.max_retries} after {delay}s: {exc}")
                    time.sleep(delay)
                else:
                    logger.error(f"[{self.name}] All retries exhausted: {exc}")
        
        raise last_exception or Exception("Unknown error in message processing")

    # Caching System
    def _get_cache_key(self, message: Dict[str, Any]) -> str:
        """Generate cache key for message."""
        import hashlib
        message_str = json.dumps(message, sort_keys=True, default=str)
        return hashlib.md5(message_str.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Dict[str, Any] | None:
        """Retrieve from cache if not expired."""
        if key not in self._cache:
            return None
        
        timestamp = self._cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.config.cache_ttl:
            self._remove_from_cache(key)
            return None
        
        return self._cache[key].copy()

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Store in cache with timestamp."""
        with self._lock:
            self._cache[key] = value.copy()
            self._cache_timestamps[key] = time.time()
            
            if len(self._cache) > self.config.cache_size:
                self._evict_oldest_cache_entry()

    def _remove_from_cache(self, key: str) -> None:
        """Remove from cache."""
        with self._lock:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    def _evict_oldest_cache_entry(self) -> None:
        """Remove oldest cache entry."""
        if not self._cache_timestamps:
            return
        
        oldest_key = min(self._cache_timestamps.keys(), 
                        key=lambda k: self._cache_timestamps[k])
        self._remove_from_cache(oldest_key)

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self.config.cache_ttl
        ]
        
        for key in expired_keys:
            self._remove_from_cache(key)

    # Circuit Breaker Pattern
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should prevent processing."""
        if not self._circuit_open:
            return False
        
        if time.time() - self._circuit_last_failure > self._circuit_reset_timeout:
            self._circuit_open = False
            self._circuit_failure_count = 0
            logger.info(f"[{self.name}] Circuit breaker reset")
            return False
        
        return True

    def _record_success(self, processing_time: float) -> None:
        """Record successful message processing."""
        with self._lock:
            self.health.messages_processed += 1
            self.health.last_success = time.time()
            self.health.processing_times.append(processing_time)
            
            self._circuit_failure_count = 0
            if self._circuit_open:
                self._circuit_open = False
                logger.info(f"[{self.name}] Circuit breaker closed after success")

    def _record_failure(self, exception: Exception) -> None:
        """Record failed message processing."""
        with self._lock:
            self.health.messages_failed += 1
            self.health.last_failure = time.time()
            self._circuit_failure_count += 1
            self._circuit_last_failure = time.time()
            
            if self._circuit_failure_count >= 5:
                self._circuit_open = True
                self.state = AgentState.DEGRADED
                logger.warning(f"[{self.name}] Circuit breaker opened due to failures")

    def _handle_failure(self, exception: Exception) -> None:
        """Handle agent failure and attempt recovery."""
        self._record_failure(exception)
        
        try:
            self._recover()
        except Exception as recovery_exc:
            logger.error(f"[{self.name}] Recovery failed: {recovery_exc}")
            self.state = AgentState.UNHEALTHY

    def _recover(self) -> None:
        """Attempt to recover from failures. Override in subclasses."""
        logger.info(f"[{self.name}] Attempting recovery...")
        time.sleep(self.config.retry_delay)

    # Health Monitoring
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name=f"{self.name}-health",
            daemon=True
        )
        self._health_thread.start()

    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self._running:
            try:
                self._update_resource_metrics()
                
                is_healthy = self._health_check()
                
                if is_healthy and self.health.success_rate > 0.8:
                    if self.state == AgentState.DEGRADED:
                        self.state = AgentState.HEALTHY
                        logger.info(f"[{self.name}] Agent recovered to healthy state")
                elif self.health.success_rate < 0.5:
                    if self.state == AgentState.HEALTHY:
                        self.state = AgentState.DEGRADED
                        logger.warning(f"[{self.name}] Agent degraded due to low success rate")
                
                self._publish_health_metrics()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as exc:
                logger.exception(f"[{self.name}] Error in health monitoring: {exc}")
                time.sleep(self.config.health_check_interval)

    def _update_resource_metrics(self) -> None:
        """Update resource usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            self.health.memory_usage = process.memory_info().rss / 1024 / 1024
            self.health.cpu_usage = process.cpu_percent()
        except ImportError:
            pass

    def _publish_health_metrics(self) -> None:
        """Publish health metrics to monitoring topic."""
        metrics = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "state": self.state.value,
            "timestamp": time.time(),
            "messages_processed": self.health.messages_processed,
            "messages_failed": self.health.messages_failed,
            "success_rate": self.health.success_rate,
            "avg_processing_time": self.health.avg_processing_time,
            "memory_usage_mb": self.health.memory_usage,
            "cpu_usage_percent": self.health.cpu_usage,
            "cache_size": len(self._cache),
            "circuit_open": self._circuit_open,
        }
        
        try:
            self.bus.publish("agent-health", metrics)
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to publish health metrics: {exc}")

    def _graceful_exit(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"[{self.name}] Caught signal {signum} - shutting down")
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        logger.info(f"[{self.name}] Initiating graceful shutdown")
        self.state = AgentState.SHUTTING_DOWN
        self._running = False
        
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)
        
        self._cleanup()
        
        self.state = AgentState.STOPPED
        logger.info(f"[{self.name}] Shutdown complete")

    def _cleanup(self) -> None:
        """Cleanup agent resources. Override in subclasses."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "health": {
                "messages_processed": self.health.messages_processed,
                "messages_failed": self.health.messages_failed,
                "success_rate": self.health.success_rate,
                "avg_processing_time": self.health.avg_processing_time,
                "memory_usage_mb": self.health.memory_usage,
                "cpu_usage_percent": self.health.cpu_usage,
            },
            "cache": {
                "size": len(self._cache),
                "max_size": self.config.cache_size,
            },
            "circuit_breaker": {
                "open": self._circuit_open,
                "failure_count": self._circuit_failure_count,
            },
        }
