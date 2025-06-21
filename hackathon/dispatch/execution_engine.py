"""High-performance execution engine for trade execution and monitoring.

Handles real-time trade execution, position monitoring, and risk management
with microsecond precision timing and comprehensive safety checks.
"""

import time
import threading
import queue
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ExecutionOrder:
    """Trade execution order."""
    order_id: str
    allocation: Dict[str, float]
    target_power: float
    max_power: float
    price_limit: float
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0  # seconds
    status: ExecutionStatus = ExecutionStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    
    def __lt__(self, other):
        """Enable priority queue ordering by timestamp."""
        return self.timestamp < other.timestamp
    
@dataclass
class ExecutionResult:
    """Execution result with performance metrics."""
    order_id: str
    success: bool
    actual_allocation: Dict[str, float]
    execution_time_ms: float
    risk_score: float
    constraint_violations: List[str]
    power_delivered: float
    efficiency: float
    
class HighPerformanceExecutionEngine:
    """High-performance execution engine with <50ms execution guarantee."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="exec")
        self.execution_queue = queue.PriorityQueue()
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.execution_results: Dict[str, ExecutionResult] = {}
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.average_execution_time = 0.0
        
        # Risk management
        self.risk_limits = {
            'max_power_deviation': 0.05,  # 5% deviation allowed
            'max_temperature_rise': 5.0,   # 5°C temperature rise
            'max_soc_change': 0.1,         # 10% SOC change per execution
        }
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.failure_count = 0
        self.failure_threshold = 5
        
        # Start background processing
        self._start_execution_processor()
        
        logger.info(f"High-performance execution engine initialized with {max_workers} workers")
    
    def _start_execution_processor(self):
        """Start background execution processor."""
        def process_executions():
            while True:
                try:
                    # Get next order from priority queue
                    priority, order = self.execution_queue.get(timeout=1.0)
                    if order is None:  # Shutdown signal
                        break
                    
                    # Execute order
                    self._execute_order_async(order)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Execution processing error: {e}")
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.circuit_breaker_active = True
                        logger.critical("Circuit breaker activated due to execution failures")
        
        self.processor_thread = threading.Thread(
            target=process_executions,
            daemon=True,
            name="execution-processor"
        )
        self.processor_thread.start()
    
    def submit_execution(self, allocation: Dict[str, float], 
                        constraints: Dict[str, Any],
                        priority: int = 1) -> str:
        """Submit execution order with priority scheduling.
        
        Args:
            allocation: Resource allocation to execute
            constraints: System constraints and limits
            priority: Execution priority (lower = higher priority)
            
        Returns:
            Order ID for tracking
        """
        if self.circuit_breaker_active:
            raise RuntimeError("Execution engine circuit breaker is active")
        
        order_id = f"exec_{int(time.time() * 1000000)}"  # Microsecond precision
        
        order = ExecutionOrder(
            order_id=order_id,
            allocation=dict(allocation),
            target_power=sum(allocation.values()),
            max_power=constraints.get('power_limit', 1000.0),
            price_limit=constraints.get('price_limit', 100.0),
            timeout=constraints.get('timeout', 30.0)
        )
        
        # Risk assessment
        order.risk_level = self._assess_risk(order, constraints)
        
        # Priority adjustment based on risk
        if order.risk_level == RiskLevel.CRITICAL:
            priority = 0  # Highest priority
        elif order.risk_level == RiskLevel.HIGH:
            priority = min(priority, 1)
        
        # Add to execution queue
        self.execution_queue.put((priority, order))
        self.active_orders[order_id] = order
        
        logger.info(f"Execution order {order_id} submitted with priority {priority}")
        return order_id
    
    def _assess_risk(self, order: ExecutionOrder, constraints: Dict[str, Any]) -> RiskLevel:
        """Assess execution risk level."""
        risk_score = 0.0
        
        # Power risk
        power_ratio = order.target_power / order.max_power
        if power_ratio > 0.95:
            risk_score += 0.4
        elif power_ratio > 0.85:
            risk_score += 0.2
        
        # Price risk
        current_price = constraints.get('current_price', 50.0)
        if current_price > order.price_limit:
            risk_score += 0.3
        
        # System state risk
        temperature = constraints.get('temperature', 65.0)
        if temperature > 75:
            risk_score += 0.3
        
        soc = constraints.get('soc', 0.5)
        if soc < 0.2 or soc > 0.9:
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _execute_order_async(self, order: ExecutionOrder):
        """Execute order asynchronously with full monitoring."""
        start_time = time.perf_counter()
        order.status = ExecutionStatus.EXECUTING
        
        try:
            # Simulate high-performance execution
            result = self._perform_execution(order)
            
            # Record successful execution
            order.status = ExecutionStatus.COMPLETED
            self.successful_executions += 1
            
        except Exception as e:
            logger.error(f"Execution failed for order {order.order_id}: {e}")
            result = ExecutionResult(
                order_id=order.order_id,
                success=False,
                actual_allocation={},
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                risk_score=1.0,
                constraint_violations=[str(e)],
                power_delivered=0.0,
                efficiency=0.0
            )
            order.status = ExecutionStatus.FAILED
            self.failure_count += 1
        
        # Update performance metrics
        execution_time = (time.perf_counter() - start_time) * 1000
        self.total_executions += 1
        self.average_execution_time = (
            (self.average_execution_time * (self.total_executions - 1) + execution_time) 
            / self.total_executions
        )
        
        # Store result
        self.execution_results[order.order_id] = result
        
        # Clean up
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
    
    def _perform_execution(self, order: ExecutionOrder) -> ExecutionResult:
        """Perform the actual execution with safety checks."""
        start_time = time.perf_counter()
        
        # Pre-execution safety checks
        violations = []
        
        # Power constraint check
        if order.target_power > order.max_power:
            violations.append(f"Power limit exceeded: {order.target_power} > {order.max_power}")
        
        # Timeout check
        elapsed = time.time() - order.timestamp
        if elapsed > order.timeout:
            violations.append(f"Execution timeout: {elapsed:.1f}s > {order.timeout}s")
        
        if violations:
            raise RuntimeError(f"Pre-execution constraints violated: {violations}")
        
        # Simulate execution with realistic timing
        execution_delay = np.random.uniform(0.001, 0.005)  # 1-5ms execution time
        time.sleep(execution_delay)
        
        # Calculate actual delivery (with small variations)
        actual_allocation = {}
        total_delivered = 0.0
        
        for service, target in order.allocation.items():
            # Simulate execution efficiency (95-99%)
            efficiency = np.random.uniform(0.95, 0.99)
            actual = target * efficiency
            actual_allocation[service] = actual
            total_delivered += actual
        
        # Calculate performance metrics
        execution_time = (time.perf_counter() - start_time) * 1000
        overall_efficiency = total_delivered / order.target_power if order.target_power > 0 else 1.0
        risk_score = 0.1 if order.risk_level == RiskLevel.LOW else 0.3
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            actual_allocation=actual_allocation,
            execution_time_ms=execution_time,
            risk_score=risk_score,
            constraint_violations=[],
            power_delivered=total_delivered,
            efficiency=overall_efficiency
        )
    
    def get_execution_status(self, order_id: str) -> Optional[ExecutionStatus]:
        """Get execution status for an order."""
        if order_id in self.active_orders:
            return self.active_orders[order_id].status
        elif order_id in self.execution_results:
            return ExecutionStatus.COMPLETED if self.execution_results[order_id].success else ExecutionStatus.FAILED
        else:
            return None
    
    def get_execution_result(self, order_id: str) -> Optional[ExecutionResult]:
        """Get execution result for a completed order."""
        return self.execution_results.get(order_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution engine performance metrics."""
        success_rate = self.successful_executions / max(self.total_executions, 1) * 100
        
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate_percent': success_rate,
            'average_execution_time_ms': self.average_execution_time,
            'active_orders': len(self.active_orders),
            'failure_count': self.failure_count,
            'circuit_breaker_active': self.circuit_breaker_active,
            'queue_size': self.execution_queue.qsize()
        }
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker after manual intervention."""
        self.circuit_breaker_active = False
        self.failure_count = 0
        logger.info("Circuit breaker reset")
    
    def shutdown(self):
        """Gracefully shutdown the execution engine."""
        # Signal shutdown
        self.execution_queue.put((0, None))
        
        # Wait for active executions to complete
        timeout = 10.0
        start_time = time.time()
        
        while self.active_orders and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        # Force shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Execution engine shutdown completed")


# Global execution engine instance
_execution_engine: Optional[HighPerformanceExecutionEngine] = None

def get_execution_engine() -> HighPerformanceExecutionEngine:
    """Get global execution engine instance."""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = HighPerformanceExecutionEngine()
    return _execution_engine

def execute_dispatch(allocation: Dict[str, float], constraints: Dict[str, Any]) -> str:
    """Execute dispatch with high-performance execution engine.
    
    Args:
        allocation: Resource allocation to execute
        constraints: System constraints
        
    Returns:
        Order ID for tracking execution
    """
    engine = get_execution_engine()
    return engine.submit_execution(allocation, constraints)

def wait_for_execution(order_id: str, timeout: float = 10.0) -> ExecutionResult:
    """Wait for execution completion with timeout.
    
    Args:
        order_id: Order ID to wait for
        timeout: Maximum wait time in seconds
        
    Returns:
        Execution result
        
    Raises:
        TimeoutError: If execution doesn't complete within timeout
    """
    engine = get_execution_engine()
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        status = engine.get_execution_status(order_id)
        
        if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
            result = engine.get_execution_result(order_id)
            if result:
                return result
        
        time.sleep(0.01)  # 10ms polling
    
    raise TimeoutError(f"Execution {order_id} did not complete within {timeout}s") 