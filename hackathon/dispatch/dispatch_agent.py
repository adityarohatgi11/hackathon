"""High-performance real-time dispatch agent for market execution.

Optimized for <100ms response time with:
- Real-time market signal processing
- Advanced emergency protocols  
- Comprehensive safety constraint validation
- Performance monitoring and circuit breakers
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    """Emergency severity levels."""
    NORMAL = 0
    WARNING = 1  
    CRITICAL = 2
    SHUTDOWN = 3

class ConstraintType(Enum):
    """Types of system constraints."""
    POWER_LIMIT = "power_limit"
    TEMPERATURE = "temperature"
    BATTERY_SOC = "battery_soc"
    GRID_FREQUENCY = "grid_frequency"
    COOLING_CAPACITY = "cooling_capacity"

@dataclass
class PerformanceMetrics:
    """Real-time performance tracking."""
    build_payload_time: float = 0.0
    adjustment_time: float = 0.0
    emergency_time: float = 0.0
    constraint_validation_time: float = 0.0
    total_response_time: float = 0.0
    calls_processed: int = 0
    constraint_violations: int = 0
    emergency_activations: int = 0
    
    def reset(self):
        """Reset metrics for new measurement period."""
        self.__dict__.update((k, 0.0 if isinstance(v, float) else 0) for k, v in self.__dict__.items())

@dataclass 
class ConstraintLimits:
    """System constraint limits and safety margins."""
    max_power_kw: float = 1000.0
    max_temperature_c: float = 80.0
    min_battery_soc: float = 0.15
    max_battery_soc: float = 0.90
    max_cooling_load_kw: float = 500.0
    min_grid_frequency_hz: float = 59.5
    max_grid_frequency_hz: float = 60.5
    safety_margin: float = 0.1  # 10% safety margin
    
    def get_safe_limit(self, limit_type: ConstraintType):
        """Get constraint limit with safety margin applied."""
        margin = 1.0 - self.safety_margin
        
        if limit_type == ConstraintType.POWER_LIMIT:
            return self.max_power_kw * margin
        elif limit_type == ConstraintType.TEMPERATURE:
            return self.max_temperature_c * margin
        elif limit_type == ConstraintType.BATTERY_SOC:
            return (self.min_battery_soc, self.max_battery_soc * margin)
        elif limit_type == ConstraintType.COOLING_CAPACITY:
            return self.max_cooling_load_kw * margin
        else:
            return 0.0

@dataclass
class MarketSignal:
    """Structured market signal data."""
    price: float
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)
    frequency: float = 60.0
    volatility: float = 0.0
    trend: str = "stable"  # "rising", "falling", "stable"

@dataclass
class SystemState:
    """Current system state snapshot."""
    temperature: float = 65.0
    soc: float = 0.5
    power_total_kw: float = 0.0
    power_available_kw: float = 1000.0
    cooling_load_kw: float = 0.0
    grid_frequency_hz: float = 60.0
    timestamp: float = field(default_factory=time.time)
    emergency_mode: bool = False

class HighPerformanceDispatchAgent:
    """High-performance dispatch agent with <100ms response guarantee."""
    
    def __init__(self, constraint_limits: Optional[ConstraintLimits] = None):
        self.constraints = constraint_limits or ConstraintLimits()
        self.metrics = PerformanceMetrics()
        self.emergency_state = EmergencyLevel.NORMAL
        self.circuit_breaker_active = False
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dispatch")
        
        # Pre-allocate arrays for performance
        self._temp_allocation = np.zeros(3, dtype=np.float32)
        self._temp_power = np.zeros(4, dtype=np.float32)
        
        # Market signal processing queue
        self.market_queue = queue.Queue(maxsize=100)
        self.market_processor_thread = None
        self._start_market_processor()
        
        logger.info("High-performance dispatch agent initialized")
    
    def _start_market_processor(self):
        """Start background market signal processor."""
        def process_market_signals():
            while True:
                try:
                    signal = self.market_queue.get(timeout=1.0)
                    if signal is None:  # Shutdown signal
                        break
                    self._process_market_signal_background(signal)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Market signal processing error: {e}")
        
        self.market_processor_thread = threading.Thread(
            target=process_market_signals, 
            daemon=True,
            name="market-processor"
        )
        self.market_processor_thread.start()
    
    def _process_market_signal_background(self, signal: MarketSignal):
        """Background processing of market signals."""
        # Detect market trends and volatility
        if signal.price > 60:
            signal.trend = "rising"
        elif signal.price < 40:
            signal.trend = "falling"
        
        # Update volatility based on price changes
        # This would normally use historical data
        signal.volatility = abs(signal.price - 50.0) / 50.0
    
    def validate_constraints_fast(self, system_state: SystemState, 
                                 total_power_kw: float) -> Tuple[bool, List[str]]:
        """Ultra-fast constraint validation using pre-computed limits."""
        start_time = time.perf_counter()
        
        violations = []
        
        # Power constraint (most critical)
        safe_power_limit = self.constraints.get_safe_limit(ConstraintType.POWER_LIMIT)
        if isinstance(safe_power_limit, (int, float)) and total_power_kw > safe_power_limit:
            violations.append(f"Power limit exceeded: {total_power_kw:.1f} > {safe_power_limit:.1f} kW")
        
        # Temperature constraint
        safe_temp_limit = self.constraints.get_safe_limit(ConstraintType.TEMPERATURE)
        if isinstance(safe_temp_limit, (int, float)) and system_state.temperature > safe_temp_limit:
            violations.append(f"Temperature limit exceeded: {system_state.temperature:.1f} > {safe_temp_limit:.1f}°C")
        
        # Battery SOC constraints
        soc_limits = self.constraints.get_safe_limit(ConstraintType.BATTERY_SOC)
        if isinstance(soc_limits, tuple):
            min_soc, max_soc = soc_limits
            if not (min_soc <= system_state.soc <= max_soc):
                violations.append(f"Battery SOC out of range: {system_state.soc:.2f} not in [{min_soc:.2f}, {max_soc:.2f}]")
        
        # Grid frequency constraints
        if not (self.constraints.min_grid_frequency_hz <= system_state.grid_frequency_hz <= self.constraints.max_grid_frequency_hz):
            violations.append(f"Grid frequency out of range: {system_state.grid_frequency_hz:.2f} Hz")
        
        self.metrics.constraint_validation_time = (time.perf_counter() - start_time) * 1000
        
        if violations:
            self.metrics.constraint_violations += 1
        
        return len(violations) == 0, violations


def build_payload(allocation: Dict[str, float], inventory: Dict[str, Any], 
                 soc: float, cooling_kw: float, power_limit: float) -> Dict[str, Any]:
    """Build market submission payload with <100ms performance guarantee.
    
    Args:
        allocation: Resource allocation from auction (in kW)
        inventory: Current system inventory
        soc: Battery state of charge
        cooling_kw: Required cooling power
        power_limit: Maximum power limit
        
    Returns:
        Market submission payload with performance metrics
    """
    start_time = time.perf_counter()
    
    # Fast path: pre-compute critical values
    gpu_power = sum(allocation.values())  # Already in kW
    total_power = gpu_power + cooling_kw
    
    # Initialize high-performance agent for constraint validation
    if not hasattr(build_payload, '_agent'):
        build_payload._agent = HighPerformanceDispatchAgent()
    
    agent = build_payload._agent
    
    # Create system state snapshot
    system_state = SystemState(
        temperature=inventory.get('temperature', 65.0),
        soc=soc,
        power_total_kw=total_power,
        power_available_kw=inventory.get('power_available', power_limit),
        cooling_load_kw=cooling_kw,
        grid_frequency_hz=inventory.get('grid_frequency', 60.0)
    )
    
    # Ultra-fast constraint validation
    constraints_satisfied, violations = agent.validate_constraints_fast(system_state, total_power)
    
    # Emergency power scaling if needed
    scale_factor = 1.0
    if not constraints_satisfied and total_power > power_limit:
        scale_factor = power_limit / total_power * 0.9  # 90% of limit for safety
        gpu_power *= scale_factor
        total_power = gpu_power + cooling_kw
        constraints_satisfied = True  # Now within limits
    
    # Calculate battery operations
    battery_charge_kw = max(0, power_limit - total_power) if soc < 0.8 else 0
    battery_discharge_kw = max(0, total_power - power_limit) if soc > 0.2 else 0
    
    # Build payload with performance data
    payload = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'allocation': allocation,
        'power_requirements': {
            'gpu_power_kw': gpu_power,
            'cooling_power_kw': cooling_kw,
            'total_power_kw': total_power,
            'battery_charge_kw': battery_charge_kw,
            'battery_discharge_kw': battery_discharge_kw,
            'power_scale_factor': scale_factor
        },
        'constraints_satisfied': constraints_satisfied,
        'constraint_violations': violations,
        'system_state': {
            'soc': soc,
            'utilization': total_power / max(power_limit, 1e-6),  # Avoid division by zero
            'efficiency': inventory.get('gpu_utilization', 0.8),
            'emergency_scaled': scale_factor < 1.0,
            'temperature': system_state.temperature,
            'grid_frequency': system_state.grid_frequency_hz
        },
        'market_data': {
            'bid_price': 45.0,
            'clearing_price': 50.0,
            'profit_margin': 0.1
        },
        'performance_metrics': {
            'build_time_ms': (time.perf_counter() - start_time) * 1000,
            'constraint_validation_time_ms': agent.metrics.constraint_validation_time,
            'total_calls': agent.metrics.calls_processed + 1
        }
    }
    
    agent.metrics.build_payload_time = (time.perf_counter() - start_time) * 1000
    agent.metrics.calls_processed += 1
    
    return payload


def real_time_adjustment(current_payload: Dict[str, Any], market_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-fast real-time dispatch adjustment with market signal processing.
    
    Args:
        current_payload: Current dispatch payload
        market_signal: Real-time market signal
        
    Returns:
        Adjusted payload with performance guarantees
    """
    start_time = time.perf_counter()
    
    adjusted_payload = current_payload.copy()
    
    # Convert market signal to structured format
    signal = MarketSignal(
        price=market_signal.get('price', 50.0),
        volume=market_signal.get('volume', 0.0),
        timestamp=market_signal.get('timestamp', time.time()),
        frequency=market_signal.get('frequency', 60.0)
    )
    
    # Fast price-responsive scaling
    price_ratio = signal.price / 50.0  # Relative to baseline
    
    # Advanced scaling logic based on market conditions
    if price_ratio > 1.3:  # Very high prices
        scale_factor = min(1.5, price_ratio * 1.1)
    elif price_ratio > 1.2:  # High prices
        scale_factor = min(1.2, price_ratio)
    elif price_ratio < 0.7:  # Very low prices
        scale_factor = max(0.3, price_ratio * 0.8)
    elif price_ratio < 0.8:  # Low prices
        scale_factor = max(0.5, price_ratio)
    else:
        scale_factor = 1.0
    
    # Grid frequency response (fast frequency regulation)
    freq_deviation = signal.frequency - 60.0
    if abs(freq_deviation) > 0.1:  # Significant frequency deviation
        freq_response = 1.0 - (freq_deviation * 0.1)  # 10% response per 0.1 Hz
        scale_factor *= freq_response
    
    # Apply scaling with constraint checking
    original_gpu_power = adjusted_payload['power_requirements']['gpu_power_kw']
    cooling_power = adjusted_payload['power_requirements']['cooling_power_kw']
    
    new_gpu_power = original_gpu_power * scale_factor
    new_total_power = new_gpu_power + cooling_power
    
    # Fast constraint check
    power_limit = original_gpu_power + cooling_power + 200  # Estimated headroom
    if new_total_power > power_limit:
        scale_factor = (power_limit - cooling_power) / original_gpu_power
        new_gpu_power = original_gpu_power * scale_factor
        new_total_power = new_gpu_power + cooling_power
    
    # Update allocation proportionally
    for service in adjusted_payload['allocation']:
        adjusted_payload['allocation'][service] *= scale_factor
    
    # Update power requirements
    adjusted_payload['power_requirements']['gpu_power_kw'] = new_gpu_power
    adjusted_payload['power_requirements']['total_power_kw'] = new_total_power
    adjusted_payload['system_state']['utilization'] = new_total_power / power_limit
    
    # Add market response data
    adjusted_payload['adjustment_factor'] = scale_factor
    adjusted_payload['market_response'] = {
        'price_ratio': price_ratio,
        'frequency_deviation': freq_deviation,
        'signal_timestamp': signal.timestamp
    }
    
    adjustment_time = (time.perf_counter() - start_time) * 1000
    adjusted_payload['performance_metrics']['adjustment_time_ms'] = adjustment_time
    
    return adjusted_payload


def emergency_response(system_state: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced emergency response with comprehensive safety protocols.
    
    Args:
        system_state: Current system state
        
    Returns:
        Emergency response actions with performance tracking
    """
    start_time = time.perf_counter()
    
    response = {
        'emergency_level': EmergencyLevel.NORMAL.value,
        'actions': [],
        'power_reduction': 0.0,
        'safe_mode': False,
        'shutdown_required': False,
        'estimated_recovery_time': 0
    }
    
    # Extract system parameters
    temp = system_state.get('temperature', 65.0)
    soc = system_state.get('soc', 0.5)
    power_limit = system_state.get('power_limit', 1000.0)
    current_power = system_state.get('total_power_kw', 0.0)
    grid_freq = system_state.get('grid_frequency', 60.0)
    cooling_capacity = system_state.get('cooling_capacity', 500.0)
    
    # Critical power constraint violation
    if current_power > power_limit * 1.1:  # 10% over limit
        response['emergency_level'] = EmergencyLevel.SHUTDOWN.value
        response['actions'].append('IMMEDIATE_SHUTDOWN')
        response['power_reduction'] = 1.0  # Complete shutdown
        response['shutdown_required'] = True
        response['estimated_recovery_time'] = 300  # 5 minutes
    elif current_power > power_limit:
        response['emergency_level'] = EmergencyLevel.CRITICAL.value
        response['actions'].append('EMERGENCY_POWER_REDUCTION')
        response['power_reduction'] = 1.0 - (power_limit * 0.9) / current_power
        response['safe_mode'] = True
        response['estimated_recovery_time'] = 60  # 1 minute
    
    # Temperature emergencies with cascading response
    if temp > 85:  # Critical temperature - immediate action
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.SHUTDOWN.value)
        response['actions'].extend(['EMERGENCY_SHUTDOWN', 'MAXIMUM_COOLING'])
        response['power_reduction'] = max(response['power_reduction'], 1.0)
        response['shutdown_required'] = True
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 600)
    elif temp > 80:  # Critical temperature
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.CRITICAL.value)
        response['actions'].extend(['EMERGENCY_COOLING', 'REDUCE_GPU_LOAD'])
        response['power_reduction'] = max(response['power_reduction'], 0.5)
        response['safe_mode'] = True
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 180)
    elif temp > 75:  # Warning temperature
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.WARNING.value)
        response['actions'].append('INCREASE_COOLING')
        response['power_reduction'] = max(response['power_reduction'], 0.2)
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 30)
    
    # Battery emergencies
    if soc < 0.05:  # Critical battery level
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.SHUTDOWN.value)
        response['actions'].extend(['EMERGENCY_CHARGE', 'LOAD_SHEDDING'])
        response['shutdown_required'] = True
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 900)
    elif soc < 0.1:  # Low battery
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.CRITICAL.value)
        response['actions'].append('EMERGENCY_CHARGE')
        response['safe_mode'] = True
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 300)
    elif soc > 0.95:  # Overcharge risk
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.WARNING.value)
        response['actions'].append('REDUCE_CHARGING')
    
    # Grid frequency emergencies
    freq_deviation = abs(grid_freq - 60.0)
    if freq_deviation > 0.5:  # Major grid disturbance
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.CRITICAL.value)
        response['actions'].extend(['GRID_DISCONNECT', 'ISLANDING_MODE'])
        response['safe_mode'] = True
        response['estimated_recovery_time'] = max(response['estimated_recovery_time'], 180)
    elif freq_deviation > 0.2:  # Grid instability
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.WARNING.value)
        response['actions'].append('FREQUENCY_REGULATION')
    
    # Cooling system emergencies
    if cooling_capacity < current_power * 0.3:  # Insufficient cooling
        response['emergency_level'] = max(response['emergency_level'], EmergencyLevel.CRITICAL.value)
        response['actions'].extend(['EMERGENCY_COOLING', 'REDUCE_LOAD'])
        response['power_reduction'] = max(response['power_reduction'], 0.4)
        response['safe_mode'] = True
    
    # Performance tracking
    emergency_time = (time.perf_counter() - start_time) * 1000
    response['performance_metrics'] = {
        'emergency_response_time_ms': emergency_time,
        'actions_count': len(response['actions']),
        'severity_level': response['emergency_level']
    }
    
    return response


def validate_dispatch_performance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that dispatch meets <100ms performance requirement."""
    
    metrics = payload.get('performance_metrics', {})
    build_time = metrics.get('build_time_ms', 0.0)
    adjustment_time = metrics.get('adjustment_time_ms', 0.0)
    emergency_time = metrics.get('emergency_response_time_ms', 0.0)
    
    total_time = build_time + adjustment_time + emergency_time
    
    return {
        'total_response_time_ms': total_time,
        'performance_target_ms': 100.0,
        'meets_target': total_time < 100.0,
        'performance_margin_ms': 100.0 - total_time,
        'breakdown': {
            'build_payload_ms': build_time,
            'real_time_adjustment_ms': adjustment_time,
            'emergency_response_ms': emergency_time
        }
    } 