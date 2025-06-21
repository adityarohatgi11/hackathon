"""Cooling system model for GPU workload management."""

from typing import Tuple, Dict, Any


def cooling_for_gpu_kW(gpu_load_kw: float) -> Tuple[float, Dict[str, Any]]:
    """Calculate cooling requirements for GPU workload.
    
    Args:
        gpu_load_kw: GPU power consumption in kW
        
    Returns:
        Tuple of (cooling_power_kw, cooling_metrics)
    """
    # STUB: Simple cooling model
    
    # Cooling efficiency curve (COP decreases with higher loads)
    base_cop = 3.5  # Coefficient of performance
    load_factor = min(1.0, gpu_load_kw / 500.0)  # Normalize to 500kW max
    cop = base_cop * (1.0 - 0.3 * load_factor)  # COP decreases with load
    
    # Cooling power required
    cooling_power = gpu_load_kw / cop
    
    # Additional metrics
    metrics = {
        'cop': cop,
        'efficiency': cop / base_cop,
        'ambient_temp': 25.0,  # Mock ambient temperature
        'coolant_temp': 15.0 + load_factor * 10,  # Coolant temp increases with load
        'fan_speed': 0.3 + load_factor * 0.7,  # Fan speed (0-1)
        'pump_power': cooling_power * 0.1  # Pump power ~10% of cooling
    }
    
    return cooling_power, metrics


def thermal_constraints(current_temp: float, target_temp: float = 65.0) -> Dict[str, float]:
    """Calculate thermal constraints for system operation.
    
    Args:
        current_temp: Current system temperature (°C)
        target_temp: Target operating temperature (°C)
        
    Returns:
        Dictionary with constraint factors
    """
    # STUB: Simple thermal constraint model
    temp_margin = target_temp - current_temp
    
    # Constraint factor (1.0 = no constraint, 0.0 = full constraint)
    if temp_margin > 10:
        factor = 1.0  # No constraint
    elif temp_margin > 0:
        factor = temp_margin / 10.0  # Linear derating
    else:
        factor = 0.1  # Emergency constraint
    
    return {
        'power_factor': factor,
        'temp_margin': temp_margin,
        'emergency_shutdown': temp_margin < -5,
        'recommended_load': factor * 100  # Percentage
    } 