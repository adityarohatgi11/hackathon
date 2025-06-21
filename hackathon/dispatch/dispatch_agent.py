"""Real-time dispatch agent for market execution."""

import pandas as pd
from typing import Dict, Any


def build_payload(allocation: Dict[str, float], inventory: Dict[str, Any], 
                 soc: float, cooling_kw: float, power_limit: float) -> Dict[str, Any]:
    """Build market submission payload.
    
    Args:
        allocation: Resource allocation from auction
        inventory: Current system inventory
        soc: Battery state of charge
        cooling_kw: Required cooling power
        power_limit: Maximum power limit
        
    Returns:
        Market submission payload
    """
    # STUB: Build mock payload
    
    # Calculate total power requirement
    gpu_power = sum(allocation.values()) * 1000  # Convert to kW
    total_power = gpu_power + cooling_kw
    
    # Check constraints
    within_limits = total_power <= power_limit
    battery_ok = 0.15 <= soc <= 0.90
    
    payload = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'allocation': allocation,
        'power_requirements': {
            'gpu_power_kw': gpu_power,
            'cooling_power_kw': cooling_kw,
            'total_power_kw': total_power,
            'battery_charge_kw': max(0, power_limit - total_power) if soc < 0.8 else 0,
            'battery_discharge_kw': max(0, total_power - power_limit) if soc > 0.2 else 0
        },
        'constraints_satisfied': within_limits and battery_ok,
        'system_state': {
            'soc': soc,
            'utilization': total_power / power_limit,
            'efficiency': inventory.get('gpu_utilization', 0.8)
        },
        'market_data': {
            'bid_price': 45.0,  # Mock bid price
            'clearing_price': 50.0,  # Mock clearing price
            'profit_margin': 0.1
        }
    }
    
    return payload


def real_time_adjustment(current_payload: Dict[str, Any], market_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust dispatch in real-time based on market signals.
    
    Args:
        current_payload: Current dispatch payload
        market_signal: Real-time market signal
        
    Returns:
        Adjusted payload
    """
    # STUB: Simple real-time adjustment
    adjusted_payload = current_payload.copy()
    
    # Adjust based on price signals
    price_ratio = market_signal.get('price', 50.0) / 50.0  # Relative to baseline
    
    # Scale allocation based on price attractiveness
    if price_ratio > 1.2:  # High prices - increase allocation
        scale_factor = min(1.2, price_ratio)
    elif price_ratio < 0.8:  # Low prices - decrease allocation
        scale_factor = max(0.5, price_ratio)
    else:
        scale_factor = 1.0
    
    # Apply scaling to allocation
    for service in adjusted_payload['allocation']:
        adjusted_payload['allocation'][service] *= scale_factor
    
    adjusted_payload['adjustment_factor'] = scale_factor
    adjusted_payload['market_response'] = market_signal
    
    return adjusted_payload


def emergency_response(system_state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle emergency situations and constraint violations.
    
    Args:
        system_state: Current system state
        
    Returns:
        Emergency response actions
    """
    # STUB: Emergency response logic
    response = {
        'emergency_level': 0,  # 0=normal, 1=warning, 2=critical
        'actions': [],
        'power_reduction': 0.0,
        'safe_mode': False
    }
    
    # Check for emergency conditions
    temp = system_state.get('temperature', 65.0)
    soc = system_state.get('soc', 0.5)
    
    if temp > 80:  # Critical temperature
        response['emergency_level'] = 2
        response['actions'].append('EMERGENCY_COOLING')
        response['power_reduction'] = 0.5
        response['safe_mode'] = True
    elif temp > 75:  # Warning temperature
        response['emergency_level'] = 1
        response['actions'].append('INCREASE_COOLING')
        response['power_reduction'] = 0.2
    
    if soc < 0.1:  # Critical battery level
        response['emergency_level'] = max(response['emergency_level'], 2)
        response['actions'].append('EMERGENCY_CHARGE')
        response['safe_mode'] = True
    
    return response 