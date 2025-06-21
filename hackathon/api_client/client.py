"""API Client for GridPilot-GT energy market integration."""

import pandas as pd
import httpx
import toml
from typing import Optional, Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    try:
        return toml.load("config.toml")
    except FileNotFoundError:
        return {"api_key": "YOUR_API_KEY_HERE", "site_power_kw": 1000}


def register_site(site_name: str) -> Dict[str, Any]:
    """Register new site and get API credentials.
    
    Args:
        site_name: Name of the site to register
        
    Returns:
        Dictionary with api_key, power, and other site information
    """
    # STUB: Return mock data for now
    return {
        "api_key": "mock_api_key_12345",
        "power": 1000000,
        "site_id": "hackfest_site_001"
    }


def get_prices(start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical energy prices.
    
    Args:
        start_time: Start time for price data (ISO format)
        end_time: End time for price data (ISO format)
        
    Returns:
        DataFrame with timestamp and price columns
    """
    # STUB: Return mock price data
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate realistic price data with some volatility
    base_price = 50
    prices = base_price + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + \
             5 * np.random.randn(len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.uniform(100, 1000, len(dates))
    })


def get_inventory() -> Dict[str, Any]:
    """Get current system inventory and status.
    
    Returns:
        Dictionary with current system state
    """
    # STUB: Return mock inventory
    return {
        "power_total": 1000.0,
        "power_available": 750.0,
        "battery_soc": 0.65,
        "gpu_utilization": 0.8,
        "cooling_load": 150.0,
        "timestamp": pd.Timestamp.now().isoformat()
    }


def submit_bid(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit bid to energy market.
    
    Args:
        payload: Bid payload with allocation and pricing
        
    Returns:
        Response from market API
    """
    # STUB: Mock successful submission
    return {
        "status": "success",
        "bid_id": "bid_12345",
        "timestamp": pd.Timestamp.now().isoformat(),
        "accepted": True
    } 