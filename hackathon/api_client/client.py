"""API Client for GridPilot-GT energy market integration."""

import pandas as pd
import httpx
import toml
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Custom exception for API client errors."""
    pass


def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    try:
        config = toml.load("config.toml")
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.warning("config.toml not found, using defaults")
        return {"api_key": "YOUR_API_KEY_HERE", "site_power_kw": 1000}


def _make_request(url: str, headers: Dict[str, str], params: Optional[Dict] = None, 
                  retries: int = 3) -> Dict[str, Any]:
    """Make HTTP request with retry logic and error handling."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers, params=params or {})
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.warning(f"Request attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise APIClientError(f"Request failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            raise APIClientError(f"HTTP error: {e}")
    
    raise APIClientError("Unexpected error in request handling")


def register_site(site_name: str) -> Dict[str, Any]:
    """Register new site and get API credentials.
    
    Args:
        site_name: Name of the site to register
        
    Returns:
        Dictionary with api_key, power, and other site information
    """
    # TODO: Replace with real API endpoint when available
    logger.info(f"Registering site: {site_name}")
    
    # For now, return enhanced mock data with realistic values
    mock_response = {
        "api_key": f"gpt_key_{int(time.time())}_{hash(site_name) % 10000:04d}",
        "power": 1000000,  # 1 MW
        "site_id": f"site_{site_name.lower().replace(' ', '_')}",
        "registration_time": datetime.now().isoformat(),
        "status": "active",
        "market_region": "CAISO",  # California ISO
        "contract_type": "real_time_energy"
    }
    
    logger.info(f"Site registered successfully: {mock_response['site_id']}")
    return mock_response


def get_prices(start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical energy prices with enhanced data quality.
    
    Args:
        start_time: Start time for price data (ISO format)
        end_time: End time for price data (ISO format)
        
    Returns:
        DataFrame with timestamp, price, volume, and market features
    """
    logger.info(f"Fetching price data from {start_time} to {end_time}")
    
    # Parse time parameters
    if end_time is None:
        end_dt = datetime.now()
    else:
        end_dt = pd.to_datetime(end_time)
    
    if start_time is None:
        start_dt = end_dt - timedelta(days=7)  # Default to 7 days
    else:
        start_dt = pd.to_datetime(start_time)
    
    # Generate realistic market data
    dates = pd.date_range(start=start_dt, end=end_dt, freq='H')
    n_periods = len(dates)
    
    # Base price with realistic daily/seasonal patterns
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    # Daily pattern: higher prices during peak hours (6-10 AM, 6-9 PM)
    daily_pattern = (
        15 * np.sin(2 * np.pi * hour_of_day / 24) +  # Basic daily cycle
        25 * (1 / (1 + np.exp(-3 * (hour_of_day - 8)))) * (1 / (1 + np.exp(3 * (hour_of_day - 20))))  # Peak hours
    )
    
    # Weekly pattern: higher on weekdays
    weekly_pattern = 5 * (5 - day_of_week) / 5 * (day_of_week < 5)
    
    # Seasonal trend
    day_of_year = dates.dayofyear
    seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Market volatility with clustering
    volatility = np.random.exponential(0.15, n_periods)
    volatility = pd.Series(volatility).rolling(window=6, min_periods=1).mean().values  # Smooth volatility
    
    # Base price around $50/MWh
    base_price = 50
    noise = np.random.normal(0, volatility * base_price)
    
    prices = base_price + daily_pattern + weekly_pattern + seasonal_pattern + noise
    prices = np.maximum(prices, 10)  # Floor price at $10/MWh
    
    # Volume patterns (higher during peak hours)
    base_volume = 500
    volume_pattern = 200 * (1 + 0.5 * np.sin(2 * np.pi * hour_of_day / 24))
    volumes = base_volume + volume_pattern + np.random.normal(0, 50, n_periods)
    volumes = np.maximum(volumes, 50)  # Minimum volume
    
    # Additional market features for forecasting
    price_ma_24 = pd.Series(prices).rolling(window=24, min_periods=1).mean()
    price_std_24 = pd.Series(prices).rolling(window=24, min_periods=1).std().fillna(0)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': (day_of_week >= 5).astype(int),
        'price_ma_24h': price_ma_24,
        'price_volatility_24h': price_std_24,
        'load_factor': volume_pattern / volume_pattern.max(),  # Normalized load
        'market_stress': (prices > price_ma_24 + 2 * price_std_24).astype(int)  # High price periods
    })
    
    logger.info(f"Retrieved {len(df)} price records with enhanced features")
    return df


def get_inventory() -> Dict[str, Any]:
    """Get current system inventory and status with realistic dynamics.
    
    Returns:
        Dictionary with current system state and operational metrics
    """
    config = load_config()
    
    # Simulate realistic system state with some variability
    base_power = config.get('site_power_kw', 1000)
    current_time = datetime.now()
    
    # Simulate daily utilization patterns
    hour = current_time.hour
    utilization_pattern = 0.7 + 0.2 * np.sin(2 * np.pi * hour / 24)  # Higher during day
    
    # Add some randomness
    utilization = np.random.normal(utilization_pattern, 0.05)
    utilization = np.clip(utilization, 0.3, 0.95)
    
    # Battery SOC simulation (slow changes)
    soc_base = 0.65 + 0.15 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
    soc = np.random.normal(soc_base, 0.05)
    soc = np.clip(soc, 0.15, 0.90)
    
    # GPU utilization correlated with power demand
    gpu_util = utilization * np.random.normal(1.0, 0.1)
    gpu_util = np.clip(gpu_util, 0.1, 1.0)
    
    inventory = {
        "power_total": float(base_power),
        "power_available": float(base_power * (1 - utilization)),
        "power_used": float(base_power * utilization),
        "battery_soc": float(soc),
        "battery_capacity_mwh": config.get('BATTERY_CAP_MWH', 1.0),
        "battery_max_power_kw": config.get('BATTERY_MAX_KW', 250.0),
        "gpu_utilization": float(gpu_util),
        "cooling_load": float(base_power * utilization * 0.15),  # ~15% for cooling
        "efficiency": float(np.random.normal(0.92, 0.02)),  # System efficiency
        "temperature": float(np.random.normal(65, 5)),  # Operating temperature
        "timestamp": current_time.isoformat(),
        "status": "operational",
        "alerts": [],
        "market_participation": True
    }
    
    # Add alerts for edge conditions
    if inventory["temperature"] > 75:
        inventory["alerts"].append("HIGH_TEMPERATURE")
    if inventory["battery_soc"] < 0.2:
        inventory["alerts"].append("LOW_BATTERY")
    if inventory["power_available"] < base_power * 0.1:
        inventory["alerts"].append("LIMITED_CAPACITY")
    
    logger.info(f"System inventory: {utilization:.1%} utilization, {soc:.1%} SOC")
    return inventory


def submit_bid(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit bid to energy market with enhanced validation and response.
    
    Args:
        payload: Bid payload with allocation and pricing
        
    Returns:
        Response from market API with detailed status
    """
    logger.info("Submitting bid to energy market")
    
    # Validate payload
    required_fields = ['allocation', 'power_requirements', 'system_state']
    for field in required_fields:
        if field not in payload:
            raise APIClientError(f"Missing required field in payload: {field}")
    
    # Extract key metrics for logging
    total_power = payload.get('power_requirements', {}).get('total_power_kw', 0)
    constraints_ok = payload.get('constraints_satisfied', False)
    
    # Simulate market response with realistic outcomes
    success_probability = 0.85 if constraints_ok else 0.3
    accepted = np.random.random() < success_probability
    
    # Generate realistic market response
    response = {
        "status": "success" if accepted else "rejected",
        "bid_id": f"bid_{int(time.time())}_{np.random.randint(1000, 9999)}",
        "timestamp": datetime.now().isoformat(),
        "accepted": accepted,
        "total_power_kw": total_power,
        "constraints_satisfied": constraints_ok,
        "market_price": np.random.normal(50, 8),  # Current market price
        "clearing_price": np.random.normal(52, 10) if accepted else None,
        "award_duration_hours": 1 if accepted else 0,
        "revenue_estimate": total_power * np.random.normal(50, 8) if accepted else 0,
        "next_auction_time": (datetime.now() + timedelta(hours=1)).isoformat()
    }
    
    if not accepted:
        response["rejection_reason"] = "PRICE_TOO_HIGH" if not constraints_ok else "MARKET_CONGESTION"
        response["suggested_price"] = response["market_price"] * 0.95
    
    logger.info(f"Bid {'accepted' if accepted else 'rejected'}: {response['bid_id']}")
    return response


def get_market_status() -> Dict[str, Any]:
    """Get current market status and conditions.
    
    Returns:
        Dictionary with market state information
    """
    logger.info("Fetching market status")
    
    # Simulate realistic market conditions
    status = {
        "market_open": True,
        "current_price": np.random.normal(50, 8),
        "price_volatility": np.random.exponential(0.15),
        "demand_level": np.random.choice(["low", "normal", "high"], p=[0.2, 0.6, 0.2]),
        "grid_frequency": np.random.normal(60.0, 0.05),  # Grid frequency in Hz
        "reserve_margin": np.random.normal(0.15, 0.03),  # Reserve capacity margin
        "congestion_level": np.random.choice(["none", "light", "moderate", "heavy"], p=[0.4, 0.3, 0.2, 0.1]),
        "forecast_accuracy": np.random.normal(0.85, 0.05),  # Recent forecast accuracy
        "last_update": datetime.now().isoformat()
    }
    
    return status 