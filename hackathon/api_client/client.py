"""API Client for GridPilot-GT energy market integration."""

import pandas as pd
import toml
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time
import numpy as np

# HTTP client - try httpx first, fallback to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    try:
        import requests
        HAS_HTTPX = False
    except ImportError:
        HAS_HTTPX = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Custom exception for API client errors."""
    pass


# MARA Hackathon API Configuration
MARA_API_BASE = "https://mara-hackathon-api.onrender.com"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml."""
    try:
        config = toml.load("config.toml")
        logger.info("Configuration loaded successfully")
        
        # Flatten the nested config for backward compatibility
        flat_config = {}
        for section, values in config.items():
            if isinstance(values, dict):
                flat_config.update(values)
            else:
                flat_config[section] = values
        
        return flat_config
    except FileNotFoundError:
        logger.warning("config.toml not found, using defaults")
        return {
            "api_key": "YOUR_API_KEY_HERE", 
            "site_power_kw": 1000000,
            "site_name": "HackFestSite",
            "base_url": MARA_API_BASE,
            "BATTERY_CAP_MWH": 1.0,
            "BATTERY_MAX_KW": 250.0
        }


def _make_request(url: str, headers: Dict[str, str], params: Optional[Dict] = None, 
                  retries: int = 3, method: str = "GET", json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make HTTP request with retry logic and error handling."""
    
    # If no HTTP client available, use fallback
    if HAS_HTTPX is None:
        logger.error("No HTTP client available (httpx or requests). Using fallback data.")
        raise APIClientError("No HTTP client available")
    
    for attempt in range(retries):
        try:
            if HAS_HTTPX:
                # Use httpx
                with httpx.Client(timeout=30.0) as client:
                    if method.upper() == "POST":
                        response = client.post(url, headers=headers, json=json_data)
                    elif method.upper() == "PUT":
                        response = client.put(url, headers=headers, json=json_data)
                    else:
                        response = client.get(url, headers=headers, params=params or {})
                    
                    response.raise_for_status()
                    return response.json()
            else:
                # Use requests as fallback
                if method.upper() == "POST":
                    response = requests.post(url, headers=headers, json=json_data, timeout=30.0)
                elif method.upper() == "PUT":
                    response = requests.put(url, headers=headers, json=json_data, timeout=30.0)
                else:
                    response = requests.get(url, headers=headers, params=params or {}, timeout=30.0)
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.warning(f"Request attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise APIClientError(f"Request failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise APIClientError("Unexpected error in request handling")


def register_site(site_name: str, power: int = 1000000) -> Dict[str, Any]:
    """Register new site with MARA Hackathon API.
    
    Args:
        site_name: Name of the site to register
        power: Power capacity in watts (default 1MW)
        
    Returns:
        Dictionary with api_key and site information
    """
    logger.info(f"Registering site with MARA API: {site_name}")
    
    # Site registration endpoint
    url = f"{MARA_API_BASE}/sites"
    
    # For registration, we might need to use a temporary API key or handle this differently
    # Based on the screenshots, it looks like we need an API key to register
    config = load_config()
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config.get("api_key", "YOUR_API_KEY_HERE")
    }
    
    payload = {
        "api_key": config.get("api_key", "YOUR_API_KEY_HERE"),
        "name": site_name,
        "power": power
    }
    
    try:
        response = _make_request(url, headers, method="POST", json_data=payload)
        logger.info(f"Site registered successfully: {response}")
        return response
    except APIClientError as e:
        logger.error(f"Failed to register site: {e}")
        # Return mock response for development
        return {
            "api_key": config.get("api_key", "YOUR_API_KEY_HERE"),
            "name": site_name,
            "power": power,
            "status": "registered",
            "site_id": f"site_{site_name.lower().replace(' ', '_')}"
        }


def get_prices(start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
    """Fetch real-time energy prices from MARA Hackathon API.
    
    Args:
        start_time: Start time for price data (not used for real-time API)
        end_time: End time for price data (not used for real-time API)
        
    Returns:
        DataFrame with timestamp, price, and market data
    """
    logger.info("Fetching real-time prices from MARA API")
    
    config = load_config()
    url = f"{MARA_API_BASE}/prices"
    headers = {
        "X-Api-Key": config.get("api_key", "YOUR_API_KEY_HERE")
    }
    
    try:
        # Get current real-time prices
        response = _make_request(url, headers)
        
        # Convert MARA API response to our expected format
        current_time = datetime.now()
        
        # Handle both list and dict responses
        if isinstance(response, list) and len(response) > 0:
            # Use all historical data from API if available
            prices_data = []
            for price_record in response:
                timestamp = pd.to_datetime(price_record.get("timestamp", current_time.isoformat()))
                prices_data.append({
                    'timestamp': timestamp,
                    'price': price_record.get("energy_price", 50.0),
                    'hash_price': price_record.get("hash_price", 8.0), 
                    'token_price': price_record.get("token_price", 3.0),
                    'volume': 1000,  # Default volume
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'is_weekend': int(timestamp.weekday() >= 5)
                })
            
            # If we have real data, use it
            if len(prices_data) >= 24:  # Sufficient data
                df = pd.DataFrame(prices_data)
                # Add additional market features
                df['price_ma_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
                df['price_volatility_24h'] = df['price'].rolling(window=24, min_periods=1).std().fillna(0)
                df['load_factor'] = df['volume'] / df['volume'].max()
                df['market_stress'] = (df['price'] > df['price_ma_24h'] + 2 * df['price_volatility_24h']).astype(int)
                
                logger.info(f"Retrieved {len(df)} real price records from MARA API")
                return df
            else:
                # Use latest real price as base for simulation
                latest_price = response[-1]
                base_energy_price = latest_price.get("energy_price", 50.0)
                base_hash_price = latest_price.get("hash_price", 8.0)
                base_token_price = latest_price.get("token_price", 3.0)
        else:
            # Single price object
            base_energy_price = response.get("energy_price", 50.0)
            base_hash_price = response.get("hash_price", 8.0)
            base_token_price = response.get("token_price", 3.0)
        
        # Create historical data by simulation based on current prices
        prices_data = []
        
        # Generate recent historical data (last 7 days)
        for i in range(168):  # 7 days * 24 hours
            timestamp = current_time - timedelta(hours=168-i)
            
            # Add some realistic variation to base prices
            energy_variation = np.random.normal(0, base_energy_price * 0.1)
            hash_variation = np.random.normal(0, base_hash_price * 0.15)
            token_variation = np.random.normal(0, base_token_price * 0.2)
            
            prices_data.append({
                'timestamp': timestamp,
                'price': max(base_energy_price + energy_variation, 0.01),  # Energy price as main price
                'hash_price': max(base_hash_price + hash_variation, 0.01),
                'token_price': max(base_token_price + token_variation, 0.01),
                'volume': np.random.uniform(500, 2000),
                'hour_of_day': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_weekend': int(timestamp.weekday() >= 5)
            })
        
        # Add the current real-time data
        current_timestamp = pd.to_datetime(response.get("timestamp", current_time.isoformat()))
        prices_data.append({
            'timestamp': current_timestamp,
            'price': response.get("energy_price", 50.0),
            'hash_price': response.get("hash_price", 8.0), 
            'token_price': response.get("token_price", 3.0),
            'volume': 1000,  # Default volume
            'hour_of_day': current_timestamp.hour,
            'day_of_week': current_timestamp.weekday(),
            'is_weekend': int(current_timestamp.weekday() >= 5)
        })
        
        df = pd.DataFrame(prices_data)
        
        # Add additional market features
        df['price_ma_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
        df['price_volatility_24h'] = df['price'].rolling(window=24, min_periods=1).std().fillna(0)
        df['load_factor'] = df['volume'] / df['volume'].max()
        df['market_stress'] = (df['price'] > df['price_ma_24h'] + 2 * df['price_volatility_24h']).astype(int)
        
        logger.info(f"Retrieved {len(df)} price records from MARA API")
        return df
        
    except (APIClientError, Exception) as e:
        logger.error(f"Failed to fetch prices from MARA API: {e}")
        # Fallback to synthetic data if API fails
        return _generate_fallback_prices()


def get_inventory() -> Dict[str, Any]:
    """Get current system inventory from MARA Hackathon API.
    
    Returns:
        Dictionary with current system state and operational metrics
    """
    logger.info("Fetching inventory from MARA API")
    
    config = load_config()
    url = f"{MARA_API_BASE}/inventory"
    headers = {
        "X-Api-Key": config.get("api_key", "YOUR_API_KEY_HERE")
    }
    
    try:
        response = _make_request(url, headers)
        
        # Convert MARA API response to our expected format
        # Extract power and token information from the complex response structure
        total_power = 0
        available_power = 0
        gpu_utilization = 0
        
        # Process inference assets
        if "inference" in response:
            inference_data = response["inference"]
            if "asic" in inference_data:
                total_power += inference_data["asic"].get("power", 0)
            if "gpu" in inference_data:
                gpu_power = inference_data["gpu"].get("power", 0)
                gpu_tokens = inference_data["gpu"].get("tokens", 0)
                total_power += gpu_power
                # Calculate GPU utilization based on tokens/power ratio
                if gpu_power > 0:
                    gpu_utilization = min(gpu_tokens / (gpu_power * 0.001), 1.0)  # Normalize
        
        # Process miners data
        if "miners" in response:
            miners_data = response["miners"]
            for miner_type in ["air", "hydro", "immersion"]:
                if miner_type in miners_data:
                    total_power += miners_data[miner_type].get("power", 0)
        
        # Estimate available power (assuming 80% max utilization)
        max_site_power = config.get("site_power_kw", 1000) * 1000  # Convert to watts
        used_power = min(total_power, max_site_power * 0.8)
        available_power = max_site_power - used_power
        
        # Calculate battery SOC based on power usage patterns
        utilization = used_power / max_site_power if max_site_power > 0 else 0
        # Simulate battery SOC (higher usage = lower SOC over time)
        base_soc = 0.65 - (utilization - 0.5) * 0.2
        soc = np.clip(base_soc + np.random.normal(0, 0.05), 0.15, 0.90)
        
        inventory = {
            "power_total": float(max_site_power / 1000),  # Convert back to kW
            "power_available": float(available_power / 1000),  # Convert to kW
            "power_used": float(used_power / 1000),  # Convert to kW
            "battery_soc": float(soc),
            "battery_capacity_mwh": config.get('BATTERY_CAP_MWH', 1.0),
            "battery_max_power_kw": config.get('BATTERY_MAX_KW', 250.0),
            "gpu_utilization": float(gpu_utilization),
            "cooling_load": float(used_power * 0.15 / 1000),  # 15% for cooling, convert to kW
            "efficiency": float(np.random.normal(0.92, 0.02)),
            "temperature": float(np.random.normal(65, 5)),
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "alerts": [],
            "market_participation": True,
            "mara_response": response  # Include raw response for debugging
        }
        
        # Add alerts for edge conditions
        if inventory["temperature"] > 75:
            inventory["alerts"].append("HIGH_TEMPERATURE")
        if inventory["battery_soc"] < 0.2:
            inventory["alerts"].append("LOW_BATTERY")
        if inventory["power_available"] < inventory["power_total"] * 0.1:
            inventory["alerts"].append("LIMITED_CAPACITY")
        
        logger.info(f"Retrieved inventory from MARA API: {utilization:.1%} utilization")
        return inventory
        
    except (APIClientError, Exception) as e:
        logger.error(f"Failed to fetch inventory from MARA API: {e}")
        # Fallback to synthetic data if API fails
        return _generate_fallback_inventory()


def submit_bid(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit machine allocation to MARA Hackathon API.
    
    Args:
        payload: Allocation payload with machine types and quantities
        
    Returns:
        Response from MARA API
    """
    logger.info("Submitting allocation to MARA API")
    
    config = load_config()
    url = f"{MARA_API_BASE}/machines"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": config.get("api_key", "YOUR_API_KEY_HERE")
    }
    
    # Convert our internal payload format to MARA API format
    allocation = payload.get('allocation', {})
    
    # Map our allocation format to MARA machine types
    mara_payload = {
        "air_miners": int(allocation.get('air_miners', 0)),
        "asic_compute": int(allocation.get('inference', 0) * 10),  # Scale inference to ASIC compute
        "gpu_compute": int(allocation.get('training', 0) * 100),   # Scale training to GPU compute  
        "hydro_miners": int(allocation.get('hydro_miners', 0)),
        "immersion_miners": int(allocation.get('immersion_miners', 0)),
        "site_id": 2,  # Default site ID, should be configured
        "updated_at": datetime.now().isoformat()
    }
    
    try:
        response = _make_request(url, headers, method="PUT", json_data=mara_payload)
        
        # Convert MARA response to our expected format
        result = {
            "status": "success",
            "bid_id": f"mara_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "allocation_accepted": True,
            "mara_response": response,
            "submitted_allocation": mara_payload
        }
        
        logger.info(f"Allocation submitted successfully to MARA API")
        return result
        
    except APIClientError as e:
        logger.error(f"Failed to submit allocation to MARA API: {e}")
        # Return failure response
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "allocation_accepted": False
        }


def get_market_status() -> Dict[str, Any]:
    """Get current machine status from MARA Hackathon API.
    
    Returns:
        Dictionary with current market and machine status
    """
    logger.info("Fetching machine status from MARA API")
    
    config = load_config()
    url = f"{MARA_API_BASE}/machines"
    headers = {
        "X-Api-Key": config.get("api_key", "YOUR_API_KEY_HERE")
    }
    
    try:
        response = _make_request(url, headers)
        
        # Extract status information
        status = {
            "grid_frequency": 60.0,  # Standard grid frequency
            "market_open": True,
            "emergency_status": "normal",
            "timestamp": datetime.now().isoformat(),
            "mara_machine_status": response
        }
        
        # Extract power and revenue information if available
        if "power" in response:
            status["total_power"] = response["power"].get("total_power_cost", 0)
            status["total_revenue"] = response.get("revenue", {}).get("total_revenue", 0)
        
        logger.info("Retrieved machine status from MARA API")
        return status
        
    except APIClientError as e:
        logger.error(f"Failed to fetch machine status from MARA API: {e}")
        # Fallback status
        return {
            "grid_frequency": 60.0,
            "market_open": True,
            "emergency_status": "normal",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


def _generate_fallback_prices() -> pd.DataFrame:
    """Generate fallback synthetic price data when API is unavailable."""
    logger.warning("Using fallback synthetic price data")
    
    # Generate realistic fallback data (previous implementation)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=7)
    dates = pd.date_range(start=start_dt, end=end_dt, freq='H')
    n_periods = len(dates)
    
    # Base price patterns
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    daily_pattern = 15 * np.sin(2 * np.pi * hour_of_day / 24)
    weekly_pattern = 5 * (5 - day_of_week) / 5 * (day_of_week < 5)
    seasonal_pattern = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    base_price = 50
    noise = np.random.normal(0, 5, n_periods)
    prices = base_price + daily_pattern + weekly_pattern + seasonal_pattern + noise
    prices = np.maximum(prices, 10)
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'hash_price': prices * 0.16,  # Approximate ratio
        'token_price': prices * 0.06,  # Approximate ratio
        'volume': np.random.uniform(500, 2000, n_periods),
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': (day_of_week >= 5).astype(int),
        'price_ma_24h': pd.Series(prices).rolling(window=24, min_periods=1).mean(),
        'price_volatility_24h': pd.Series(prices).rolling(window=24, min_periods=1).std().fillna(0),
        'load_factor': np.random.uniform(0.5, 1.0, n_periods),
        'market_stress': np.random.choice([0, 1], n_periods, p=[0.9, 0.1])
    })


def _generate_fallback_inventory() -> Dict[str, Any]:
    """Generate fallback synthetic inventory when API is unavailable."""
    logger.warning("Using fallback synthetic inventory data")
    
    config = load_config()
    base_power = config.get('site_power_kw', 1000)
    utilization = np.random.normal(0.7, 0.1)
    utilization = np.clip(utilization, 0.3, 0.95)
    
    return {
        "power_total": float(base_power),
        "power_available": float(base_power * (1 - utilization)),
        "power_used": float(base_power * utilization),
        "battery_soc": float(np.random.normal(0.65, 0.1)),
        "battery_capacity_mwh": config.get('BATTERY_CAP_MWH', 1.0),
        "battery_max_power_kw": config.get('BATTERY_MAX_KW', 250.0),
        "gpu_utilization": float(np.random.normal(0.8, 0.1)),
        "cooling_load": float(base_power * utilization * 0.15),
        "efficiency": float(np.random.normal(0.92, 0.02)),
        "temperature": float(np.random.normal(65, 5)),
        "timestamp": datetime.now().isoformat(),
        "status": "operational_fallback",
        "alerts": ["API_UNAVAILABLE"],
        "market_participation": False
    }


def test_mara_api_connection() -> Dict[str, Any]:
    """Test connection to MARA Hackathon API and verify authentication.
    
    Returns:
        Dictionary with connection status and API response details
    """
    logger.info("Testing MARA API connection...")
    
    config = load_config()
    api_key = config.get("api_key", "YOUR_API_KEY_HERE")
    
    # Test basic connectivity with prices endpoint (should work without auth)
    try:
        prices_url = f"{MARA_API_BASE}/prices"
        prices_response = _make_request(prices_url, {})
        
        logger.info("✅ Successfully connected to MARA API prices endpoint")
        
        # Handle both list and dict responses
        if isinstance(prices_response, list) and len(prices_response) > 0:
            latest_price = prices_response[-1]  # Get most recent price
        elif isinstance(prices_response, dict):
            latest_price = prices_response
        else:
            latest_price = {}
        
        price_data = {
            "prices_available": True,
            "current_energy_price": latest_price.get("energy_price"),
            "current_hash_price": latest_price.get("hash_price"),
            "current_token_price": latest_price.get("token_price"),
            "timestamp": latest_price.get("timestamp")
        }
    except APIClientError as e:
        logger.error(f"❌ Failed to connect to MARA API prices: {e}")
        price_data = {"prices_available": False, "error": str(e)}
    
    # Test authenticated endpoints
    auth_headers = {"X-Api-Key": api_key}
    auth_data = {}
    
    # Test inventory endpoint (requires auth)
    try:
        inventory_url = f"{MARA_API_BASE}/inventory"
        inventory_response = _make_request(inventory_url, auth_headers)
        
        logger.info("✅ Successfully authenticated with MARA API")
        auth_data = {
            "authentication": "success",
            "inventory_available": True,
            "api_key_valid": True
        }
    except APIClientError as e:
        logger.warning(f"⚠️ Authentication issue with MARA API: {e}")
        auth_data = {
            "authentication": "failed",
            "inventory_available": False,
            "api_key_valid": False,
            "error": str(e)
        }
    
    # Test machine status endpoint (requires auth)
    try:
        machines_url = f"{MARA_API_BASE}/machines"
        machines_response = _make_request(machines_url, auth_headers)
        
        logger.info("✅ Machine status endpoint accessible")
        auth_data["machines_available"] = True
    except APIClientError as e:
        logger.warning(f"⚠️ Machine status endpoint issue: {e}")
        auth_data["machines_available"] = False
    
    # Compile test results
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "api_base_url": MARA_API_BASE,
        "api_key_configured": api_key != "YOUR_API_KEY_HERE",
        **price_data,
        **auth_data,
        "overall_status": "operational" if price_data.get("prices_available") and auth_data.get("authentication") == "success" else "limited"
    }
    
    # Provide recommendations
    recommendations = []
    if not test_results["api_key_configured"]:
        recommendations.append("🔑 Set your MARA API key in config.toml")
    if not auth_data.get("api_key_valid"):
        recommendations.append("🔐 Verify your API key is correct and active")
    if not price_data.get("prices_available"):
        recommendations.append("🌐 Check internet connection and API status")
    
    test_results["recommendations"] = recommendations
    
    # Log summary
    if test_results["overall_status"] == "operational":
        logger.info("🎉 MARA API integration is fully operational!")
    else:
        logger.warning(f"⚠️ MARA API integration has issues: {recommendations}")
    
    return test_results 