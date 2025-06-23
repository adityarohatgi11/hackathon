"""Enhanced data agent with intelligent caching and minimal API dependencies."""

from __future__ import annotations

import logging
import time
import pickle
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    from api_client import client as api_client
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False

from forecasting.feature_engineering import FeatureEngineer
from .enhanced_base_agent import EnhancedBaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class EnhancedDataAgent(EnhancedBaseAgent):
    """Enhanced data agent with intelligent caching and minimal API calls."""

    subscribe_topics = []  # No subscriptions - autonomous data source
    publish_topic = "feature-vector"

    def __init__(self, fetch_interval: int = 60, cache_dir: str = "data/cache"):
        # Configure for high-performance data operations
        config = AgentConfig(
            cache_size=5000,  # Large cache for data
            cache_ttl=300.0,  # 5 minute cache for features
            enable_caching=True,
            enable_metrics=True,
            max_retries=5,  # More retries for API calls
            retry_delay=2.0,
        )
        
        super().__init__(name="EnhancedDataAgent", config=config)
        
        self._feature_engineer = FeatureEngineer()
        self._fetch_interval = fetch_interval
        self._last_fetch = 0.0
        self._cache_dir = cache_dir
        self._data_persistence_file = os.path.join(cache_dir, "price_data.pkl")
        
        # Internal data storage
        self._price_history: List[Dict[str, Any]] = []
        self._last_api_success = 0.0
        self._api_failure_count = 0
        self._use_synthetic_data = False
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _initialize(self) -> None:
        """Initialize data sources and load persisted data."""
        logger.info(f"[{self.name}] Initializing data sources")
        
        # Load persisted data
        self._load_persisted_data()
        
        # Test API connectivity
        if HAS_API_CLIENT:
            self._test_api_connectivity()
        else:
            logger.warning(f"[{self.name}] No API client available, using synthetic data")
            self._use_synthetic_data = True

    def _health_check(self) -> bool:
        """Check data agent health including API connectivity and data freshness."""
        try:
            # Check if we have recent data
            if not self._price_history:
                return False
            
            # Check data freshness
            last_data_time = self._price_history[-1].get("timestamp", 0)
            if isinstance(last_data_time, str):
                last_data_time = pd.to_datetime(last_data_time).timestamp()
            
            data_age = time.time() - last_data_time
            if data_age > 3600:  # 1 hour
                return False
            
            return True
            
        except Exception as exc:
            logger.error(f"[{self.name}] Health check failed: {exc}")
            return False

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Not used - this agent is autonomous."""
        return None

    def start(self) -> None:
        """Override start to run autonomous data fetching loop."""
        logger.info(f"[{self.name}] Starting autonomous data collection")
        self.state = self.state.__class__.HEALTHY
        self._running = True

        # Start health monitoring
        if self.config.enable_metrics:
            self._start_health_monitoring()

        # Initialize
        try:
            self._initialize()
        except Exception as exc:
            logger.error(f"[{self.name}] Initialization failed: {exc}")
            self.state = self.state.__class__.UNHEALTHY
            return

        # Main autonomous loop
        while self._running:
            try:
                current_time = time.time()
                if current_time - self._last_fetch >= self._fetch_interval:
                    self._fetch_and_publish()
                    self._last_fetch = current_time
                    
                # Persist data periodically
                if len(self._price_history) % 50 == 0:  # Every 50 data points
                    self._persist_data()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as exc:
                logger.exception(f"[{self.name}] Error in autonomous loop: {exc}")
                self._handle_failure(exc)
                time.sleep(10)  # Wait before retrying

    def _fetch_and_publish(self) -> None:
        """Fetch data and publish with intelligent analysis."""
        try:
            # Try to fetch real data first
            prices_df, inventory_data = self._get_data()
            
            if prices_df.empty:
                logger.warning(f"[{self.name}] No data received, skipping publish")
                return

            # Engineer features
            features_df = self._feature_engineer.engineer_features(prices_df)

            # Perform intelligent analysis
            market_insights = self._analyze_market_intelligence(prices_df, inventory_data, features_df)

            # Prepare comprehensive message
            message = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "prices": self._make_json_serializable(prices_df.to_dict(orient="records")),
                "inventory": self._make_json_serializable(inventory_data),
                "features": self._make_json_serializable(features_df.to_dict(orient="records")),
                "market_intelligence": market_insights,
                "data_quality": self._assess_data_quality(prices_df),
                "source": self.name,
                "agent_id": self.agent_id,
                "using_synthetic": self._use_synthetic_data,
            }

            # Publish to message bus
            self.bus.publish(self.publish_topic, message)
            logger.info(f"[{self.name}] Published intelligent data with {len(prices_df)} records")

        except Exception as exc:
            logger.exception(f"[{self.name}] Failed to fetch and publish data: {exc}")
            self._api_failure_count += 1
            raise

    def _get_data(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Get data from API or synthetic source with caching."""
        if self._use_synthetic_data or not HAS_API_CLIENT:
            return self._generate_synthetic_data()
        
        try:
            # Try real API
            prices_df = api_client.get_prices()
            inventory_data = api_client.get_inventory()
            
            if not prices_df.empty:
                # Success - store in history
                self._store_price_data(prices_df.to_dict(orient="records"))
                self._last_api_success = time.time()
                return prices_df, inventory_data
            else:
                raise Exception("Empty data from API")
                
        except Exception as exc:
            logger.warning(f"[{self.name}] API fetch failed: {exc}")
            self._api_failure_count += 1
            
            # Fallback to synthetic
            logger.info(f"[{self.name}] Generating synthetic data as fallback")
            self._use_synthetic_data = True
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate realistic synthetic data for testing and fallback."""
        now = pd.Timestamp.now()
        
        # Generate realistic price data
        num_points = 24  # Last 24 hours
        timestamps = [now - pd.Timedelta(hours=i) for i in range(num_points, 0, -1)]
        
        # Base price with trend and noise
        base_price = 3.5
        trend = np.random.normal(0, 0.01, num_points)
        noise = np.random.normal(0, 0.1, num_points)
        
        prices = []
        current_price = base_price
        
        for i, ts in enumerate(timestamps):
            current_price = max(0.1, current_price + trend[i] + noise[i])
            
            prices.append({
                "timestamp": ts.isoformat(),
                "energy_price": current_price,
                "hash_price": current_price * 0.8,
                "token_price": current_price * 1.2,
                "volume": np.random.uniform(100, 1000),
                "synthetic": True
            })
        
        # Store for continuity
        self._store_price_data(prices)
        
        prices_df = pd.DataFrame(prices)
        
        # Generate synthetic inventory
        inventory_data = {
            "timestamp": now.isoformat(),
            "utilization_rate": np.random.uniform(40, 80),
            "battery_soc": np.random.uniform(0.2, 0.8),
            "power_available": np.random.uniform(400, 600),
            "synthetic": True
        }
        
        return prices_df, inventory_data

    def _store_price_data(self, price_data: List[Dict[str, Any]]) -> None:
        """Store price data in history with size limits."""
        self._price_history.extend(price_data)
        
        # Keep only last 1000 records to manage memory
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]

    def _persist_data(self) -> None:
        """Persist data to disk for recovery."""
        try:
            with open(self._data_persistence_file, 'wb') as f:
                pickle.dump(self._price_history[-500:], f)  # Save last 500 records
                
            logger.debug(f"[{self.name}] Data persisted to disk")
            
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to persist data: {exc}")

    def _load_persisted_data(self) -> None:
        """Load persisted data from disk."""
        try:
            if os.path.exists(self._data_persistence_file):
                with open(self._data_persistence_file, 'rb') as f:
                    self._price_history = pickle.load(f)
                logger.info(f"[{self.name}] Loaded {len(self._price_history)} price records from cache")
                
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to load persisted data: {exc}")
            self._price_history = []

    def _test_api_connectivity(self) -> None:
        """Test API connectivity and set operational mode."""
        try:
            test_data = api_client.get_prices()
            if not test_data.empty:
                logger.info(f"[{self.name}] API connectivity confirmed")
                self._use_synthetic_data = False
            else:
                logger.warning(f"[{self.name}] API returned empty data, using synthetic mode")
                self._use_synthetic_data = True
        except Exception as exc:
            logger.warning(f"[{self.name}] API test failed: {exc}, using synthetic mode")
            self._use_synthetic_data = True

    def _analyze_market_intelligence(self, prices_df, inventory_data, features_df) -> Dict[str, Any]:
        """Perform comprehensive market analysis using local algorithms."""
        try:
            insights = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_source": "synthetic" if self._use_synthetic_data else "api",
                "market_regime": self._identify_market_regime(prices_df),
                "system_health": self._analyze_system_health(inventory_data),
                "alerts": self._generate_smart_alerts(prices_df, inventory_data),
                "confidence": self._calculate_analysis_confidence()
            }
            
            return insights
            
        except Exception as exc:
            logger.warning(f"[{self.name}] Market intelligence analysis failed: {exc}")
            return {"error": str(exc), "timestamp": pd.Timestamp.now().isoformat()}

    def _identify_market_regime(self, prices_df) -> Dict[str, Any]:
        """Identify current market regime using technical analysis."""
        if prices_df.empty or 'energy_price' not in prices_df.columns:
            return {"regime": "unknown", "confidence": 0.0}
        
        prices = prices_df['energy_price'].values
        
        # Moving averages for trend
        if len(prices) >= 10:
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-10:])
            
            if short_ma > long_ma * 1.02:
                regime = "bullish"
            elif short_ma < long_ma * 0.98:
                regime = "bearish"
            else:
                regime = "sideways"
        else:
            regime = "unknown"
        
        # Volatility regime
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * 100
        else:
            volatility = 0
        
        vol_regime = "high" if volatility > 5 else "medium" if volatility > 2 else "low"
        
        return {
            "trend_regime": regime,
            "volatility_regime": vol_regime,
            "volatility_pct": round(volatility, 2),
            "current_price": round(prices[-1], 2),
            "confidence": 0.8 if len(prices) >= 10 else 0.5
        }

    def _analyze_system_health(self, inventory_data) -> Dict[str, Any]:
        """Analyze system health from inventory data."""
        if not inventory_data:
            return {"status": "unknown"}
        
        utilization = inventory_data.get("utilization_rate", 50.0)
        battery_soc = inventory_data.get("battery_soc", 0.5)
        power_available = inventory_data.get("power_available", 500)
        
        health_score = 1.0
        issues = []
        
        # Check utilization
        if utilization > 90:
            issues.append("high_utilization")
            health_score -= 0.2
        elif utilization < 20:
            issues.append("low_utilization")
            health_score -= 0.1
        
        # Check battery
        if battery_soc < 0.2:
            issues.append("low_battery")
            health_score -= 0.3
        
        status = "excellent" if health_score > 0.9 else "good" if health_score > 0.7 else "poor"
        
        return {
            "status": status,
            "score": max(0.0, health_score),
            "issues": issues,
            "utilization": utilization,
            "battery_level": battery_soc,
            "power_available": power_available
        }

    def _generate_smart_alerts(self, prices_df, inventory_data) -> List[Dict[str, Any]]:
        """Generate intelligent alerts based on data analysis."""
        alerts = []
        
        # Price-based alerts
        if not prices_df.empty and 'energy_price' in prices_df.columns:
            current_price = prices_df['energy_price'].iloc[-1]
            avg_price = prices_df['energy_price'].mean()
            
            if current_price > avg_price * 1.5:
                alerts.append({
                    "type": "price_spike",
                    "severity": "high",
                    "message": f"Energy price spike detected: {current_price:.2f} (avg: {avg_price:.2f})",
                    "value": current_price
                })
        
        # System-based alerts
        if inventory_data:
            utilization = inventory_data.get("utilization_rate", 50)
            battery_soc = inventory_data.get("battery_soc", 0.5)
            
            if utilization > 95:
                alerts.append({
                    "type": "high_utilization",
                    "severity": "high",
                    "message": f"System utilization critical: {utilization:.1f}%",
                    "value": utilization
                })
            
            if battery_soc < 0.1:
                alerts.append({
                    "type": "low_battery",
                    "severity": "critical",
                    "message": f"Battery critically low: {battery_soc:.1%}",
                    "value": battery_soc
                })
        
        return alerts

    def _calculate_analysis_confidence(self) -> float:
        """Calculate confidence in analysis based on data quality and availability."""
        confidence = 1.0
        
        # Reduce confidence for synthetic data
        if self._use_synthetic_data:
            confidence *= 0.7
        
        # Reduce confidence for API issues
        if self._api_failure_count > 0:
            failure_impact = min(0.3, self._api_failure_count * 0.05)
            confidence -= failure_impact
        
        return max(0.1, confidence)

    def _assess_data_quality(self, prices_df) -> Dict[str, Any]:
        """Assess overall data quality."""
        if prices_df.empty:
            return {"status": "poor", "score": 0.0, "issues": ["no_data"]}
        
        issues = []
        score = 1.0
        
        # Check for synthetic data
        if self._use_synthetic_data:
            issues.append("synthetic_data")
            score *= 0.8
        
        status = "excellent" if score > 0.9 else "good" if score > 0.7 else "poor"
        
        return {
            "status": status,
            "score": max(0.0, score),
            "issues": issues,
            "data_points": len(prices_df),
            "using_synthetic": self._use_synthetic_data
        }

    def _make_json_serializable(self, data):
        """Make data JSON serializable."""
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.floating, np.ndarray)):
            return data.item() if hasattr(data, 'item') else float(data)
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif pd.isna(data):
            return None
        else:
            return data

    def _cleanup(self) -> None:
        """Cleanup resources and persist data."""
        logger.info(f"[{self.name}] Cleaning up and persisting data")
        self._persist_data()
        
    def _recover(self) -> None:
        """Enhanced recovery for data agent."""
        logger.info(f"[{self.name}] Attempting data agent recovery")
        
        # Reset API failure count periodically
        if self._api_failure_count > 10:
            self._api_failure_count = max(0, self._api_failure_count - 5)
        
        # Test API connectivity again
        if HAS_API_CLIENT and self._use_synthetic_data:
            try:
                test_data = api_client.get_prices()
                if not test_data.empty:
                    self._use_synthetic_data = False
                    logger.info(f"[{self.name}] API recovered, switching back to real data")
            except:
                pass  # Continue with synthetic data
        
        super()._recover()
