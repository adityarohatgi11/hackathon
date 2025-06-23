from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from api_client import client as api_client
from forecasting.feature_engineering import FeatureEngineer

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Agent that fetches MARA API data and publishes engineered features."""

    subscribe_topics = []  # No subscriptions - this is a data source
    publish_topic = "feature-vector"

    def __init__(self, fetch_interval: int = 60):
        super().__init__(name="DataAgent")
        self._feature_engineer = FeatureEngineer()
        self._fetch_interval = fetch_interval
        self._last_fetch = 0.0

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Not used - this agent generates data autonomously."""
        return None

    def start(self) -> None:
        """Override start to run autonomous data fetching loop."""
        logger.info("[%s] Starting autonomous data fetching", self.name)
        self._running = True

        while self._running:
            try:
                current_time = time.time()
                if current_time - self._last_fetch >= self._fetch_interval:
                    self._fetch_and_publish()
                    self._last_fetch = current_time
                time.sleep(5)  # Check every 5 seconds
            except Exception as exc:
                logger.exception("[%s] Error in data fetch loop: %s", self.name, exc)
                time.sleep(10)  # Wait before retrying

    def _fetch_and_publish(self) -> None:
        """Fetch data from MARA API and publish feature vector with intelligent analysis."""
        try:
            # Fetch raw price data
            prices_df = api_client.get_prices()
            if prices_df.empty:
                logger.warning("No price data received from MARA API")
                return

            # Fetch inventory data
            inventory_data = api_client.get_inventory()

            # Engineer features
            features_df = self._feature_engineer.engineer_features(prices_df)

            # Intelligent market analysis
            market_insights = self._analyze_market_intelligence(prices_df, inventory_data, features_df)

            # Prepare message with JSON-serializable data and intelligence
            message = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "prices": self._make_json_serializable(prices_df.to_dict(orient="records")),
                "inventory": self._make_json_serializable(inventory_data),
                "features": self._make_json_serializable(features_df.to_dict(orient="records")),
                "market_intelligence": market_insights,  # New intelligent analysis
                "source": self.name,
            }

            # Publish to message bus
            self.bus.publish(self.publish_topic, message)
            logger.info("[%s] Published intelligent feature vector with %d price records and market insights", 
                       self.name, len(prices_df))

        except Exception as exc:
            logger.exception("[%s] Failed to fetch and publish data: %s", self.name, exc)

    def _analyze_market_intelligence(self, prices_df, inventory_data, features_df) -> Dict[str, Any]:
        """Perform intelligent market analysis beyond basic data collection."""
        try:
            insights = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_quality": self._assess_data_quality(prices_df),
                "market_state": self._identify_market_state(prices_df),
                "system_health": self._analyze_system_health(inventory_data),
                "feature_insights": self._analyze_feature_patterns(features_df),
                "alerts": self._generate_market_alerts(prices_df, inventory_data),
                "confidence": 0.8
            }
            
            return insights
            
        except Exception as exc:
            logger.warning("[%s] Market intelligence analysis failed: %s", self.name, exc)
            return {"error": str(exc), "timestamp": pd.Timestamp.now().isoformat()}

    def _assess_data_quality(self, prices_df) -> Dict[str, Any]:
        """Assess the quality and reliability of incoming data."""
        if prices_df.empty:
            return {"status": "poor", "issues": ["no_data"], "score": 0.0}
        
        issues = []
        score = 1.0
        
        # Check for missing values
        missing_pct = prices_df.isnull().sum().sum() / (len(prices_df) * len(prices_df.columns))
        if missing_pct > 0.1:
            issues.append("high_missing_data")
            score -= 0.3
        
        # Check data freshness
        if 'timestamp' in prices_df.columns:
            latest_time = pd.to_datetime(prices_df['timestamp']).max()
            age_minutes = (pd.Timestamp.now() - latest_time).total_seconds() / 60
            if age_minutes > 30:
                issues.append("stale_data")
                score -= 0.2
        
        # Check for outliers in prices
        if 'energy_price' in prices_df.columns:
            q1, q3 = prices_df['energy_price'].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((prices_df['energy_price'] < (q1 - 1.5 * iqr)) | 
                       (prices_df['energy_price'] > (q3 + 1.5 * iqr))).sum()
            outlier_pct = outliers / len(prices_df)
            if outlier_pct > 0.1:
                issues.append("price_outliers")
                score -= 0.1
        
        status = "excellent" if score > 0.9 else "good" if score > 0.7 else "poor"
        
        return {
            "status": status,
            "score": max(0.0, score),
            "issues": issues,
            "data_points": len(prices_df),
            "completeness": 1.0 - missing_pct
        }

    def _identify_market_state(self, prices_df) -> Dict[str, Any]:
        """Identify current market regime and conditions."""
        if prices_df.empty or 'energy_price' not in prices_df.columns:
            return {"regime": "unknown", "volatility": "unknown", "trend": "unknown"}
        
        prices = prices_df['energy_price'].values
        
        # Calculate volatility
        if len(prices) > 1:
            price_changes = np.diff(prices) / prices[:-1]
            volatility = np.std(price_changes) * 100  # As percentage
        else:
            volatility = 0
        
        # Determine volatility regime
        vol_regime = "high" if volatility > 5 else "medium" if volatility > 1 else "low"
        
        # Calculate trend
        if len(prices) >= 10:
            recent_trend = (prices[-1] - prices[-10]) / prices[-10]
            trend = "bullish" if recent_trend > 0.05 else "bearish" if recent_trend < -0.05 else "sideways"
        else:
            trend = "unknown"
        
        # Market regime classification
        current_price = prices[-1]
        if current_price > 5.0:
            regime = "high_price"
        elif current_price < 2.0:
            regime = "low_price"
        else:
            regime = "normal"
        
        return {
            "regime": regime,
            "volatility": vol_regime,
            "volatility_pct": round(volatility, 2),
            "trend": trend,
            "current_price": round(current_price, 2),
            "price_range": {
                "min": round(prices.min(), 2),
                "max": round(prices.max(), 2),
                "avg": round(prices.mean(), 2)
            }
        }

    def _analyze_system_health(self, inventory_data) -> Dict[str, Any]:
        """Analyze system health and operational status."""
        utilization = inventory_data.get("utilization_rate", 50.0)
        battery_soc = inventory_data.get("battery_soc", 0.5)
        power_available = inventory_data.get("power_available", 500)
        
        health_score = 1.0
        issues = []
        
        # Analyze utilization
        if utilization > 90:
            health_score -= 0.3
            issues.append("high_utilization_risk")
        elif utilization < 20:
            issues.append("underutilization")
        
        # Analyze battery
        if battery_soc < 0.2:
            health_score -= 0.4
            issues.append("low_battery_critical")
        elif battery_soc < 0.4:
            health_score -= 0.2
            issues.append("low_battery_warning")
        
        # Analyze power availability
        if power_available < 100:
            health_score -= 0.3
            issues.append("limited_power_capacity")
        
        status = "critical" if health_score < 0.5 else "warning" if health_score < 0.8 else "healthy"
        
        return {
            "status": status,
            "score": max(0.0, health_score),
            "issues": issues,
            "metrics": {
                "utilization_rate": utilization,
                "battery_soc": battery_soc,
                "power_available": power_available
            }
        }

    def _analyze_feature_patterns(self, features_df) -> Dict[str, Any]:
        """Analyze patterns in engineered features for insights."""
        if features_df.empty:
            return {"status": "no_features", "insights": []}
        
        insights = []
        
        # Analyze price momentum features
        momentum_features = [col for col in features_df.columns if 'momentum' in col.lower()]
        if momentum_features:
            momentum_avg = features_df[momentum_features].mean().mean()
            if momentum_avg > 0.1:
                insights.append("Strong positive price momentum detected")
            elif momentum_avg < -0.1:
                insights.append("Strong negative price momentum detected")
        
        # Analyze volatility features
        vol_features = [col for col in features_df.columns if 'volatility' in col.lower() or 'std' in col.lower()]
        if vol_features:
            vol_avg = features_df[vol_features].mean().mean()
            if vol_avg > features_df[vol_features].std().mean():
                insights.append("Elevated volatility conditions")
        
        # Analyze trend features
        trend_features = [col for col in features_df.columns if 'trend' in col.lower() or 'ma' in col.lower()]
        if trend_features:
            trend_direction = features_df[trend_features].mean().mean()
            if abs(trend_direction) > 0.05:
                direction = "upward" if trend_direction > 0 else "downward"
                insights.append(f"Clear {direction} trend pattern identified")
        
        return {
            "status": "analyzed",
            "feature_count": len(features_df.columns),
            "insights": insights,
            "key_patterns": {
                "momentum_signal": momentum_avg if 'momentum_avg' in locals() else None,
                "volatility_level": vol_avg if 'vol_avg' in locals() else None,
                "trend_strength": abs(trend_direction) if 'trend_direction' in locals() else None
            }
        }

    def _generate_market_alerts(self, prices_df, inventory_data) -> List[Dict[str, Any]]:
        """Generate intelligent alerts based on market conditions."""
        alerts = []
        
        if not prices_df.empty and 'energy_price' in prices_df.columns:
            current_price = prices_df['energy_price'].iloc[-1]
            
            # Price alerts
            if current_price > 8.0:
                alerts.append({
                    "type": "price_spike",
                    "severity": "high",
                    "message": f"Energy price spike detected: ${current_price:.2f}/MWh",
                    "action": "Consider energy trading opportunities"
                })
            elif current_price < 1.0:
                alerts.append({
                    "type": "price_drop",
                    "severity": "medium",
                    "message": f"Energy price drop: ${current_price:.2f}/MWh",
                    "action": "Consider reducing energy trading exposure"
                })
        
        # System alerts
        utilization = inventory_data.get("utilization_rate", 50)
        if utilization > 95:
            alerts.append({
                "type": "system_overload",
                "severity": "critical",
                "message": f"System utilization critical: {utilization}%",
                "action": "Reduce load immediately"
            })
        
        battery_soc = inventory_data.get("battery_soc", 0.5)
        if battery_soc < 0.15:
            alerts.append({
                "type": "battery_critical",
                "severity": "high",
                "message": f"Battery SOC critical: {battery_soc*100:.1f}%",
                "action": "Prioritize battery charging"
            })
        
        return alerts

    def _make_json_serializable(self, data):
        """Convert pandas data structures to JSON-serializable format."""
        if isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if hasattr(value, 'isoformat'):  # datetime/timestamp objects
                    result[key] = value.isoformat()
                elif pd.isna(value):  # NaN values
                    result[key] = None
                else:
                    result[key] = self._make_json_serializable(value)
            return result
        else:
            return data


if __name__ == "__main__":
    DataAgent(fetch_interval=30).start()  # Fetch every 30 seconds 