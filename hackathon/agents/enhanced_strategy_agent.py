"""Enhanced strategy agent with robust decision-making and minimal dependencies."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Try to import Q-learning components
try:
    from forecasting.advanced_qlearning import AdvancedQNetwork
    import torch
    HAS_QLEARNING = True
except ImportError:
    HAS_QLEARNING = False

# Try to import game theory components
try:
    from game_theory.vcg_auction import VCGAuction
    from game_theory.mpc_controller import MPCController
    HAS_GAME_THEORY = True
except ImportError:
    HAS_GAME_THEORY = False

from .enhanced_base_agent import EnhancedBaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class EnhancedStrategyAgent(EnhancedBaseAgent):
    """Enhanced strategy agent with robust decision-making."""

    subscribe_topics = ["feature-vector", "forecast"]
    publish_topic = "strategy-action"

    def __init__(self, cache_dir: str = "data/cache"):
        config = AgentConfig(
            cache_size=2000,
            cache_ttl=180.0,
            enable_caching=True,
            enable_metrics=True,
            max_retries=3,
            retry_delay=1.0,
        )
        
        super().__init__(name="EnhancedStrategyAgent", config=config)
        
        self._cache_dir = cache_dir
        self._strategy_history_file = os.path.join(cache_dir, "strategy_history.pkl")
        
        # Strategy components
        self._qlearning_model: Optional[AdvancedQNetwork] = None
        
        # Internal state
        self._last_features: Optional[Dict[str, Any]] = None
        self._last_forecast: Optional[Dict[str, Any]] = None
        self._strategy_history: List[Dict[str, Any]] = []
        
        # Strategy weights (adaptive)
        self._strategy_weights = {
            "qlearning": 0.4,
            "game_theory": 0.3,
            "heuristic": 0.3
        }
        
        # Risk management
        self._risk_tolerance = 0.7
        self._max_allocation = 0.8
        self._min_battery_reserve = 0.2
        
        os.makedirs(cache_dir, exist_ok=True)

    def _initialize(self) -> None:
        """Initialize strategy components."""
        logger.info(f"[{self.name}] Initializing strategy components")
        
        self._load_strategy_history()
        
        if HAS_QLEARNING:
            self._initialize_qlearning()
        else:
            logger.warning(f"[{self.name}] Q-learning not available, using fallback strategies")

    def _health_check(self) -> bool:
        """Check strategy agent health."""
        try:
            if not self._last_features and not self._last_forecast:
                return False
            
            if self._strategy_history:
                recent_performance = self._calculate_recent_performance()
                if recent_performance < 0.3:
                    return False
            
            return True
            
        except Exception as exc:
            logger.error(f"[{self.name}] Health check failed: {exc}")
            return False

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Process incoming data and generate strategic decisions."""
        try:
            message_type = self._identify_message_type(message)
            
            if message_type == "feature-vector":
                self._last_features = message
            elif message_type == "forecast":
                self._last_forecast = message
            
            if self._last_features is not None:
                strategy = self._generate_comprehensive_strategy()
                self._record_strategy(strategy)
                
                return {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "action": strategy["action"],
                    "strategy_analysis": strategy["analysis"],
                    "confidence": strategy["confidence"],
                    "risk_assessment": strategy["risk"],
                    "source": self.name,
                    "agent_id": self.agent_id,
                    "methods_used": strategy["methods_used"]
                }
            
            return None
            
        except Exception as exc:
            logger.exception(f"[{self.name}] Error processing message: {exc}")
            return None

    def _identify_message_type(self, message: Dict[str, Any]) -> str:
        """Identify the type of incoming message."""
        if "features" in message or "prices" in message:
            return "feature-vector"
        elif "forecast" in message:
            return "forecast"
        else:
            return "unknown"

    def _generate_comprehensive_strategy(self) -> Dict[str, Any]:
        """Generate strategy using multiple methods."""
        methods_used = []
        strategies = {}
        
        # Method 1: Q-learning strategy
        if HAS_QLEARNING and self._qlearning_model:
            try:
                qlearning_strategy = self._generate_qlearning_strategy()
                strategies["qlearning"] = qlearning_strategy
                methods_used.append("qlearning")
            except Exception as exc:
                logger.warning(f"[{self.name}] Q-learning strategy failed: {exc}")
        
        # Method 2: Heuristic strategy (always available)
        try:
            heuristic_strategy = self._generate_heuristic_strategy()
            strategies["heuristic"] = heuristic_strategy
            methods_used.append("heuristic")
        except Exception as exc:
            logger.warning(f"[{self.name}] Heuristic strategy failed: {exc}")
            # Fallback to safe default
            heuristic_strategy = self._generate_safe_default_strategy()
            strategies["heuristic"] = heuristic_strategy
            methods_used.append("safe_default")
        
        # Combine strategies intelligently
        combined_strategy = self._combine_strategies(strategies, methods_used)
        
        return combined_strategy

    def _generate_qlearning_strategy(self) -> Dict[str, Any]:
        """Generate strategy using Q-learning model."""
        if not self._qlearning_model or not self._last_features:
            raise Exception("Q-learning model or features not available")
        
        # Prepare state vector
        state_vector = self._prepare_qlearning_state()
        
        # Get Q-learning action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            q_values = self._qlearning_model(state_tensor)
            action_raw = q_values.numpy().flatten()
        
        # Convert to strategy
        energy_allocation = np.clip((action_raw[0] + 1) / 2, 0, self._max_allocation)
        hash_allocation = np.clip((action_raw[1] + 1) / 2, 0, self._max_allocation)
        battery_charge_rate = np.clip(action_raw[2], -1, 1)
        
        # Ensure allocations don't exceed 100%
        total_allocation = energy_allocation + hash_allocation
        if total_allocation > 1.0:
            scale_factor = 1.0 / total_allocation
            energy_allocation *= scale_factor
            hash_allocation *= scale_factor
        
        return {
            "energy_allocation": float(energy_allocation),
            "hash_allocation": float(hash_allocation),
            "battery_charge_rate": float(battery_charge_rate),
            "confidence": 0.8,
            "method": "qlearning"
        }

    def _generate_heuristic_strategy(self) -> Dict[str, Any]:
        """Generate strategy using heuristic rules."""
        if not self._last_features:
            raise Exception("No features available for heuristic strategy")
        
        features = self._last_features
        prices = features.get("prices", [])
        inventory = features.get("inventory", {})
        market_intelligence = features.get("market_intelligence", {})
        
        # Default allocations
        energy_allocation = 0.5
        hash_allocation = 0.3
        battery_charge_rate = 0.0
        
        if prices:
            latest_price = prices[-1]
            energy_price = latest_price.get("energy_price", 3.0)
            hash_price = latest_price.get("hash_price", 2.4)
            
            # Price-based allocation
            if energy_price > 4.0:  # High energy prices
                energy_allocation = min(0.7, self._max_allocation)
                hash_allocation = min(0.2, 1.0 - energy_allocation)
            elif hash_price > 3.0:  # High hash prices
                hash_allocation = min(0.6, self._max_allocation)
                energy_allocation = min(0.3, 1.0 - hash_allocation)
            
            # Market regime adjustment
            market_regime = market_intelligence.get("market_regime", {})
            if market_regime.get("volatility_regime") == "high":
                energy_allocation *= 0.8
                hash_allocation *= 0.8
        
        # Battery management
        if inventory:
            battery_soc = inventory.get("battery_soc", 0.5)
            utilization = inventory.get("utilization_rate", 50.0)
            
            if battery_soc < self._min_battery_reserve:
                battery_charge_rate = 0.5
            elif battery_soc > 0.8:
                battery_charge_rate = -0.3
            elif utilization > 80:
                battery_charge_rate = -0.2
        
        # Risk adjustment
        risk_factor = self._calculate_risk_factor()
        energy_allocation *= risk_factor
        hash_allocation *= risk_factor
        
        return {
            "energy_allocation": float(energy_allocation),
            "hash_allocation": float(hash_allocation),
            "battery_charge_rate": float(battery_charge_rate),
            "confidence": 0.6,
            "method": "heuristic"
        }

    def _generate_safe_default_strategy(self) -> Dict[str, Any]:
        """Generate a safe default strategy."""
        return {
            "energy_allocation": 0.4,
            "hash_allocation": 0.3,
            "battery_charge_rate": 0.0,
            "confidence": 0.3,
            "method": "safe_default"
        }

    def _combine_strategies(self, strategies: Dict[str, Dict[str, Any]], methods_used: List[str]) -> Dict[str, Any]:
        """Intelligently combine multiple strategies."""
        if not strategies:
            return {
                "action": self._generate_safe_default_strategy(),
                "analysis": {"error": "No strategies available"},
                "confidence": 0.1,
                "risk": {"level": "high", "reason": "no_strategy"},
                "methods_used": ["safe_default"]
            }
        
        # Weighted combination
        combined_action = {
            "energy_allocation": 0.0,
            "hash_allocation": 0.0,
            "battery_charge_rate": 0.0
        }
        
        total_weight = 0.0
        combined_confidence = 0.0
        
        for method in methods_used:
            if method in strategies:
                weight = self._strategy_weights.get(method, 0.3)
                strategy = strategies[method]
                
                combined_action["energy_allocation"] += strategy["energy_allocation"] * weight
                combined_action["hash_allocation"] += strategy["hash_allocation"] * weight
                combined_action["battery_charge_rate"] += strategy["battery_charge_rate"] * weight
                
                combined_confidence += strategy["confidence"] * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for key in combined_action:
                combined_action[key] /= total_weight
            combined_confidence /= total_weight
        
        # Risk assessment
        risk_assessment = self._assess_strategy_risk(combined_action)
        
        analysis = {
            "individual_strategies": strategies,
            "total_allocation": combined_action["energy_allocation"] + combined_action["hash_allocation"],
            "diversification": len(methods_used)
        }
        
        return {
            "action": combined_action,
            "analysis": analysis,
            "confidence": combined_confidence,
            "risk": risk_assessment,
            "methods_used": methods_used
        }

    def _prepare_qlearning_state(self) -> np.ndarray:
        """Prepare state vector for Q-learning model."""
        if not self._last_features:
            return np.zeros(23)
        
        features = self._last_features
        prices = features.get("prices", [])
        inventory = features.get("inventory", {})
        
        # Price features
        if prices:
            latest_price = prices[-1]
            price_features = [
                latest_price.get("energy_price", 3.0),
                latest_price.get("hash_price", 2.4),
                latest_price.get("token_price", 3.6)
            ]
        else:
            price_features = [3.0, 2.4, 3.6]
        
        # Inventory features
        inventory_features = [
            inventory.get("utilization_rate", 50.0) / 100.0,
            inventory.get("battery_soc", 0.5),
            inventory.get("power_available", 500.0) / 1000.0
        ]
        
        # Market features
        market_intelligence = features.get("market_intelligence", {})
        market_regime = market_intelligence.get("market_regime", {})
        
        market_features = [
            1.0 if market_regime.get("trend_regime") == "bullish" else 0.0,
            1.0 if market_regime.get("volatility_regime") == "high" else 0.0,
            market_regime.get("confidence", 0.5)
        ]
        
        # Technical indicators
        if len(prices) > 1:
            recent_prices = [p.get("energy_price", 3.0) for p in prices[-5:]]
            ma_short = np.mean(recent_prices[-3:]) if len(recent_prices) >= 3 else recent_prices[-1]
            ma_long = np.mean(recent_prices)
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if len(recent_prices) > 1 else 0.0
            volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0.0
        else:
            ma_short, ma_long, momentum, volatility = 3.0, 3.0, 0.0, 0.0
        
        technical_features = [ma_short, ma_long, momentum, volatility]
        
        # Time features
        now = pd.Timestamp.now()
        time_features = [now.hour / 24.0, now.dayofweek / 7.0]
        
        # Combine all features
        state_vector = np.array(price_features + inventory_features + market_features + 
                               technical_features + time_features)
        
        # Pad or truncate to expected size
        if len(state_vector) > 23:
            state_vector = state_vector[:23]
        elif len(state_vector) < 23:
            state_vector = np.pad(state_vector, (0, 23 - len(state_vector)), 'constant')
        
        return state_vector

    def _calculate_risk_factor(self) -> float:
        """Calculate risk adjustment factor."""
        risk_factor = self._risk_tolerance
        
        if self._last_features:
            market_intelligence = self._last_features.get("market_intelligence", {})
            alerts = market_intelligence.get("alerts", [])
            
            critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
            if critical_alerts:
                risk_factor *= 0.5
            
            market_regime = market_intelligence.get("market_regime", {})
            if market_regime.get("volatility_regime") == "high":
                risk_factor *= 0.8
        
        return risk_factor

    def _assess_strategy_risk(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of proposed strategy."""
        total_allocation = action["energy_allocation"] + action["hash_allocation"]
        battery_rate = abs(action["battery_charge_rate"])
        
        risk_level = "low"
        risk_factors = []
        
        if total_allocation > 0.9:
            risk_level = "high"
            risk_factors.append("high_total_allocation")
        elif total_allocation > 0.7:
            risk_level = "medium"
            risk_factors.append("moderate_allocation")
        
        if battery_rate > 0.8:
            risk_level = "high"
            risk_factors.append("aggressive_battery_usage")
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "total_allocation": total_allocation,
            "battery_usage": battery_rate
        }

    def _record_strategy(self, strategy: Dict[str, Any]) -> None:
        """Record strategy for performance tracking."""
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "strategy": strategy,
            "market_data": self._last_features,
            "forecast_data": self._last_forecast
        }
        
        self._strategy_history.append(record)
        
        if len(self._strategy_history) > 1000:
            self._strategy_history = self._strategy_history[-1000:]

    def _calculate_recent_performance(self) -> float:
        """Calculate recent strategy performance."""
        if len(self._strategy_history) < 5:
            return 0.6
        
        recent_strategies = self._strategy_history[-10:]
        
        # Diversity score
        methods_used = set()
        for record in recent_strategies:
            methods_used.update(record["strategy"].get("methods_used", []))
        diversity_score = len(methods_used) / 3.0
        
        # Risk management score
        risk_scores = []
        for record in recent_strategies:
            risk_level = record["strategy"]["risk"]["level"]
            if risk_level == "low":
                risk_scores.append(1.0)
            elif risk_level == "medium":
                risk_scores.append(0.7)
            else:
                risk_scores.append(0.3)
        
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0.5
        
        # Confidence score
        confidence_scores = [record["strategy"]["confidence"] for record in recent_strategies]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Combined performance
        performance = (diversity_score * 0.3 + avg_risk_score * 0.4 + avg_confidence * 0.3)
        return performance

    def _initialize_qlearning(self) -> None:
        """Initialize Q-learning components."""
        try:
            self._qlearning_model = AdvancedQNetwork(
                state_dim=23,
                action_dim=3,
                hidden_dim=128
            )
            
            # Try to load pre-trained model
            model_path = "best_qlearning_model_episode_99.pth"
            if os.path.exists(model_path):
                self._qlearning_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"[{self.name}] Loaded pre-trained Q-learning model")
            else:
                logger.info(f"[{self.name}] Using untrained Q-learning model")
            
            self._qlearning_model.eval()
            
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to initialize Q-learning: {exc}")
            self._qlearning_model = None

    def _load_strategy_history(self) -> None:
        """Load strategy history from disk."""
        try:
            if os.path.exists(self._strategy_history_file):
                with open(self._strategy_history_file, 'rb') as f:
                    self._strategy_history = pickle.load(f)
                logger.info(f"[{self.name}] Loaded {len(self._strategy_history)} strategy records")
                
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to load strategy history: {exc}")
            self._strategy_history = []

    def _persist_strategy_history(self) -> None:
        """Persist strategy history to disk."""
        try:
            with open(self._strategy_history_file, 'wb') as f:
                pickle.dump(self._strategy_history[-500:], f)
            logger.debug(f"[{self.name}] Strategy history persisted")
            
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to persist strategy history: {exc}")

    def _cleanup(self) -> None:
        """Cleanup resources and persist data."""
        logger.info(f"[{self.name}] Cleaning up and persisting strategy data")
        self._persist_strategy_history()
        
    def _recover(self) -> None:
        """Enhanced recovery for strategy agent."""
        logger.info(f"[{self.name}] Attempting strategy agent recovery")
        
        if self._calculate_recent_performance() < 0.3:
            self._strategy_weights = {
                "qlearning": 0.3,
                "game_theory": 0.3,
                "heuristic": 0.4
            }
            logger.info(f"[{self.name}] Reset to safe strategy weights")
        
        super()._recover()
