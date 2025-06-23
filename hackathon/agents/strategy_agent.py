from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

# Try to import Ray RLlib components
try:
    import ray
    from ray.rllib.algorithms.sac import SAC
    from ray.rllib.algorithms.sac.sac import SACConfig
    from ray.rllib.env.env_context import EnvContext
    import gymnasium as gym
    from gymnasium import spaces
    HAS_RLLIB = True
except ImportError:
    HAS_RLLIB = False
    # Create dummy classes to prevent import errors
    class gym:
        class Env: 
            def __init__(self, config=None): pass
            def reset(self, seed=None, options=None): return np.zeros(20), {}
            def step(self, action): return np.zeros(20), 0.0, True, False, {}
    
    class spaces:
        @staticmethod
        def Box(low, high, shape, dtype): return None
    
    class EnvContext(dict): pass
    
    logger = logging.getLogger(__name__)
    logger.warning("Ray RLlib not available, using simple strategy fallback")

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EnergyTradingEnv(gym.Env):
    """Custom Gymnasium environment for energy trading with battery constraints."""
    
    def __init__(self, config: EnvContext = None):
        super().__init__()
        
        if not HAS_RLLIB:
            return  # Skip initialization if dependencies missing
        
        # Action space: [energy_allocation, hash_allocation, battery_charge_rate]
        # Values between -1 and 1, normalized
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: price features + battery state + inventory
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        # Environment state
        self.battery_soc = 0.5  # Start at 50% state of charge
        self.max_battery_capacity = 1000.0  # kWh
        self.current_prices = np.zeros(3)  # [energy, hash, token]
        self.price_history = np.zeros((10, 3))  # Last 10 price observations
        self.inventory_utilization = 0.6
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        if not HAS_RLLIB:
            return np.zeros(20), {}
            
        super().reset(seed=seed)
        self.battery_soc = 0.5
        self.current_prices = np.random.uniform(1.0, 5.0, 3)
        self.price_history = np.random.uniform(1.0, 5.0, (10, 3))
        self.inventory_utilization = np.random.uniform(0.4, 0.8)
        self.step_count = 0
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        if not HAS_RLLIB:
            return np.zeros(20), 0.0, True, False, {}
            
        energy_alloc, hash_alloc, battery_rate = action
        
        # Normalize allocations to [0, 1]
        energy_alloc = (energy_alloc + 1) / 2
        hash_alloc = (hash_alloc + 1) / 2
        battery_rate = battery_rate  # Keep as [-1, 1] for charge/discharge
        
        # Update battery state
        old_soc = self.battery_soc
        self.battery_soc = np.clip(
            self.battery_soc + battery_rate * 0.1, 0.0, 1.0
        )
        
        # Calculate reward
        reward = self._calculate_reward(energy_alloc, hash_alloc, battery_rate, old_soc)
        
        # Update environment state
        self._update_state()
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward(self, energy_alloc, hash_alloc, battery_rate, old_soc):
        """Calculate reward based on revenue, risk, and battery efficiency."""
        # Revenue from allocations
        revenue = (
            energy_alloc * self.current_prices[0] * 100 +  # Energy trading
            hash_alloc * self.current_prices[1] * 50      # Hash trading
        )
        
        # Battery efficiency bonus/penalty
        battery_efficiency = 1.0 - abs(battery_rate) * 0.1  # Penalty for frequent changes
        
        # Risk penalty (volatility)
        price_volatility = np.std(self.price_history[-5:])
        risk_penalty = price_volatility * (energy_alloc + hash_alloc) * 0.1
        
        # Battery state penalty (avoid extremes)
        battery_penalty = 0.0
        if self.battery_soc < 0.2 or self.battery_soc > 0.8:
            battery_penalty = 10.0
        
        total_reward = revenue * battery_efficiency - risk_penalty - battery_penalty
        return total_reward
    
    def _update_state(self):
        """Update environment state (prices, inventory)."""
        # Simulate price changes with some persistence
        price_change = np.random.normal(0, 0.1, 3)
        self.current_prices = np.clip(
            self.current_prices + price_change, 0.5, 10.0
        )
        
        # Update price history
        self.price_history = np.roll(self.price_history, -1, axis=0)
        self.price_history[-1] = self.current_prices
        
        # Update inventory (simulate external changes)
        self.inventory_utilization += np.random.normal(0, 0.02)
        self.inventory_utilization = np.clip(self.inventory_utilization, 0.1, 0.9)
    
    def _get_observation(self):
        """Get current observation vector."""
        obs = np.concatenate([
            self.current_prices,  # 3 values
            self.price_history.flatten()[:10],  # Take first 10 values from flattened history
            [self.battery_soc],  # 1 value
            [self.inventory_utilization],  # 1 value
        ])
        # Pad or truncate to exactly 20 dimensions
        if len(obs) > 20:
            obs = obs[:20]
        elif len(obs) < 20:
            obs = np.pad(obs, (0, 20 - len(obs)), 'constant')
        return obs.astype(np.float32)


class StrategyAgent(BaseAgent):
    """RL-based strategy agent for optimal resource allocation."""

    subscribe_topics = ["feature-vector", "forecast"]
    publish_topic = "strategy-action"

    def __init__(self):
        super().__init__(name="StrategyAgent")
        self._use_rl = HAS_RLLIB
        self._last_features = None
        self._last_forecast = None
        
        if self._use_rl:
            self._initialize_rl()
        else:
            logger.warning("[%s] Using simple heuristic strategy (RLlib not available)", self.name)

    def _initialize_rl(self):
        """Initialize Ray RLlib SAC algorithm."""
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            
            # Configure SAC algorithm
            config = (
                SACConfig()
                .environment(EnergyTradingEnv)
                .framework("torch")
                .training(
                    lr=3e-4,
                    train_batch_size=256,
                    replay_buffer_config={"capacity": 10000},
                )
                .rollouts(num_rollout_workers=0)  # Single-threaded for simplicity
                .evaluation(evaluation_interval=None)  # Disable evaluation
            )
            
            # Load existing model or create new one
            model_path = "models/strategy_agent_sac"
            if os.path.exists(model_path):
                self._rl_agent = SAC.from_checkpoint(model_path)
                logger.info("[%s] Loaded existing RL model", self.name)
            else:
                self._rl_agent = config.build()
                logger.info("[%s] Created new RL model", self.name)
                
        except Exception as exc:
            logger.warning("[%s] Failed to initialize RL: %s. Using heuristics.", self.name, exc)
            self._use_rl = False

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Process feature vectors and forecasts to generate strategy actions."""
        try:
            # Store latest data
            if "features" in message:
                self._last_features = message
            elif "forecast" in message:
                self._last_forecast = message
            
            # Only generate strategy if we have both features and forecast
            if self._last_features is None or self._last_forecast is None:
                return None
                
            # Generate strategy action
            if self._use_rl:
                action = self._generate_rl_action()
            else:
                action = self._generate_heuristic_action()
            
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "action": action,
                "confidence": self._calculate_confidence(),
                "source": self.name,
            }
            
        except Exception as exc:
            logger.exception("[%s] Error generating strategy: %s", self.name, exc)
            return None

    def _generate_rl_action(self) -> Dict[str, float]:
        """Generate action using trained RL policy."""
        try:
            # Prepare observation from latest data
            observation = self._prepare_observation()
            
            # Get action from RL agent
            action = self._rl_agent.compute_single_action(observation)
            
            # Convert to interpretable action
            energy_alloc = (action[0] + 1) / 2  # Convert to [0, 1]
            hash_alloc = (action[1] + 1) / 2
            battery_rate = action[2]  # Keep as [-1, 1]
            
            return {
                "energy_allocation": float(energy_alloc),
                "hash_allocation": float(hash_alloc), 
                "battery_charge_rate": float(battery_rate),
                "method": "rl_sac"
            }
            
        except Exception as exc:
            logger.warning("[%s] RL action failed: %s. Using heuristic.", self.name, exc)
            return self._generate_heuristic_action()

    def _generate_heuristic_action(self) -> Dict[str, float]:
        """Generate action using intelligent heuristics and market analysis."""
        try:
            # Get current prices and historical context
            prices = self._last_features.get("prices", [])
            inventory = self._last_features.get("inventory", {})
            forecast = self._last_forecast.get("forecast", []) if self._last_forecast else []
            
            if not prices:
                return {"energy_allocation": 0.5, "hash_allocation": 0.5, "battery_charge_rate": 0.0, "method": "heuristic_default"}
            
            # Advanced market analysis
            market_analysis = self._analyze_market_conditions(prices, forecast)
            system_analysis = self._analyze_system_state(inventory)
            
            # Strategic decision making based on analysis
            strategy = self._make_strategic_decision(market_analysis, system_analysis)
            
            return {
                "energy_allocation": float(strategy["energy_alloc"]),
                "hash_allocation": float(strategy["hash_alloc"]),
                "battery_charge_rate": float(strategy["battery_rate"]),
                "method": "intelligent_heuristic",
                "market_regime": market_analysis["regime"],
                "confidence": strategy["confidence"],
                "reasoning": strategy["reasoning"]
            }
            
        except Exception as exc:
            logger.exception("[%s] Heuristic action failed: %s", self.name, exc)
            return {"energy_allocation": 0.5, "hash_allocation": 0.5, "battery_charge_rate": 0.0, "method": "fallback"}

    def _analyze_market_conditions(self, prices: list, forecast: list) -> Dict[str, Any]:
        """Analyze current market conditions and trends."""
        if len(prices) < 5:
            return {"regime": "unknown", "volatility": "medium", "trend": "neutral", "signals": []}
        
        # Extract recent price data
        recent_prices = [p.get("energy_price", 3.0) for p in prices[-24:]]  # Last 24 hours
        current_price = recent_prices[-1]
        
        # Price momentum analysis
        short_ma = sum(recent_prices[-3:]) / min(3, len(recent_prices[-3:]))
        long_ma = sum(recent_prices[-12:]) / min(12, len(recent_prices[-12:]))
        
        # Volatility analysis
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
        volatility = sum(price_changes) / len(price_changes) if price_changes else 0
        
        # Market regime identification
        if short_ma > long_ma * 1.05:
            regime = "bullish"
        elif short_ma < long_ma * 0.95:
            regime = "bearish"
        else:
            regime = "sideways"
        
        # Incorporate forecast information
        forecast_signal = "neutral"
        if forecast:
            next_prices = [f.get("predicted_price", current_price) for f in forecast[:3]]
            if len(next_prices) > 0:
                forecast_trend = (next_prices[-1] - current_price) / current_price
                if forecast_trend > 0.05:
                    forecast_signal = "bullish"
                elif forecast_trend < -0.05:
                    forecast_signal = "bearish"
        
        # Trading signals
        signals = []
        if regime == "bullish" and forecast_signal in ["bullish", "neutral"]:
            signals.append("favor_energy_trading")
        elif regime == "bearish":
            signals.append("reduce_energy_exposure")
        
        if volatility > 1.0:  # High volatility
            signals.append("increase_battery_activity")
            signals.append("hedge_positions")
        
        return {
            "regime": regime,
            "volatility": "high" if volatility > 1.0 else "medium" if volatility > 0.5 else "low",
            "trend": forecast_signal,
            "price_momentum": (short_ma - long_ma) / long_ma if long_ma > 0 else 0,
            "current_price": current_price,
            "signals": signals
        }

    def _analyze_system_state(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state and constraints."""
        utilization = inventory.get("utilization_rate", 60.0) / 100.0
        battery_soc = inventory.get("battery_soc", 0.6)
        power_available = inventory.get("power_available", 500)
        
        # System health assessment
        health_score = 1.0
        constraints = []
        
        if utilization > 0.9:
            health_score -= 0.3
            constraints.append("high_utilization")
        elif utilization < 0.3:
            constraints.append("low_utilization")
        
        if battery_soc < 0.2:
            health_score -= 0.4
            constraints.append("low_battery")
        elif battery_soc > 0.9:
            constraints.append("high_battery")
        
        if power_available < 100:
            health_score -= 0.3
            constraints.append("limited_power")
        
        return {
            "health_score": max(0.0, health_score),
            "utilization": utilization,
            "battery_soc": battery_soc,
            "power_available": power_available,
            "constraints": constraints,
            "capacity_stress": utilization > 0.8
        }

    def _make_strategic_decision(self, market: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
        """Make strategic allocation decision based on market and system analysis."""
        
        # Base allocations
        energy_alloc = 0.5
        hash_alloc = 0.5
        battery_rate = 0.0
        confidence = 0.5
        reasoning = []
        
        # Market-driven adjustments
        if "favor_energy_trading" in market["signals"]:
            energy_alloc = 0.7
            hash_alloc = 0.3
            confidence += 0.2
            reasoning.append("Bullish market favors energy trading")
        elif "reduce_energy_exposure" in market["signals"]:
            energy_alloc = 0.3
            hash_alloc = 0.7
            confidence += 0.15
            reasoning.append("Bearish market, shift to hash mining")
        
        # System constraint adjustments
        if "high_utilization" in system["constraints"]:
            # Reduce allocations when system is stressed
            energy_alloc *= 0.8
            hash_alloc *= 0.8
            reasoning.append("High utilization, reducing load")
        
        if "low_battery" in system["constraints"]:
            battery_rate = 0.6  # Aggressive charging
            confidence += 0.1
            reasoning.append("Low battery, prioritizing charging")
        elif "high_battery" in system["constraints"]:
            if market["regime"] == "bullish":
                battery_rate = -0.4  # Discharge to trade
                reasoning.append("High battery + bullish market, discharging to trade")
        
        # Volatility-based battery strategy
        if market["volatility"] == "high":
            if abs(battery_rate) < 0.3:  # No strong battery signal yet
                battery_rate = 0.3 if system["battery_soc"] < 0.7 else -0.2
                reasoning.append("High volatility, active battery management")
        
        # Risk management
        if system["health_score"] < 0.5:
            # Conservative approach when system health is poor
            energy_alloc = min(energy_alloc, 0.4)
            hash_alloc = min(hash_alloc, 0.4)
            battery_rate *= 0.5
            confidence *= 0.7
            reasoning.append("Poor system health, conservative approach")
        
        # Ensure allocations sum to reasonable values
        total_alloc = energy_alloc + hash_alloc
        if total_alloc > 1.0:
            energy_alloc /= total_alloc
            hash_alloc /= total_alloc
        
        # Confidence adjustment
        confidence = min(0.95, max(0.1, confidence))
        
        return {
            "energy_alloc": energy_alloc,
            "hash_alloc": hash_alloc,
            "battery_rate": battery_rate,
            "confidence": confidence,
            "reasoning": "; ".join(reasoning) if reasoning else "Standard balanced allocation"
        }

    def _prepare_observation(self) -> np.ndarray:
        """Prepare observation vector for RL agent."""
        try:
            # Extract price data
            prices = self._last_features.get("prices", [])
            if len(prices) >= 10:
                price_history = np.array([[p.get("energy_price", 3.0), p.get("hash_price", 3.0), p.get("token_price", 2.0)] for p in prices[-10:]])
            else:
                # Pad with default values
                price_history = np.full((10, 3), [3.0, 3.0, 2.0])
            
            current_prices = price_history[-1] if len(price_history) > 0 else np.array([3.0, 3.0, 2.0])
            
            # Battery state (mock for now)
            battery_soc = 0.6  # Could be extracted from system state
            
            # Inventory utilization
            inventory = self._last_features.get("inventory", {})
            inventory_util = inventory.get("utilization_rate", 60.0) / 100.0
            
            # Build observation
            obs = np.concatenate([
                current_prices,
                price_history[:-1].flatten()[:10],  # Last 10 price features
                [battery_soc],
                [inventory_util],
            ])
            
            # Ensure exactly 20 dimensions
            if len(obs) > 20:
                obs = obs[:20]
            elif len(obs) < 20:
                obs = np.pad(obs, (0, 20 - len(obs)), 'constant', constant_values=0.0)
                
            return obs.astype(np.float32)
            
        except Exception as exc:
            logger.warning("[%s] Failed to prepare observation: %s", self.name, exc)
            return np.zeros(20, dtype=np.float32)

    def _calculate_confidence(self) -> float:
        """Calculate confidence in the strategy decision."""
        try:
            # Base confidence on data quality and recency
            features_age = 0.9  # Mock: could check timestamp
            forecast_quality = 0.8  # Mock: could check forecast uncertainty
            
            return float(features_age * forecast_quality)
        except Exception:
            return 0.5


if __name__ == "__main__":
    StrategyAgent().start() 