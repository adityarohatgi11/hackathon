#!/usr/bin/env python3
"""
Advanced Q-Learning Implementation for GridPilot-GT
===================================================

This module provides sophisticated reinforcement learning capabilities including:
- Deep Q-Networks (DQN) with experience replay
- Double DQN and Dueling DQN architectures
- Prioritized Experience Replay
- Multi-step learning
- Advanced state representation and reward shaping
- Real-time adaptation and continuous learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque, namedtuple
import random
from datetime import datetime, timedelta
import json
import pickle

# Initialize logger first
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    
    # Test PyTorch functionality
    try:
        test_tensor = torch.tensor([1.0])
        TORCH_AVAILABLE = True
        logger.info("PyTorch backend initialized successfully")
    except Exception as e:
        TORCH_AVAILABLE = False
        logger.warning(f"PyTorch available but not functional: {e}")
        
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using numpy-based Q-learning")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class AdvancedStateEncoder:
    """Advanced state encoder for energy trading environments."""
    
    def __init__(self, lookback_hours: int = 24, include_technical_indicators: bool = True):
        """
        Initialize state encoder.
        
        Args:
            lookback_hours: Number of historical hours to include
            include_technical_indicators: Whether to compute technical indicators
        """
        self.lookback_hours = lookback_hours
        self.include_technical_indicators = include_technical_indicators
        self.price_history = deque(maxlen=lookback_hours)
        self.volume_history = deque(maxlen=lookback_hours)
        self.demand_history = deque(maxlen=lookback_hours)
        
        # Technical indicator parameters
        self.sma_periods = [5, 10, 20]
        self.ema_alpha = 0.2
        self.rsi_period = 14
        
        logger.info(f"Advanced state encoder initialized: {lookback_hours}h lookback")
    
    def encode_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Encode market state into feature vector.
        
        Args:
            market_data: Dictionary with current market information
            
        Returns:
            Encoded state vector
        """
        features = []
        
        # Current market conditions
        current_price = market_data.get('price', 50.0)
        current_soc = market_data.get('soc', 0.5)
        current_demand = market_data.get('demand', 0.5)
        current_volatility = market_data.get('volatility', 0.1)
        
        # Update histories
        self.price_history.append(current_price)
        self.demand_history.append(current_demand)
        
        # Basic features
        features.extend([
            current_price / 100.0,  # Normalized price
            current_soc,
            current_demand,
            current_volatility,
            market_data.get('hour_of_day', 12) / 24.0,  # Time of day
            market_data.get('day_of_week', 3) / 7.0,    # Day of week
        ])
        
        # Price statistics
        if len(self.price_history) > 1:
            prices = np.array(self.price_history)
            features.extend([
                (current_price - np.mean(prices)) / (np.std(prices) + 1e-8),  # Price z-score
                (current_price - prices[-2]) / (prices[-2] + 1e-8),           # Price return
                np.std(prices) / (np.mean(prices) + 1e-8),                    # Coefficient of variation
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Technical indicators
        if self.include_technical_indicators and len(self.price_history) >= max(self.sma_periods):
            prices = np.array(self.price_history)
            
            # Simple Moving Averages
            for period in self.sma_periods:
                if len(prices) >= period:
                    sma = np.mean(prices[-period:])
                    features.append((current_price - sma) / (sma + 1e-8))
                else:
                    features.append(0.0)
            
            # Exponential Moving Average
            ema = self._calculate_ema(prices)
            features.append((current_price - ema) / (ema + 1e-8))
            
            # RSI
            rsi = self._calculate_rsi(prices)
            features.append(rsi / 100.0)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            features.extend([
                (current_price - bb_upper) / (bb_upper + 1e-8),
                (current_price - bb_lower) / (bb_lower + 1e-8)
            ])
        else:
            # Pad with zeros if not enough data
            features.extend([0.0] * 7)
        
        # System state features
        features.extend([
            market_data.get('power_available', 1000000) / 1000000.0,  # Normalized power
            market_data.get('battery_capacity', 100000) / 100000.0,   # Normalized battery
            market_data.get('cooling_efficiency', 3.0) / 5.0,         # Normalized COP
            market_data.get('grid_stability', 0.95),                  # Grid stability
        ])
        
        # Market regime features
        features.extend([
            market_data.get('market_trend', 0.0),        # Bull/bear indicator
            market_data.get('volatility_regime', 0.0),   # High/low vol regime
            market_data.get('liquidity_score', 0.5),     # Market liquidity
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_ema(self, prices: np.ndarray) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) == 0:
            return 0.0
        
        ema = prices[0]
        for price in prices[1:]:
            ema = self.ema_alpha * price + (1 - self.ema_alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < 2:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < self.rsi_period:
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
        else:
            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, lower_band
    
    def get_state_size(self) -> int:
        """Get the size of the encoded state vector."""
        # Count features: basic(6) + price_stats(3) + technical(7) + system(4) + market(3)
        return 23


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for DQN."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        logger.info(f"Prioritized replay buffer initialized: {capacity} capacity")
    
    def add(self, experience: Experience):
        """Add experience to buffer."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority
    
    def __len__(self):
        return len(self.buffer)


if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """Dueling Deep Q-Network architecture."""
        
        def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
            """
            Initialize Dueling DQN.
            
            Args:
                state_size: Input state dimension
                action_size: Number of actions
                hidden_sizes: Hidden layer sizes
            """
            super(DuelingDQN, self).__init__()
            
            self.state_size = state_size
            self.action_size = action_size
            
            # Feature extraction layers
            layers = []
            prev_size = state_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_size = hidden_size
            
            self.feature_layers = nn.Sequential(*layers)
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_size, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            """Initialize network weights."""
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        def forward(self, state):
            """Forward pass through the network."""
            features = self.feature_layers(state)
            
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            return q_values


class AdvancedDQNAgent:
    """Advanced Deep Q-Network Agent with modern RL techniques."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 memory_size: int = 100000,
                 target_update_freq: int = 1000,
                 double_dqn: bool = True,
                 n_step: int = 3):
        """
        Initialize Advanced DQN Agent.
        
        Args:
            state_size: State space dimension
            action_size: Action space dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
            batch_size: Training batch size
            memory_size: Replay buffer size
            target_update_freq: Target network update frequency
            double_dqn: Whether to use Double DQN
            n_step: N-step learning horizon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.n_step = n_step
        
        # Networks
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DuelingDQN(state_size, action_size).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Fallback to numpy-based Q-table
            self.q_table = np.random.normal(0, 0.1, (1000, action_size))  # Simplified
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        # Action mapping for energy trading
        self.action_meanings = [
            "conservative",      # Low risk, low allocation
            "moderate_low",      # Below average allocation
            "balanced",          # Average allocation
            "moderate_high",     # Above average allocation
            "aggressive"         # High risk, high allocation
        ]
        
        logger.info(f"Advanced DQN Agent initialized: {state_size}D state, {action_size} actions")
        logger.info(f"Using {'PyTorch' if TORCH_AVAILABLE else 'NumPy'} backend")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        if TORCH_AVAILABLE:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        else:
            # Simplified state mapping for numpy version
            state_hash = hash(str(state)) % len(self.q_table)
            return np.argmax(self.q_table[state_hash])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done, 1.0)
        
        # Add to n-step buffer
        self.n_step_buffer.append(experience)
        
        # If buffer is full, compute n-step return and add to memory
        if len(self.n_step_buffer) == self.n_step:
            n_step_experience = self._compute_n_step_experience()
            self.memory.add(n_step_experience)
    
    def _compute_n_step_experience(self) -> Experience:
        """Compute n-step experience from buffer."""
        first_exp = self.n_step_buffer[0]
        last_exp = self.n_step_buffer[-1]
        
        # Compute n-step return
        n_step_return = 0
        for i, exp in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * exp.reward
        
        return Experience(
            first_exp.state,
            first_exp.action,
            n_step_return,
            last_exp.next_state,
            last_exp.done,
            1.0
        )
    
    def replay(self):
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size or not TORCH_AVAILABLE:
            return
        
        # Sample batch
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        if not experiences:
            return
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * next_q_values * (~dones).unsqueeze(1)
        
        # Compute loss with importance sampling weights
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        priorities = td_errors.abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(indices, priorities)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store metrics
        self.losses.append(loss.item())
    
    def get_trading_strategy(self, state: np.ndarray) -> Dict[str, float]:
        """
        Convert action to trading strategy parameters.
        
        Args:
            state: Current market state
            
        Returns:
            Trading strategy parameters
        """
        action = self.act(state, training=False)
        
        # Map action to strategy
        strategies = {
            0: {"allocation_pct": 0.2, "risk_tolerance": 0.1, "aggressiveness": 0.2},
            1: {"allocation_pct": 0.4, "risk_tolerance": 0.2, "aggressiveness": 0.4},
            2: {"allocation_pct": 0.6, "risk_tolerance": 0.3, "aggressiveness": 0.6},
            3: {"allocation_pct": 0.8, "risk_tolerance": 0.4, "aggressiveness": 0.8},
            4: {"allocation_pct": 1.0, "risk_tolerance": 0.5, "aggressiveness": 1.0},
        }
        
        strategy = strategies.get(action, strategies[2])
        strategy["action_name"] = self.action_meanings[action]
        strategy["confidence"] = 1.0 - self.epsilon  # Higher confidence as exploration decreases
        
        return strategy
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if TORCH_AVAILABLE:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_rewards': self.episode_rewards,
                'losses': self.losses
            }, filepath)
        else:
            # Save numpy version
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'episode_rewards': self.episode_rewards
                }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            self.episode_rewards = checkpoint['episode_rewards']
            self.losses = checkpoint['losses']
        else:
            # Load numpy version
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
                self.episode_rewards = data['episode_rewards']
        
        logger.info(f"Model loaded from {filepath}")


class EnergyTradingRewardFunction:
    """Sophisticated reward function for energy trading."""
    
    def __init__(self, 
                 profit_weight: float = 1.0,
                 risk_penalty: float = 0.3,
                 efficiency_bonus: float = 0.2,
                 stability_bonus: float = 0.1):
        """
        Initialize reward function.
        
        Args:
            profit_weight: Weight for profit component
            risk_penalty: Penalty for high risk actions
            efficiency_bonus: Bonus for energy efficiency
            stability_bonus: Bonus for grid stability
        """
        self.profit_weight = profit_weight
        self.risk_penalty = risk_penalty
        self.efficiency_bonus = efficiency_bonus
        self.stability_bonus = stability_bonus
    
    def calculate_reward(self, 
                        state: Dict[str, Any], 
                        action: int, 
                        next_state: Dict[str, Any],
                        allocation_result: Dict[str, Any]) -> float:
        """
        Calculate sophisticated reward for energy trading action.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            allocation_result: Result of allocation strategy
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Profit component
        revenue = allocation_result.get('revenue', 0)
        cost = allocation_result.get('cost', 0)
        profit = revenue - cost
        reward += self.profit_weight * profit / 1000.0  # Normalize
        
        # Risk penalty
        volatility = next_state.get('volatility', 0.1)
        allocation_pct = allocation_result.get('allocation_pct', 0.5)
        risk_score = volatility * allocation_pct
        reward -= self.risk_penalty * risk_score
        
        # Efficiency bonus
        power_efficiency = allocation_result.get('efficiency', 0.8)
        reward += self.efficiency_bonus * power_efficiency
        
        # Stability bonus
        grid_stability = next_state.get('grid_stability', 0.95)
        reward += self.stability_bonus * grid_stability
        
        # Battery management bonus/penalty
        soc_current = state.get('soc', 0.5)
        soc_next = next_state.get('soc', 0.5)
        
        # Penalty for extreme SOC levels
        if soc_next < 0.1 or soc_next > 0.9:
            reward -= 0.5
        
        # Bonus for maintaining optimal SOC range
        if 0.3 <= soc_next <= 0.7:
            reward += 0.1
        
        # Market timing bonus
        price_current = state.get('price', 50.0)
        price_next = next_state.get('price', 50.0)
        
        # Bonus for buying low, selling high
        if action >= 3 and price_current < price_next:  # Aggressive when price rising
            reward += 0.2
        elif action <= 1 and price_current > price_next:  # Conservative when price falling
            reward += 0.2
        
        return float(reward)


# Factory function
class SimpleNumpyQAgent:
    """Simple numpy-based Q-learning agent as fallback."""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        
        # Simple Q-table approximation using state hash
        self.q_values = {}
        self.action_counts = np.zeros(action_size)
        
        logger.info(f"Simple numpy Q-agent initialized: {state_size}D state, {action_size} actions")
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Create hash key for state."""
        # Discretize continuous state for Q-table
        discretized = np.round(state * 10).astype(int)
        return str(discretized.tolist())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state_key = self._hash_state(state)
            if state_key in self.q_values:
                action = np.argmax(self.q_values[state_key])
            else:
                action = np.random.randint(self.action_size)
        
        self.action_counts[action] += 1
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Update Q-values using Q-learning update rule."""
        state_key = self._hash_state(state)
        next_state_key = self._hash_state(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_values[state_key][action]
        next_max_q = np.max(self.q_values[next_state_key]) if not done else 0
        target_q = reward + self.gamma * next_max_q
        
        self.q_values[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_trading_strategy(self, state: np.ndarray) -> Dict[str, float]:
        """Get trading strategy based on current state."""
        action = self.act(state, training=False)
        
        # Map actions to trading strategies
        strategies = {
            0: {"conservative": 0.8, "moderate": 0.2, "aggressive": 0.0},
            1: {"conservative": 0.6, "moderate": 0.4, "aggressive": 0.0},
            2: {"conservative": 0.3, "moderate": 0.5, "aggressive": 0.2},
            3: {"conservative": 0.1, "moderate": 0.4, "aggressive": 0.5},
            4: {"conservative": 0.0, "moderate": 0.2, "aggressive": 0.8}
        }
        
        return strategies.get(action, strategies[2])  # Default to moderate
    
    def save_model(self, filepath: str):
        """Save Q-values to file."""
        data = {
            'q_values': self.q_values,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_model(self, filepath: str):
        """Load Q-values from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.q_values = data.get('q_values', {})
            self.epsilon = data.get('epsilon', self.epsilon)
            self.action_counts = np.array(data.get('action_counts', [0] * self.action_size))
        except FileNotFoundError:
            logger.warning(f"Model file {filepath} not found, using fresh model")


def create_advanced_qlearning_system(state_lookback: int = 24,
                                   action_size: int = 5,
                                   learning_rate: float = 0.001) -> Tuple[Any, AdvancedStateEncoder, EnergyTradingRewardFunction]:
    """
    Create a complete advanced Q-learning system for energy trading.
    
    Args:
        state_lookback: Hours of historical data to include in state
        action_size: Number of possible actions
        learning_rate: Learning rate for neural network
        
    Returns:
        Tuple of (agent, state_encoder, reward_function)
    """
    # Initialize components
    state_encoder = AdvancedStateEncoder(lookback_hours=state_lookback)
    state_size = state_encoder.get_state_size()
    
    # Choose agent based on PyTorch availability
    if TORCH_AVAILABLE:
        try:
            agent = AdvancedDQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                double_dqn=True,
                n_step=3
            )
            logger.info("Using PyTorch backend")
        except Exception as e:
            logger.warning(f"PyTorch agent failed to initialize: {e}")
            logger.info("Falling back to numpy backend")
            agent = SimpleNumpyQAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate
            )
    else:
        logger.info("Using NumPy backend")
        agent = SimpleNumpyQAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate
        )
    
    reward_function = EnergyTradingRewardFunction()
    
    logger.info("Advanced Q-learning system created successfully")
    
    return agent, state_encoder, reward_function


if __name__ == "__main__":
    # Example usage
    print("Advanced Q-Learning for GridPilot-GT")
    print("===================================")
    
    # Create system
    agent, encoder, reward_fn = create_advanced_qlearning_system()
    
    print(f"Created system with {agent.state_size}D state space and {agent.action_size} actions")
    print(f"Using {'PyTorch' if TORCH_AVAILABLE else 'NumPy'} backend")
    
    # Quick test
    test_state = encoder.encode_state({
        'price': 55.0,
        'soc': 0.6,
        'demand': 0.4,
        'volatility': 0.12,
        'hour_of_day': 14,
        'day_of_week': 2
    })
    
    strategy = agent.get_trading_strategy(test_state)
    print(f"Sample trading strategy: {strategy}") 