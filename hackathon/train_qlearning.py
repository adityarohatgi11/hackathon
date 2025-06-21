#!/usr/bin/env python3
"""
Q-Learning Training Script for GridPilot-GT
==========================================

This script trains the advanced Q-learning agent using historical market data
and simulated energy trading environments.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, Any
import argparse

from forecasting.advanced_qlearning import (
    create_advanced_qlearning_system,
    AdvancedDQNAgent,
    AdvancedStateEncoder,
    EnergyTradingRewardFunction
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QLearningTrainer:
    """Advanced Q-learning trainer for energy trading."""
    
    def __init__(self, episodes: int = 500, max_steps_per_episode: int = 24):
        """
        Initialize trainer.
        
        Args:
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
        """
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Initialize Q-learning system
        self.agent, self.state_encoder, self.reward_function = create_advanced_qlearning_system(
            state_lookback=24,
            action_size=5,
            learning_rate=0.001
        )
        
        # Training metrics
        self.episode_rewards = []
        self.training_start_time = None
        
        logger.info(f"Q-learning trainer initialized: {episodes} episodes")
    
    def generate_synthetic_data(self, n_days: int = 30) -> pd.DataFrame:
        """Generate synthetic market data for training."""
        logger.info(f"Generating {n_days} days of synthetic market data...")
        
        # Time series
        timestamps = pd.date_range(start='2024-01-01', periods=n_days * 24, freq='H')
        
        # Price simulation with realistic patterns
        np.random.seed(42)
        base_price = 50.0
        
        # Daily pattern (higher during day, lower at night)
        hourly_pattern = 10 * np.sin(2 * np.pi * np.arange(24) / 24) + 5
        daily_pattern = np.tile(hourly_pattern, n_days)
        
        # Random walk component
        random_walk = np.cumsum(np.random.normal(0, 2, n_days * 24))
        
        # Volatility clustering
        volatility = np.abs(np.random.normal(0.1, 0.05, n_days * 24))
        volatility = np.maximum(volatility, 0.01)  # Minimum volatility
        
        # Combine all components
        prices = base_price + daily_pattern + random_walk
        prices = np.maximum(prices, 10.0)  # Minimum price
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'price_volatility_24h': volatility,
        })
        
        logger.info(f"Generated {len(data)} data points")
        return data
    
    def allocation_simulator(self, market_data: Dict[str, Any], action: int) -> Dict[str, Any]:
        """Simulate allocation results for training."""
        # Map action to allocation percentage
        allocation_pcts = [0.2, 0.4, 0.6, 0.8, 1.0]
        allocation_pct = allocation_pcts[min(action, len(allocation_pcts) - 1)]
        
        price = market_data['price']
        power_available = market_data['power_available']
        
        # Calculate allocation
        allocation_kw = allocation_pct * power_available
        
        # Revenue model (simplified)
        base_revenue_rate = 0.001  # $/kW
        price_factor = price / 50.0  # Normalize around $50
        revenue = allocation_kw * base_revenue_rate * price_factor
        
        # Cost model
        base_cost_rate = 0.0005  # $/kW
        efficiency_factor = 0.8 + 0.1 * (5 - action)  # More efficient for conservative actions
        cost = allocation_kw * base_cost_rate / efficiency_factor
        
        return {
            'allocation_pct': allocation_pct,
            'allocation_kw': allocation_kw,
            'revenue': revenue,
            'cost': cost,
            'efficiency': efficiency_factor,
            'profit': revenue - cost
        }
    
    def extract_market_data(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """Extract market data from DataFrame row."""
        return {
            'price': float(row['price']),
            'soc': 0.5 + 0.3 * np.sin(index * 0.1),  # Simulated battery SOC
            'demand': 0.5 + 0.2 * np.cos(index * 0.05),  # Simulated demand
            'volatility': float(row['price_volatility_24h']),
            'hour_of_day': index % 24,
            'day_of_week': (index // 24) % 7,
            'power_available': 1000000,  # 1 MW
            'battery_capacity': 100000,
            'cooling_efficiency': 3.0,
            'grid_stability': 0.95,
            'market_trend': np.random.normal(0, 0.1),
            'volatility_regime': 1 if float(row['price_volatility_24h']) > 0.15 else 0,
            'liquidity_score': 0.8
        }
    
    def train(self, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train the Q-learning agent.
        
        Args:
            market_data: Historical market data (if None, generates synthetic data)
            
        Returns:
            Training results and metrics
        """
        self.training_start_time = datetime.now()
        logger.info("Starting Q-learning training...")
        
        # Use provided data or generate synthetic data
        if market_data is None:
            market_data = self.generate_synthetic_data(n_days=30)
        
        logger.info(f"Training on {len(market_data)} data points")
        
        best_reward = float('-inf')
        training_history = []
        
        for episode in range(self.episodes):
            episode_reward = 0
            episode_steps = 0
            
            # Random starting point in data
            start_idx = np.random.randint(0, len(market_data) - self.max_steps_per_episode)
            
            # Initial state
            initial_market_data = self.extract_market_data(market_data.iloc[start_idx], start_idx)
            state = self.state_encoder.encode_state(initial_market_data)
            
            for step in range(self.max_steps_per_episode):
                if start_idx + step + 1 >= len(market_data):
                    break
                
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Simulate action
                current_market_data = self.extract_market_data(
                    market_data.iloc[start_idx + step], start_idx + step
                )
                next_market_data = self.extract_market_data(
                    market_data.iloc[start_idx + step + 1], start_idx + step + 1
                )
                
                allocation_result = self.allocation_simulator(current_market_data, action)
                
                # Calculate reward
                reward = self.reward_function.calculate_reward(
                    current_market_data, action, next_market_data, allocation_result
                )
                
                # Next state
                next_state = self.state_encoder.encode_state(next_market_data)
                
                # Store experience
                done = (step == self.max_steps_per_episode - 1)
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                self.agent.replay()
                
                # Update for next iteration
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            
            # Update best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                if hasattr(self.agent, 'save_model'):
                    self.agent.save_model(f"best_qlearning_model_episode_{episode}.pth")
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                logger.info(f"Episode {episode}/{self.episodes}, Avg Reward: {avg_reward:.3f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Store training history
            training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'epsilon': self.agent.epsilon,
                'avg_loss': np.mean(self.agent.losses[-10:]) if self.agent.losses else 0
            })
        
        training_time = datetime.now() - self.training_start_time
        
        # Final model save
        if hasattr(self.agent, 'save_model'):
            self.agent.save_model("final_qlearning_model.pth")
        
        results = {
            'training_completed': True,
            'total_episodes': self.episodes,
            'training_time': training_time.total_seconds(),
            'best_reward': best_reward,
            'final_epsilon': self.agent.epsilon,
            'average_reward': np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'training_history': training_history,
        }
        
        logger.info(f"Q-learning training completed in {training_time}")
        logger.info(f"Best reward: {best_reward:.3f}, Average reward: {results['average_reward']:.3f}")
        
        return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Q-Learning Agent for GridPilot-GT")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    
    args = parser.parse_args()
    
    print("🧠 Q-Learning Training for GridPilot-GT")
    print("=====================================")
    print(f"Episodes: {args.episodes}")
    
    # Initialize trainer
    trainer = QLearningTrainer(episodes=args.episodes)
    
    # Train agent
    print("🚀 Starting training...")
    training_results = trainer.train()
    
    # Save results
    results_file = f"qlearning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_file}")
    print("🎉 Q-Learning training completed successfully!")
    
    # Summary
    print("\n📊 TRAINING SUMMARY:")
    print(f"  • Total Episodes: {training_results['total_episodes']}")
    print(f"  • Training Time: {training_results['training_time']:.1f}s")
    print(f"  • Best Reward: {training_results['best_reward']:.3f}")
    print(f"  • Average Reward: {training_results['average_reward']:.3f}")
    print(f"  • Final Epsilon: {training_results['final_epsilon']:.3f}")


if __name__ == "__main__":
    main() 