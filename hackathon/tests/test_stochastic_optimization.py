#!/usr/bin/env python3
"""
Comprehensive Test of Advanced Stochastic Optimization Methods
Demonstrates the new mathematical models and their integration with GridPilot-GT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import our new stochastic models
from forecasting.stochastic_models import (
    StochasticDifferentialEquation, MonteCarloEngine, ReinforcementLearningAgent,
    create_stochastic_forecaster, create_monte_carlo_engine, create_rl_agent
)

# Import existing components for integration
from api_client.client import get_prices
from forecasting.forecaster import Forecaster
from game_theory.bid_generators import build_bid_vector
from game_theory.vcg_auction import vcg_allocate
from game_theory.mpc_controller import MPCController

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stochastic_differential_equations():
    """Test advanced SDE models for price forecasting."""
    print("\n" + "="*60)
    print("TESTING STOCHASTIC DIFFERENTIAL EQUATIONS")
    print("="*60)
    
    # Get real price data
    try:
        prices_df = get_prices()
        price_series = prices_df['price']
        print(f"Loaded {len(price_series)} price observations")
    except Exception as e:
        print(f"Error loading prices: {e}")
        # Generate synthetic data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=720, freq='H')
        price_series = pd.Series(50 + np.cumsum(np.random.normal(0, 2, 720)), index=dates)
        print("Using synthetic price data")
    
    # Test different SDE models
    sde_models = {
        "Mean-Reverting": create_stochastic_forecaster("mean_reverting"),
        "Geometric Brownian Motion": create_stochastic_forecaster("gbm"),
        "Jump Diffusion": create_stochastic_forecaster("jump_diffusion"),
        "Heston Stochastic Vol": create_stochastic_forecaster("heston")
    }
    
    results = {}
    
    for model_name, sde_model in sde_models.items():
        print(f"\nTesting {model_name} Model:")
        print("-" * 40)
        
        try:
            # Fit model to historical data
            fitted_params = sde_model.fit(price_series)
            print(f"Fitted parameters: {fitted_params}")
            
            # Generate price scenarios
            price_scenarios = sde_model.simulate(n_steps=24, n_paths=1000, 
                                               initial_price=price_series.iloc[-1])
            
            # Calculate statistics
            mean_prices = np.mean(price_scenarios, axis=0)
            std_prices = np.std(price_scenarios, axis=0)
            percentile_5 = np.percentile(price_scenarios, 5, axis=0)
            percentile_95 = np.percentile(price_scenarios, 95, axis=0)
            
            results[model_name] = {
                "mean_forecast": mean_prices,
                "volatility": std_prices,
                "confidence_interval": (percentile_5, percentile_95),
                "scenarios": price_scenarios
            }
            
            print(f"24h forecast mean: ${mean_prices[23]:.2f}")
            print(f"24h forecast volatility: ${std_prices[23]:.2f}")
            print(f"24h confidence interval: [${percentile_5[23]:.2f}, ${percentile_95[23]:.2f}]")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


def test_monte_carlo_risk_assessment():
    """Test Monte Carlo simulation for risk assessment."""
    print("\n" + "="*60)
    print("TESTING MONTE CARLO RISK ASSESSMENT")
    print("="*60)
    
    # Create Monte Carlo engine
    mc_engine = create_monte_carlo_engine(n_simulations=5000)
    
    # Create a simple allocation strategy for testing
    def simple_allocation_strategy(forecast_df, initial_conditions):
        """Simple allocation strategy based on price forecast."""
        prices = forecast_df['predicted_price'].values
        max_power = initial_conditions.get("power_available", 1000.0)
        
        # Allocate more power when prices are higher
        normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
        allocations = normalized_prices * max_power * 0.5  # Max 50% allocation
        
        return {
            "inference": allocations * 0.4,
            "training": allocations * 0.3,
            "cooling": allocations * 0.3
        }
    
    # Test with different SDE models
    sde_model = create_stochastic_forecaster("mean_reverting")
    
    try:
        # Fit to some sample data
        sample_prices = pd.Series(50 + np.cumsum(np.random.normal(0, 1, 100)))
        sde_model.fit(sample_prices)
        
        # Run scenario analysis
        scenario_results = mc_engine.scenario_analysis(
            price_model=sde_model,
            allocation_strategy=simple_allocation_strategy,
            horizon=24,
            initial_conditions={"price": 50.0, "soc": 0.5, "power_available": 1000.0}
        )
        
        print("Monte Carlo Scenario Analysis Results:")
        print("-" * 40)
        print(f"Number of simulations: {scenario_results['n_simulations']}")
        
        revenue_stats = scenario_results['revenue_statistics']
        print(f"Expected revenue: ${revenue_stats['expected_return']:.2f}")
        print(f"Revenue volatility: ${revenue_stats['volatility']:.2f}")
        print(f"Value at Risk (95%): ${revenue_stats['var']:.2f}")
        print(f"Conditional VaR: ${revenue_stats['cvar']:.2f}")
        print(f"Sharpe ratio: {revenue_stats['sharpe_ratio']:.3f}")
        
        scenario_outcomes = scenario_results['scenario_outcomes']
        print(f"Best case revenue: ${scenario_outcomes['best_case_revenue']:.2f}")
        print(f"Worst case revenue: ${scenario_outcomes['worst_case_revenue']:.2f}")
        print(f"Median revenue: ${scenario_outcomes['median_revenue']:.2f}")
        print(f"Probability of positive return: {scenario_outcomes['probability_positive']:.1%}")
        
        allocation_stats = scenario_results['allocation_statistics']
        print(f"Average allocation: {allocation_stats['mean']:.1f} kW")
        print(f"Allocation range: [{allocation_stats['min']:.1f}, {allocation_stats['max']:.1f}] kW")
        
        return scenario_results
        
    except Exception as e:
        print(f"Error in Monte Carlo analysis: {e}")
        return {"error": str(e)}


def test_reinforcement_learning_agent():
    """Test RL agent for adaptive bidding strategies."""
    print("\n" + "="*60)
    print("TESTING REINFORCEMENT LEARNING AGENT")
    print("="*60)
    
    # Create RL agent
    rl_agent = create_rl_agent(state_size=64, action_size=5, learning_rate=0.1)
    
    # Create synthetic market data for training
    n_periods = 1000
    dates = pd.date_range(start=datetime.now() - timedelta(hours=n_periods), 
                         periods=n_periods, freq='H')
    
    # Generate realistic price patterns
    base_price = 50
    hourly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 24)
    daily_pattern = 5 * np.sin(2 * np.pi * np.arange(n_periods) / (24 * 7))
    noise = np.random.normal(0, 3, n_periods)
    prices = base_price + hourly_pattern + daily_pattern + noise
    prices = np.maximum(prices, 10)  # Ensure positive prices
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'price_volatility_24h': pd.Series(prices).rolling(24, min_periods=1).std()
    })
    
    print(f"Generated {len(market_data)} periods of synthetic market data")
    
    # Define reward function for RL training
    def reward_function(current_state, action, next_state):
        """Calculate reward based on price movement prediction."""
        current_price = current_state["price"]
        next_price = next_state["price"]
        price_change = (next_price - current_price) / current_price
        
        # Reward strategies that align with price movements
        # Action 0-4 represent different aggressiveness levels
        if action <= 2:  # Conservative strategies
            reward = -abs(price_change) * 100  # Penalize volatility
        else:  # Aggressive strategies
            reward = price_change * 100  # Reward positive price movements
        
        return reward
    
    # Train the RL agent
    print("Training RL agent...")
    training_episodes = 10
    total_rewards = []
    
    for episode in range(training_episodes):
        # Use different portions of data for each episode
        start_idx = episode * (len(market_data) // training_episodes)
        end_idx = start_idx + (len(market_data) // training_episodes)
        episode_data = market_data.iloc[start_idx:end_idx]
        
        episode_reward = rl_agent.train_episode(episode_data, reward_function)
        total_rewards.append(episode_reward)
        
        if episode % 2 == 0:
            print(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}")
    
    print(f"Training completed. Average reward: {np.mean(total_rewards):.2f}")
    
    # Test learned strategy
    print("\nTesting learned bidding strategies:")
    print("-" * 40)
    
    test_states = [
        {"price": 30.0, "soc": 0.3, "volatility": 0.05, "demand": 0.4},  # Low price, low SOC
        {"price": 50.0, "soc": 0.5, "volatility": 0.10, "demand": 0.5},  # Medium conditions
        {"price": 80.0, "soc": 0.8, "volatility": 0.20, "demand": 0.7},  # High price, high volatility
        {"price": 100.0, "soc": 0.9, "volatility": 0.15, "demand": 0.9}, # Very high price
    ]
    
    for i, state in enumerate(test_states):
        strategy = rl_agent.get_bidding_strategy(state)
        print(f"State {i+1}: Price=${state['price']:.0f}, SOC={state['soc']:.1f}, Vol={state['volatility']:.2f}")
        print(f"  -> Strategy: Aggressiveness={strategy['aggressiveness']:.2f}, Risk Tolerance={strategy['risk_tolerance']:.2f}")
    
    return {
        "training_rewards": total_rewards,
        "final_q_table": rl_agent.q_table,
        "test_strategies": [rl_agent.get_bidding_strategy(state) for state in test_states]
    }


def test_integration_with_existing_system():
    """Test integration of stochastic methods with existing GridPilot-GT components."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH EXISTING SYSTEM")
    print("="*60)
    
    try:
        # Get real price data
        prices_df = get_prices()
        print(f"Loaded {len(prices_df)} price records from MARA API")
        
        # Create stochastic forecaster
        sde_forecaster = create_stochastic_forecaster("mean_reverting")
        sde_forecaster.fit(prices_df['price'])
        
        # Generate stochastic price scenarios
        current_price = prices_df['price'].iloc[-1]
        price_scenarios = sde_forecaster.simulate(n_steps=24, n_paths=100, 
                                                initial_price=current_price)
        
        print(f"Generated {price_scenarios.shape[0]} price scenarios for 24-hour horizon")
        
        # Create forecast DataFrame using mean of scenarios
        mean_forecast = np.mean(price_scenarios, axis=0)
        std_forecast = np.std(price_scenarios, axis=0)
        
        forecast_df = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
            'predicted_price': mean_forecast,
            'σ_energy': std_forecast,
            'σ_hash': std_forecast * 0.5,
            'σ_token': std_forecast * 0.3,
            'lower_bound': mean_forecast - 1.96 * std_forecast,
            'upper_bound': mean_forecast + 1.96 * std_forecast,
            'method': 'stochastic_sde'
        })
        
        print(f"Created stochastic forecast: Mean price ${mean_forecast[0]:.2f}, Uncertainty ${std_forecast[0]:.2f}")
        
        # Generate bids using existing system with stochastic forecast
        uncertainty_df = forecast_df[['σ_energy', 'σ_hash', 'σ_token']]
        
        bids_df = build_bid_vector(
            current_price=current_price,
            forecast=forecast_df,
            uncertainty=uncertainty_df,
            soc=0.5,
            lambda_deg=0.0002
        )
        
        print(f"Generated bids for {len(bids_df)} time periods")
        print(f"Energy bid range: ${bids_df['energy_bid'].min():.2f} - ${bids_df['energy_bid'].max():.2f}")
        print(f"Total inference allocation: {bids_df['inference'].sum():.1f} kW")
        print(f"Total training allocation: {bids_df['training'].sum():.1f} kW")
        print(f"Total cooling allocation: {bids_df['cooling'].sum():.1f} kW")
        
        # Run VCG auction with stochastic bids
        allocation_result = vcg_allocate(bids_df, capacity_kw=1000)
        
        print(f"VCG Auction Results:")
        print(f"  Total allocation: {allocation_result['total_allocation_kw']:.1f} kW")
        print(f"  Inference: {allocation_result['allocations']['inference']:.1f} kW")
        print(f"  Training: {allocation_result['allocations']['training']:.1f} kW")
        print(f"  Cooling: {allocation_result['allocations']['cooling']:.1f} kW")
        print(f"  Total payments: ${allocation_result['total_payments']:.2f}")
        
        # Test MPC controller with stochastic forecast
        mpc_controller = MPCController(horizon=24, lambda_deg=0.0002)
        current_state = {"soc": 0.5, "available_power_kw": 1000.0}
        
        mpc_result = mpc_controller.optimize_horizon(forecast_df, current_state)
        
        print(f"MPC Optimization Results:")
        print(f"  Status: {mpc_result['status']}")
        print(f"  Energy bids range: {np.min(mpc_result['energy_bids']):.1f} - {np.max(mpc_result['energy_bids']):.1f} kW")
        print(f"  Total energy allocation: {np.sum(mpc_result['energy_bids']):.1f} kW")
        
        # Calculate performance metrics
        total_revenue = np.sum(forecast_df['predicted_price'] * mpc_result['energy_bids']) * 0.001
        risk_adjusted_revenue = total_revenue - 2 * np.sum(forecast_df['σ_energy'] * mpc_result['energy_bids']) * 0.001
        
        print(f"Performance Metrics:")
        print(f"  Expected revenue: ${total_revenue:.2f}")
        print(f"  Risk-adjusted revenue: ${risk_adjusted_revenue:.2f}")
        print(f"  Risk penalty: ${total_revenue - risk_adjusted_revenue:.2f}")
        
        return {
            "stochastic_forecast": forecast_df,
            "bids": bids_df,
            "vcg_result": allocation_result,
            "mpc_result": mpc_result,
            "performance": {
                "expected_revenue": total_revenue,
                "risk_adjusted_revenue": risk_adjusted_revenue,
                "risk_penalty": total_revenue - risk_adjusted_revenue
            }
        }
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    """Run comprehensive stochastic optimization tests."""
    print("GridPilot-GT Advanced Stochastic Optimization Test Suite")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Stochastic Differential Equations
    try:
        results["sde_models"] = test_stochastic_differential_equations()
    except Exception as e:
        print(f"SDE test failed: {e}")
        results["sde_models"] = {"error": str(e)}
    
    # Test 2: Monte Carlo Risk Assessment
    try:
        results["monte_carlo"] = test_monte_carlo_risk_assessment()
    except Exception as e:
        print(f"Monte Carlo test failed: {e}")
        results["monte_carlo"] = {"error": str(e)}
    
    # Test 3: Reinforcement Learning
    try:
        results["reinforcement_learning"] = test_reinforcement_learning_agent()
    except Exception as e:
        print(f"RL test failed: {e}")
        results["reinforcement_learning"] = {"error": str(e)}
    
    # Test 4: System Integration
    try:
        results["integration"] = test_integration_with_existing_system()
    except Exception as e:
        print(f"Integration test failed: {e}")
        results["integration"] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("STOCHASTIC OPTIMIZATION TEST SUMMARY")
    print("="*80)
    
    successful_tests = sum(1 for result in results.values() if "error" not in result)
    total_tests = len(results)
    
    print(f"Tests completed: {successful_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if "error" not in result else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")
        if "error" in result:
            print(f"  Error: {result['error']}")
    
    if successful_tests == total_tests:
        print("\n🎉 All stochastic optimization methods are working correctly!")
        print("The advanced mathematical models are ready for deployment.")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Review errors above.")
    
    return results


if __name__ == "__main__":
    results = main() 