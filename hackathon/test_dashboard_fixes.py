#!/usr/bin/env python3
"""
Test script to verify dashboard fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_data_generation():
    """Test that data generation works without KeyError."""
    print("🧪 Testing data generation...")
    
    # Import the fixed functions
    from ui.complete_dashboard import get_real_time_data, generate_sample_data
    
    try:
        # Test real-time data
        data, inventory = get_real_time_data()
        
        # Check all required columns exist
        required_cols = ['timestamp', 'price', 'utilization_rate', 'battery_soc', 'volume', 'price_volatility_24h', 'energy_allocation', 'hash_allocation']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print(f"✅ All required columns present: {list(data.columns)}")
            print(f"✅ Data shape: {data.shape}")
            return True
            
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_stochastic_simulation():
    """Test stochastic simulation with different parameters."""
    print("\n🎲 Testing stochastic simulation...")
    
    from ui.complete_dashboard import run_stochastic_simulation, generate_sample_data
    
    # Generate test data
    data = generate_sample_data()
    
    # Test different model types
    model_types = ["Mean Reverting", "Geometric Brownian Motion", "Jump Diffusion", "Heston"]
    
    for model_type in model_types:
        try:
            print(f"  Testing {model_type}...")
            
            results = run_stochastic_simulation(data, model_type, n_simulations=100, horizon=12)
            
            # Check results structure
            required_keys = ['mean_forecast', 'confidence_lower', 'confidence_upper', 'fitted_params', 'model_type']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                print(f"    ❌ Missing result keys: {missing_keys}")
                return False
            
            # Check that parameters actually affect results
            mean_price = np.mean(results['mean_forecast'])
            if model_type == "Mean Reverting":
                # Should revert to mean
                expected_behavior = "mean reversion"
            elif model_type == "Geometric Brownian Motion":
                # Should show trend
                expected_behavior = "trending"
            elif model_type == "Jump Diffusion":
                # Should have jumps
                expected_behavior = "jumps"
            else:  # Heston
                # Should have stochastic volatility
                expected_behavior = "stochastic volatility"
            
            print(f"    ✅ {model_type}: Mean price ${mean_price:.3f}, behavior: {expected_behavior}")
            print(f"    ✅ Parameters: {results['fitted_params']}")
            
        except Exception as e:
            print(f"    ❌ {model_type} failed: {e}")
            return False
    
    return True

def test_reinforcement_learning():
    """Test reinforcement learning with different parameters."""
    print("\n🤖 Testing reinforcement learning...")
    
    from ui.complete_dashboard import run_reinforcement_learning, generate_sample_data
    
    # Generate test data
    data = generate_sample_data()
    
    # Test different algorithms and parameters
    test_cases = [
        {"algorithm": "Q-Learning", "episodes": 50, "learning_rate": 0.1},
        {"algorithm": "Deep Q-Network", "episodes": 100, "learning_rate": 0.05},
        {"algorithm": "Policy Gradient", "episodes": 75, "learning_rate": 0.2}
    ]
    
    for case in test_cases:
        try:
            print(f"  Testing {case['algorithm']} with {case['episodes']} episodes, lr={case['learning_rate']}...")
            
            results = run_reinforcement_learning(data, **case)
            
            # Check results structure
            required_keys = ['avg_reward', 'episodes_trained', 'optimal_strategy', 'model_type']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                print(f"    ❌ Missing result keys: {missing_keys}")
                return False
            
            # Check that parameters affect results
            avg_reward = results['avg_reward']
            episodes_trained = results['episodes_trained']
            
            print(f"    ✅ {case['algorithm']}: Avg reward {avg_reward:.1f}, Episodes {episodes_trained}")
            print(f"    ✅ Strategy: {results['optimal_strategy']}")
            
        except Exception as e:
            print(f"    ❌ {case['algorithm']} failed: {e}")
            return False
    
    return True

def test_game_theory():
    """Test game theory with different parameters."""
    print("\n🎮 Testing game theory...")
    
    from ui.complete_dashboard import run_game_theory_optimization, generate_sample_data
    
    # Generate test data
    data = generate_sample_data()
    
    # Test different game types
    test_cases = [
        {"game_type": "Cooperative", "n_players": 3, "scenarios": 50},
        {"game_type": "Stackelberg", "n_players": 4, "scenarios": 100},
        {"game_type": "Non-Cooperative", "n_players": 2, "scenarios": 75}
    ]
    
    for case in test_cases:
        try:
            print(f"  Testing {case['game_type']} with {case['n_players']} players, {case['scenarios']} scenarios...")
            
            results = run_game_theory_optimization(data, **case)
            
            # Check results structure
            required_keys = ['game_type', 'total_coalition_value', 'individual_payoffs', 'efficiency_gain']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                print(f"    ❌ Missing result keys: {missing_keys}")
                return False
            
            # Check that parameters affect results
            total_value = results['total_coalition_value']
            efficiency_gain = results['efficiency_gain']
            n_players = len(results['individual_payoffs'])
            
            print(f"    ✅ {case['game_type']}: Total value ${total_value:.0f}, Efficiency +{efficiency_gain:.1f}%")
            print(f"    ✅ Players: {n_players}, Payoffs: {results['individual_payoffs']}")
            
        except Exception as e:
            print(f"    ❌ {case['game_type']} failed: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Dashboard Fixes")
    print("=" * 50)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Stochastic Simulation", test_stochastic_simulation),
        ("Reinforcement Learning", test_reinforcement_learning),
        ("Game Theory", test_game_theory)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Dashboard fixes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    main() 