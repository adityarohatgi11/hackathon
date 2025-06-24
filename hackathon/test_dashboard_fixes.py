#!/usr/bin/env python3
"""
Test script to validate dashboard fixes and check for remaining issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_generation():
    """Test the data generation functions."""
    print("🧪 Testing data generation...")
    
    try:
        # Import and test the dashboard data functions
        from ui.complete_dashboard import get_real_time_data, generate_sample_data
        
        print("✅ Import successful")
        
        # Test sample data generation
        sample_data = generate_sample_data()
        print(f"✅ Sample data generated: {len(sample_data)} rows")
        print(f"✅ Columns: {list(sample_data.columns)}")
        
        # Check for required columns
        required_cols = ['timestamp', 'price', 'utilization_rate', 'battery_soc', 'volume', 'price_volatility_24h']
        missing_cols = [col for col in required_cols if col not in sample_data.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✅ All required columns present")
        
        # Test real-time data
        real_data, inventory = get_real_time_data()
        print(f"✅ Real-time data generated: {len(real_data)} rows")
        print(f"✅ Real-time columns: {list(real_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data generation: {e}")
        return False

def test_plotly_usage():
    """Test plotly usage to check for variable scope issues."""
    print("\n🧪 Testing plotly usage...")
    
    try:
        # Test basic plotly figure creation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='test'))
        print("✅ Basic plotly figure creation works")
        
        # Test in a function scope similar to the dashboard
        def test_figure_creation():
            data = pd.DataFrame({
                'x': range(10),
                'y': np.random.random(10)
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['x'], y=data['y']))
            return fig
        
        test_fig = test_figure_creation()
        print("✅ Figure creation in function scope works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in plotly usage: {e}")
        return False

def test_stochastic_simulation():
    """Test stochastic simulation function."""
    print("\n🧪 Testing stochastic simulation...")
    
    try:
        from ui.complete_dashboard import run_stochastic_simulation
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq='H'),
            'price': np.random.uniform(2.5, 4.0, 24),
            'utilization_rate': np.random.uniform(50, 90, 24),
            'battery_soc': np.random.uniform(0.3, 0.9, 24)
        })
        
        print("✅ Test data created")
        
        # Test simulation
        results = run_stochastic_simulation(test_data, "mean_reverting", 100, 12)
        print("✅ Stochastic simulation completed")
        print(f"✅ Results keys: {list(results.keys())}")
        
        # Check results structure
        required_keys = ['mean_forecast', 'confidence_upper', 'confidence_lower', 'fitted_params', 'model_type']
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            print(f"❌ Missing result keys: {missing_keys}")
            return False
        else:
            print("✅ All required result keys present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in stochastic simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_and_rl():
    """Test machine learning and reinforcement learning functions."""
    print("\n🧪 Testing ML and RL functions...")
    
    try:
        from ui.complete_dashboard import run_neural_network_training, run_reinforcement_learning_real
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=100), periods=100, freq='H'),
            'price': np.random.uniform(2.5, 4.0, 100),
            'utilization_rate': np.random.uniform(50, 90, 100),
            'battery_soc': np.random.uniform(0.3, 0.9, 100)
        })
        
        # Test neural network training
        print("Testing neural network training...")
        ml_results = run_neural_network_training(test_data, "LSTM", 10, 16, 0.001)
        print("✅ Neural network training completed")
        print(f"✅ ML results keys: {list(ml_results.keys())}")
        
        # Test reinforcement learning
        print("Testing reinforcement learning...")
        rl_results = run_reinforcement_learning_real(test_data, "Q-Learning", 10, 0.1, 0.1)
        print("✅ Reinforcement learning completed")
        print(f"✅ RL results keys: {list(rl_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in ML/RL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_game_theory():
    """Test game theory optimization."""
    print("\n🧪 Testing game theory optimization...")
    
    try:
        from ui.complete_dashboard import run_game_theory_optimization
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq='H'),
            'price': np.random.uniform(2.5, 4.0, 24),
            'utilization_rate': np.random.uniform(50, 90, 24),
            'battery_soc': np.random.uniform(0.3, 0.9, 24)
        })
        
        # Test game theory
        game_results = run_game_theory_optimization(test_data, "Cooperative", 3, 50)
        print("✅ Game theory optimization completed")
        print(f"✅ Game results keys: {list(game_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in game theory: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting dashboard validation tests...\n")
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Plotly Usage", test_plotly_usage),
        ("Stochastic Simulation", test_stochastic_simulation),
        ("ML and RL", test_ml_and_rl),
        ("Game Theory", test_game_theory)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running {test_name} Test")
        print(f"{'='*50}")
        
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should be working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main() 