"""Interface validation tests for Lane A integration with lanes B, C, and D."""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Lane A imports
from api_client.client import get_prices, get_inventory
from forecasting.forecaster import create_forecaster
from forecasting.advanced_forecaster import create_advanced_forecaster

# Lane B imports
from game_theory.bid_generators import build_bid_vector

# Lane C imports
from game_theory.vcg_auction import vcg_allocate
from dispatch.dispatch_agent import build_payload

# Lane D compatibility
from control.cooling_model import cooling_for_gpu_kW


class TestLaneAInterfaceContracts:
    """Test that Lane A provides exactly what lanes B, C, D expect."""
    
    def test_forecast_output_interface_contract(self):
        """Test that forecast output meets exact interface requirements."""
        prices = get_prices()
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        # CRITICAL: Exact column requirements for lanes B, C, D
        required_columns = [
            'timestamp',        # Required by all lanes for time synchronization
            'predicted_price',  # Required by Lane B for bid optimization
            'σ_energy',        # Required by Lane B for uncertainty-aware MPC
            'σ_hash',          # Required by Lane B for hash rate optimization
            'σ_token',         # Required by Lane B for token allocation
            'lower_bound',     # Required by Lane D for confidence intervals
            'upper_bound'      # Required by Lane D for risk visualization
        ]
        
        for col in required_columns:
            assert col in forecast.columns, f"CRITICAL: Missing required column '{col}' for lane integration"
        
        # Value constraints validation
        assert len(forecast) == 24, "Must return exactly 24 periods for standard horizon"
        assert all(forecast['predicted_price'] > 0), "All prices must be positive"
        assert all(forecast['σ_energy'] >= 0), "All uncertainties must be non-negative"
        assert all(forecast['σ_hash'] >= 0), "Hash uncertainty must be non-negative"
        assert all(forecast['σ_token'] >= 0), "Token uncertainty must be non-negative"
        assert all(forecast['upper_bound'] >= forecast['lower_bound']), "Upper bound must be >= lower bound"
        
        # Uncertainty relationships (critical for Lane B optimization)
        assert all(forecast['σ_hash'] <= forecast['σ_energy']), "Hash uncertainty should be <= energy uncertainty"
        assert all(forecast['σ_token'] <= forecast['σ_energy']), "Token uncertainty should be <= energy uncertainty"
        
    def test_inventory_interface_contract(self):
        """Test that inventory data meets exact interface requirements."""
        inventory = get_inventory()
        
        # CRITICAL: Exact field requirements for lanes B, C, D
        required_fields = [
            'power_total', 'power_available', 'power_used',
            'battery_soc', 'gpu_utilization', 'timestamp', 'status'
        ]
        
        for field in required_fields:
            assert field in inventory, f"CRITICAL: Missing required field '{field}' in inventory"
        
        # Value constraints validation
        assert 0.0 <= inventory['battery_soc'] <= 1.0, "SOC must be between 0 and 1"
        assert 0.0 <= inventory['gpu_utilization'] <= 1.0, "GPU utilization must be between 0 and 1"
        assert inventory['power_used'] <= inventory['power_total'], "Used power cannot exceed total"
        assert inventory['power_available'] >= 0, "Available power must be non-negative"


class TestLaneBIntegrationInterface:
    """Test Lane B (Bidding & MPC) integration interface."""
    
    def test_bid_vector_generation_interface(self):
        """Test bid vector generation meets Lane B requirements."""
        prices = get_prices()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        inventory = get_inventory()
        
        # Generate bids using Lane A outputs
        bids = build_bid_vector(
            current_price=prices['price'].iloc[-1],
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=inventory['battery_soc'],
            lambda_deg=0.0002
        )
        
        # Validate bid vector structure
        required_bid_columns = [
            'timestamp', 'energy_bid', 'regulation_bid', 'spinning_reserve_bid',
            'inference', 'training', 'cooling'
        ]
        
        for col in required_bid_columns:
            assert col in bids.columns, f"CRITICAL: Missing bid column '{col}'"
        
        # Validate bid data
        assert len(bids) == 24, "Bid vector must match forecast horizon"
        assert all(bids['energy_bid'] > 0), "Energy bids must be positive"


class TestLaneCIntegrationInterface:
    """Test Lane C (Auction & Dispatch) integration interface."""
    
    def test_vcg_auction_interface(self):
        """Test VCG auction interface requirements."""
        prices = get_prices()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=12)
        inventory = get_inventory()
        
        # Generate bids
        bids = build_bid_vector(
            current_price=prices['price'].iloc[-1],
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=inventory['battery_soc'],
            lambda_deg=0.0002
        )
        
        # Test VCG auction
        allocation, payments = vcg_allocate(bids, inventory['power_total'])
        
        # Validate allocation structure
        assert isinstance(allocation, dict), "Allocation must be dictionary"
        assert isinstance(payments, dict), "Payments must be dictionary"
        
        # Validate allocation keys
        expected_services = ['inference', 'training', 'cooling']
        for service in expected_services:
            assert service in allocation, f"Missing allocation for service '{service}'"
            assert allocation[service] >= 0, f"Allocation for '{service}' must be non-negative"


class TestLaneDIntegrationInterface:
    """Test Lane D (UI & LLM) integration interface."""
    
    def test_json_serialization_interface(self):
        """Test JSON serialization for UI compatibility."""
        prices = get_prices()
        inventory = get_inventory()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        # Create UI data package
        ui_data = {
            'timestamp': datetime.now().isoformat(),
            'prices': prices.to_dict('records'),
            'forecast': forecast.to_dict('records'),
            'inventory': inventory,
            'feature_importance': forecaster.feature_importance(),
            'model_performance': forecaster.get_model_performance()
        }
        
        # Test JSON serialization
        json_str = json.dumps(ui_data, default=str)
        assert isinstance(json_str, str), "Must be JSON serializable"
        
        # Test deserialization
        reconstructed = json.loads(json_str)
        
        # Validate structure preservation
        assert 'prices' in reconstructed
        assert 'forecast' in reconstructed
        assert 'inventory' in reconstructed
        assert len(reconstructed['forecast']) == 24


class TestInterfaceRobustness:
    """Test interface robustness under various conditions."""
    
    def test_interface_consistency_across_forecasters(self):
        """Test that both basic and advanced forecasters provide same interface."""
        prices = get_prices()
        
        basic_forecaster = create_forecaster()
        advanced_forecaster = create_advanced_forecaster()
        
        basic_forecast = basic_forecaster.predict_next(prices, periods=12)
        advanced_forecast = advanced_forecaster.predict_next(prices, periods=12)
        
        # Both should have identical column structure
        assert set(basic_forecast.columns) == set(advanced_forecast.columns), "Forecaster interfaces must be identical"
        
        # Both should have same length
        assert len(basic_forecast) == len(advanced_forecast), "Forecast lengths must match"
        
    def test_backward_compatibility(self):
        """Test backward compatibility of interfaces."""
        prices = get_prices()
        forecaster = create_forecaster()
        
        # Old-style method should still work
        old_forecast = forecaster.forecast(prices, horizon_hours=12)
        
        # New-style method
        new_forecast = forecaster.predict_next(prices, periods=12)
        
        # Both should provide compatible data
        assert isinstance(old_forecast, dict), "Old interface returns dict"
        assert isinstance(new_forecast, pd.DataFrame), "New interface returns DataFrame"
        
        # Data should be compatible
        assert len(old_forecast['energy_price']) == len(new_forecast), "Interface data length mismatch"
