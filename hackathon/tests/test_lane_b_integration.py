"""Comprehensive Lane B integration tests with Lane A forecasting."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Lane A imports
from api_client.client import get_prices, get_inventory
from forecasting.forecaster import create_forecaster

# Lane B imports  
from game_theory.bid_generators import build_bid_vector, portfolio_optimization
from game_theory.mpc_controller import MPCController
from game_theory.risk_models import historical_var, historical_cvar, risk_adjustment_factor


class TestMPCController:
    """Test MPC controller functionality."""
    
    def test_mpc_controller_initialization(self):
        """Test MPC controller initializes properly."""
        controller = MPCController()
        assert controller.horizon == 24
        assert controller.lambda_deg is not None
        
    def test_mpc_optimization_with_forecast(self):
        """Test MPC optimization with real forecast data."""
        prices = get_prices()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        controller = MPCController(horizon=24)
        current_state = {'soc': 0.5, 'available_power_kw': 1000.0}
        
        result = controller.optimize_horizon(forecast, current_state)
        
        assert 'status' in result
        assert 'energy_bids' in result
        assert len(result['energy_bids']) == 24
        assert all(bid >= 0 for bid in result['energy_bids'])


class TestRiskModels:
    """Test risk modeling functions."""
    
    def test_historical_var_calculation(self):
        """Test VaR calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.05, 0.02])
        var_95 = historical_var(returns, alpha=0.95)
        assert var_95 > 0
        
    def test_risk_adjustment_factor(self):
        """Test risk adjustment factor."""
        returns = pd.Series([0.1, -0.15, 0.08, -0.12])
        factor = risk_adjustment_factor(returns, target_risk=0.05)
        assert 0 < factor <= 1.0


class TestLaneABIntegration:
    """Test Lane A + Lane B integration."""
    
    def test_end_to_end_integration(self):
        """Test complete forecast to optimized bids pipeline."""
        prices = get_prices()
        inventory = get_inventory()
        
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        bids = build_bid_vector(
            current_price=prices['price'].iloc[-1],
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=inventory['battery_soc'],
            lambda_deg=0.0002
        )
        
        constraints = {'max_power': 1.0, 'cvar_limit': 0.05}
        optimized_bids = portfolio_optimization(bids, constraints)
        
        assert len(optimized_bids) == 24
        assert all(optimized_bids['energy_bid'] > 0)
        
        total_allocation = optimized_bids[['inference', 'training', 'cooling']].sum(axis=1)
        assert all(total_allocation <= 1.1)  # Allow small numerical errors
