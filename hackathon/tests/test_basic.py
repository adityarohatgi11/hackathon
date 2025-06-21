"""Basic tests for GridPilot-GT system."""

import pandas as pd
from api_client import get_prices, get_inventory
from forecasting import Forecaster
from game_theory.bid_generators import build_bid_vector
from game_theory.vcg_auction import vcg_allocate
from control.cooling_model import cooling_for_gpu_kW
from dispatch.dispatch_agent import build_payload


def test_api_client():
    """Test API client basic functionality."""
    prices = get_prices()
    assert isinstance(prices, pd.DataFrame)
    assert len(prices) > 0
    assert 'price' in prices.columns
    
    inventory = get_inventory()
    assert isinstance(inventory, dict)
    assert 'power_total' in inventory


def test_forecaster():
    """Test forecasting functionality."""
    forecaster = Forecaster()
    prices = get_prices()
    forecast = forecaster.predict_next(prices)
    
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) > 0
    assert 'predicted_price' in forecast.columns


def test_bid_generation():
    """Test bid generation."""
    forecaster = Forecaster()
    prices = get_prices()
    forecast = forecaster.predict_next(prices)
    
    bids = build_bid_vector(
        current_price=50.0,
        forecast=forecast,
        uncertainty=forecast[["σ_energy","σ_hash","σ_token"]],
        soc=0.5,
        lambda_deg=0.0002
    )
    
    assert isinstance(bids, pd.DataFrame)
    assert len(bids) > 0
    assert 'energy_bid' in bids.columns


def test_vcg_auction():
    """Test VCG auction mechanism."""
    forecaster = Forecaster()
    prices = get_prices()
    forecast = forecaster.predict_next(prices)
    
    bids = build_bid_vector(50.0, forecast, forecast[["σ_energy","σ_hash","σ_token"]], 0.5, 0.0002)
    allocation, payments = vcg_allocate(bids, 1000.0)
    
    assert isinstance(allocation, dict)
    assert isinstance(payments, dict)


def test_cooling_model():
    """Test cooling calculations."""
    cooling_kw, metrics = cooling_for_gpu_kW(100.0)
    
    assert cooling_kw > 0
    assert isinstance(metrics, dict)
    assert 'cop' in metrics


def test_dispatch_payload():
    """Test dispatch payload generation."""
    allocation = {'inference': 0.3, 'training': 0.2, 'cooling': 0.1}
    inventory = get_inventory()
    
    payload = build_payload(allocation, inventory, 0.5, 50.0, 1000.0)
    
    assert isinstance(payload, dict)
    assert 'allocation' in payload
    assert 'power_requirements' in payload


def test_main_integration():
    """Test main integration flow."""
    # Import main function
    from main import main
    
    # Run in simulation mode
    result = main(simulate=True)
    
    assert result is not None
    assert isinstance(result, dict) 