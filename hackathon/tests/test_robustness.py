import pytest
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

# Import factory to get best forecaster
from forecasting import create_advanced_forecaster

# Import main orchestrator for end-to-end simulation
from main import main as run_main


@pytest.fixture
def synthetic_prices():
    """Generate synthetic price data spanning two weeks for robustness tests."""
    periods = 336  # 2 weeks hourly data
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='H')
    # Daily seasonality + random noise + occasional spikes/outliers
    base_price = 60 + 15 * np.sin(2 * np.pi * (np.arange(periods) % 24) / 24)
    noise = np.random.normal(0, 4, periods)
    spikes = np.random.choice([0, 30, -20], size=periods, p=[0.95, 0.03, 0.02])
    prices = base_price + noise + spikes
    
    # Ensure no negative prices
    prices = np.clip(prices, 5, None)
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.uniform(500, 2000, periods)
    })


def test_quantitative_forecaster_prediction(synthetic_prices):
    """Test QuantitativeForecaster end-to-end prediction pipeline."""
    forecaster = create_advanced_forecaster()
    forecast = forecaster.predict_next(synthetic_prices, periods=24)

    # Interface checks
    required_cols = [
        'timestamp', 'predicted_price', 'lower_bound', 'upper_bound',
        'σ_energy', 'σ_hash', 'σ_token'
    ]
    for col in required_cols:
        assert col in forecast.columns, f"Missing column {col}"

    # Basic robustness checks
    assert len(forecast) == 24
    assert forecast['predicted_price'].min() > 0
    assert all(forecast['upper_bound'] >= forecast['lower_bound'])

    # Ensemble weights sanity (should sum approximately to 1)
    if hasattr(forecaster, 'ensemble_weights'):
        total_weight = sum(forecaster.ensemble_weights.values())
        assert 0.8 <= total_weight <= 1.2  # allow some tolerance


@pytest.mark.skipif(
    not hasattr(run_main, '__call__'),
    reason="Main orchestrator not callable"
)
def test_end_to_end_simulation():
    """Run the main orchestrator in simulation mode to ensure end-to-end robustness."""
    payload = run_main(simulate=True)

    assert isinstance(payload, dict)
    assert payload.get('constraints_satisfied', False) is True
    assert payload['power_requirements']['total_power_kw'] >= 0
    # Utilization can be zero if no allocation – that's acceptable
    assert 0 <= payload['system_state']['utilization'] <= 1


def test_forecaster_handles_missing_data(synthetic_prices):
    """Ensure forecaster gracefully handles missing timestamps and prices."""
    # Introduce missing values
    corrupted = synthetic_prices.copy()
    corrupted.loc[10:20, 'price'] = np.nan
    corrupted.loc[30:35, 'timestamp'] = pd.NaT

    forecaster = create_advanced_forecaster()

    # Should not raise and should fill / drop appropriately
    forecast = forecaster.predict_next(corrupted.dropna(subset=['timestamp']), periods=12)
    assert len(forecast) == 12
    assert not forecast['predicted_price'].isna().any()


@pytest.mark.parametrize('outlier_factor', [5, 10, 20])
def test_forecaster_resilient_to_price_outliers(synthetic_prices, outlier_factor):
    """Test robustness of forecast under extreme price outliers."""
    extreme_prices = synthetic_prices.copy()
    # Introduce a single extreme outlier
    extreme_prices.loc[100, 'price'] *= outlier_factor

    forecaster = create_advanced_forecaster()
    forecast = forecaster.predict_next(extreme_prices, periods=24)

    # Forecast should remain finite and positive
    assert np.isfinite(forecast['predicted_price']).all()
    assert (forecast['predicted_price'] > 0).all() 