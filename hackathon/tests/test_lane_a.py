"""Comprehensive tests for Lane A: Data & Forecasting."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from api_client import get_prices, get_inventory, submit_bid, get_market_status
from forecasting import Forecaster, FeatureEngineer


class TestAPIClient:
    """Test enhanced API client functionality."""
    
    def test_get_prices_enhanced(self):
        """Test enhanced price data retrieval."""
        prices = get_prices()
        
        # Basic structure
        assert isinstance(prices, pd.DataFrame)
        assert len(prices) > 0
        assert 'timestamp' in prices.columns
        assert 'price' in prices.columns
        
        # Enhanced features
        expected_features = [
            'volume', 'hour_of_day', 'day_of_week', 'is_weekend',
            'price_ma_24h', 'price_volatility_24h', 'load_factor', 'market_stress'
        ]
        for feature in expected_features:
            assert feature in prices.columns, f"Missing feature: {feature}"
        
        # Data quality
        assert prices['price'].min() >= 10  # Minimum price floor
        assert prices['timestamp'].is_monotonic_increasing  # Time ordering
        assert not prices['price'].isna().any()  # No missing prices
    
    def test_get_inventory_enhanced(self):
        """Test enhanced inventory data."""
        inventory = get_inventory()
        
        # Required fields for other lanes
        required_fields = [
            'power_total', 'power_available', 'battery_soc', 'gpu_utilization',
            'timestamp', 'temperature', 'efficiency', 'alerts'
        ]
        for field in required_fields:
            assert field in inventory, f"Missing inventory field: {field}"
        
        # Data constraints
        assert 0 <= inventory['battery_soc'] <= 1
        assert 0 <= inventory['gpu_utilization'] <= 1
        assert inventory['power_total'] >= 0
        assert isinstance(inventory['alerts'], list)
    
    def test_submit_bid_validation(self):
        """Test bid submission with validation."""
        # Valid payload
        valid_payload = {
            'allocation': {'inference': 0.3, 'training': 0.2, 'cooling': 0.1},
            'power_requirements': {'total_power_kw': 100},
            'system_state': {'soc': 0.5},
            'constraints_satisfied': True
        }
        
        response = submit_bid(valid_payload)
        assert response['status'] in ['success', 'rejected']
        assert 'bid_id' in response
        assert 'timestamp' in response
        
        # Invalid payload should raise error
        with pytest.raises(Exception):
            submit_bid({'invalid': 'payload'})


class TestFeatureEngineer:
    """Test feature engineering capabilities."""
    
    @pytest.fixture
    def sample_prices(self):
        """Sample price data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=168, freq='H')  # 1 week
        prices = 50 + 10 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 5, 168)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.uniform(100, 1000, 168)
        })
    
    def test_feature_engineering(self, sample_prices):
        """Test comprehensive feature engineering."""
        fe = FeatureEngineer()
        features = fe.engineer_features(sample_prices)
        
        # Should have many more features than input
        assert len(features.columns) > len(sample_prices.columns) * 10
        
        # Check specific feature categories
        temporal_features = [col for col in features.columns if any(
            keyword in col for keyword in ['hour', 'day', 'month', 'weekend', 'peak']
        )]
        assert len(temporal_features) >= 10
        
        price_features = [col for col in features.columns if 'price' in col]
        assert len(price_features) >= 20
        
        technical_features = [col for col in features.columns if any(
            keyword in col for keyword in ['bb_', 'rsi', 'macd']
        )]
        assert len(technical_features) >= 5
    
    def test_feature_selection(self, sample_prices):
        """Test feature selection functionality."""
        fe = FeatureEngineer()
        features = fe.engineer_features(sample_prices)
        selected = fe.select_features(features, max_features=10)
        
        assert len(selected) <= 10
        assert all(feat in features.columns for feat in selected)
    
    def test_prepare_forecast_data(self, sample_prices):
        """Test data preparation for forecasting."""
        fe = FeatureEngineer()
        features = fe.engineer_features(sample_prices)
        X, y = fe.prepare_forecast_data(features)
        
        assert len(X) == len(y)
        assert len(X) == len(features)
        assert not X.isna().any().any()  # No missing values


class TestForecaster:
    """Test advanced forecasting capabilities."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for forecasting tests."""
        dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
        prices = 50 + 10 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 3, 168)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.uniform(100, 1000, 168)
        })
    
    def test_forecaster_initialization(self):
        """Test forecaster initialization options."""
        # Default initialization
        forecaster = Forecaster()
        assert forecaster.use_prophet
        assert forecaster.use_ensemble
        
        # Custom initialization
        forecaster_simple = Forecaster(use_prophet=False, use_ensemble=False)
        assert not forecaster_simple.use_prophet
        assert not forecaster_simple.use_ensemble
    
    def test_predict_next_interface_compatibility(self, sample_data):
        """Test that predict_next maintains interface compatibility with other lanes."""
        forecaster = Forecaster(use_prophet=False, use_ensemble=False)  # Use simple for speed
        forecast = forecaster.predict_next(sample_data, periods=24)
        
        # Required columns for Lane B (bidding)
        required_columns = [
            'timestamp', 'predicted_price', 'σ_energy', 'σ_hash', 'σ_token'
        ]
        for col in required_columns:
            assert col in forecast.columns, f"Missing required column: {col}"
        
        # Data quality
        assert len(forecast) == 24
        assert forecast['predicted_price'].min() > 0
        assert not forecast['predicted_price'].isna().any()
        assert forecast['timestamp'].is_monotonic_increasing
    
    def test_predict_volatility(self, sample_data):
        """Test volatility prediction."""
        forecaster = Forecaster()
        volatility = forecaster.predict_volatility(sample_data)
        
        assert len(volatility) == len(sample_data)
        assert 'vol_forecast' in volatility.columns
        assert volatility['vol_forecast'].iloc[-1] >= 0
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        forecaster = Forecaster()
        importance = forecaster.feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(0 <= score <= 1 for score in importance.values())
    
    def test_model_performance_tracking(self, sample_data):
        """Test model performance metrics."""
        forecaster = Forecaster()
        forecaster.fit(sample_data)  # Explicit training
        
        performance = forecaster.get_model_performance()
        assert isinstance(performance, dict)


class TestIntegrationWithOtherLanes:
    """Test integration compatibility with Lanes B, C, D."""
    
    def test_lane_b_compatibility(self):
        """Test data format compatibility with Lane B (bidding)."""
        # Get Lane A outputs
        prices = get_prices()
        forecaster = Forecaster(use_prophet=False, use_ensemble=False)  # Simple for speed
        forecast = forecaster.predict_next(prices)
        
        # Simulate Lane B input requirements
        current_price = prices['price'].iloc[-1]
        uncertainty = forecast[["σ_energy","σ_hash","σ_token"]]
        
        # These should not raise errors in Lane B
        assert isinstance(current_price, (int, float))
        assert len(uncertainty) == len(forecast)
        assert uncertainty.columns.tolist() == ["σ_energy","σ_hash","σ_token"]
    
    def test_lane_c_compatibility(self):
        """Test data format compatibility with Lane C (auction/dispatch)."""
        inventory = get_inventory()
        
        # Required fields for Lane C
        assert 'power_total' in inventory
        assert 'power_available' in inventory
        assert 'battery_soc' in inventory
        assert isinstance(inventory['power_total'], (int, float))
    
    def test_lane_d_compatibility(self):
        """Test data format compatibility with Lane D (UI/LLM)."""
        # Get data that Lane D will visualize
        prices = get_prices()
        inventory = get_inventory()
        forecaster = Forecaster(use_prophet=False, use_ensemble=False)
        forecast = forecaster.predict_next(prices, periods=12)
        
        # Data should be JSON serializable for web UI
        import json
        
        # Test serialization of key data
        price_sample = {
            'timestamp': prices['timestamp'].iloc[-1].isoformat(),
            'price': float(prices['price'].iloc[-1]),
            'forecast': float(forecast['predicted_price'].iloc[0])
        }
        json.dumps(price_sample)  # Should not raise
        
        # Inventory should be JSON compatible
        inventory_serializable = {k: v for k, v in inventory.items() 
                                 if isinstance(v, (str, int, float, list))}
        json.dumps(inventory_serializable)  # Should not raise


class TestDataQuality:
    """Test data quality and robustness."""
    
    def test_missing_data_handling(self):
        """Test handling of missing or corrupted data."""
        # Test with empty data
        empty_df = pd.DataFrame()
        forecaster = Forecaster()
        
        # Should not crash
        forecast = forecaster.predict_next(empty_df)
        assert isinstance(forecast, pd.DataFrame)
    
    def test_extreme_values(self):
        """Test handling of extreme price values."""
        dates = pd.date_range(start='2024-01-01', periods=48, freq='H')
        extreme_prices = [1000, 0.01] * 24  # Extreme high and low
        
        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'price': extreme_prices,
            'volume': [100] * 48
        })
        
        forecaster = Forecaster(use_prophet=False, use_ensemble=False)
        forecast = forecaster.predict_next(extreme_data)
        
        # Should produce reasonable forecasts despite extreme inputs
        assert forecast['predicted_price'].min() > 0
        assert forecast['predicted_price'].max() < 10000
    
    def test_time_consistency(self):
        """Test time series consistency."""
        prices = get_prices()
        
        # Check time ordering
        assert prices['timestamp'].is_monotonic_increasing
        
        # Check reasonable time gaps
        time_diffs = prices['timestamp'].diff().dropna()
        assert all(time_diffs <= pd.Timedelta(hours=2))  # No gaps > 2 hours 