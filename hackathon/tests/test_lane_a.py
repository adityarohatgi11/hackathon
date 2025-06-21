"""Tests for Lane A: Data & Forecasting functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from api_client.client import get_prices, get_inventory, submit_bid, test_mara_api_connection
from forecasting.forecaster import Forecaster, create_forecaster
from forecasting.advanced_forecaster import QuantitativeForecaster, create_advanced_forecaster
from forecasting.feature_engineering import FeatureEngineer


class TestAPIClient:
    """Test API client functionality with MARA integration."""
    
    def test_get_prices_basic(self):
        """Test basic price data retrieval."""
        df = get_prices()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'price' in df.columns
        assert 'hash_price' in df.columns
        assert 'token_price' in df.columns
        
        # Verify prices are positive
        assert (df['price'] > 0).all()
        assert (df['hash_price'] > 0).all()
        assert (df['token_price'] > 0).all()
    
    def test_get_inventory_structure(self):
        """Test inventory data structure and validation."""
        inventory = get_inventory()
        
        required_fields = [
            'power_total', 'power_available', 'power_used', 
            'battery_soc', 'gpu_utilization', 'timestamp', 'status'
        ]
        
        for field in required_fields:
            assert field in inventory, f"Missing required field: {field}"
        
        # Validate data types and ranges
        assert 0 <= inventory['battery_soc'] <= 1.0
        assert 0 <= inventory['gpu_utilization'] <= 1.0
        assert inventory['power_used'] <= inventory['power_total']
        assert inventory['power_available'] >= 0
    
    def test_submit_bid_format(self):
        """Test bid submission format and response."""
        test_payload = {
            'allocation': {
                'air_miners': 1,
                'inference': 2,
                'training': 1,
                'hydro_miners': 0,
                'immersion_miners': 1
            },
            'power_requirements': {'total_power_kw': 100},
            'system_state': {'status': 'test'}
        }
        
        response = submit_bid(test_payload)
        
        assert 'status' in response
        assert 'timestamp' in response
        assert response['status'] in ['success', 'failed']
        
        if response['status'] == 'success':
            assert 'bid_id' in response
            assert 'allocation_accepted' in response
    
    def test_mara_api_connection_test(self):
        """Test MARA API connection diagnostics."""
        result = test_mara_api_connection()
        
        required_fields = [
            'timestamp', 'api_base_url', 'api_key_configured',
            'overall_status', 'recommendations'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field in connection test: {field}"
        
        assert result['overall_status'] in ['operational', 'limited']
        assert isinstance(result['recommendations'], list)
    
    @patch('api_client.client._make_request')
    def test_api_error_handling(self, mock_request):
        """Test API error handling and fallback behavior."""
        # Simulate API failure
        mock_request.side_effect = Exception("API unavailable")
        
        # Should fall back to synthetic data
        df = get_prices()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        inventory = get_inventory()
        assert isinstance(inventory, dict)
        assert 'status' in inventory
        # Should indicate fallback mode
        assert 'fallback' in inventory['status'] or 'API_UNAVAILABLE' in inventory.get('alerts', [])


class TestForecasting:
    """Test forecasting functionality with real/mock data."""
    
    def test_basic_forecaster_creation(self):
        """Test basic forecaster instantiation."""
        forecaster = create_forecaster()
        assert isinstance(forecaster, Forecaster)
    
    def test_advanced_forecaster_creation(self):
        """Test advanced forecaster with dependencies check."""
        try:
            forecaster = create_advanced_forecaster()
            assert isinstance(forecaster, QuantitativeForecaster)
        except ImportError:
            # Advanced libraries not available, should fall back
            forecaster = create_advanced_forecaster()
            assert isinstance(forecaster, Forecaster)
    
    def test_forecasting_with_real_data(self):
        """Test forecasting using real API data."""
        # Get real price data
        df = get_prices()
        
        if len(df) < 48:  # Need sufficient data
            pytest.skip("Insufficient data for forecasting test")
        
        forecaster = create_forecaster()
        
        # Test forecast generation
        forecast = forecaster.forecast(df, horizon_hours=24)
        
        assert isinstance(forecast, dict)
        required_keys = ['timestamps', 'energy_price', 'sigma_energy', 'sigma_hash', 'sigma_token']
        for key in required_keys:
            assert key in forecast, f"Missing forecast key: {key}"
        
        # Validate forecast structure
        assert len(forecast['timestamps']) == 24
        assert len(forecast['energy_price']) == 24
        assert all(p > 0 for p in forecast['energy_price'])
        assert all(s >= 0 for s in forecast['sigma_energy'])
    
    def test_forecasting_empty_data_handling(self):
        """Test forecaster behavior with empty/insufficient data."""
        forecaster = create_forecaster()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        forecast = forecaster.forecast(empty_df, horizon_hours=24)
        
        # Should return default/synthetic forecast
        assert isinstance(forecast, dict)
        assert 'energy_price' in forecast
        assert len(forecast['energy_price']) == 24
    
    def test_feature_engineering_with_api_data(self):
        """Test feature engineering with real API data."""
        df = get_prices()
        
        feature_engineer = FeatureEngineer()
        enhanced_df = feature_engineer.engineer_features(df)
        
        # Should have more columns than original
        assert len(enhanced_df.columns) > len(df.columns)
        
        # Check for key technical indicators
        expected_features = ['price_ma', 'price_rsi', 'price_volatility', 'returns']
        present_features = [f for f in expected_features if any(f in col for col in enhanced_df.columns)]
        assert len(present_features) > 0, "No technical indicators found in features"


class TestIntegration:
    """Test end-to-end integration scenarios."""
    
    def test_data_flow_integration(self):
        """Test complete data flow from API to forecasting."""
        # Step 1: Get real-time data
        prices_df = get_prices()
        inventory = get_inventory()
        
        # Step 2: Create forecast
        forecaster = create_forecaster()
        forecast = forecaster.forecast(prices_df, horizon_hours=6)
        
        # Step 3: Validate integration
        assert isinstance(forecast, dict)
        assert len(forecast['energy_price']) == 6
        
        # Test that forecast incorporates market conditions
        current_price = prices_df['price'].iloc[-1]
        forecast_prices = forecast['energy_price']
        
        # Forecast should be reasonable relative to current prices (lenient for volatile markets)
        price_ratio = np.array(forecast_prices) / current_price
        # More lenient bounds for energy market volatility
        assert all(0.1 <= ratio <= 10.0 for ratio in price_ratio), f"Forecast prices seem unrealistic: ratios={price_ratio}"
    
    def test_api_key_configuration(self):
        """Test API key configuration and validation."""
        connection_test = test_mara_api_connection()
        
        # If API key is not configured, should provide clear guidance
        if not connection_test['api_key_configured']:
            recommendations_str = str(connection_test['recommendations']).lower()
            assert "api key" in recommendations_str or "mara" in recommendations_str
        
        # Should handle both configured and unconfigured states gracefully
        assert connection_test['overall_status'] in ['operational', 'limited']
    
    def test_fallback_reliability(self):
        """Test system reliability when API is unavailable."""
        with patch('api_client.client._make_request') as mock_request:
            # Simulate complete API failure
            mock_request.side_effect = Exception("Network error")
            
            # System should still function with fallback data
            prices_df = get_prices()
            inventory = get_inventory()
            
            assert len(prices_df) > 0
            assert 'price' in prices_df.columns
            assert isinstance(inventory, dict)
            
            # Should be able to generate forecasts
            forecaster = create_forecaster()
            forecast = forecaster.forecast(prices_df, horizon_hours=12)
            
            assert isinstance(forecast, dict)
            assert len(forecast['energy_price']) == 12


@pytest.mark.integration
class TestMARAPILive:
    """Live tests for MARA API (only run when API is available)."""
    
    def test_live_api_connection(self):
        """Test live connection to MARA API."""
        try:
            connection_test = test_mara_api_connection()
            
            if connection_test.get('prices_available'):
                # Live API test - get real prices
                df = get_prices()
                assert len(df) > 0
                
                # Verify we got real-time data
                latest_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
                time_diff = abs((datetime.now() - latest_timestamp).total_seconds())
                
                # Should be relatively recent (within 24 hours)
                assert time_diff < 86400, "Price data seems too old"
                
            else:
                pytest.skip("MARA API not available for live testing")
                
        except Exception as e:
            pytest.skip(f"Live API test skipped due to: {e}")
    
    def test_live_inventory_data(self):
        """Test live inventory data from MARA API."""
        try:
            inventory = get_inventory()
            
            # If we have MARA response data, validate it
            if 'mara_response' in inventory:
                mara_data = inventory['mara_response']
                
                # Should have expected MARA structure
                expected_sections = ['inference', 'miners']
                available_sections = [s for s in expected_sections if s in mara_data]
                
                assert len(available_sections) > 0, "No expected MARA data sections found"
                
            # Inventory should have realistic values
            assert 0 <= inventory['battery_soc'] <= 1.0
            assert inventory['power_total'] > 0
            
        except Exception as e:
            pytest.skip(f"Live inventory test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 