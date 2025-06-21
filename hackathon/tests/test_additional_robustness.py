"""Additional robustness tests for specific edge cases and integration issues."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from api_client.client import get_prices, get_inventory
from forecasting.forecaster import create_forecaster
from forecasting.advanced_forecaster import create_advanced_forecaster
from game_theory.bid_generators import build_bid_vector


class TestCriticalRobustnessIssues:
    """Test critical robustness issues that could break system integration."""
    
    def test_negative_price_predictions_robustness(self):
        """Test handling of negative price predictions that break downstream systems."""
        # Create data that could lead to negative predictions
        extreme_crash_data = []
        base_time = datetime.now()
        
        # Simulate market crash scenario
        for i in range(50):
            price = max(0.1, 100.0 - i * 3.0)  # Rapidly declining prices
            extreme_crash_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8,
                'token_price': price * 0.6
            })
        
        crash_df = pd.DataFrame(extreme_crash_data)
        
        # Test both forecasters
        basic_forecaster = create_forecaster()
        advanced_forecaster = create_advanced_forecaster()
        
        basic_forecast = basic_forecaster.predict_next(crash_df, periods=24)
        advanced_forecast = advanced_forecaster.predict_next(crash_df, periods=24)
        
        # CRITICAL: All predictions must be positive
        assert all(basic_forecast['predicted_price'] > 0), "Basic forecaster produced negative prices!"
        assert all(advanced_forecast['predicted_price'] > 0), "Advanced forecaster produced negative prices!"
        
        # Test bid generation doesn't break with crash scenario
        bids = build_bid_vector(
            current_price=crash_df['price'].iloc[-1],
            forecast=basic_forecast,
            uncertainty=basic_forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=0.5,
            lambda_deg=0.0002
        )
        
        # CRITICAL: All bids must be positive
        assert all(bids['energy_bid'] > 0), "Bid generation produced negative bids!"
        
    def test_zero_and_negative_periods_handling(self):
        """Test system behavior when asked to forecast 0 or negative periods."""
        prices = get_prices()
        forecaster = create_forecaster()
        
        # Test zero periods
        zero_forecast = forecaster.predict_next(prices, periods=0)
        assert len(zero_forecast) == 0, "Zero periods should return empty DataFrame"
        assert list(zero_forecast.columns) == [
            'timestamp', 'predicted_price', 'lower_bound', 'upper_bound',
            'σ_energy', 'σ_hash', 'σ_token', 'method'
        ], "Empty forecast should have correct columns"
        
        # Test negative periods
        negative_forecast = forecaster.predict_next(prices, periods=-5)
        assert len(negative_forecast) == 0, "Negative periods should return empty DataFrame"
        
    def test_extreme_volatility_handling(self):
        """Test system behavior with extreme market volatility."""
        # Create extremely volatile data
        volatile_data = []
        base_time = datetime.now()
        
        for i in range(100):
            # Alternating extreme highs and lows
            price = 1000.0 if i % 2 == 0 else 1.0
            volatile_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8,
                'token_price': price * 0.6
            })
        
        volatile_df = pd.DataFrame(volatile_data)
        
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(volatile_df, periods=24)
        
        # Should handle extreme volatility gracefully
        assert len(forecast) == 24
        assert all(forecast['predicted_price'] > 0)
        assert all(forecast['predicted_price'] <= 500.0), "Should cap extreme predictions"
        
        # Uncertainty should be high but reasonable
        assert all(forecast['σ_energy'] > 0)
        assert all(forecast['σ_energy'] < 200), "Uncertainty should be bounded"
        
    def test_missing_forecast_columns_robustness(self):
        """Test robustness when forecast DataFrames are missing expected columns."""
        prices = get_prices()
        
        # Create a mock forecast missing critical columns
        incomplete_forecast = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=24, freq='H'),
            'predicted_price': np.random.uniform(40, 60, 24)
            # Missing σ_energy, σ_hash, σ_token columns
        })
        
        # This should not crash the bid generation
        try:
            # Add missing columns with defaults
            if 'σ_energy' not in incomplete_forecast.columns:
                incomplete_forecast['σ_energy'] = incomplete_forecast['predicted_price'] * 0.1
            if 'σ_hash' not in incomplete_forecast.columns:
                incomplete_forecast['σ_hash'] = incomplete_forecast['σ_energy'] * 0.5
            if 'σ_token' not in incomplete_forecast.columns:
                incomplete_forecast['σ_token'] = incomplete_forecast['σ_energy'] * 0.3
            
            bids = build_bid_vector(
                current_price=50.0,
                forecast=incomplete_forecast,
                uncertainty=incomplete_forecast[['σ_energy', 'σ_hash', 'σ_token']],
                soc=0.5,
                lambda_deg=0.0002
            )
            
            assert len(bids) == 24
            assert all(bids['energy_bid'] > 0)
            
        except Exception as e:
            pytest.fail(f"System should handle missing columns gracefully: {e}")
    
    def test_extreme_outlier_data_robustness(self):
        """Test system robustness with extreme outlier data points."""
        # Create data with extreme outliers
        outlier_data = []
        base_time = datetime.now()
        
        for i in range(50):
            if i == 25:  # Single extreme outlier
                price = 10000.0  # Unrealistic spike
            else:
                price = np.random.uniform(40, 60)
            
            outlier_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8,
                'token_price': price * 0.6
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(outlier_df, periods=24)
        
        # Should not propagate extreme outliers
        max_predicted = forecast['predicted_price'].max()
        assert max_predicted <= 500.0, f"Extreme outlier propagated: {max_predicted}"
        
        # All predictions should be reasonable
        assert all(forecast['predicted_price'] >= 0.01)
        assert all(forecast['predicted_price'] <= 500.0)
    
    def test_pandas_futurewarning_fixes(self):
        """Test that pandas FutureWarnings are properly handled."""
        prices = get_prices()
        
        # This should not generate FutureWarnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            forecaster = create_forecaster()
            forecast = forecaster.predict_next(prices, periods=12)
            
            # Check for FutureWarnings related to pct_change
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            pct_change_warnings = [w for w in future_warnings if 'pct_change' in str(w.message)]
            
            assert len(pct_change_warnings) == 0, f"Found pct_change FutureWarnings: {[str(w.message) for w in pct_change_warnings]}"
    
    def test_interface_consistency_under_stress(self):
        """Test that interfaces remain consistent under stress conditions."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [50.0],
            'hash_price': [40.0],
            'token_price': [30.0]
        })
        
        # Test with large forecast horizon
        large_horizon_data = get_prices()
        
        forecasters = [create_forecaster(), create_advanced_forecaster()]
        
        for forecaster in forecasters:
            # Test minimal data
            minimal_forecast = forecaster.predict_next(minimal_data, periods=6)
            self._validate_forecast_interface(minimal_forecast, 6)
            
            # Test large horizon
            large_forecast = forecaster.predict_next(large_horizon_data, periods=168)  # 1 week
            self._validate_forecast_interface(large_forecast, 168)
    
    def _validate_forecast_interface(self, forecast: pd.DataFrame, expected_periods: int):
        """Validate that forecast has the correct interface."""
        required_columns = [
            'timestamp', 'predicted_price', 'lower_bound', 'upper_bound',
            'σ_energy', 'σ_hash', 'σ_token'
        ]
        
        for col in required_columns:
            assert col in forecast.columns, f"Missing required column: {col}"
        
        assert len(forecast) == expected_periods, f"Expected {expected_periods} periods, got {len(forecast)}"
        assert all(forecast['predicted_price'] > 0), "All prices must be positive"
        assert all(forecast['σ_energy'] >= 0), "All uncertainties must be non-negative"
        assert all(forecast['upper_bound'] >= forecast['lower_bound']), "Upper bound must be >= lower bound"


class TestAdvancedIntegrationRobustness:
    """Test advanced integration scenarios that could cause failures."""
    
    def test_concurrent_forecasting_robustness(self):
        """Test robustness when multiple forecasting requests happen concurrently."""
        import threading
        import time
        
        prices = get_prices()
        results = []
        errors = []
        
        def forecast_worker():
            try:
                forecaster = create_forecaster()
                forecast = forecaster.predict_next(prices, periods=12)
                results.append(forecast)
            except Exception as e:
                errors.append(e)
        
        # Start multiple concurrent forecasting threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=forecast_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent forecasting errors: {errors}"
        assert len(results) == 5, "All forecasting threads should complete"
        
        # All results should have consistent interface
        for result in results:
            self._validate_forecast_interface(result, 12)
    
    def test_memory_leak_prevention(self):
        """Test that repeated forecasting doesn't cause memory leaks."""
        import gc
        import psutil
        import os
        
        prices = get_prices()
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many forecasting operations
        for _ in range(50):
            forecaster = create_forecaster()
            forecast = forecaster.predict_next(prices, periods=24)
            del forecaster, forecast
            
            # Force garbage collection every 10 iterations
            if _ % 10 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 150MB due to Prophet models)
        assert memory_increase < 150, f"Potential memory leak: {memory_increase:.1f}MB increase"
    
    def _validate_forecast_interface(self, forecast: pd.DataFrame, expected_periods: int):
        """Validate forecast interface consistency."""
        required_columns = [
            'timestamp', 'predicted_price', 'lower_bound', 'upper_bound',
            'σ_energy', 'σ_hash', 'σ_token'
        ]
        
        for col in required_columns:
            assert col in forecast.columns, f"Missing column: {col}"
        
        assert len(forecast) == expected_periods
        assert all(forecast['predicted_price'] > 0)
        assert all(forecast['σ_energy'] >= 0) 