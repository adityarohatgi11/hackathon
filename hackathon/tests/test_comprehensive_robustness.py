"""Comprehensive end-to-end robustness tests for Lane A integration."""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

# Lane A imports
from api_client.client import get_prices, get_inventory, submit_bid, test_mara_api_connection
from forecasting.forecaster import Forecaster, create_forecaster
from forecasting.advanced_forecaster import QuantitativeForecaster, create_advanced_forecaster
from forecasting.feature_engineering import FeatureEngineer

# Lane B imports
from game_theory.bid_generators import build_bid_vector, portfolio_optimization, dynamic_pricing_strategy

# Lane C imports  
from game_theory.vcg_auction import vcg_allocate, auction_efficiency_metrics
from dispatch.dispatch_agent import build_payload, real_time_adjustment, emergency_response

# Lane D compatibility tests
from control.cooling_model import cooling_for_gpu_kW


class TestEndToEndIntegration:
    """Test complete end-to-end system integration."""
    
    def test_complete_market_cycle_simulation(self):
        """Test complete market cycle from data ingestion to dispatch."""
        # Step 1: Data ingestion
        prices = get_prices()
        inventory = get_inventory()
        
        assert isinstance(prices, pd.DataFrame)
        assert len(prices) > 0
        assert isinstance(inventory, dict)
        
        # Step 2: Forecasting
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        # Validate forecast interface
        required_cols = ['timestamp', 'predicted_price', 'σ_energy', 'σ_hash', 'σ_token']
        for col in required_cols:
            assert col in forecast.columns, f"Missing forecast column: {col}"
        
        assert len(forecast) == 24
        assert all(forecast['predicted_price'] > 0)
        assert all(forecast['σ_energy'] >= 0)
        
        # Step 3: Bid generation (Lane B integration)
        current_price = prices['price'].iloc[-1]
        soc = inventory['battery_soc']
        lambda_deg = 0.0002
        
        bids = build_bid_vector(
            current_price=current_price,
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=soc,
            lambda_deg=lambda_deg
        )
        
        assert isinstance(bids, pd.DataFrame)
        assert len(bids) == 24
        assert 'energy_bid' in bids.columns
        assert all(bids['energy_bid'] > 0)
        
        # Step 4: VCG Auction (Lane C integration)
        allocation, payments = vcg_allocate(bids, inventory['power_total'])
        
        assert isinstance(allocation, dict)
        assert isinstance(payments, dict)
        assert all(v >= 0 for v in allocation.values())
        assert all(v >= 0 for v in payments.values())
        
        # Step 5: Cooling calculation
        inference_power = allocation.get('inference', 0) * 1000
        cooling_kw, cooling_metrics = cooling_for_gpu_kW(inference_power)
        
        assert cooling_kw >= 0
        assert isinstance(cooling_metrics, dict)
        assert 'cop' in cooling_metrics
        
        # Step 6: Dispatch payload (Lane C integration)
        payload = build_payload(
            allocation=allocation,
            inventory=inventory,
            soc=soc,
            cooling_kw=cooling_kw,
            power_limit=inventory['power_total']
        )
        
        assert isinstance(payload, dict)
        assert 'allocation' in payload
        assert 'power_requirements' in payload
        assert 'constraints_satisfied' in payload
        
        # Step 7: JSON serialization (Lane D compatibility)
        json_payload = json.dumps(payload, default=str)
        assert isinstance(json_payload, str)
        
        # Deserialize and validate
        reconstructed = json.loads(json_payload)
        assert reconstructed['allocation'] == allocation
        
    def test_multi_forecaster_consistency(self):
        """Test consistency across different forecaster implementations."""
        prices = get_prices()
        
        # Test both basic and advanced forecasters
        basic_forecaster = create_forecaster()
        advanced_forecaster = create_advanced_forecaster()
        
        basic_forecast = basic_forecaster.predict_next(prices, periods=12)
        advanced_forecast = advanced_forecaster.predict_next(prices, periods=12)
        
        # Both should have same interface
        for df in [basic_forecast, advanced_forecast]:
            assert 'timestamp' in df.columns
            assert 'predicted_price' in df.columns
            assert 'σ_energy' in df.columns
            assert 'σ_hash' in df.columns
            assert 'σ_token' in df.columns
            assert len(df) == 12
        
        # Predictions should be reasonable (within 50% of each other)
        basic_prices = basic_forecast['predicted_price'].values
        advanced_prices = advanced_forecast['predicted_price'].values
        
        ratio = np.abs(basic_prices - advanced_prices) / (basic_prices + 1e-6)
        assert np.mean(ratio) < 0.5, "Forecasters produce very different results"
        
    def test_stress_testing_extreme_conditions(self):
        """Test system behavior under extreme market conditions."""
        # Create extreme price data
        extreme_prices = []
        base_time = datetime.now()
        
        # Scenario 1: Price spike
        for i in range(100):
            price = 1000.0 if 20 <= i <= 30 else 50.0  # 10-hour price spike
            extreme_prices.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8,
                'token_price': price * 0.6
            })
        
        extreme_df = pd.DataFrame(extreme_prices)
        
        # Test forecasting with extreme data
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(extreme_df, periods=24)
        
        # Should handle extreme values gracefully
        assert len(forecast) == 24
        assert all(forecast['predicted_price'] > 0)
        assert all(forecast['σ_energy'] > 0)
        
        # Uncertainty should be higher during volatile periods
        uncertainty_mean = forecast['σ_energy'].mean()
        assert uncertainty_mean > 5.0, "Uncertainty should be higher for extreme data"
        
        # Test bid generation with extreme forecasts
        bids = build_bid_vector(
            current_price=1000.0,
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=0.5,
            lambda_deg=0.0002
        )
        
        assert isinstance(bids, pd.DataFrame)
        assert all(bids['energy_bid'] > 0)
        
    def test_missing_data_resilience(self):
        """Test system resilience to missing/corrupted data."""
        # Test with minimal data
        minimal_prices = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [50.0],
            'hash_price': [40.0],
            'token_price': [30.0]
        })
        
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(minimal_prices, periods=6)
        
        assert len(forecast) == 6
        assert all(forecast['predicted_price'] > 0)
        
        # Test with missing columns
        incomplete_prices = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now(), periods=50, freq='H'),
            'price': np.random.uniform(40, 60, 50)
            # Missing hash_price and token_price
        })
        
        # Should handle gracefully
        forecast2 = forecaster.predict_next(incomplete_prices, periods=12)
        assert len(forecast2) == 12
        
        # Test with NaN values
        nan_prices = get_prices().copy()
        nan_prices.loc[10:20, 'price'] = np.nan  # Introduce NaN values
        
        forecast3 = forecaster.predict_next(nan_prices, periods=6)
        assert len(forecast3) == 6
        assert all(~np.isnan(forecast3['predicted_price']))
        
    def test_api_failure_fallback_robustness(self):
        """Test system behavior when APIs fail."""
        with patch('api_client.client._make_request') as mock_request:
            # Test complete API failure
            mock_request.side_effect = Exception("Network error")
            
            # Should still get data (fallback)
            prices = get_prices()
            inventory = get_inventory()
            
            assert isinstance(prices, pd.DataFrame)
            assert len(prices) > 0
            assert isinstance(inventory, dict)
            
            # System should continue to work
            forecaster = create_forecaster()
            forecast = forecaster.predict_next(prices, periods=12)
            
            assert len(forecast) == 12
            assert all(forecast['predicted_price'] > 0)
            
    def test_concurrent_operation_safety(self):
        """Test system safety under concurrent operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                prices = get_prices()
                forecaster = create_forecaster()
                forecast = forecaster.predict_next(prices, periods=6)
                results.append(len(forecast))
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple concurrent operations
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        # All operations should succeed
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 5
        assert all(r == 6 for r in results)
        
    def test_memory_usage_and_performance(self):
        """Test memory usage and performance characteristics."""
        import psutil
        import time
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset
        large_prices = []
        base_time = datetime.now()
        for i in range(1000):  # 1000 hours of data
            large_prices.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': 50.0 + 10 * np.sin(i / 24) + np.random.normal(0, 2),
                'hash_price': 40.0 + 8 * np.sin(i / 24) + np.random.normal(0, 1.5),
                'token_price': 30.0 + 6 * np.sin(i / 24) + np.random.normal(0, 1)
            })
        
        large_df = pd.DataFrame(large_prices)
        
        # Test performance
        start_time = time.time()
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(large_df, periods=24)
        end_time = time.time()
        
        processing_time = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Performance assertions
        assert processing_time < 30.0, f"Processing took too long: {processing_time:.2f}s"
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"
        assert len(forecast) == 24
        
    def test_interface_backward_compatibility(self):
        """Test backward compatibility of interfaces."""
        prices = get_prices()
        
        # Test old-style forecast method
        forecaster = create_forecaster()
        
        # Both methods should work
        forecast_new = forecaster.predict_next(prices, periods=12)
        forecast_old = forecaster.forecast(prices, horizon_hours=12)
        
        # Old method returns dict, new returns DataFrame
        assert isinstance(forecast_new, pd.DataFrame)
        assert isinstance(forecast_old, dict)
        
        # Should have compatible data
        assert len(forecast_old['energy_price']) == 12
        assert len(forecast_new) == 12
        
        # Test feature importance interface
        importance = forecaster.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
    def test_configuration_robustness(self):
        """Test robustness to different configuration scenarios."""
        # Test connection diagnostics
        connection_test = test_mara_api_connection()
        
        assert isinstance(connection_test, dict)
        assert 'overall_status' in connection_test
        assert connection_test['overall_status'] in ['operational', 'limited']
        
        # Should handle missing API key gracefully
        assert 'api_key_configured' in connection_test
        assert 'recommendations' in connection_test
        assert isinstance(connection_test['recommendations'], list)


class TestLaneBCDIntegration:
    """Test specific integration points with lanes B, C, and D."""
    
    def test_lane_b_bidding_integration(self):
        """Test integration with Lane B bidding system."""
        prices = get_prices()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=24)
        
        # Test bid generation
        bids = build_bid_vector(
            current_price=prices['price'].iloc[-1],
            forecast=forecast,
            uncertainty=forecast[['σ_energy', 'σ_hash', 'σ_token']],
            soc=0.7,
            lambda_deg=0.0002
        )
        
        # Validate bid structure for Lane B
        required_bid_cols = ['timestamp', 'energy_bid', 'regulation_bid', 'spinning_reserve_bid']
        for col in required_bid_cols:
            assert col in bids.columns, f"Missing bid column: {col}"
        
        # Test portfolio optimization
        constraints = {'max_power': 1.0, 'min_soc': 0.15, 'max_soc': 0.90}
        optimized_bids = portfolio_optimization(bids, constraints)
        
        assert isinstance(optimized_bids, pd.DataFrame)
        assert len(optimized_bids) == len(bids)
        
        # Test dynamic pricing strategy
        market_conditions = {
            'volatility': forecast['σ_energy'].mean(),
            'trend': 'upward' if forecast['predicted_price'].iloc[-1] > forecast['predicted_price'].iloc[0] else 'downward',
            'liquidity': 'high'
        }
        
        strategy = dynamic_pricing_strategy(market_conditions)
        assert isinstance(strategy, dict)
        assert 'aggressiveness' in strategy
        assert 0 <= strategy['aggressiveness'] <= 1
        
    def test_lane_c_auction_dispatch_integration(self):
        """Test integration with Lane C auction and dispatch system."""
        prices = get_prices()
        inventory = get_inventory()
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(prices, periods=12)
        
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
        
        assert isinstance(allocation, dict)
        assert isinstance(payments, dict)
        
        # Test auction efficiency metrics
        efficiency = auction_efficiency_metrics(allocation, bids)
        assert isinstance(efficiency, dict)
        assert 'allocation_efficiency' in efficiency
        assert 0 <= efficiency['allocation_efficiency'] <= 1
        
        # Test dispatch payload building
        cooling_kw, _ = cooling_for_gpu_kW(allocation.get('inference', 0) * 1000)
        
        payload = build_payload(
            allocation=allocation,
            inventory=inventory,
            soc=inventory['battery_soc'],
            cooling_kw=cooling_kw,
            power_limit=inventory['power_total']
        )
        
        # Validate payload structure
        assert 'allocation' in payload
        assert 'power_requirements' in payload
        assert 'constraints_satisfied' in payload
        assert 'system_state' in payload
        
        # Test real-time adjustment
        market_signal = {'price': prices['price'].iloc[-1] * 1.2, 'urgency': 'high'}
        adjusted_payload = real_time_adjustment(payload, market_signal)
        
        assert 'adjustment_factor' in adjusted_payload
        assert 'market_response' in adjusted_payload
        
        # Test emergency response
        emergency_state = {
            'temperature': 85.0,  # High temperature
            'soc': 0.05,         # Low battery
            'total_power_kw': inventory['power_total'] * 1.1  # Over capacity
        }
        
        emergency_response_result = emergency_response(emergency_state)
        assert isinstance(emergency_response_result, dict)
        assert 'emergency_level' in emergency_response_result
        assert emergency_response_result['emergency_level'] >= 1  # Should detect emergency
        
    def test_lane_d_ui_llm_integration(self):
        """Test integration with Lane D UI and LLM system."""
        prices = get_prices()
        inventory = get_inventory()
        forecaster = create_forecaster()
        
        # Test data serialization for UI
        forecast = forecaster.predict_next(prices, periods=24)
        
        # Should be JSON serializable
        ui_data = {
            'prices': prices.to_dict('records'),
            'forecast': forecast.to_dict('records'),
            'inventory': inventory,
            'feature_importance': forecaster.feature_importance(),
            'model_performance': forecaster.get_model_performance()
        }
        
        json_str = json.dumps(ui_data, default=str)
        assert isinstance(json_str, str)
        
        # Deserialize and validate
        reconstructed = json.loads(json_str)
        assert 'prices' in reconstructed
        assert 'forecast' in reconstructed
        assert len(reconstructed['forecast']) == 24
        
        # Test feature importance for LLM interpretation
        importance = forecaster.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Top features should make sense
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_names = [name for name, _ in top_features]
        
        # Should include time-based or price-based features (more flexible)
        meaningful_features = [f for f in feature_names if any(keyword in f.lower() for keyword in ['hour', 'day', 'time', 'peak', 'price', 'lag', 'ma'])]
        assert len(meaningful_features) > 0, "Should include meaningful features for interpretation"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        forecaster = create_forecaster()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        forecast = forecaster.predict_next(empty_df, periods=6)
        
        assert len(forecast) == 6
        assert all(forecast['predicted_price'] > 0)
        
        # Single row
        single_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [75.0],
            'hash_price': [60.0],
            'token_price': [45.0]
        })
        
        forecast2 = forecaster.predict_next(single_row, periods=12)
        assert len(forecast2) == 12
        
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        forecaster = create_forecaster()
        prices = get_prices()
        
        # Test negative periods - should return empty DataFrame gracefully
        forecast_negative = forecaster.predict_next(prices, periods=-5)
        assert len(forecast_negative) == 0
        assert 'predicted_price' in forecast_negative.columns
        
        # Test zero periods
        forecast_zero = forecaster.predict_next(prices, periods=0)
        assert len(forecast_zero) == 0
        
        # Test very large periods
        forecast_large = forecaster.predict_next(prices, periods=100)
        assert len(forecast_large) == 100
        assert all(forecast_large['predicted_price'] > 0)
        
    def test_data_quality_validation(self):
        """Test data quality validation and cleaning."""
        # Create problematic data
        problematic_data = []
        base_time = datetime.now()
        
        for i in range(50):
            price = np.random.uniform(20, 100) if i % 5 != 0 else np.nan  # 20% NaN
            problematic_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8 if not np.isnan(price) else np.nan,
                'token_price': price * 0.6 if not np.isnan(price) else np.nan
            })
        
        problematic_df = pd.DataFrame(problematic_data)
        
        # Should handle NaN values gracefully
        forecaster = create_forecaster()
        forecast = forecaster.predict_next(problematic_df, periods=12)
        
        assert len(forecast) == 12
        assert all(~np.isnan(forecast['predicted_price']))
        assert all(forecast['predicted_price'] > 0)
        
    def test_extreme_value_handling(self):
        """Test handling of extreme values."""
        # Create data with extreme outliers
        extreme_data = []
        base_time = datetime.now()
        
        for i in range(100):
            if i == 50:
                price = 10000.0  # Extreme outlier
            elif i == 51:
                price = 0.001    # Near-zero price
            else:
                price = np.random.uniform(40, 60)
            
            extreme_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': price,
                'hash_price': price * 0.8,
                'token_price': price * 0.6
            })
        
        extreme_df = pd.DataFrame(extreme_data)
        
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(extreme_df, periods=24)
        
        # Should produce reasonable forecasts despite outliers
        assert len(forecast) == 24
        assert all(forecast['predicted_price'] > 0)
        assert all(forecast['predicted_price'] <= 500)  # Should cap extreme predictions at reasonable levels
        
        # Uncertainty should reflect the extreme conditions
        assert forecast['σ_energy'].mean() > 1.0


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Generate large dataset (1 week of hourly data)
        large_data = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(24 * 7):  # 168 hours
            large_data.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': 50.0 + 10 * np.sin(i / 24) + np.random.normal(0, 3),
                'hash_price': 40.0 + 8 * np.sin(i / 24) + np.random.normal(0, 2),
                'token_price': 30.0 + 6 * np.sin(i / 24) + np.random.normal(0, 1.5)
            })
        
        large_df = pd.DataFrame(large_data)
        
        import time
        start_time = time.time()
        
        forecaster = create_advanced_forecaster()
        forecast = forecaster.predict_next(large_df, periods=48)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 30.0, f"Large dataset processing took too long: {processing_time:.2f}s"
        assert len(forecast) == 48
        
    def test_repeated_forecasting_performance(self):
        """Test performance of repeated forecasting calls."""
        prices = get_prices()
        forecaster = create_forecaster()
        
        import time
        times = []
        
        # Run multiple forecasts
        for _ in range(5):
            start = time.time()
            forecast = forecaster.predict_next(prices, periods=12)
            end = time.time()
            times.append(end - start)
            
            assert len(forecast) == 12
        
        # Performance should be consistent
        avg_time = np.mean(times)
        assert avg_time < 10.0, f"Average forecasting time too high: {avg_time:.2f}s"


# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 