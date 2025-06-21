"""Comprehensive test suite for VCG auction and dispatch integration.

Tests include:
- Cross-module integration (A→B→C→D pipeline)
- Performance benchmarking with CI gates
- Safety protocol and constraint violation testing
- Chaos testing and recovery scenarios
- Interface contract validation
"""

import pytest
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_theory.vcg_auction import vcg_allocate, auction_efficiency_metrics, validate_vcg_properties
from dispatch.dispatch_agent import build_payload, real_time_adjustment, emergency_response, validate_dispatch_performance
from dispatch.execution_engine import get_execution_engine, execute_dispatch, wait_for_execution
import main


class TestVCGAuctionIntegration:
    """Test VCG auction mechanism integration."""
    
    @pytest.fixture
    def sample_bids(self):
        """Sample bid data for testing."""
        return pd.DataFrame({
            'inference': [100, 200, 50, 150],
            'training': [150, 100, 80, 120], 
            'cooling': [50, 30, 20, 40],
            'inference_bid': [45, 55, 40, 50],
            'training_bid': [50, 48, 42, 47],
            'cooling_bid': [35, 38, 30, 33],
            'energy_bid': [50, 52, 45, 48]
        })
    
    @pytest.fixture
    def test_inventory(self):
        """Sample inventory data."""
        return {
            'power_total': 1000.0,
            'power_available': 750.0,
            'battery_soc': 0.65,
            'gpu_utilization': 0.8,
            'temperature': 68.0,
            'grid_frequency': 60.0
        }
    
    def test_vcg_auction_basic_functionality(self, sample_bids):
        """Test basic VCG auction functionality."""
        allocation, payments = vcg_allocate(sample_bids, 1000.0)
        
        # Verify allocation structure
        assert isinstance(allocation, dict)
        assert isinstance(payments, dict)
        assert set(allocation.keys()) == {'inference', 'training', 'cooling'}
        assert set(payments.keys()) == {'inference', 'training', 'cooling'}
        
        # Verify allocation constraints
        total_allocation = sum(allocation.values())
        assert total_allocation <= 1000.0  # Within capacity
        assert all(v >= 0 for v in allocation.values())  # Non-negative
        
        # Verify payments
        assert all(v >= 0 for v in payments.values())  # Non-negative payments
    
    def test_vcg_auction_economic_properties(self, sample_bids):
        """Test VCG auction economic properties."""
        allocation, payments = vcg_allocate(sample_bids, 1000.0)
        
        # Test VCG properties
        properties = validate_vcg_properties(sample_bids, allocation, payments)
        assert properties['individual_rationality']
        assert properties['truthfulness']
        assert properties['pareto_efficiency']
        
        # Test efficiency metrics
        metrics = auction_efficiency_metrics(allocation, sample_bids)
        assert metrics['social_welfare'] > 0
        assert 0 <= metrics['allocation_efficiency'] <= 5.0  # Reasonable range
        assert metrics['truthfulness_score'] == 1.0  # VCG guarantee
        assert metrics['pareto_efficiency'] == 1.0  # VCG guarantee
    
    @pytest.mark.benchmark
    def test_vcg_auction_performance(self, sample_bids, benchmark):
        """Benchmark VCG auction performance."""
        def run_auction():
            return vcg_allocate(sample_bids, 1000.0)
        
        # Benchmark the auction
        result = benchmark(run_auction)
        allocation, payments = result
        
        # Verify performance target (should be < 10ms for small instances)
        assert benchmark.stats.stats.mean < 0.01  # < 10ms
        
        # Verify correctness
        assert sum(allocation.values()) <= 1000.0


class TestDispatchAgentIntegration:
    """Test dispatch agent integration and performance."""
    
    @pytest.fixture
    def sample_allocation(self):
        """Sample allocation from VCG auction."""
        return {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
    
    @pytest.fixture
    def test_inventory(self):
        """Test inventory data."""
        return {
            'power_total': 1000.0,
            'power_available': 750.0,
            'battery_soc': 0.65,
            'gpu_utilization': 0.8,
            'temperature': 68.0,
            'grid_frequency': 60.0
        }
    
    @pytest.mark.benchmark
    def test_build_payload_performance(self, sample_allocation, test_inventory, benchmark):
        """Test build_payload performance meets <100ms requirement."""
        def run_build_payload():
            return build_payload(sample_allocation, test_inventory, 0.65, 150.0, 1000.0)
        
        result = benchmark(run_build_payload)
        
        # Performance gate: must be < 100ms
        assert benchmark.stats.stats.mean < 0.1  # < 100ms
        
        # Verify result structure
        assert 'performance_metrics' in result
        assert result['performance_metrics']['build_time_ms'] < 100
    
    @pytest.mark.benchmark
    def test_real_time_adjustment_performance(self, sample_allocation, test_inventory, benchmark):
        """Test real-time adjustment performance."""
        payload = build_payload(sample_allocation, test_inventory, 0.65, 150.0, 1000.0)
        market_signal = {'price': 65.0, 'volume': 150, 'frequency': 59.7}
        
        def run_adjustment():
            return real_time_adjustment(payload, market_signal)
        
        result = benchmark(run_adjustment)
        
        # Performance gate: must be < 50ms
        assert benchmark.stats.stats.mean < 0.05  # < 50ms
        
        # Verify market response
        assert 'market_response' in result
        assert 'adjustment_factor' in result
    
    @pytest.mark.benchmark
    def test_emergency_response_performance(self, benchmark):
        """Test emergency response performance."""
        system_state = {
            'temperature': 78.0,
            'soc': 0.35,
            'power_limit': 1000.0,
            'total_power_kw': 850.0,
            'grid_frequency': 59.8,
            'cooling_capacity': 400.0
        }
        
        def run_emergency():
            return emergency_response(system_state)
        
        result = benchmark(run_emergency)
        
        # Performance gate: must be < 10ms
        assert benchmark.stats.stats.mean < 0.01  # < 10ms
        
        # Verify emergency response
        assert 'emergency_level' in result
        assert 'actions' in result
    
    def test_dispatch_pipeline_performance(self, sample_allocation, test_inventory):
        """Test complete dispatch pipeline performance."""
        start_time = time.perf_counter()
        
        # Build payload
        payload = build_payload(sample_allocation, test_inventory, 0.65, 150.0, 1000.0)
        
        # Real-time adjustment
        market_signal = {'price': 65.0, 'volume': 150, 'frequency': 59.7}
        adjusted_payload = real_time_adjustment(payload, market_signal)
        
        # Emergency response
        system_state = {'temperature': 70.0, 'soc': 0.65, 'power_limit': 1000.0, 'total_power_kw': 800.0}
        emergency_resp = emergency_response(system_state)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Performance gate: total pipeline < 100ms
        assert total_time < 100.0
        
        # Validate performance tracking
        perf_validation = validate_dispatch_performance(adjusted_payload)
        assert perf_validation['meets_target']
        assert perf_validation['total_response_time_ms'] < 100.0


class TestSafetyProtocols:
    """Test safety protocols and constraint violation handling."""
    
    def test_power_constraint_violations(self):
        """Test handling of power constraint violations."""
        # Extreme allocation exceeding limits
        extreme_allocation = {'inference': 2000.0, 'training': 1500.0, 'cooling': 500.0}
        inventory = {'power_total': 1000.0, 'power_available': 500.0, 'battery_soc': 0.5}
        
        payload = build_payload(extreme_allocation, inventory, 0.5, 200.0, 1000.0)
        
        # Should automatically scale down
        assert payload['system_state']['emergency_scaled']
        assert payload['power_requirements']['total_power_kw'] <= 1000.0
        assert payload['power_requirements']['power_scale_factor'] < 1.0
    
    def test_temperature_emergency_response(self):
        """Test temperature emergency cascading."""
        # Critical temperature scenario
        critical_state = {
            'temperature': 85.0,  # Critical temperature
            'soc': 0.5,
            'power_limit': 1000.0,
            'total_power_kw': 800.0
        }
        
        response = emergency_response(critical_state)
        
        # Should trigger shutdown
        assert response['emergency_level'] == 3  # SHUTDOWN
        assert response['shutdown_required']
        assert 'EMERGENCY_SHUTDOWN' in response['actions']
        assert response['power_reduction'] == 1.0  # Complete shutdown
        assert response['estimated_recovery_time'] >= 600  # 10+ minutes
    
    def test_battery_emergency_scenarios(self):
        """Test battery emergency handling."""
        # Critical low battery
        low_battery_state = {
            'temperature': 70.0,
            'soc': 0.03,  # Critical low
            'power_limit': 1000.0,
            'total_power_kw': 800.0
        }
        
        response = emergency_response(low_battery_state)
        
        assert response['emergency_level'] == 3  # SHUTDOWN
        assert response['shutdown_required']
        assert 'EMERGENCY_CHARGE' in response['actions']
        assert 'LOAD_SHEDDING' in response['actions']
    
    def test_grid_frequency_emergency(self):
        """Test grid frequency disturbance handling."""
        # Major grid disturbance
        grid_disturbance_state = {
            'temperature': 70.0,
            'soc': 0.5,
            'power_limit': 1000.0,
            'total_power_kw': 800.0,
            'grid_frequency': 60.7  # Major deviation
        }
        
        response = emergency_response(grid_disturbance_state)
        
        assert response['emergency_level'] >= 2  # CRITICAL
        assert 'GRID_DISCONNECT' in response['actions']
        assert 'ISLANDING_MODE' in response['actions']
        assert response['safe_mode']
    
    def test_cascading_failures(self):
        """Test handling of multiple simultaneous failures."""
        # Multiple failure scenario
        multi_failure_state = {
            'temperature': 82.0,      # High temperature
            'soc': 0.08,             # Low battery
            'power_limit': 1000.0,
            'total_power_kw': 1200.0, # Over power limit
            'grid_frequency': 60.6,   # Frequency deviation
            'cooling_capacity': 200.0 # Insufficient cooling
        }
        
        response = emergency_response(multi_failure_state)
        
        # Should trigger maximum emergency level
        assert response['emergency_level'] == 3  # SHUTDOWN
        assert response['shutdown_required']
        assert len(response['actions']) >= 3  # Multiple actions
        assert response['power_reduction'] >= 0.5  # Significant reduction


class TestChaosEngineering:
    """Chaos testing for system resilience."""
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        # Empty bids
        empty_bids = pd.DataFrame()
        allocation, payments = vcg_allocate(empty_bids, 1000.0)
        assert allocation == {}
        assert payments == {}
        
        # Negative values
        negative_bids = pd.DataFrame({
            'inference': [-100, 200],
            'training': [150, -100],
            'cooling': [50, 30],
            'inference_bid': [45, 55],
            'training_bid': [50, 48],
            'cooling_bid': [35, 38]
        })
        
        # Should handle gracefully
        allocation, payments = vcg_allocate(negative_bids, 1000.0)
        assert isinstance(allocation, dict)
        assert isinstance(payments, dict)
    
    def test_extreme_market_conditions(self):
        """Test extreme market condition handling."""
        # Extreme price volatility
        extreme_signal = {
            'price': 1000.0,  # 20x normal price
            'frequency': 58.0,  # Major grid disturbance
            'volume': 0
        }
        
        payload = build_payload(
            {'inference': 400.0, 'training': 300.0, 'cooling': 100.0},
            {'power_total': 1000.0, 'battery_soc': 0.5},
            0.5, 150.0, 1000.0
        )
        
        adjusted = real_time_adjustment(payload, extreme_signal)
        
        # Should handle extreme conditions
        assert 'adjustment_factor' in adjusted
        assert adjusted['adjustment_factor'] > 0  # Still operational
    
    def test_system_recovery_drill(self):
        """Test system recovery from emergency state."""
        # Start in emergency state
        emergency_state = {
            'temperature': 78.0,  # Warning level
            'soc': 0.15,         # Low but not critical
            'power_limit': 1000.0,
            'total_power_kw': 950.0  # Near limit
        }
        
        response = emergency_response(emergency_state)
        initial_level = response['emergency_level']
        initial_reduction = response['power_reduction']
        
        # Simulate recovery (temperature decreases, power reduced)
        recovery_state = {
            'temperature': 72.0,  # Improved
            'soc': 0.25,         # Charging
            'power_limit': 1000.0,
            'total_power_kw': 700.0  # Reduced load
        }
        
        recovery_response = emergency_response(recovery_state)
        
        # Should show recovery
        assert recovery_response['emergency_level'] <= initial_level
        assert recovery_response['power_reduction'] <= initial_reduction
        assert recovery_response['estimated_recovery_time'] <= 60


class TestEndToEndIntegration:
    """End-to-end integration testing."""
    
    def test_full_pipeline_simulation(self):
        """Test complete A→B→C→D pipeline."""
        # Run multiple simulation cycles
        for cycle in range(3):
            try:
                result = main.main(simulate=True)
                assert result is not None
                
                # Verify payload structure
                assert 'power_requirements' in result
                assert 'constraints_satisfied' in result
                assert 'performance_metrics' in result
                
                # Verify performance
                if 'performance_metrics' in result:
                    build_time = result['performance_metrics'].get('build_time_ms', 0)
                    assert build_time < 100  # Performance gate
                
            except Exception as e:
                pytest.fail(f"Pipeline simulation failed on cycle {cycle}: {e}")
    
    def test_interface_contract_validation(self):
        """Test interface contracts between modules."""
        # Test VCG → Dispatch interface
        sample_bids = pd.DataFrame({
            'inference': [100, 200],
            'training': [150, 100],
            'cooling': [50, 30],
            'inference_bid': [45, 55],
            'training_bid': [50, 48],
            'cooling_bid': [35, 38]
        })
        
        allocation, payments = vcg_allocate(sample_bids, 1000.0)
        
        # Verify allocation contract
        assert isinstance(allocation, dict)
        for service in ['inference', 'training', 'cooling']:
            assert service in allocation
            assert isinstance(allocation[service], (int, float))
            assert allocation[service] >= 0
        
        # Test Dispatch → Execution interface
        inventory = {'power_total': 1000.0, 'battery_soc': 0.5}
        payload = build_payload(allocation, inventory, 0.5, 150.0, 1000.0)
        
        # Verify payload contract
        required_fields = [
            'allocation', 'power_requirements', 'constraints_satisfied',
            'system_state', 'performance_metrics'
        ]
        for field in required_fields:
            assert field in payload


# Performance regression tracking
class TestPerformanceRegression:
    """Track performance over time and detect regressions."""
    
    PERFORMANCE_BASELINES = {
        'vcg_auction_time_ms': 5.0,
        'build_payload_time_ms': 1.0,
        'adjustment_time_ms': 0.1,
        'emergency_response_time_ms': 0.1,
        'execution_engine_time_ms': 10.0
    }
    
    def test_performance_regression_gates(self):
        """Test that performance hasn't regressed beyond thresholds."""
        # VCG Auction
        bids = pd.DataFrame({
            'inference': [100, 200, 50],
            'training': [150, 100, 80],
            'cooling': [50, 30, 20],
            'inference_bid': [45, 55, 40],
            'training_bid': [50, 48, 42],
            'cooling_bid': [35, 38, 30]
        })
        
        start_time = time.perf_counter()
        vcg_allocate(bids, 1000.0)
        vcg_time = (time.perf_counter() - start_time) * 1000
        
        assert vcg_time < self.PERFORMANCE_BASELINES['vcg_auction_time_ms']
        
        # Dispatch Agent
        allocation = {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
        inventory = {'power_total': 1000.0, 'battery_soc': 0.5}
        
        start_time = time.perf_counter()
        payload = build_payload(allocation, inventory, 0.5, 150.0, 1000.0)
        build_time = (time.perf_counter() - start_time) * 1000
        
        assert build_time < self.PERFORMANCE_BASELINES['build_payload_time_ms']
        
        # Check payload performance metrics
        if 'performance_metrics' in payload:
            assert payload['performance_metrics']['build_time_ms'] < self.PERFORMANCE_BASELINES['build_payload_time_ms']


# Pytest configuration
@pytest.fixture(scope="session")
def benchmark_config():
    """Configure benchmark settings."""
    return {
        'min_rounds': 10,
        'max_time': 5.0,
        'warmup': True
    } 