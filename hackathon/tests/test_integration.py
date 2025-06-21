"""Integration tests for GridPilot-GT end-to-end functionality.

Includes:
- End-to-end simulation testing
- Mocked API server testing
- CI performance gates
- Cross-module interface validation
"""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import requests
import threading
import http.server
import socketserver
import json
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import main
from api_client import get_prices, get_inventory, submit_bid
from game_theory.vcg_auction import vcg_allocate
from dispatch.dispatch_agent import build_payload


class MockAPIServer:
    """Mock API server for testing API interactions."""
    
    def __init__(self, port=8000):
        self.port = port
        self.server = None
        self.thread = None
        self.responses = {
            '/prices': {
                'status': 200,
                'data': [
                    {'timestamp': '2025-01-01T00:00:00', 'price': 50.0, 'volume': 100},
                    {'timestamp': '2025-01-01T01:00:00', 'price': 52.0, 'volume': 95}
                ]
            },
            '/inventory': {
                'status': 200,
                'data': {
                    'power_total': 1000.0,
                    'power_available': 750.0,
                    'battery_soc': 0.65,
                    'gpu_utilization': 0.8
                }
            },
            '/submit': {
                'status': 201,
                'data': {'order_id': 'test_order_123', 'status': 'accepted'}
            }
        }
    
    def set_response(self, endpoint: str, status: int, data: Any):
        """Set custom response for an endpoint."""
        self.responses[endpoint] = {'status': status, 'data': data}
    
    def start(self):
        """Start the mock server."""
        responses = self.responses  # Capture responses in closure
        
        class MockHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in responses:
                    response = responses[self.path]
                    self.send_response(response['status'])
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response['data']).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path in responses:
                    response = responses[self.path]
                    self.send_response(response['status'])
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response['data']).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        self.server = socketserver.TCPServer(("", self.port), MockHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)  # Give server time to start
    
    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()


class TestEndToEndIntegration:
    """Comprehensive end-to-end integration tests."""
    
    @pytest.fixture
    def mock_server(self):
        """Fixture to provide mock API server."""
        server = MockAPIServer(port=8001)
        server.start()
        yield server
        server.stop()
    
    def test_end_to_end_simulation_multiple_cycles(self):
        """Test multiple end-to-end simulation cycles."""
        results = []
        
        for cycle in range(5):
            start_time = time.perf_counter()
            result = main.main(simulate=True)
            cycle_time = (time.perf_counter() - start_time) * 1000
            
            # Verify successful execution
            assert result is not None, f"Cycle {cycle} failed"
            
            # Performance gate: each cycle < 5 seconds
            assert cycle_time < 5000, f"Cycle {cycle} took {cycle_time:.2f}ms, exceeding 5s limit"
            
            # Verify result structure
            required_fields = ['power_requirements', 'constraints_satisfied', 'system_state']
            for field in required_fields:
                assert field in result, f"Missing field {field} in cycle {cycle}"
            
            # Verify constraints
            assert result['constraints_satisfied'], f"Constraints violated in cycle {cycle}"
            
            # Verify power limits
            total_power = result['power_requirements']['total_power_kw']
            assert total_power <= 1000.0, f"Power limit exceeded in cycle {cycle}: {total_power} kW"
            
            results.append({
                'cycle': cycle,
                'time_ms': cycle_time,
                'total_power': total_power,
                'utilization': result['system_state']['utilization']
            })
        
        # Verify consistency across cycles (lenient for market-responsive system)
        power_values = [r['total_power'] for r in results]
        power_std = np.std(power_values)
        # More lenient threshold for dynamic market-responsive system
        assert power_std < 200, f"Power allocation too variable across cycles: std={power_std:.2f}"
    
    @patch('api_client.client.requests.get')
    @patch('api_client.client.requests.post')
    def test_api_error_handling(self, mock_post, mock_get):
        """Test graceful handling of API errors."""
        # Test 500 server error
        mock_get.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        # Should handle gracefully in simulation mode
        result = main.main(simulate=True)
        assert result is not None  # Should still work with fallback data
    
    @patch('api_client.client.requests.get')
    def test_api_timeout_handling(self, mock_get):
        """Test handling of API timeouts."""
        # Simulate timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        # Should handle gracefully
        result = main.main(simulate=True)
        assert result is not None
    
    def test_interface_contract_validation(self):
        """Validate interfaces between modules match specifications."""
        # Test A→B interface (prices → bidding)
        sample_prices = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=24, freq='h'),
            'price': np.random.uniform(40, 60, 24),
            'volume': np.random.uniform(80, 120, 24)
        })
        
        # Should have required columns
        required_price_cols = ['timestamp', 'price', 'volume']
        for col in required_price_cols:
            assert col in sample_prices.columns
        
        # Test B→C interface (bids → auction)
        sample_bids = pd.DataFrame({
            'inference': [0.1, 0.3, 0.2],
            'training': [0.2, 0.1, 0.15],
            'cooling': [0.05, 0.08, 0.06],
            'inference_bid': [45, 55, 50],
            'training_bid': [50, 48, 47],
            'cooling_bid': [35, 38, 36]
        })
        
        allocation, payments = vcg_allocate(sample_bids, 1000.0)
        
        # Validate allocation interface
        assert isinstance(allocation, dict)
        assert set(allocation.keys()) == {'inference', 'training', 'cooling'}
        for service, value in allocation.items():
            assert isinstance(value, (int, float))
            assert value >= 0
        
        # Test C→D interface (allocation → dispatch)
        inventory = {
            'power_total': 1000.0,
            'power_available': 750.0,
            'battery_soc': 0.65,
            'gpu_utilization': 0.8
        }
        
        payload = build_payload(allocation, inventory, 0.65, 150.0, 1000.0)
        
        # Validate payload interface
        required_payload_fields = [
            'allocation', 'power_requirements', 'constraints_satisfied',
            'system_state', 'market_data', 'performance_metrics'
        ]
        for field in required_payload_fields:
            assert field in payload, f"Missing payload field: {field}"


class TestPerformanceGates:
    """CI performance gates to prevent regression."""
    
    # Performance thresholds (in milliseconds)
    PERFORMANCE_GATES = {
        'vcg_auction_max_ms': 10.0,
        'dispatch_build_max_ms': 5.0,
        'dispatch_adjustment_max_ms': 2.0,
        'emergency_response_max_ms': 1.0,
        'end_to_end_max_ms': 1000.0,  # 1 second for full pipeline
    }
    
    @pytest.mark.performance
    def test_vcg_auction_performance_gate(self):
        """CI gate: VCG auction must complete within time limit."""
        bids = pd.DataFrame({
            'inference': np.random.uniform(0.1, 0.5, 50),  # Larger dataset
            'training': np.random.uniform(0.1, 0.3, 50),
            'cooling': np.random.uniform(0.05, 0.15, 50),
            'inference_bid': np.random.uniform(40, 60, 50),
            'training_bid': np.random.uniform(45, 55, 50),
            'cooling_bid': np.random.uniform(30, 40, 50)
        })
        
        # Measure performance
        times = []
        for _ in range(10):  # Multiple runs for stability
            start_time = time.perf_counter()
            allocation, payments = vcg_allocate(bids, 1000.0)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
            
            # Verify correctness
            assert sum(allocation.values()) <= 1000.0
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Performance gates
        assert avg_time < self.PERFORMANCE_GATES['vcg_auction_max_ms'], \
            f"VCG auction average time {avg_time:.2f}ms exceeds limit {self.PERFORMANCE_GATES['vcg_auction_max_ms']}ms"
        
        assert max_time < self.PERFORMANCE_GATES['vcg_auction_max_ms'] * 2, \
            f"VCG auction max time {max_time:.2f}ms exceeds limit {self.PERFORMANCE_GATES['vcg_auction_max_ms'] * 2}ms"
    
    @pytest.mark.performance
    def test_dispatch_performance_gates(self):
        """CI gate: Dispatch functions must meet performance requirements."""
        allocation = {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
        inventory = {'power_total': 1000.0, 'battery_soc': 0.65, 'gpu_utilization': 0.8}
        
        # Test build_payload performance
        build_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            payload = build_payload(allocation, inventory, 0.65, 150.0, 1000.0)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            build_times.append(elapsed_ms)
        
        avg_build_time = np.mean(build_times)
        assert avg_build_time < self.PERFORMANCE_GATES['dispatch_build_max_ms'], \
            f"Build payload average time {avg_build_time:.2f}ms exceeds limit {self.PERFORMANCE_GATES['dispatch_build_max_ms']}ms"
    
    @pytest.mark.performance  
    def test_end_to_end_performance_gate(self):
        """CI gate: Full end-to-end pipeline must complete within time limit."""
        times = []
        
        for _ in range(5):  # Multiple runs
            start_time = time.perf_counter()
            result = main.main(simulate=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            times.append(elapsed_ms)
            
            assert result is not None
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        assert avg_time < self.PERFORMANCE_GATES['end_to_end_max_ms'], \
            f"End-to-end average time {avg_time:.2f}ms exceeds limit {self.PERFORMANCE_GATES['end_to_end_max_ms']}ms"


class TestChaosEngineering:
    """Chaos engineering tests for system resilience."""
    
    def test_random_input_chaos(self):
        """Test system resilience with random/malformed inputs."""
        for _ in range(20):  # Multiple chaos runs
            # Generate random/invalid data
            chaos_bids = pd.DataFrame({
                'inference': np.random.uniform(-100, 1000, np.random.randint(0, 10)),
                'training': np.random.uniform(-50, 500, np.random.randint(0, 10)),
                'cooling': np.random.uniform(-25, 250, np.random.randint(0, 10)),
                'inference_bid': np.random.uniform(0, 200, np.random.randint(0, 10)),
                'training_bid': np.random.uniform(0, 150, np.random.randint(0, 10)),
                'cooling_bid': np.random.uniform(0, 100, np.random.randint(0, 10))
            })
            
            try:
                # Should handle gracefully
                allocation, payments = vcg_allocate(chaos_bids, np.random.uniform(100, 2000))
                
                # Basic sanity checks
                assert isinstance(allocation, dict)
                assert isinstance(payments, dict)
                assert all(v >= 0 for v in allocation.values())  # Non-negative
                
            except Exception as e:
                # If it fails, should be a handled exception, not a crash
                assert not isinstance(e, (KeyError, AttributeError, IndexError)), \
                    f"Unhandled exception type: {type(e).__name__}: {e}"
    
    def test_resource_exhaustion_simulation(self):
        """Test behavior under resource exhaustion."""
        # Simulate very limited resources
        limited_inventory = {
            'power_total': 100.0,      # Very limited power
            'power_available': 50.0,    # Even less available
            'battery_soc': 0.05,       # Critical battery
            'gpu_utilization': 0.95     # Near capacity
        }
        
        high_demand_allocation = {
            'inference': 500.0,  # Much higher than available
            'training': 400.0,
            'cooling': 200.0
        }
        
        # Should handle gracefully
        payload = build_payload(high_demand_allocation, limited_inventory, 0.05, 200.0, 100.0)
        
        # Should scale down appropriately
        assert payload['system_state']['emergency_scaled']
        assert payload['power_requirements']['total_power_kw'] <= 100.0
        assert payload['constraints_satisfied']
    
    def test_concurrent_access_chaos(self):
        """Test system behavior under concurrent access."""
        def run_simulation():
            try:
                return main.main(simulate=True)
            except Exception:
                return None
        
        # Run multiple simulations concurrently
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_simulation) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        successful = [r for r in results if r is not None]
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% threshold"


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.integration,  # Mark all tests as integration tests
] 