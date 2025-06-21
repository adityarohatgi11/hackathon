#!/usr/bin/env python3
"""
Performance testing script for GridPilot-GT Lane C implementation.

This script can be run locally to validate performance gates before CI.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game_theory.vcg_auction import vcg_allocate
from dispatch.dispatch_agent import build_payload, real_time_adjustment, emergency_response
from dispatch.execution_engine import get_execution_engine, execute_dispatch
import main as main_module


class PerformanceTester:
    """Performance testing class with built-in gates."""
    
    # Performance thresholds (in milliseconds)
    PERFORMANCE_GATES = {
        'vcg_auction_max_ms': 10.0,
        'dispatch_build_max_ms': 5.0,
        'dispatch_adjustment_max_ms': 2.0,
        'emergency_response_max_ms': 1.0,
        'execution_engine_max_ms': 15.0,
        'end_to_end_max_ms': 1000.0,
    }
    
    def __init__(self):
        self.results = {}
        self.passed_gates = 0
        self.total_gates = len(self.PERFORMANCE_GATES)
    
    def run_benchmark(self, func, *args, runs=10, **kwargs) -> Tuple[float, float, List[float]]:
        """Run a function multiple times and return timing statistics."""
        times = []
        
        # Warmup run
        func(*args, **kwargs)
        
        # Actual benchmark runs
        for _ in range(runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
            times.append(elapsed)
        
        avg_time = float(np.mean(times))
        max_time = float(np.max(times))
        
        return avg_time, max_time, times
    
    def test_vcg_auction_performance(self) -> bool:
        """Test VCG auction performance."""
        print("🔄 Testing VCG Auction Performance...")
        
        # Create test data
        bids = pd.DataFrame({
            'inference': np.random.uniform(0.1, 0.5, 50),
            'training': np.random.uniform(0.1, 0.3, 50),
            'cooling': np.random.uniform(0.05, 0.15, 50),
            'inference_bid': np.random.uniform(40, 60, 50),
            'training_bid': np.random.uniform(45, 55, 50),
            'cooling_bid': np.random.uniform(30, 40, 50)
        })
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(vcg_allocate, bids, 1000.0, runs=20)
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['vcg_auction_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['vcg_auction'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def test_dispatch_build_performance(self) -> bool:
        """Test dispatch payload building performance."""
        print("\n🔄 Testing Dispatch Build Performance...")
        
        allocation = {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
        inventory = {'power_total': 1000.0, 'battery_soc': 0.65, 'gpu_utilization': 0.8}
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(
            build_payload, allocation, inventory, 0.65, 150.0, 1000.0, runs=50
        )
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['dispatch_build_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['dispatch_build'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def test_dispatch_adjustment_performance(self) -> bool:
        """Test real-time adjustment performance."""
        print("\n🔄 Testing Real-Time Adjustment Performance...")
        
        # Setup
        allocation = {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
        inventory = {'power_total': 1000.0, 'battery_soc': 0.65, 'gpu_utilization': 0.8}
        payload = build_payload(allocation, inventory, 0.65, 150.0, 1000.0)
        market_signal = {'price': 65.0, 'volume': 150, 'frequency': 59.7}
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(
            real_time_adjustment, payload, market_signal, runs=100
        )
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['dispatch_adjustment_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['dispatch_adjustment'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def test_emergency_response_performance(self) -> bool:
        """Test emergency response performance."""
        print("\n🔄 Testing Emergency Response Performance...")
        
        system_state = {
            'temperature': 78.0,
            'soc': 0.35,
            'power_limit': 1000.0,
            'total_power_kw': 850.0,
            'grid_frequency': 59.8,
            'cooling_capacity': 400.0
        }
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(
            emergency_response, system_state, runs=100
        )
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['emergency_response_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['emergency_response'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def test_execution_engine_performance(self) -> bool:
        """Test execution engine performance."""
        print("\n🔄 Testing Execution Engine Performance...")
        
        # Setup
        allocation = {'inference': 400.0, 'training': 300.0, 'cooling': 100.0}
        inventory = {'power_total': 1000.0, 'battery_soc': 0.65}
        payload = build_payload(allocation, inventory, 0.65, 150.0, 1000.0)
        
        # Simple test without iteration  
        def run_execution():
            engine = get_execution_engine()
            constraints = {'power_limit': 1000.0, 'max_temp': 80.0}
            return execute_dispatch(payload['allocation'], constraints)
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(run_execution, runs=20)
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['execution_engine_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['execution_engine'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def test_end_to_end_performance(self) -> bool:
        """Test end-to-end pipeline performance."""
        print("\n🔄 Testing End-to-End Performance...")
        
        # Run benchmark
        avg_time, max_time, times = self.run_benchmark(
            main_module.main, simulate=True, runs=10
        )
        
        # Check performance gate
        gate_limit = self.PERFORMANCE_GATES['end_to_end_max_ms']
        passed = avg_time < gate_limit
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Gate limit: {gate_limit}ms")
        print(f"  Status: {status}")
        
        self.results['end_to_end'] = {
            'avg_time_ms': avg_time,
            'max_time_ms': max_time,
            'gate_limit_ms': gate_limit,
            'passed': passed
        }
        
        if passed:
            self.passed_gates += 1
        
        return passed
    
    def run_all_tests(self) -> bool:
        """Run all performance tests."""
        print("🚀 GridPilot-GT Performance Test Suite")
        print("=" * 50)
        
        tests = [
            self.test_vcg_auction_performance,
            self.test_dispatch_build_performance,
            self.test_dispatch_adjustment_performance,
            self.test_emergency_response_performance,
            self.test_execution_engine_performance,
            self.test_end_to_end_performance,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Performance Test Summary")
        print("=" * 50)
        
        for component, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{component:20} | {result['avg_time_ms']:8.2f}ms | {status}")
        
        pass_rate = (self.passed_gates / self.total_gates) * 100
        overall_status = "✅ PASS" if self.passed_gates == self.total_gates else "❌ FAIL"
        
        print("-" * 50)
        print(f"Overall Result: {overall_status} ({self.passed_gates}/{self.total_gates} gates passed)")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.passed_gates == self.total_gates:
            print("\n🎉 All performance gates passed! System is ready for production.")
        else:
            print(f"\n⚠️  {self.total_gates - self.passed_gates} performance gate(s) failed. Please optimize before deployment.")
        
        return self.passed_gates == self.total_gates


def main():
    """Main function to run performance tests."""
    tester = PerformanceTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code for CI
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 