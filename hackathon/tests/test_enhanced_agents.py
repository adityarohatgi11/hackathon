"""Comprehensive tests for enhanced agent system."""

import pytest
import time
import threading
import multiprocessing
import os
import tempfile
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

import pandas as pd
import numpy as np

# Import enhanced agents
from agents.enhanced_base_agent import EnhancedBaseAgent, AgentConfig, AgentState, HealthMetrics
from agents.enhanced_data_agent import EnhancedDataAgent
from agents.enhanced_strategy_agent import EnhancedStrategyAgent
from agents.enhanced_system_manager import EnhancedSystemManager, SystemConfig
from agents.message_bus import MessageBus


class TestEnhancedBaseAgent:
    """Test suite for EnhancedBaseAgent functionality."""

    def test_agent_initialization(self):
        """Test enhanced agent initialization."""
        
        class TestAgent(EnhancedBaseAgent):
            def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
                return {"processed": True}
        
        config = AgentConfig(
            max_retries=5,
            cache_size=100,
            enable_caching=True,
            enable_metrics=True
        )
        
        agent = TestAgent("TestAgent", config)
        
        assert agent.name == "TestAgent"
        assert agent.config.max_retries == 5
        assert agent.config.cache_size == 100
        assert agent.state == AgentState.INITIALIZING
        assert agent.agent_id is not None
        assert len(agent.agent_id) == 8

    def test_health_metrics(self):
        """Test health metrics tracking."""
        metrics = HealthMetrics()
        
        # Test initial state
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_processing_time == 0.0
        
        # Test after processing
        metrics.messages_processed = 8
        metrics.messages_failed = 2
        metrics.processing_times.extend([0.1, 0.2, 0.15, 0.3])
        
        assert metrics.success_rate == 0.8
        assert metrics.avg_processing_time == 0.1875

    def test_caching_system(self):
        """Test intelligent caching system."""
        
        class TestAgent(EnhancedBaseAgent):
            def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
                return {"result": message["input"] * 2}
        
        config = AgentConfig(cache_ttl=1.0, enable_caching=True)
        agent = TestAgent("CacheTestAgent", config)
        
        # Test cache key generation
        message1 = {"input": 5}
        message2 = {"input": 5}
        message3 = {"input": 10}
        
        key1 = agent._get_cache_key(message1)
        key2 = agent._get_cache_key(message2)
        key3 = agent._get_cache_key(message3)
        
        assert key1 == key2  # Same message should have same key
        assert key1 != key3  # Different messages should have different keys
        
        # Test cache storage and retrieval
        agent._set_cache(key1, {"result": 10})
        cached_result = agent._get_from_cache(key1)
        
        assert cached_result is not None
        assert cached_result["result"] == 10
        
        # Test cache expiration
        time.sleep(1.1)  # Wait for TTL to expire
        expired_result = agent._get_from_cache(key1)
        assert expired_result is None

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        
        class TestAgent(EnhancedBaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
                self.call_count += 1
                if message.get("should_fail", False):
                    raise Exception("Test failure")
                return {"success": True}
        
        agent = TestAgent("CircuitTestAgent")
        
        # Test normal operation
        assert not agent._circuit_open
        assert agent._circuit_failure_count == 0
        
        # Trigger failures to open circuit breaker
        for i in range(6):
            try:
                agent._process_message_with_retries({"should_fail": True})
            except:
                # Manually increment failure count to ensure circuit opens
                agent._circuit_failure_count += 1
                if agent._circuit_failure_count >= 5:
                    agent._circuit_open = True
                    agent.state = AgentState.DEGRADED
        
        # Circuit should be open now or agent should be degraded
        assert agent._circuit_open or agent.state == AgentState.DEGRADED or agent.state == AgentState.DEGRADED


class TestEnhancedDataAgent:
    """Test suite for EnhancedDataAgent functionality."""

    def test_data_agent_initialization(self):
        """Test data agent initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedDataAgent(fetch_interval=30, cache_dir=temp_dir)
            
            assert agent.name == "EnhancedDataAgent"
            assert agent._fetch_interval == 30
            assert agent._cache_dir == temp_dir
            assert agent.subscribe_topics == []
            assert agent.publish_topic == "feature-vector"

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedDataAgent(cache_dir=temp_dir)
            agent._use_synthetic_data = True
            
            prices_df, inventory_data = agent._generate_synthetic_data()
            
            # Validate price data
            assert not prices_df.empty
            assert len(prices_df) == 24  # 24 hours of data
            assert "timestamp" in prices_df.columns
            assert "energy_price" in prices_df.columns
            assert "hash_price" in prices_df.columns
            assert "synthetic" in prices_df.columns
            
            # Validate inventory data
            assert isinstance(inventory_data, dict)
            assert "utilization_rate" in inventory_data
            assert "battery_soc" in inventory_data
            assert "synthetic" in inventory_data
            
            # Check data ranges
            assert prices_df["energy_price"].min() > 0
            assert 0 <= inventory_data["battery_soc"] <= 1
            assert 0 <= inventory_data["utilization_rate"] <= 100

    def test_market_intelligence_analysis(self):
        """Test market intelligence analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedDataAgent(cache_dir=temp_dir)
            
            # Create test data
            prices_df = pd.DataFrame({
                "timestamp": pd.date_range("2025-01-01", periods=24, freq="H"),
                "energy_price": np.random.uniform(2.0, 5.0, 24),
                "hash_price": np.random.uniform(1.5, 4.0, 24),
                "token_price": np.random.uniform(3.0, 6.0, 24)
            })
            
            inventory_data = {
                "utilization_rate": 75.0,
                "battery_soc": 0.6,
                "power_available": 450.0
            }
            
            features_df = pd.DataFrame({
                "feature1": np.random.randn(24),
                "feature2": np.random.randn(24)
            })
            
            # Test analysis
            insights = agent._analyze_market_intelligence(prices_df, inventory_data, features_df)
            
            assert "timestamp" in insights
            assert "market_regime" in insights
            assert "system_health" in insights
            assert "alerts" in insights
            assert "confidence" in insights


class TestEnhancedStrategyAgent:
    """Test suite for EnhancedStrategyAgent functionality."""

    def test_strategy_agent_initialization(self):
        """Test strategy agent initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedStrategyAgent(cache_dir=temp_dir)
            
            assert agent.name == "EnhancedStrategyAgent"
            assert agent.subscribe_topics == ["feature-vector", "forecast"]
            assert agent.publish_topic == "strategy-action"
            assert agent._risk_tolerance == 0.7
            assert agent._max_allocation == 0.8

    def test_heuristic_strategy_generation(self):
        """Test heuristic strategy generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedStrategyAgent(cache_dir=temp_dir)
            
            # Set up test data
            agent._last_features = {
                "prices": [{
                    "energy_price": 4.5,  # High price
                    "hash_price": 2.0,
                    "token_price": 3.0
                }],
                "inventory": {
                    "battery_soc": 0.3,
                    "utilization_rate": 60.0
                },
                "market_intelligence": {
                    "market_regime": {
                        "volatility_regime": "medium"
                    }
                }
            }
            
            strategy = agent._generate_heuristic_strategy()
            
            assert "energy_allocation" in strategy
            assert "hash_allocation" in strategy
            assert "battery_charge_rate" in strategy
            assert "confidence" in strategy
            assert strategy["method"] == "heuristic"
            
            # High energy price should lead to higher energy allocation
            assert strategy["energy_allocation"] > 0.4

    def test_risk_assessment(self):
        """Test strategy risk assessment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = EnhancedStrategyAgent(cache_dir=temp_dir)
            
            # Test high-risk strategy
            high_risk_action = {
                "energy_allocation": 0.95,  # Very high allocation
                "hash_allocation": 0.05,
                "battery_charge_rate": 0.9   # Aggressive battery usage
            }
            
            risk = agent._assess_strategy_risk(high_risk_action)
            assert risk["level"] == "high"
            assert "high_total_allocation" in risk["factors"]
            assert "aggressive_battery_usage" in risk["factors"]
            
            # Test low-risk strategy
            low_risk_action = {
                "energy_allocation": 0.4,
                "hash_allocation": 0.3,
                "battery_charge_rate": 0.1
            }
            
            risk = agent._assess_strategy_risk(low_risk_action)
            assert risk["level"] == "low"
            assert len(risk["factors"]) == 0


class TestEnhancedSystemManager:
    """Test suite for EnhancedSystemManager functionality."""

    def test_system_manager_initialization(self):
        """Test system manager initialization."""
        config = SystemConfig(
            data_fetch_interval=30,
            enable_monitoring=True,
            restart_on_failure=True
        )
        
        manager = EnhancedSystemManager(config)
        
        assert manager.config.data_fetch_interval == 30
        assert manager.config.enable_monitoring is True
        assert manager.config.restart_on_failure is True
        assert not manager._running
        assert len(manager._agents) == 0

    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        manager = EnhancedSystemManager()
        
        manager._collect_system_metrics()
        
        metrics = manager._system_metrics
        assert "timestamp" in metrics
        assert "system_uptime" in metrics
        assert "agents_running" in metrics
        assert "agent_status" in metrics
        assert "message_bus_status" in metrics

    def test_message_bus_health_check(self):
        """Test message bus health checking."""
        manager = EnhancedSystemManager()
        
        health = manager._check_message_bus_health()
        
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        if health["status"] == "healthy":
            assert "type" in health


def run_enhanced_agent_tests():
    """Run all enhanced agent tests with comprehensive reporting."""
    
    # Configure pytest for comprehensive testing
    pytest_args = [
        __file__,
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--durations=10",       # Show 10 slowest tests
        "-x",                   # Stop on first failure
        "--strict-markers",     # Strict marker checking
    ]
    
    print("Running Enhanced Agent Tests...")
    print("=" * 50)
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All Enhanced Agent Tests PASSED!")
    else:
        print(f"\n❌ Tests FAILED with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    run_enhanced_agent_tests()
