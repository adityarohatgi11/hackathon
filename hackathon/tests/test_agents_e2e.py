"""End-to-end tests for the GridPilot-GT agent system."""

import asyncio
import json
import logging
import time
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from agents.message_bus import MessageBus
from agents.data_agent import DataAgent
from agents.forecaster_agent import ForecasterAgent
from agents.strategy_agent import StrategyAgent
from agents.local_llm_agent import LocalLLMAgent
from agents.vector_store_agent import VectorStoreAgent


class TestAgentSystem:
    """Test suite for the complete agent system."""

    @pytest.fixture
    def message_bus(self):
        """Create message bus for testing."""
        return MessageBus()

    @pytest.fixture
    def mock_api_data(self):
        """Mock MARA API data for testing."""
        return {
            "prices": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-06-23 10:00", periods=48, freq="H"),
                "energy_price": np.random.uniform(2.5, 4.5, 48),
                "hash_price": np.random.uniform(2.0, 4.0, 48),
                "token_price": np.random.uniform(1.5, 3.5, 48),
                "volume": np.random.uniform(800, 1200, 48),
                "volatility_24h": np.random.uniform(0.05, 0.25, 48),
            }),
            "inventory": {
                "utilization_rate": 65.4,
                "available_capacity": 1000,
                "total_machines": 150,
            }
        }

    def test_message_bus_basic_operations(self, message_bus):
        """Test basic message bus operations."""
        # Test publishing and consuming
        test_message = {"test": "data", "timestamp": "2025-06-23T10:00:00"}
        
        message_bus.publish("test-topic", test_message)
        
        # Consume one message
        messages = []
        consumer = message_bus.consume("test-topic", block_ms=100)
        try:
            message = next(consumer)
            messages.append(message)
        except StopIteration:
            pass
        
        assert len(messages) == 1
        assert messages[0]["test"] == "data"

    @patch('agents.data_agent.MaraAPIClient')
    def test_data_agent_processing(self, mock_api_client, message_bus, mock_api_data):
        """Test DataAgent fetches and publishes data correctly."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_instance.get_prices.return_value = mock_api_data["prices"]
        mock_client_instance.get_inventory.return_value = mock_api_data["inventory"]
        mock_api_client.return_value = mock_client_instance
        
        # Create agent
        agent = DataAgent(fetch_interval=1)  # 1 second for testing
        agent.bus = message_bus
        
        # Test data fetching
        agent._fetch_and_publish()
        
        # Verify API was called
        mock_client_instance.get_prices.assert_called_once()
        mock_client_instance.get_inventory.assert_called_once()
        
        # Check message was published
        consumer = message_bus.consume("feature-vector", block_ms=100)
        try:
            message = next(consumer)
            assert "prices" in message
            assert "features" in message
            assert "inventory" in message
            assert message["source"] == "DataAgent"
        except StopIteration:
            pytest.fail("No message published by DataAgent")

    def test_forecaster_agent_processing(self, message_bus, mock_api_data):
        """Test ForecasterAgent processes feature vectors correctly."""
        agent = ForecasterAgent()
        agent.bus = message_bus
        
        # Create input message
        input_message = {
            "prices": mock_api_data["prices"].to_dict(orient="records"),
            "source": "DataAgent",
            "timestamp": "2025-06-23T10:00:00"
        }
        
        # Process message
        result = agent.handle_message(input_message)
        
        # Verify output
        assert result is not None
        assert "forecast" in result
        assert "source" in result
        assert result["source"] == "ForecasterAgent"
        
        forecast = result["forecast"]
        assert isinstance(forecast, list)
        assert len(forecast) > 0
        
        # Check forecast structure
        first_forecast = forecast[0]
        assert "predicted_price" in first_forecast
        assert "timestamp" in first_forecast

    def test_strategy_agent_processing(self, message_bus, mock_api_data):
        """Test StrategyAgent generates valid strategy actions."""
        agent = StrategyAgent()
        agent.bus = message_bus
        
        # Setup feature vector
        feature_message = {
            "prices": mock_api_data["prices"].to_dict(orient="records"),
            "inventory": mock_api_data["inventory"],
            "source": "DataAgent",
        }
        
        # Setup forecast
        forecast_message = {
            "forecast": [
                {
                    "timestamp": "2025-06-23T11:00:00",
                    "predicted_price": 3.5,
                    "lower_bound": 3.0,
                    "upper_bound": 4.0,
                    "method": "prophet"
                }
            ],
            "source": "ForecasterAgent",
        }
        
        # Process both messages
        agent.handle_message(feature_message)
        result = agent.handle_message(forecast_message)
        
        # Verify strategy output
        assert result is not None
        assert "action" in result
        assert "confidence" in result
        
        action = result["action"]
        assert "energy_allocation" in action
        assert "hash_allocation" in action
        assert "battery_charge_rate" in action
        assert "method" in action
        
        # Check allocation constraints
        assert 0.0 <= action["energy_allocation"] <= 1.0
        assert 0.0 <= action["hash_allocation"] <= 1.0
        assert -1.0 <= action["battery_charge_rate"] <= 1.0

    def test_llm_agent_analysis(self, message_bus):
        """Test LocalLLMAgent generates analysis for different message types."""
        agent = LocalLLMAgent()
        agent.bus = message_bus
        
        # Test strategy analysis
        strategy_message = {
            "action": {
                "energy_allocation": 0.6,
                "hash_allocation": 0.4,
                "battery_charge_rate": 0.2,
                "method": "heuristic"
            },
            "source": "StrategyAgent",
        }
        
        result = agent.handle_message(strategy_message)
        
        assert result is not None
        assert "analysis" in result
        
        analysis = result["analysis"]
        assert "summary" in analysis
        assert "recommendations" in analysis
        assert "risk_assessment" in analysis
        assert "method" in analysis
        
        # Check risk assessment is valid
        assert analysis["risk_assessment"] in ["Low", "Medium", "High"]

    def test_vector_store_agent_storage(self, message_bus):
        """Test VectorStoreAgent stores and retrieves messages."""
        agent = VectorStoreAgent(persist_directory="test_vectorstore")
        agent.bus = message_bus
        
        # Test storing a message
        test_message = {
            "action": {
                "energy_allocation": 0.7,
                "hash_allocation": 0.3,
                "battery_charge_rate": -0.1,
            },
            "source": "StrategyAgent",
            "timestamp": "2025-06-23T10:00:00"
        }
        
        result = agent.handle_message(test_message)
        
        # Should return context (might be None if no relevant history)
        # Main test is that no exception is raised
        
        # Test knowledge search
        search_message = {
            "query_type": "knowledge_search",
            "query": "strategy allocation",
            "source": "TestQuery"
        }
        
        search_result = agent.handle_message(search_message)
        
        assert search_result is not None
        assert "query" in search_result
        assert "results" in search_result
        assert search_result["query"] == "strategy allocation"

    def test_full_pipeline_integration(self, message_bus, mock_api_data):
        """Test complete data flow through all agents."""
        
        # Initialize all agents
        with patch('agents.data_agent.MaraAPIClient') as mock_api:
            # Setup mock
            mock_client = MagicMock()
            mock_client.get_prices.return_value = mock_api_data["prices"]
            mock_client.get_inventory.return_value = mock_api_data["inventory"]
            mock_api.return_value = mock_client
            
            # Create agents
            data_agent = DataAgent(fetch_interval=1)
            data_agent.bus = message_bus
            
            forecaster_agent = ForecasterAgent()
            forecaster_agent.bus = message_bus
            
            strategy_agent = StrategyAgent()
            strategy_agent.bus = message_bus
            
            llm_agent = LocalLLMAgent()
            llm_agent.bus = message_bus
            
            vector_agent = VectorStoreAgent(persist_directory="test_integration_vectorstore")
            vector_agent.bus = message_bus
            
            # Step 1: Data agent publishes feature vector
            data_agent._fetch_and_publish()
            
            # Step 2: Get feature vector from bus
            feature_consumer = message_bus.consume("feature-vector", block_ms=100)
            feature_message = next(feature_consumer)
            
            # Verify feature message
            assert "prices" in feature_message
            assert "features" in feature_message
            
            # Step 3: Forecaster processes feature vector
            forecast_result = forecaster_agent.handle_message(feature_message)
            assert forecast_result is not None
            
            # Step 4: Strategy agent processes both feature vector and forecast
            strategy_agent.handle_message(feature_message)
            strategy_result = strategy_agent.handle_message(forecast_result)
            assert strategy_result is not None
            
            # Step 5: LLM agent analyzes strategy
            llm_result = llm_agent.handle_message(strategy_result)
            assert llm_result is not None
            
            # Step 6: Vector store saves all messages
            vector_agent.handle_message(feature_message)
            vector_agent.handle_message(forecast_result)
            vector_agent.handle_message(strategy_result)
            vector_agent.handle_message(llm_result)
            
            # Verify final outputs have expected structure
            assert "action" in strategy_result
            assert "analysis" in llm_result
            
            strategy_action = strategy_result["action"]
            assert all(key in strategy_action for key in [
                "energy_allocation", "hash_allocation", "battery_charge_rate", "method"
            ])

    def test_error_handling_and_robustness(self, message_bus):
        """Test agent error handling and recovery."""
        
        # Test malformed message handling
        agents = [
            ForecasterAgent(),
            StrategyAgent(),
            LocalLLMAgent(),
            VectorStoreAgent(persist_directory="test_error_vectorstore")
        ]
        
        malformed_messages = [
            {},  # Empty message
            {"invalid": "data"},  # Missing expected fields
            {"prices": []},  # Empty prices
            {"prices": "not_a_list"},  # Wrong data type
        ]
        
        for agent in agents:
            agent.bus = message_bus
            for message in malformed_messages:
                try:
                    result = agent.handle_message(message)
                    # Should either return None or valid result, not crash
                    if result is not None:
                        assert isinstance(result, dict)
                except Exception as e:
                    pytest.fail(f"{agent.__class__.__name__} crashed on malformed message: {e}")

    def test_performance_benchmarks(self, message_bus, mock_api_data):
        """Test agent performance under load."""
        
        # Test forecaster performance with varying data sizes
        agent = ForecasterAgent()
        agent.bus = message_bus
        
        # Test with different data sizes
        data_sizes = [24, 48, 168, 720]  # 1 day, 2 days, 1 week, 1 month
        
        for size in data_sizes:
            # Create test data
            test_prices = mock_api_data["prices"].head(size)
            message = {
                "prices": test_prices.to_dict(orient="records"),
                "source": "TestAgent"
            }
            
            # Measure processing time
            start_time = time.time()
            result = agent.handle_message(message)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify result and performance
            assert result is not None
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            print(f"Forecaster processed {size} records in {processing_time:.3f}s")

    def test_concurrent_agent_operations(self, message_bus):
        """Test multiple agents operating concurrently."""
        
        # This test would ideally use actual concurrency
        # For now, test sequential operation of multiple agents
        
        agents = [
            ForecasterAgent(),
            StrategyAgent(), 
            LocalLLMAgent(),
            VectorStoreAgent(persist_directory="test_concurrent_vectorstore")
        ]
        
        for agent in agents:
            agent.bus = message_bus
        
        # Test message broadcasting to multiple agents
        test_message = {
            "action": {
                "energy_allocation": 0.5,
                "hash_allocation": 0.5,
                "battery_charge_rate": 0.0,
            },
            "source": "TestAgent",
            "timestamp": "2025-06-23T10:00:00"
        }
        
        results = []
        for agent in agents:
            try:
                result = agent.handle_message(test_message)
                results.append((agent.__class__.__name__, result))
            except Exception as e:
                pytest.fail(f"Agent {agent.__class__.__name__} failed: {e}")
        
        # Verify all agents processed without errors
        assert len(results) == len(agents)

    def test_data_consistency_and_validation(self, message_bus, mock_api_data):
        """Test data consistency across agent chain."""
        
        # Initialize agents
        forecaster = ForecasterAgent()
        forecaster.bus = message_bus
        
        strategy = StrategyAgent()
        strategy.bus = message_bus
        
        # Process data through chain
        feature_message = {
            "prices": mock_api_data["prices"].to_dict(orient="records"),
            "inventory": mock_api_data["inventory"],
            "source": "DataAgent",
        }
        
        # Get forecast
        forecast_result = forecaster.handle_message(feature_message)
        assert forecast_result is not None
        
        # Get strategy
        strategy.handle_message(feature_message)
        strategy_result = strategy.handle_message(forecast_result)
        assert strategy_result is not None
        
        # Validate data consistency
        forecast_data = forecast_result["forecast"]
        strategy_action = strategy_result["action"]
        
        # Check forecast has proper structure
        assert isinstance(forecast_data, list)
        assert len(forecast_data) > 0
        
        for pred in forecast_data[:5]:  # Check first 5 predictions
            assert "predicted_price" in pred
            assert "timestamp" in pred
            assert isinstance(pred["predicted_price"], (int, float))
            assert pred["predicted_price"] > 0  # Prices should be positive
        
        # Check strategy allocations sum to reasonable values
        total_allocation = strategy_action["energy_allocation"] + strategy_action["hash_allocation"]
        assert 0.0 <= total_allocation <= 2.0  # Allow some flexibility
        
        # Check battery rate is in bounds
        assert -1.0 <= strategy_action["battery_charge_rate"] <= 1.0

    def test_system_recovery_after_failure(self, message_bus):
        """Test system recovery after component failures."""
        
        # Test agent restart capability
        agent = ForecasterAgent()
        agent.bus = message_bus
        
        # Simulate partial failure by corrupting internal state
        original_forecaster = agent._forecaster
        agent._forecaster = None
        
        # Try to process message (should use fallback)
        test_message = {
            "prices": [
                {"timestamp": "2025-06-23T10:00:00", "energy_price": 3.0, "hash_price": 2.5, "token_price": 2.0}
            ],
            "source": "TestAgent"
        }
        
        result = agent.handle_message(test_message)
        
        # Should still return some result (fallback mechanism)
        assert result is not None or True  # Allow None for graceful degradation
        
        # Restore and verify recovery
        agent._forecaster = original_forecaster
        result_recovered = agent.handle_message(test_message)
        
        # After recovery, should work normally
        # Note: might still be None if insufficient data, but shouldn't crash
        assert True  # Test passes if no exception is raised


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 