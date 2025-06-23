#!/usr/bin/env python3
"""Comprehensive test script for validating the GridPilot-GT agent system."""

import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from agents.message_bus import MessageBus
from agents.data_agent import DataAgent
from agents.forecaster_agent import ForecasterAgent
from agents.strategy_agent import StrategyAgent
from agents.local_llm_agent import LocalLLMAgent
from agents.vector_store_agent import VectorStoreAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSystemTester:
    """Comprehensive tester for the agent system."""
    
    def __init__(self):
        self.bus = MessageBus()
        self.test_results = {}
        self.start_time = time.time()
        
    def generate_mock_data(self, hours: int = 48) -> Dict[str, Any]:
        """Generate realistic mock data for testing."""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            periods=hours,
            freq="H"
        )
        
        # Generate correlated price data with realistic patterns
        base_energy = 3.0
        base_hash = 2.5
        base_token = 2.0
        
        # Add daily seasonality and random walk
        hourly_pattern = np.sin(2 * np.pi * np.arange(hours) / 24) * 0.5
        random_walk_energy = np.cumsum(np.random.normal(0, 0.1, hours))
        random_walk_hash = np.cumsum(np.random.normal(0, 0.08, hours))
        random_walk_token = np.cumsum(np.random.normal(0, 0.12, hours))
        
        energy_prices = base_energy + hourly_pattern + random_walk_energy * 0.3
        hash_prices = base_hash + hourly_pattern * 0.7 + random_walk_hash * 0.25
        token_prices = base_token + hourly_pattern * 0.3 + random_walk_token * 0.4
        
        # Ensure positive prices
        energy_prices = np.maximum(energy_prices, 0.5)
        hash_prices = np.maximum(hash_prices, 0.5)
        token_prices = np.maximum(token_prices, 0.5)
        
        return {
            "prices": pd.DataFrame({
                "timestamp": timestamps,
                "price": energy_prices,  # Main price column expected by feature engineering
                "energy_price": energy_prices,
                "hash_price": hash_prices,
                "token_price": token_prices,
                "volume": np.random.uniform(800, 1200, hours),
                "volatility_24h": np.random.uniform(0.05, 0.3, hours),
            }),
            "inventory": {
                "utilization_rate": np.random.uniform(50, 80),
                "power_used": np.random.uniform(400, 700),
                "power_available": np.random.uniform(300, 600),
                "available_capacity": 1000,
                "total_machines": 150,
                "battery_soc": np.random.uniform(20, 80),
            }
        }
    
    def test_individual_agents(self) -> Dict[str, bool]:
        """Test each agent individually."""
        logger.info("🧪 Testing individual agents...")
        results = {}
        
        mock_data = self.generate_mock_data()
        
        # Test ForecasterAgent
        try:
            forecaster = ForecasterAgent()
            forecaster.bus = self.bus
            
            input_message = {
                "prices": mock_data["prices"].to_dict(orient="records"),
                "source": "TestDataAgent",
                "timestamp": datetime.now().isoformat()
            }
            
            forecast_result = forecaster.handle_message(input_message)
            
            if forecast_result and "forecast" in forecast_result:
                forecast_data = forecast_result["forecast"]
                if isinstance(forecast_data, list) and len(forecast_data) > 0:
                    results["ForecasterAgent"] = True
                    logger.info("✅ ForecasterAgent: PASS")
                else:
                    results["ForecasterAgent"] = False
                    logger.error("❌ ForecasterAgent: Empty forecast")
            else:
                results["ForecasterAgent"] = False
                logger.error("❌ ForecasterAgent: No forecast returned")
                
        except Exception as e:
            results["ForecasterAgent"] = False
            logger.error(f"❌ ForecasterAgent: Exception - {e}")
        
        # Test StrategyAgent
        try:
            strategy = StrategyAgent()
            strategy.bus = self.bus
            
            # Setup required messages
            feature_message = {
                "prices": mock_data["prices"].to_dict(orient="records"),
                "features": [],  # Add empty features list - would be populated by feature engineering in real usage
                "inventory": mock_data["inventory"],
                "source": "TestDataAgent",
            }
            
            forecast_message = {
                "forecast": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "predicted_price": 3.5,
                        "lower_bound": 3.0,
                        "upper_bound": 4.0,
                        "method": "test"
                    }
                ],
                "source": "TestForecasterAgent",
            }
            
            # Process messages
            strategy.handle_message(feature_message)
            strategy_result = strategy.handle_message(forecast_message)
            
            if strategy_result and "action" in strategy_result:
                action = strategy_result["action"]
                required_keys = ["energy_allocation", "hash_allocation", "battery_charge_rate", "method"]
                if all(key in action for key in required_keys):
                    # Check allocation constraints
                    energy_valid = 0.0 <= action["energy_allocation"] <= 1.0
                    hash_valid = 0.0 <= action["hash_allocation"] <= 1.0
                    battery_valid = -1.0 <= action["battery_charge_rate"] <= 1.0
                    
                    if energy_valid and hash_valid and battery_valid:
                        results["StrategyAgent"] = True
                        logger.info("✅ StrategyAgent: PASS")
                    else:
                        results["StrategyAgent"] = False
                        logger.error("❌ StrategyAgent: Invalid allocation values")
                else:
                    results["StrategyAgent"] = False
                    logger.error("❌ StrategyAgent: Missing required action fields")
            else:
                results["StrategyAgent"] = False
                logger.error("❌ StrategyAgent: No action returned")
                
        except Exception as e:
            results["StrategyAgent"] = False
            logger.error(f"❌ StrategyAgent: Exception - {e}")
        
        # Test LocalLLMAgent
        try:
            llm = LocalLLMAgent()
            llm.bus = self.bus
            
            test_strategy_message = {
                "action": {
                    "energy_allocation": 0.6,
                    "hash_allocation": 0.4,
                    "battery_charge_rate": 0.2,
                    "method": "test"
                },
                "source": "TestStrategyAgent",
                "timestamp": datetime.now().isoformat()
            }
            
            llm_result = llm.handle_message(test_strategy_message)
            
            if llm_result and "analysis" in llm_result:
                analysis = llm_result["analysis"]
                required_fields = ["summary", "recommendations", "risk_assessment", "method"]
                if all(field in analysis for field in required_fields):
                    if analysis["risk_assessment"] in ["Low", "Medium", "High"]:
                        results["LocalLLMAgent"] = True
                        logger.info("✅ LocalLLMAgent: PASS")
                    else:
                        results["LocalLLMAgent"] = False
                        logger.error("❌ LocalLLMAgent: Invalid risk assessment")
                else:
                    results["LocalLLMAgent"] = False
                    logger.error("❌ LocalLLMAgent: Missing analysis fields")
            else:
                results["LocalLLMAgent"] = False
                logger.error("❌ LocalLLMAgent: No analysis returned")
                
        except Exception as e:
            results["LocalLLMAgent"] = False
            logger.error(f"❌ LocalLLMAgent: Exception - {e}")
        
        # Test VectorStoreAgent
        try:
            vector_store = VectorStoreAgent(persist_directory="test_vector_store")
            vector_store.bus = self.bus
            
            test_message = {
                "action": {
                    "energy_allocation": 0.5,
                    "hash_allocation": 0.5,
                    "battery_charge_rate": 0.0,
                },
                "source": "TestAgent",
                "timestamp": datetime.now().isoformat()
            }
            
            # Test storage
            store_result = vector_store.handle_message(test_message)
            
            # Test search
            search_message = {
                "query_type": "knowledge_search",
                "query": "strategy allocation",
                "source": "TestQuery"
            }
            
            search_result = vector_store.handle_message(search_message)
            
            if search_result and "results" in search_result:
                results["VectorStoreAgent"] = True
                logger.info("✅ VectorStoreAgent: PASS")
            else:
                results["VectorStoreAgent"] = False
                logger.error("❌ VectorStoreAgent: Search failed")
                
        except Exception as e:
            results["VectorStoreAgent"] = False
            logger.error(f"❌ VectorStoreAgent: Exception - {e}")
        
        return results
    
    def test_message_bus(self) -> bool:
        """Test message bus functionality."""
        logger.info("🧪 Testing message bus...")
        
        try:
            # Test basic publish/consume
            test_message = {
                "test_id": "bus_test_1",
                "timestamp": datetime.now().isoformat(),
                "data": {"value": 42}
            }
            
            self.bus.publish("test-topic", test_message)
            
            # Try to consume
            consumer = self.bus.consume("test-topic", block_ms=1000)
            try:
                received = next(consumer)
                if received.get("test_id") == "bus_test_1":
                    logger.info("✅ Message Bus: PASS")
                    return True
                else:
                    logger.error("❌ Message Bus: Message content mismatch")
                    return False
            except StopIteration:
                logger.error("❌ Message Bus: No message received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Message Bus: Exception - {e}")
            return False
    
    @patch('api_client.client.get_prices')
    @patch('api_client.client.get_inventory')
    def test_data_agent_with_mocks(self, mock_inventory, mock_prices) -> bool:
        """Test DataAgent with mocked API calls."""
        logger.info("🧪 Testing DataAgent with mocked API...")
        
        try:
            # Setup mocks
            mock_data = self.generate_mock_data()
            mock_prices.return_value = mock_data["prices"]
            mock_inventory.return_value = mock_data["inventory"]
            
            # Create and test data agent
            data_agent = DataAgent(fetch_interval=1)
            data_agent.bus = self.bus
            
            # Test data fetching
            data_agent._fetch_and_publish()
            
            # Verify API was called
            mock_prices.assert_called_once()
            mock_inventory.assert_called_once()
            
            # Check message was published
            consumer = self.bus.consume("feature-vector", block_ms=100)
            try:
                message = next(consumer)
                if ("prices" in message and "features" in message and 
                    "inventory" in message and message["source"] == "DataAgent"):
                    logger.info("✅ DataAgent: PASS")
                    return True
                else:
                    logger.error("❌ DataAgent: Invalid message structure")
                    return False
            except StopIteration:
                logger.error("❌ DataAgent: No message published")
                return False
                
        except Exception as e:
            logger.error(f"❌ DataAgent: Exception - {e}")
            return False
    
    def test_end_to_end_flow(self) -> bool:
        """Test complete data flow through agent pipeline."""
        logger.info("🧪 Testing end-to-end flow...")
        
        try:
            # Generate test data
            mock_data = self.generate_mock_data(48)
            
            # Initialize agents
            forecaster = ForecasterAgent()
            forecaster.bus = self.bus
            
            strategy = StrategyAgent()
            strategy.bus = self.bus
            
            llm = LocalLLMAgent()
            llm.bus = self.bus
            
            vector_store = VectorStoreAgent(persist_directory="test_e2e_vector_store")
            vector_store.bus = self.bus
            
            # Step 1: Simulate data agent output
            feature_message = {
                "prices": mock_data["prices"].to_dict(orient="records"),
                "features": [],  # Add empty features list - would be populated by feature engineering in real usage
                "inventory": mock_data["inventory"],
                "source": "DataAgent",
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 2: Forecaster processes data
            forecast_result = forecaster.handle_message(feature_message)
            if not forecast_result:
                logger.error("❌ E2E: Forecaster failed")
                return False
            
            # Step 3: Strategy processes data and forecast
            strategy.handle_message(feature_message)
            strategy_result = strategy.handle_message(forecast_result)
            if not strategy_result:
                logger.error("❌ E2E: Strategy failed")
                return False
            
            # Step 4: LLM analyzes strategy
            llm_result = llm.handle_message(strategy_result)
            if not llm_result:
                logger.error("❌ E2E: LLM analysis failed")
                return False
            
            # Step 5: Vector store saves all
            vector_store.handle_message(feature_message)
            vector_store.handle_message(forecast_result)
            vector_store.handle_message(strategy_result)
            vector_store.handle_message(llm_result)
            
            # Verify final outputs
            if ("action" in strategy_result and 
                "analysis" in llm_result and
                "forecast" in forecast_result):
                logger.info("✅ End-to-End Flow: PASS")
                return True
            else:
                logger.error("❌ E2E: Missing expected outputs")
                return False
                
        except Exception as e:
            logger.error(f"❌ End-to-End Flow: Exception - {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_performance(self) -> Dict[str, float]:
        """Test agent performance metrics."""
        logger.info("🧪 Testing performance...")
        
        performance_results = {}
        
        # Test forecaster performance with different data sizes
        forecaster = ForecasterAgent()
        forecaster.bus = self.bus
        
        data_sizes = [24, 48, 168]  # 1 day, 2 days, 1 week
        
        for size in data_sizes:
            mock_data = self.generate_mock_data(size)
            message = {
                "prices": mock_data["prices"].to_dict(orient="records"),
                "source": "TestAgent"
            }
            
            start_time = time.time()
            result = forecaster.handle_message(message)
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_results[f"forecaster_{size}h"] = processing_time
            
            logger.info(f"Forecaster ({size}h): {processing_time:.3f}s")
        
        return performance_results
    
    def test_error_handling(self) -> Dict[str, bool]:
        """Test error handling and robustness."""
        logger.info("🧪 Testing error handling...")
        
        error_results = {}
        
        agents = {
            "ForecasterAgent": ForecasterAgent(),
            "StrategyAgent": StrategyAgent(),
            "LocalLLMAgent": LocalLLMAgent(),
            "VectorStoreAgent": VectorStoreAgent(persist_directory="test_error_vector_store")
        }
        
        for agent_name, agent in agents.items():
            agent.bus = self.bus
            
            # Test malformed messages
            malformed_messages = [
                {},  # Empty
                {"invalid": "data"},  # Wrong structure
                {"prices": []},  # Empty data
                {"prices": "not_a_list"},  # Wrong type
            ]
            
            passed_tests = 0
            total_tests = len(malformed_messages)
            
            for i, message in enumerate(malformed_messages):
                try:
                    result = agent.handle_message(message)
                    # Should not crash - result can be None or valid dict
                    if result is None or isinstance(result, dict):
                        passed_tests += 1
                except Exception as e:
                    logger.warning(f"{agent_name} failed on malformed message {i}: {e}")
            
            error_results[agent_name] = passed_tests == total_tests
            
            if error_results[agent_name]:
                logger.info(f"✅ {agent_name} Error Handling: PASS")
            else:
                logger.error(f"❌ {agent_name} Error Handling: FAIL")
        
        return error_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite."""
        logger.info("🚀 Starting comprehensive agent system tests...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "message_bus": None,
            "data_agent": None,
            "individual_agents": {},
            "end_to_end": None,
            "performance": {},
            "error_handling": {},
            "overall_pass": False
        }
        
        # Test message bus
        results["message_bus"] = self.test_message_bus()
        
        # Test data agent with mocks
        results["data_agent"] = self.test_data_agent_with_mocks()
        
        # Test individual agents
        results["individual_agents"] = self.test_individual_agents()
        
        # Test end-to-end flow
        results["end_to_end"] = self.test_end_to_end_flow()
        
        # Test performance
        results["performance"] = self.test_performance()
        
        # Test error handling
        results["error_handling"] = self.test_error_handling()
        
        # Calculate overall pass/fail
        individual_pass = all(results["individual_agents"].values())
        error_handling_pass = all(results["error_handling"].values())
        
        results["overall_pass"] = (
            results["message_bus"] and
            results["data_agent"] and
            individual_pass and
            results["end_to_end"] and
            error_handling_pass
        )
        
        results["end_time"] = datetime.now().isoformat()
        results["total_duration"] = time.time() - self.start_time
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 60)
        report.append("GridPilot-GT Agent System Test Report")
        report.append("=" * 60)
        report.append(f"Test Start: {results['start_time']}")
        report.append(f"Test End: {results['end_time']}")
        report.append(f"Total Duration: {results['total_duration']:.2f} seconds")
        report.append("")
        
        # Overall result
        status = "✅ PASS" if results["overall_pass"] else "❌ FAIL"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Message Bus
        bus_status = "✅ PASS" if results["message_bus"] else "❌ FAIL"
        report.append(f"Message Bus: {bus_status}")
        
        # Data Agent
        data_status = "✅ PASS" if results["data_agent"] else "❌ FAIL"
        report.append(f"Data Agent: {data_status}")
        report.append("")
        
        # Individual Agents
        report.append("Individual Agent Tests:")
        for agent, passed in results["individual_agents"].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"  {agent}: {status}")
        report.append("")
        
        # End-to-End
        e2e_status = "✅ PASS" if results["end_to_end"] else "❌ FAIL"
        report.append(f"End-to-End Flow: {e2e_status}")
        report.append("")
        
        # Performance
        report.append("Performance Metrics:")
        for test, duration in results["performance"].items():
            report.append(f"  {test}: {duration:.3f}s")
        report.append("")
        
        # Error Handling
        report.append("Error Handling Tests:")
        for agent, passed in results["error_handling"].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"  {agent}: {status}")
        report.append("")
        
        # Summary
        total_tests = (
            1 +  # Message bus
            1 +  # Data agent
            len(results["individual_agents"]) +
            1 +  # End-to-end
            len(results["error_handling"])
        )
        
        passed_tests = (
            (1 if results["message_bus"] else 0) +
            (1 if results["data_agent"] else 0) +
            sum(results["individual_agents"].values()) +
            (1 if results["end_to_end"] else 0) +
            sum(results["error_handling"].values())
        )
        
        report.append(f"Test Summary: {passed_tests}/{total_tests} tests passed")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main test function."""
    tester = AgentSystemTester()
    
    try:
        results = tester.run_all_tests()
        report = tester.generate_report(results)
        
        print(report)
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open("test_report.txt", "w") as f:
            f.write(report)
        
        logger.info("📝 Test results saved to test_results.json and test_report.txt")
        
        # Exit with appropriate code
        exit_code = 0 if results["overall_pass"] else 1
        return exit_code
        
    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 