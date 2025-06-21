"""
Mock LLM Interface for Testing
Provides simulated responses to test the LLM integration functionality
without requiring the actual llama-cpp-python installation.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLM:
    """Mock LLM class that simulates llama-cpp-python functionality for testing."""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        logger.info(f"Mock LLM initialized with model path: {model_path}")
    
    def create_completion(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs):
        """Mock completion that returns simulated responses."""
        # Extract the actual query from the prompt (after "Query: ")
        query_start = prompt.find("Query: ")
        if query_start != -1:
            query = prompt[query_start + 7:].strip()
        else:
            query = prompt
        
        # Generate appropriate response based on query content
        query_lower = query.lower()
        
        if "demand response" in query_lower:
            response = "Demand response programs help reduce peak electricity usage by shifting consumption to off-peak hours. This lowers costs, improves grid stability, and reduces the need for expensive peaking power plants."
        elif "battery" in query_lower or "soc" in query_lower:
            response = "Battery management involves monitoring state of charge, optimizing charge and discharge cycles, and maintaining battery health. Proper management extends battery life and maximizes energy storage efficiency."
        elif "optimize" in query_lower or "cost" in query_lower:
            response = "Energy cost optimization includes load shifting to off-peak hours, demand response participation, and efficient equipment operation. This can reduce electricity bills by 10 to 30 percent."
        elif "risk" in query_lower:
            response = "Current energy usage patterns show several risks: high peak demand during business hours increases costs, limited flexibility to respond to price signals, and dependence on grid power without backup options."
        elif "explain" in query_lower:
            response = "This is a simulated explanation of the decision-making process. In a real scenario, the model would analyze the energy data, consider factors like demand patterns, grid conditions, and optimization goals to provide detailed reasoning for the recommended actions."
        else:
            response = "This is a simulated response from the Claude API. The model would analyze the provided context and generate relevant insights, explanations, or recommendations based on the energy management scenario."
        
        return {
            "choices": [{
                "text": response,
                "finish_reason": "length"
            }]
        }


class MockLLMInterface:
    """
    Mock interface for testing LLM functionality without requiring llama-cpp-python.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the mock LLM interface.
        
        Args:
            model_path: Path to the model file (for compatibility, not used)
        """
        self.model = MockLLM("mock-model")
        logger.info("Mock LLM interface initialized for testing")
    
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        return self.model is not None
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user query with optional context.
        
        Args:
            query: User's question or request
            context: Optional context data (energy data, system state, etc.)
            
        Returns:
            Model's response as a string
        """
        if not self.is_available():
            return "Model not available. Please check the model installation."
        
        try:
            # Build the prompt with context
            prompt = self._build_prompt(query, context)
            
            # Get response from model
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["</s>", "Human:", "Assistant:"]
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    def generate_insights(self, energy_data: pd.DataFrame, system_state: Optional[Dict] = None) -> str:
        """
        Generate insights from energy data.
        
        Args:
            energy_data: DataFrame containing energy consumption/production data
            system_state: Optional current system state information
            
        Returns:
            Generated insights as a string
        """
        if energy_data.empty:
            return "No energy data provided for analysis."
        
        # Create context from energy data
        context = {
            "data_summary": {
                "total_consumption": energy_data.get('consumption', pd.Series()).sum() if 'consumption' in energy_data.columns else "N/A",
                "peak_demand": energy_data.get('demand', pd.Series()).max() if 'demand' in energy_data.columns else "N/A",
                "data_points": len(energy_data)
            },
            "system_state": system_state or {}
        }
        
        query = "Analyze this energy data and provide key insights about consumption patterns, efficiency opportunities, and optimization recommendations."
        
        return self.process_query(query, context)
    
    def explain_decision(self, decision: str, context: Dict[str, Any]) -> str:
        """
        Explain a decision made by the energy management system.
        
        Args:
            decision: The decision that was made
            context: Context information about the decision
            
        Returns:
            Explanation of the decision
        """
        query = f"Explain why the following decision was made: {decision}"
        
        return self.process_query(query, context)
    
    def _build_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a prompt for the model with context.
        
        Args:
            query: User's query
            context: Optional context data
            
        Returns:
            Formatted prompt string
        """
        prompt = "<s>[INST] You are an AI assistant specialized in energy management and optimization. "
        prompt += "You help analyze energy data, provide insights, and explain decisions related to energy systems. "
        prompt += "Be concise, practical, and focus on actionable recommendations.\n\n"
        
        if context:
            prompt += "Context:\n"
            if isinstance(context, dict):
                for key, value in context.items():
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        prompt += f"Query: {query}\n\n"
        prompt += "Please provide a helpful response based on the context and query. [/INST]"
        
        return prompt


def test_mock_interface():
    """Test the mock LLM interface with sample data."""
    print("Testing Mock LLM Interface...")
    
    # Initialize interface
    interface = MockLLMInterface()
    print(f"Model available: {interface.is_available()}")
    
    # Test basic query
    print("\n--- Testing Basic Query ---")
    response = interface.process_query("What are the benefits of demand response in energy management?")
    print(f"Response: {response}")
    
    # Test with context
    print("\n--- Testing with Context ---")
    context = {
        "current_demand": "500 kW",
        "grid_price": "$0.15/kWh",
        "battery_charge": "80%"
    }
    response = interface.process_query("Should we discharge the battery now?", context)
    print(f"Response: {response}")
    
    # Test insights generation
    print("\n--- Testing Insights Generation ---")
    # Create sample energy data
    energy_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
        'consumption': [100 + i*10 for i in range(24)],
        'demand': [150 + i*5 for i in range(24)]
    })
    
    insights = interface.generate_insights(energy_data)
    print(f"Insights: {insights}")
    
    # Test decision explanation
    print("\n--- Testing Decision Explanation ---")
    decision_context = {
        "decision": "Discharge battery to reduce grid demand",
        "current_price": "$0.20/kWh",
        "battery_efficiency": "95%",
        "demand_reduction": "50 kW"
    }
    
    explanation = interface.explain_decision("Discharge battery", decision_context)
    print(f"Explanation: {explanation}")
    
    # Test different query types
    print("\n--- Testing Different Query Types ---")
    queries = [
        "What are the current energy prices?",
        "How can we optimize our energy consumption?",
        "Explain the battery management strategy",
        "What are the risks of current energy usage patterns?"
    ]
    
    for query in queries:
        response = interface.process_query(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")


if __name__ == "__main__":
    test_mock_interface() 