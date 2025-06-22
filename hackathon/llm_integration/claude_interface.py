"""
Claude API Integration for Energy Management System
Provides chat interface and decision explanation capabilities using Anthropic's Claude API.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import anthropic, but provide fallback if not available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available. Install with: pip install anthropic")


class ClaudeInterface:
    """
    Claude API Interface for Energy Management System
    
    Provides natural language processing capabilities for:
    - Query processing and response generation
    - Energy data insights generation
    - Decision explanation and reasoning
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the Claude API interface.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-3-haiku-20240307 for speed)
        """
        self.client = None
        self.model = model
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self._initialize_client()
        else:
            logger.warning("Claude API not available. Check API key and anthropic package installation.")
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Successfully initialized Claude API client with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Claude API client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if the Claude API is available and configured."""
        return self.client is not None
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            query: User's question or request
            
        Returns:
            Generated response from Claude
        """
        if not self.is_available():
            return "Sorry, the Claude API is not available. Please check your API key and internet connection."
        
        try:
            # Create a system prompt for energy management context
            system_prompt = """You are an energy management assistant. Provide brief, actionable advice in plain text only. 
            
            IMPORTANT: 
            - Use only plain text, no mathematical notation, LaTeX, or special formatting
            - Keep responses under 2-3 sentences
            - Focus on key points only
            - Use simple language that doesn't require special rendering
            - Avoid symbols like %, $, or mathematical expressions that might trigger formatting
            - Write percentages as "75 percent" instead of "75%"
            - Write prices as "15 cents per kilowatt hour" instead of "$0.15/kWh"
            """
            
            # Generate response using Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,  # Keep responses concise
                temperature=0.3,  # Low temperature for focused responses
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            
            # Extract the response text
            if response and hasattr(response, 'content') and len(response.content) > 0:
                return response.content[0].text.strip()
            else:
                return "Sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing query with Claude: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def generate_insights(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate insights from energy data.
        
        Args:
            data: DataFrame containing energy consumption, prices, etc.
            context: Optional context information for the analysis
            
        Returns:
            Generated insights about the energy data
        """
        if not self.is_available():
            return "Claude API not available for insights generation."
        
        try:
            # Handle string prompts directly
            if isinstance(data, str):
                return self.process_query(data)
            
            # Create a summary of the data using plain text
            total_consumption = data['consumption'].sum()
            avg_consumption = data['consumption'].mean()
            peak_demand = data['demand'].max()
            avg_price = data['price'].mean()
            min_price = data['price'].min()
            max_price = data['price'].max()
            current_soc = data['battery_soc'].iloc[-1]
            
            data_summary = f"""
            Energy Data Summary:
            - Total consumption: {total_consumption:.0f} kilowatt hours
            - Average consumption: {avg_consumption:.1f} kilowatts
            - Peak demand: {peak_demand:.0f} kilowatts
            - Average price: {avg_price:.3f} dollars per kilowatt hour
            - Price range: {min_price:.3f} to {max_price:.3f} dollars per kilowatt hour
            - Current battery state of charge: {current_soc:.1%}
            """
            
            # Add context information if provided
            if context:
                context_str = "\n".join([f"- {key}: {value}" for key, value in context.items()])
                data_summary += f"\n\nAdditional Context:\n{context_str}"
            
            query = f"Briefly analyze this energy data and provide 2-3 key insights in plain text only: {data_summary}"
            
            return self.process_query(query)
            
        except Exception as e:
            logger.error(f"Error generating insights with Claude: {e}")
            return f"Error generating insights: {str(e)}"
    
    def explain_decision(self, decision: str, context: Dict[str, Any]) -> str:
        """
        Explain a decision made by the energy management system.
        
        Args:
            decision: The decision that was made
            context: Context information about the decision
            
        Returns:
            Explanation of the decision
        """
        if not self.is_available():
            return "Claude API not available for decision explanation."
        
        try:
            # Format context for the query using plain text
            context_str = "\n".join([f"- {key}: {value}" for key, value in context.items()])
            
            query = f"Briefly explain this energy decision in 1-2 sentences using plain text only: {decision}\nContext:\n{context_str}"
            
            return self.process_query(query)
            
        except Exception as e:
            logger.error(f"Error explaining decision with Claude: {e}")
            return f"Error explaining decision: {str(e)}"
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get information about API usage and costs.
        
        Returns:
            Dictionary with usage information
        """
        if not self.is_available():
            return {"error": "Claude API not available"}
        
        try:
            # Note: Anthropic doesn't provide usage info in the same way as OpenAI
            # This is a placeholder for future implementation
            return {
                "model": self.model,
                "status": "active",
                "note": "Usage tracking not available in current API version"
            }
        except Exception as e:
            return {"error": str(e)}


def test_claude_interface():
    """Test the Claude API interface with sample queries."""
    print("Testing Claude API Interface...")
    
    interface = ClaudeInterface()
    
    if interface.is_available():
        print("✅ Claude API connected successfully")
    else:
        print("❌ Claude API not available")
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    # Test queries
    test_queries = [
        "What are the benefits of demand response?",
        "Explain the battery management strategy",
        "How can I optimize energy costs?",
        "What are the current energy prices?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        response = interface.process_query(query)
        response_time = time.time() - start_time
        print(f"Response ({response_time:.2f}s): {response}")


if __name__ == "__main__":
    test_claude_interface() 