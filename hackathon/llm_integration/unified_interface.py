"""
Unified LLM Interface for Energy Management System
Automatically chooses between Claude API and mock interface based on availability.
"""

import logging
import os
from typing import Optional, Dict, Any, List
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Claude interface
try:
    from .claude_interface import ClaudeInterface
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Claude interface not available")

# Import mock interface as fallback
try:
    from .mock_interface import MockLLMInterface
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False
    logger.warning("Mock interface not available")


class UnifiedLLMInterface:
    """
    Unified LLM Interface that automatically selects the best available LLM provider.
    
    Priority order:
    1. Claude API (if API key available)
    2. Mock interface (fallback)
    """
    
    def __init__(self, preferred_provider: Optional[str] = None):
        """
        Initialize the unified LLM interface.
        
        Args:
            preferred_provider: Force specific provider ('claude', 'mock')
        """
        self.provider = None
        self.interface = None
        self.preferred_provider = preferred_provider
        
        self._initialize_interface()
    
    def _initialize_interface(self):
        """Initialize the best available LLM interface."""
        
        # Check if preferred provider is specified
        if self.preferred_provider:
            if self.preferred_provider == 'claude' and CLAUDE_AVAILABLE:
                self._try_claude()
            elif self.preferred_provider == 'mock' and MOCK_AVAILABLE:
                self._try_mock()
            else:
                logger.warning(f"Preferred provider '{self.preferred_provider}' not available, trying alternatives")
                self._auto_select_provider()
        else:
            self._auto_select_provider()
    
    def _auto_select_provider(self):
        """Automatically select the best available provider."""
        
        # Try Claude API first (fastest and most capable)
        if self._try_claude():
            return
        
        # Fall back to mock interface
        if self._try_mock():
            return
        
        # No providers available
        logger.error("No LLM providers available")
        self.provider = "none"
        self.interface = None
    
    def _try_claude(self) -> bool:
        """Try to initialize Claude API interface."""
        try:
            claude_interface = ClaudeInterface()
            if claude_interface.is_available():
                self.interface = claude_interface
                self.provider = "claude"
                logger.info("✅ Using Claude API for LLM functionality")
                return True
        except Exception as e:
            logger.debug(f"Claude API not available: {e}")
        return False
    
    def _try_mock(self) -> bool:
        """Try to initialize mock interface."""
        try:
            mock_interface = MockLLMInterface()
            self.interface = mock_interface
            self.provider = "mock"
            logger.info("✅ Using mock interface for LLM functionality")
            return True
        except Exception as e:
            logger.debug(f"Mock interface not available: {e}")
        return False
    
    def is_available(self) -> bool:
        """Check if any LLM interface is available."""
        return self.interface is not None and self.provider != "none"
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        if not self.is_available():
            return {"provider": "none", "status": "unavailable"}
        
        info = {
            "provider": self.provider,
            "status": "available"
        }
        
        # Add provider-specific info
        if self.provider == "claude":
            info.update(self.interface.get_usage_info())
        elif self.provider == "mock":
            info["model"] = "mock-interface"
            info["backend"] = "simulated"
        
        return info
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using the selected LLM provider.
        
        Args:
            query: User's question or request
            
        Returns:
            Generated response from the LLM
        """
        if not self.is_available():
            return "Sorry, no LLM providers are available. Please check your configuration."
        
        try:
            start_time = time.time()
            response = self.interface.process_query(query)
            response_time = time.time() - start_time
            
            logger.debug(f"Query processed by {self.provider} in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query with {self.provider}: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def generate_insights(self, data: pd.DataFrame) -> str:
        """
        Generate insights from energy data.
        
        Args:
            data: DataFrame containing energy consumption, prices, etc.
            
        Returns:
            Generated insights about the energy data
        """
        if not self.is_available():
            return "No LLM providers available for insights generation."
        
        try:
            start_time = time.time()
            insights = self.interface.generate_insights(data)
            response_time = time.time() - start_time
            
            logger.debug(f"Insights generated by {self.provider} in {response_time:.2f}s")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights with {self.provider}: {e}")
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
            return "No LLM providers available for decision explanation."
        
        try:
            start_time = time.time()
            explanation = self.interface.explain_decision(decision, context)
            response_time = time.time() - start_time
            
            logger.debug(f"Decision explained by {self.provider} in {response_time:.2f}s")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining decision with {self.provider}: {e}")
            return f"Error explaining decision: {str(e)}"


def test_unified_interface():
    """Test the unified LLM interface."""
    print("🧪 Testing Unified LLM Interface")
    print("=" * 50)
    
    # Test automatic provider selection
    print("1️⃣ Testing automatic provider selection...")
    interface = UnifiedLLMInterface()
    
    if interface.is_available():
        provider_info = interface.get_provider_info()
        print(f"✅ Using provider: {provider_info['provider']}")
        print(f"   Status: {provider_info['status']}")
        if 'model' in provider_info:
            print(f"   Model: {provider_info['model']}")
    else:
        print("❌ No LLM providers available")
        return
    
    # Test query processing
    print("\n2️⃣ Testing query processing...")
    test_queries = [
        "What is demand response?",
        "How to optimize energy costs?",
        "Explain battery management"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = interface.process_query(query)
        print(f"Response: {response[:100]}...")
    
    # Test provider-specific selection
    print("\n3️⃣ Testing provider-specific selection...")
    for provider in ['claude', 'mock']:
        print(f"\nTrying {provider} provider...")
        try:
            specific_interface = UnifiedLLMInterface(preferred_provider=provider)
            if specific_interface.is_available():
                provider_info = specific_interface.get_provider_info()
                print(f"✅ {provider} provider available: {provider_info['provider']}")
            else:
                print(f"❌ {provider} provider not available")
        except Exception as e:
            print(f"❌ Error with {provider} provider: {e}")
    
    print("\n✅ Unified LLM Interface test completed!")


if __name__ == "__main__":
    test_unified_interface() 