#!/usr/bin/env python3
"""
Test Claude API Integration
Demonstrates how to use Claude API as an alternative to local LLM.
"""

import os
import sys
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_claude_setup():
    """Test Claude API setup and configuration."""
    print("🔧 Claude API Integration Test")
    print("=" * 50)
    
    # Check if anthropic package is available
    try:
        import anthropic
        print("✅ anthropic package is available")
    except ImportError:
        print("❌ anthropic package not available")
        print("Install with: pip install anthropic")
        return False
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ ANTHROPIC_API_KEY environment variable is set")
        print(f"   Key starts with: {api_key[:10]}...")
    else:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return False
    
    return True

def test_claude_interface():
    """Test the Claude interface functionality."""
    print("\n🤖 Testing Claude Interface...")
    
    try:
        from llm_integration.claude_interface import ClaudeInterface
        
        interface = ClaudeInterface()
        
        if interface.is_available():
            print("✅ Claude API interface initialized successfully")
            
            # Test a simple query
            print("\n📝 Testing query processing...")
            start_time = time.time()
            response = interface.process_query("What is energy efficiency?")
            response_time = time.time() - start_time
            
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"💬 Response: {response}")
            
            return True
        else:
            print("❌ Claude API interface not available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Claude interface: {e}")
        return False

def test_unified_interface():
    """Test the unified interface with Claude preference."""
    print("\n🔄 Testing Unified Interface with Claude...")
    
    try:
        from llm_integration.unified_interface import UnifiedLLMInterface
        
        # Test automatic selection
        print("1️⃣ Testing automatic provider selection...")
        interface = UnifiedLLMInterface()
        
        if interface.is_available():
            provider_info = interface.get_provider_info()
            print(f"✅ Selected provider: {provider_info['provider']}")
            
            # Test query
            response = interface.process_query("Explain demand response")
            print(f"💬 Response: {response[:100]}...")
            
            return True
        else:
            print("❌ No LLM providers available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing unified interface: {e}")
        return False

def test_provider_comparison():
    """Compare different LLM providers."""
    print("\n⚖️ Comparing LLM Providers...")
    
    try:
        from llm_integration.unified_interface import UnifiedLLMInterface
        
        providers = ['claude', 'local', 'mock']
        results = {}
        
        for provider in providers:
            print(f"\n🔍 Testing {provider} provider...")
            try:
                interface = UnifiedLLMInterface(preferred_provider=provider)
                if interface.is_available():
                    provider_info = interface.get_provider_info()
                    print(f"✅ {provider} provider available")
                    
                    # Test response time
                    start_time = time.time()
                    response = interface.process_query("What is peak shaving?")
                    response_time = time.time() - start_time
                    
                    results[provider] = {
                        'available': True,
                        'response_time': response_time,
                        'response_length': len(response)
                    }
                    print(f"   Response time: {response_time:.2f}s")
                    print(f"   Response length: {len(response)} chars")
                else:
                    print(f"❌ {provider} provider not available")
                    results[provider] = {'available': False}
                    
            except Exception as e:
                print(f"❌ Error with {provider} provider: {e}")
                results[provider] = {'available': False, 'error': str(e)}
        
        # Summary
        print("\n📊 Provider Comparison Summary:")
        for provider, result in results.items():
            if result.get('available'):
                print(f"   {provider}: ✅ {result['response_time']:.2f}s")
            else:
                print(f"   {provider}: ❌ Not available")
        
        return results
        
    except Exception as e:
        print(f"❌ Error comparing providers: {e}")
        return {}

def main():
    """Run all Claude integration tests."""
    print("🚀 Claude API Integration Test Suite")
    print("=" * 60)
    
    # Test setup
    if not test_claude_setup():
        print("\n❌ Claude API setup failed. Please check your configuration.")
        return
    
    # Test Claude interface
    if not test_claude_interface():
        print("\n❌ Claude interface test failed.")
        return
    
    # Test unified interface
    if not test_unified_interface():
        print("\n❌ Unified interface test failed.")
        return
    
    # Compare providers
    results = test_provider_comparison()
    
    print("\n✅ Claude API Integration Test Completed!")
    print("\n💡 Next steps:")
    print("   1. Set your ANTHROPIC_API_KEY environment variable")
    print("   2. Run the dashboard: python main.py --mode dashboard")
    print("   3. The system will automatically use Claude API when available")

if __name__ == "__main__":
    main() 