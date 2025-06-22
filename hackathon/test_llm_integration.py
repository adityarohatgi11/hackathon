#!/usr/bin/env python3
"""
Test script to verify LLM integration functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_integration.unified_interface import create_llm_interface
import json

def test_llm_integration():
    """Test the LLM integration functionality."""
    print("=" * 60)
    print("LLM Integration Test")
    print("=" * 60)
    
    try:
        # Create LLM interface
        print("1. Creating LLM interface...")
        llm_interface = create_llm_interface()
        print(f"   ✓ Interface created: {type(llm_interface).__name__}")
        
        # Check service availability
        print("2. Checking service availability...")
        is_available = llm_interface.is_service_available()
        print(f"   ✓ Service available: {is_available}")
        
        # Get model info
        print("3. Getting model information...")
        model_info = llm_interface.get_model_info()
        print(f"   ✓ Model info: {json.dumps(model_info, indent=2)}")
        
        # Test basic insight generation
        print("4. Testing basic insight generation...")
        test_prompt = "Analyze energy trading performance and provide 3 key insights."
        insights = llm_interface.generate_insights(test_prompt)
        print(f"   ✓ Generated insights ({len(insights)} characters)")
        print(f"   Preview: {insights[:200]}...")
        
        # Test Q-learning analysis
        print("5. Testing Q-learning analysis...")
        qlearning_data = {
            'Episodes': 100,
            'Best Reward': 28.5,
            'Average Reward': 21.3,
            'Convergence': 'Achieved'
        }
        qlearning_prompt = f"""
        Analyze these Q-Learning training results:
        {qlearning_data}
        
        Provide performance assessment and recommendations.
        """
        qlearning_insights = llm_interface.generate_insights(qlearning_prompt)
        print(f"   ✓ Q-learning analysis generated ({len(qlearning_insights)} characters)")
        print(f"   Preview: {qlearning_insights[:200]}...")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - LLM Integration Working!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting LLM Integration Tests...\n")
    
    # Run basic tests
    basic_success = test_llm_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic Integration: {'✅ PASS' if basic_success else '❌ FAIL'}")
    
    if basic_success:
        print("\n🎉 LLM integration is working correctly!")
        print("\nNext steps:")
        print("1. Open the Streamlit dashboard at http://localhost:8507")
        print("2. Navigate to the 'AI Insights' tab")
        print("3. Use the 'Test LLM Integration' button")
        print("4. Try the AI analysis buttons in other tabs")
    else:
        print("\n⚠️  Test failed. Check the error messages above.")
        
    sys.exit(0 if basic_success else 1)
